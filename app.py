"""
Production Scheduling Demo – Flask Backend v3
Supports:
  - CSV dataset loading with weekly grouping (Exit Factory Year & Week)
  - Advanced constraints (Rear Loader 2512, PKRML, PKRRLSB, DC REFUSE STOCK)
  - Full demand coverage (all allocated models must be arranged)
  - Model balance (alpha) and line balance (beta) objectives
  - AI Agent Chat powered by Azure OpenAI with daily token budget control
"""

from flask import Flask, request, jsonify, send_from_directory
from ortools.sat.python import cp_model
import os, json, math, time, threading
from datetime import date
import pandas as pd

app = Flask(__name__, static_folder="static")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ND = len(DAYS)

# Day indices for special constraints
TUE, THU, FRI = 1, 3, 4
PREFERRED_DAYS_PKRML_PKRRLSB = [TUE, THU]
PREFERRED_DAY_DC_REFUSE = FRI

# ─────────────────────────────────────────────────────────────────────────────
# Azure OpenAI Configuration (set via /config endpoint or env vars)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ai_config.json")
TOKEN_LOG_PATH = os.path.join(os.path.dirname(__file__), "token_usage.json")

_config_lock = threading.Lock()
_token_lock = threading.Lock()

DEFAULT_CONFIG = {
    "azure_endpoint":      "",       # e.g. https://your-resource.openai.azure.com/
    "azure_api_key":       "",       # Your Azure OpenAI API key
    "azure_api_version":   "2024-12-01-preview",
    "azure_deployment":    "gpt-4o", # Deployment name
    "daily_token_limit":   100000,   # Max tokens per day (prompt + completion)
    "max_tokens_per_reply": 1000,    # Max tokens per single reply
}

def load_ai_config():
    """Load AI config from file, falling back to defaults."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            saved = json.load(f)
        cfg = {**DEFAULT_CONFIG, **saved}
    else:
        cfg = dict(DEFAULT_CONFIG)
    # Override with env vars if present
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        cfg["azure_endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        cfg["azure_api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
    if os.environ.get("AZURE_OPENAI_DEPLOYMENT"):
        cfg["azure_deployment"] = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    if os.environ.get("AZURE_OPENAI_API_VERSION"):
        cfg["azure_api_version"] = os.environ["AZURE_OPENAI_API_VERSION"]
    return cfg

def save_ai_config(cfg):
    """Persist AI config to file."""
    with _config_lock:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Token Usage Tracking
# ─────────────────────────────────────────────────────────────────────────────
def _load_token_log():
    if os.path.exists(TOKEN_LOG_PATH):
        with open(TOKEN_LOG_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_token_log(log):
    with open(TOKEN_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

def get_today_usage():
    """Return tokens used today."""
    with _token_lock:
        log = _load_token_log()
        today = date.today().isoformat()
        return log.get(today, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0})

def record_token_usage(prompt_tokens, completion_tokens):
    """Add token usage for today."""
    with _token_lock:
        log = _load_token_log()
        today = date.today().isoformat()
        if today not in log:
            log[today] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
        log[today]["prompt_tokens"] += prompt_tokens
        log[today]["completion_tokens"] += completion_tokens
        log[today]["total_tokens"] += (prompt_tokens + completion_tokens)
        log[today]["requests"] += 1
        _save_token_log(log)
    return log[today]

def check_budget(cfg):
    """Return (ok, remaining, used) for today's token budget."""
    usage = get_today_usage()
    limit = int(cfg.get("daily_token_limit", 100000))
    used = usage.get("total_tokens", 0)
    remaining = max(0, limit - used)
    return remaining > 0, remaining, used


# ─────────────────────────────────────────────────────────────────────────────
# AI Context Builder
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the AI Assistant for the **Production Scheduling Optimizer** tool.
You help users understand:
1. **The Data**: What models, lines, quantities, and constraints are loaded.
2. **The Algorithm**: How the CP-SAT (Constraint Programming - Satisfiability) solver works.
3. **How to Use the Tool**: How to select weeks, adjust parameters, and interpret results.
4. **The Constraints**: Rear Loader 2512 spacing, PKRML/PKRRLSB day preferences, DC Refuse Friday rule.

Key Technical Details:
- Solver: Google OR-Tools CP-SAT (Constraint Programming with SAT backend)
- Objective: Minimize weighted sum of model imbalance (alpha) + line imbalance (beta) + soft constraint penalties
- Hard Constraints: Full demand coverage, Rear Loader 2512 max 1/day, small model spread
- Soft Constraints: PKRML prefer Tue/Thu, PKRRLSB prefer Tue/Thu, DC Refuse prefer Friday (penalty weight configurable)
- Parameters: Alpha (model balance weight), Beta (line balance weight), Small Model Threshold, Time Limit, Penalty Weight
- Data is grouped by "Exit Factory Year" & "Exit Factory Week"

Be concise, helpful, and use plain language suitable for non-technical executives.
When discussing data, reference specific model names and numbers from the context provided.
"""

def build_context_message(solver_context):
    """Build a context message from the current app state."""
    parts = []
    if solver_context.get("week_key"):
        parts.append(f"Currently viewing: {solver_context['week_key']}")
    if solver_context.get("demand_summary"):
        parts.append(f"Demand Summary:\n{solver_context['demand_summary']}")
    if solver_context.get("solve_result"):
        r = solver_context["solve_result"]
        parts.append(f"Latest Solve Result: Status={r.get('status','N/A')}, "
                      f"Objective={r.get('objective','N/A')}, "
                      f"Total Units={r.get('total_units','N/A')}")
        if r.get("compliance"):
            comp_lines = []
            for c in r["compliance"]:
                status = "MET" if c["met"] else "NOT MET"
                comp_lines.append(f"  - {c['constraint']}: {c['unit']} → {c['scheduled_days']} [{status}]")
            parts.append("Constraint Compliance:\n" + "\n".join(comp_lines))
    if not parts:
        parts.append("No data loaded yet. User should select a week and run the optimizer first.")
    return "Current Application State:\n" + "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Data Loading
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data",
                         "daily_tactical_scheduler_v2_2_2026.csv")

def load_dataset():
    """Load the CSV and return a dict of available weeks."""
    df = pd.read_csv(DATA_PATH)
    weeks = {}
    for _, row in df.iterrows():
        year = int(row["Exit Factory Year"])
        week = int(row["Exit Factory Week"])
        key = f"{year}-W{week:02d}"
        if key not in weeks:
            weeks[key] = {
                "label": f"{year} – Week {week}",
                "year": year,
                "week": week,
                "rows": []
            }
        weeks[key]["rows"].append({
            "model":        str(row["Model"]).strip(),
            "line":         str(int(row["Line Sort"])),
            "qty":          1,
            "rl2512":       int(row.get("Rear Loader 2512", 0)),
            "pkrrlsb":      int(row.get("Rear Loader (PKRRLSB)", 0)),
            "pkrml":        int(row.get("Manual Side Loader (PKRML)", 0)),
            "dc_refuse":    int(row.get("DC REFUSE STOCK", 0)),
        })
    return weeks


# ─────────────────────────────────────────────────────────────────────────────
# Solver (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────
def solve_schedule(demand_rows, alpha=1, beta=1,
                   small_model_threshold=5, time_limit_sec=10,
                   penalty_weight=50):
    q = {}
    flags = {}
    for row in demand_rows:
        m   = str(row["model"]).strip()
        s   = str(row["line"]).strip()
        qty = int(row.get("qty", 1))
        if qty <= 0:
            continue
        key = (m, s)
        q[key] = q.get(key, 0) + qty
        if key not in flags:
            flags[key] = {"rl2512": 0, "pkrrlsb": 0, "pkrml": 0, "dc_refuse": 0}
        for f in ("rl2512", "pkrrlsb", "pkrml", "dc_refuse"):
            flags[key][f] = max(flags[key][f], int(row.get(f, 0)))

    if not q:
        raise ValueError("No valid demand rows provided.")

    models = sorted(set(m for m, s in q))
    lines  = sorted(set(s for m, s in q))
    Qm = {m: sum(q.get((m, s), 0) for s in lines) for m in models}
    Qs = {s: sum(q.get((m, s), 0) for m in models) for s in lines}
    total_units = sum(Qm.values())

    cp = cp_model.CpModel()
    x = {}
    for (m, s), q_ms in q.items():
        for d in range(ND):
            x[(m, s, d)] = cp.new_int_var(0, q_ms, f"x[{m},{s},{d}]")

    y = {}
    for m in models:
        for d in range(ND):
            terms = [x[(m, s, d)] for s in lines if (m, s, d) in x]
            y[(m, d)] = cp.new_int_var(0, Qm[m], f"y[{m},{d}]")
            cp.add(y[(m, d)] == sum(terms)) if terms else cp.add(y[(m, d)] == 0)

    z = {}
    for s in lines:
        for d in range(ND):
            terms = [x[(m, s, d)] for m in models if (m, s, d) in x]
            z[(s, d)] = cp.new_int_var(0, Qs[s], f"z[{s},{d}]")
            cp.add(z[(s, d)] == sum(terms)) if terms else cp.add(z[(s, d)] == 0)

    for (m, s), q_ms in q.items():
        cp.add(sum(x[(m, s, d)] for d in range(ND)) == q_ms)

    for m in models:
        if Qm[m] < small_model_threshold:
            lo = Qm[m] // ND
            hi = math.ceil(Qm[m] / ND)
            for d in range(ND):
                cp.add(y[(m, d)] >= lo)
                cp.add(y[(m, d)] <= hi)

    rl2512_pairs = [(m, s) for (m, s), f in flags.items() if f["rl2512"] == 1]
    if rl2512_pairs:
        for d in range(ND):
            daily_rl2512 = [x[(m, s, d)] for (m, s) in rl2512_pairs if (m, s, d) in x]
            if daily_rl2512:
                cp.add(sum(daily_rl2512) <= 1)

    penalty_terms = []
    def add_preferred_days_soft(pairs, preferred_days, label):
        for (m, s) in pairs:
            q_ms = q.get((m, s), 0)
            if q_ms == 0:
                continue
            on_preferred = []
            for d in preferred_days:
                if (m, s, d) in x:
                    b = cp.new_bool_var(f"pref_{label}_{m}_{s}_{d}")
                    cp.add(x[(m, s, d)] >= 1).only_enforce_if(b)
                    cp.add(x[(m, s, d)] == 0).only_enforce_if(b.negated())
                    on_preferred.append(b)
            if on_preferred:
                penalty = cp.new_bool_var(f"pen_{label}_{m}_{s}")
                cp.add(sum(on_preferred) >= 1).only_enforce_if(penalty.negated())
                cp.add(sum(on_preferred) == 0).only_enforce_if(penalty)
                penalty_terms.append(penalty_weight * penalty)

    pkrrlsb_pairs = [(m, s) for (m, s), f in flags.items() if f["pkrrlsb"] == 1]
    add_preferred_days_soft(pkrrlsb_pairs, PREFERRED_DAYS_PKRML_PKRRLSB, "pkrrlsb")
    pkrml_pairs = [(m, s) for (m, s), f in flags.items() if f["pkrml"] == 1]
    add_preferred_days_soft(pkrml_pairs, PREFERRED_DAYS_PKRML_PKRRLSB, "pkrml")
    dc_refuse_pairs = [(m, s) for (m, s), f in flags.items() if f["dc_refuse"] == 1]
    add_preferred_days_soft(dc_refuse_pairs, [PREFERRED_DAY_DC_REFUSE], "dc_refuse")

    u = {}
    v = {}
    for m in models:
        for d in range(ND):
            u[(m, d)] = cp.new_int_var(0, ND * Qm[m], f"u[{m},{d}]")
            expr = ND * y[(m, d)] - Qm[m]
            cp.add(u[(m, d)] >= expr)
            cp.add(u[(m, d)] >= -expr)

    for s in lines:
        for d in range(ND):
            v[(s, d)] = cp.new_int_var(0, ND * Qs[s], f"v[{s},{d}]")
            expr = ND * z[(s, d)] - Qs[s]
            cp.add(v[(s, d)] >= expr)
            cp.add(v[(s, d)] >= -expr)

    balance_obj = alpha * sum(u.values()) + beta * sum(v.values())
    soft_penalty = sum(penalty_terms) if penalty_terms else 0
    cp.minimize(balance_obj + soft_penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 4
    status = solver.solve(cp)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found.")

    status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    detail = []
    for (m, s, d), var in x.items():
        val = solver.value(var)
        if val > 0:
            pair_flags = flags.get((m, s), {})
            detail.append({
                "model": m, "line": s, "day": DAYS[d], "day_idx": d, "qty": val,
                "rl2512": pair_flags.get("rl2512", 0),
                "pkrrlsb": pair_flags.get("pkrrlsb", 0),
                "pkrml": pair_flags.get("pkrml", 0),
                "dc_refuse": pair_flags.get("dc_refuse", 0),
            })
    detail.sort(key=lambda r: (r["day_idx"], r["line"], r["model"]))

    model_day = {}
    for row in detail:
        key = (row["model"], row["day"])
        model_day[key] = model_day.get(key, 0) + row["qty"]
    model_day_list = [{"model": m, "day": d, "qty": v} for (m, d), v in sorted(model_day.items())]

    line_day = {}
    for row in detail:
        key = (row["line"], row["day"])
        line_day[key] = line_day.get(key, 0) + row["qty"]
    line_day_list = [{"line": s, "day": d, "qty": v} for (s, d), v in sorted(line_day.items())]

    compliance = build_compliance_report(detail, flags, q)

    return {
        "status": status_str, "objective": solver.objective_value,
        "total_units": total_units, "detail": detail,
        "model_day": model_day_list, "line_day": line_day_list,
        "models": models, "lines": lines,
        "Qm": Qm, "Qs": Qs, "compliance": compliance,
    }


def build_compliance_report(detail, flags, q):
    scheduled = {}
    for row in detail:
        key = (row["model"], row["line"])
        if key not in scheduled:
            scheduled[key] = []
        scheduled[key].append(row["day"])

    report = []
    for (m, s), f in flags.items():
        q_ms = q.get((m, s), 0)
        days_used = scheduled.get((m, s), [])

        if f["rl2512"]:
            day_counts = {}
            for row in detail:
                if row["model"] == m and row["line"] == s:
                    day_counts[row["day"]] = day_counts.get(row["day"], 0) + row["qty"]
            max_day = max(day_counts.values()) if day_counts else 0
            report.append({
                "constraint": "Rear Loader 2512 – Max 1/Day + Spaced",
                "unit": f"{m} (Line {s})", "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Evenly spaced, ≤1/day", "met": max_day <= 1,
            })
        if f["pkrrlsb"]:
            report.append({
                "constraint": "Rear Loader PKRRLSB – Prefer Tue/Thu",
                "unit": f"{m} (Line {s})", "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Tue / Thu",
                "met": any(d in ["Tue", "Thu"] for d in days_used),
            })
        if f["pkrml"]:
            report.append({
                "constraint": "Manual Side Loader PKRML – Prefer Tue/Thu",
                "unit": f"{m} (Line {s})", "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Tue / Thu",
                "met": any(d in ["Tue", "Thu"] for d in days_used),
            })
        if f["dc_refuse"]:
            report.append({
                "constraint": "DC Refuse Stock – Prefer Friday",
                "unit": f"{m} (Line {s})", "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Fri", "met": "Fri" in days_used,
            })
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/weeks", methods=["GET"])
def get_weeks():
    try:
        weeks = load_dataset()
        result = [
            {"key": k, "label": v["label"], "year": v["year"],
             "week": v["week"], "count": len(v["rows"])}
            for k, v in sorted(weeks.items())
        ]
        return jsonify({"ok": True, "weeks": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/week_data/<week_key>", methods=["GET"])
def get_week_data(week_key):
    try:
        weeks = load_dataset()
        if week_key not in weeks:
            return jsonify({"ok": False, "error": f"Week '{week_key}' not found."}), 404
        return jsonify({"ok": True, "rows": weeks[week_key]["rows"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/solve", methods=["POST"])
def solve():
    try:
        body = request.get_json(force=True)
        demand       = body.get("demand", [])
        alpha        = float(body.get("alpha", 1))
        beta         = float(body.get("beta", 1))
        threshold    = int(body.get("threshold", 5))
        time_limit   = int(body.get("time_limit", 10))
        penalty_wt   = int(body.get("penalty_weight", 50))
        result = solve_schedule(
            demand, alpha=alpha, beta=beta,
            small_model_threshold=threshold,
            time_limit_sec=time_limit,
            penalty_weight=penalty_wt
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
# AI Chat Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/ai/config", methods=["GET"])
def get_ai_config():
    """Return current AI config (mask the API key for security)."""
    cfg = load_ai_config()
    masked = dict(cfg)
    if masked.get("azure_api_key"):
        key = masked["azure_api_key"]
        masked["azure_api_key"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
    masked["is_configured"] = bool(cfg.get("azure_api_key") and cfg.get("azure_endpoint"))
    return jsonify({"ok": True, "config": masked})


@app.route("/ai/config", methods=["POST"])
def set_ai_config():
    """Update AI config (API key, endpoint, limits)."""
    body = request.get_json(force=True)
    cfg = load_ai_config()
    for key in ("azure_endpoint", "azure_api_key", "azure_api_version",
                "azure_deployment", "daily_token_limit", "max_tokens_per_reply"):
        if key in body and body[key] is not None:
            cfg[key] = body[key]
    save_ai_config(cfg)
    return jsonify({"ok": True, "message": "Configuration saved."})


@app.route("/ai/usage", methods=["GET"])
def get_ai_usage():
    """Return today's token usage and budget."""
    cfg = load_ai_config()
    usage = get_today_usage()
    limit = int(cfg.get("daily_token_limit", 100000))
    return jsonify({
        "ok": True,
        "date": date.today().isoformat(),
        "usage": usage,
        "daily_limit": limit,
        "remaining": max(0, limit - usage.get("total_tokens", 0)),
    })


@app.route("/ai/usage/history", methods=["GET"])
def get_ai_usage_history():
    """Return full token usage history."""
    log = _load_token_log()
    return jsonify({"ok": True, "history": log})


@app.route("/ai/chat", methods=["POST"])
def ai_chat():
    """
    Handle AI chat request.
    Body: { message, solver_context: { week_key, demand_summary, solve_result } }
    """
    cfg = load_ai_config()

    if not cfg.get("azure_api_key") or not cfg.get("azure_endpoint"):
        return jsonify({
            "ok": False,
            "error": "Azure OpenAI is not configured. Please set your API key and endpoint in Settings."
        }), 400

    # Check daily budget
    ok, remaining, used = check_budget(cfg)
    if not ok:
        return jsonify({
            "ok": False,
            "error": f"Daily token budget exhausted ({used:,} / {int(cfg['daily_token_limit']):,} tokens used). Resets at midnight."
        }), 429

    body = request.get_json(force=True)
    user_message = body.get("message", "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "Empty message."}), 400

    solver_context = body.get("solver_context", {})
    conversation_history = body.get("history", [])

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=cfg["azure_endpoint"],
            api_key=cfg["azure_api_key"],
            api_version=cfg["azure_api_version"],
        )

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": build_context_message(solver_context)},
        ]

        # Add conversation history (last 10 exchanges max)
        for msg in conversation_history[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        max_reply = int(cfg.get("max_tokens_per_reply", 1000))

        response = client.chat.completions.create(
            model=cfg["azure_deployment"],
            messages=messages,
            max_tokens=min(max_reply, remaining),
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        usage = response.usage

        # Record token usage
        today_usage = record_token_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

        return jsonify({
            "ok": True,
            "reply": reply,
            "tokens_used": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
            "daily_usage": today_usage,
            "daily_limit": int(cfg["daily_token_limit"]),
        })

    except Exception as e:
        return jsonify({"ok": False, "error": f"Azure OpenAI error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
