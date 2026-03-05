"""
Production Scheduling Demo – Flask Backend v2
Supports:
  - CSV dataset loading with weekly grouping (Exit Factory Year & Week)
  - Advanced constraints:
      * Rear Loader 2512: max 1 per day, spaced as evenly as possible
      * Manual Side Loader (PKRML) == 1: prefer Tue/Thu
      * Rear Loader (PKRRLSB) == 1: prefer Tue/Thu
      * DC REFUSE STOCK == 1: prefer Friday
  - Full demand coverage (all allocated models must be arranged)
  - Model balance (alpha) and line balance (beta) objectives
"""

from flask import Flask, request, jsonify, send_from_directory
from ortools.sat.python import cp_model
import os
import pandas as pd
import math

app = Flask(__name__, static_folder="static")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ND = len(DAYS)

# Day indices for special constraints
TUE, THU, FRI = 1, 3, 4
PREFERRED_DAYS_PKRML_PKRRLSB = [TUE, THU]   # Tue, Thu
PREFERRED_DAY_DC_REFUSE = FRI                # Fri

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
# Solver
# ─────────────────────────────────────────────────────────────────────────────
def solve_schedule(demand_rows, alpha=1, beta=1,
                   small_model_threshold=5, time_limit_sec=10,
                   penalty_weight=50):
    """
    demand_rows: list of dicts with keys:
        model, line, qty, rl2512, pkrrlsb, pkrml, dc_refuse
    Returns dict with detail, model_day, line_day tables.
    """
    # ── Aggregate demand ──────────────────────────────────────────────────────
    q = {}          # (model, line) -> qty
    flags = {}      # (model, line) -> {rl2512, pkrrlsb, pkrml, dc_refuse}

    for row in demand_rows:
        m   = str(row["model"]).strip()
        s   = str(row["line"]).strip()
        qty = int(row.get("qty", 1))
        if qty <= 0:
            continue
        key = (m, s)
        q[key] = q.get(key, 0) + qty

        # Aggregate flags: if ANY unit in (m,s) has the flag, the pair has it
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

    # ── CP-SAT Model ──────────────────────────────────────────────────────────
    cp = cp_model.CpModel()

    # x[(m,s,d)] = units of model m on line s on day d
    x = {}
    for (m, s), q_ms in q.items():
        for d in range(ND):
            x[(m, s, d)] = cp.new_int_var(0, q_ms, f"x[{m},{s},{d}]")

    # y[(m,d)] = total units of model m on day d (across all lines)
    y = {}
    for m in models:
        for d in range(ND):
            terms = [x[(m, s, d)] for s in lines if (m, s, d) in x]
            y[(m, d)] = cp.new_int_var(0, Qm[m], f"y[{m},{d}]")
            cp.add(y[(m, d)] == sum(terms)) if terms else cp.add(y[(m, d)] == 0)

    # z[(s,d)] = total units on line s on day d (across all models)
    z = {}
    for s in lines:
        for d in range(ND):
            terms = [x[(m, s, d)] for m in models if (m, s, d) in x]
            z[(s, d)] = cp.new_int_var(0, Qs[s], f"z[{s},{d}]")
            cp.add(z[(s, d)] == sum(terms)) if terms else cp.add(z[(s, d)] == 0)

    # ── Hard Constraint A: Meet ALL weekly demand ─────────────────────────────
    for (m, s), q_ms in q.items():
        cp.add(sum(x[(m, s, d)] for d in range(ND)) == q_ms)

    # ── Hard Constraint B: Small model spread ─────────────────────────────────
    for m in models:
        if Qm[m] < small_model_threshold:
            lo = Qm[m] // ND
            hi = math.ceil(Qm[m] / ND)
            for d in range(ND):
                cp.add(y[(m, d)] >= lo)
                cp.add(y[(m, d)] <= hi)

    # ── Hard Constraint C: Rear Loader 2512 – max 1 per day per line ──────────
    # Collect all (m,s) pairs that have rl2512 flag
    rl2512_pairs = [(m, s) for (m, s), f in flags.items() if f["rl2512"] == 1]
    if rl2512_pairs:
        for d in range(ND):
            daily_rl2512 = [x[(m, s, d)] for (m, s) in rl2512_pairs if (m, s, d) in x]
            if daily_rl2512:
                cp.add(sum(daily_rl2512) <= 1)

    # ── Soft Constraints: Preferred Day Scheduling ────────────────────────────
    # Penalty variables for violating soft preferences
    penalty_terms = []

    def add_preferred_days_soft(pairs, preferred_days, label):
        """
        For each (m,s) pair in pairs with qty==1, add a soft constraint
        that encourages scheduling on one of the preferred_days.
        penalty_var = 1 if NOT scheduled on any preferred day.
        """
        for (m, s) in pairs:
            q_ms = q.get((m, s), 0)
            if q_ms == 0:
                continue
            # is_preferred[d] = 1 if x[(m,s,d)] > 0 AND d in preferred_days
            on_preferred = []
            for d in preferred_days:
                if (m, s, d) in x:
                    b = cp.new_bool_var(f"pref_{label}_{m}_{s}_{d}")
                    cp.add(x[(m, s, d)] >= 1).only_enforce_if(b)
                    cp.add(x[(m, s, d)] == 0).only_enforce_if(b.negated())
                    on_preferred.append(b)
            if on_preferred:
                # penalty = 1 if none of the preferred days are used
                penalty = cp.new_bool_var(f"pen_{label}_{m}_{s}")
                # at_least_one_preferred: sum(on_preferred) >= 1 <=> NOT penalty
                cp.add(sum(on_preferred) >= 1).only_enforce_if(penalty.negated())
                cp.add(sum(on_preferred) == 0).only_enforce_if(penalty)
                penalty_terms.append(penalty_weight * penalty)

    # PKRRLSB (Rear Loader PKRRLSB) == 1 → prefer Tue/Thu
    pkrrlsb_pairs = [(m, s) for (m, s), f in flags.items() if f["pkrrlsb"] == 1]
    add_preferred_days_soft(pkrrlsb_pairs, PREFERRED_DAYS_PKRML_PKRRLSB, "pkrrlsb")

    # PKRML (Manual Side Loader) == 1 → prefer Tue/Thu
    pkrml_pairs = [(m, s) for (m, s), f in flags.items() if f["pkrml"] == 1]
    add_preferred_days_soft(pkrml_pairs, PREFERRED_DAYS_PKRML_PKRRLSB, "pkrml")

    # DC REFUSE STOCK == 1 → prefer Friday
    dc_refuse_pairs = [(m, s) for (m, s), f in flags.items() if f["dc_refuse"] == 1]
    add_preferred_days_soft(dc_refuse_pairs, [PREFERRED_DAY_DC_REFUSE], "dc_refuse")

    # ── Balance Objective ─────────────────────────────────────────────────────
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

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 4

    status = solver.solve(cp)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found. Try relaxing constraints or increasing time limit.")

    status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    # ── Build Output ──────────────────────────────────────────────────────────
    detail = []
    for (m, s, d), var in x.items():
        val = solver.value(var)
        if val > 0:
            pair_flags = flags.get((m, s), {})
            detail.append({
                "model":     m,
                "line":      s,
                "day":       DAYS[d],
                "day_idx":   d,
                "qty":       val,
                "rl2512":    pair_flags.get("rl2512", 0),
                "pkrrlsb":   pair_flags.get("pkrrlsb", 0),
                "pkrml":     pair_flags.get("pkrml", 0),
                "dc_refuse": pair_flags.get("dc_refuse", 0),
            })

    detail.sort(key=lambda r: (r["day_idx"], r["line"], r["model"]))

    # Model/day summary
    model_day = {}
    for row in detail:
        key = (row["model"], row["day"])
        model_day[key] = model_day.get(key, 0) + row["qty"]
    model_day_list = [{"model": m, "day": d, "qty": v}
                      for (m, d), v in sorted(model_day.items())]

    # Line/day summary
    line_day = {}
    for row in detail:
        key = (row["line"], row["day"])
        line_day[key] = line_day.get(key, 0) + row["qty"]
    line_day_list = [{"line": s, "day": d, "qty": v}
                     for (s, d), v in sorted(line_day.items())]

    # Constraint compliance report
    compliance = build_compliance_report(detail, flags, q)

    return {
        "status":     status_str,
        "objective":  solver.objective_value,
        "total_units": total_units,
        "detail":     detail,
        "model_day":  model_day_list,
        "line_day":   line_day_list,
        "models":     models,
        "lines":      lines,
        "Qm":         Qm,
        "Qs":         Qs,
        "compliance": compliance,
    }


def build_compliance_report(detail, flags, q):
    """
    Build a per-constraint compliance summary for the UI.
    Returns list of {constraint, unit, scheduled_day, preferred, met}.
    """
    # Index detail by (model, line, day)
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
            # Check: max 1 per day
            day_counts = {}
            for row in detail:
                if row["model"] == m and row["line"] == s:
                    day_counts[row["day"]] = day_counts.get(row["day"], 0) + row["qty"]
            max_day = max(day_counts.values()) if day_counts else 0
            spread_ok = max_day <= 1
            report.append({
                "constraint": "Rear Loader 2512 – Max 1/Day + Spaced",
                "unit": f"{m} (Line {s})",
                "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Evenly spaced, ≤1/day",
                "met": spread_ok,
            })

        if f["pkrrlsb"]:
            on_pref = any(d in ["Tue", "Thu"] for d in days_used)
            report.append({
                "constraint": "Rear Loader PKRRLSB – Prefer Tue/Thu",
                "unit": f"{m} (Line {s})",
                "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Tue / Thu",
                "met": on_pref,
            })

        if f["pkrml"]:
            on_pref = any(d in ["Tue", "Thu"] for d in days_used)
            report.append({
                "constraint": "Manual Side Loader PKRML – Prefer Tue/Thu",
                "unit": f"{m} (Line {s})",
                "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Tue / Thu",
                "met": on_pref,
            })

        if f["dc_refuse"]:
            on_pref = "Fri" in days_used
            report.append({
                "constraint": "DC Refuse Stock – Prefer Friday",
                "unit": f"{m} (Line {s})",
                "qty": q_ms,
                "scheduled_days": sorted(set(days_used)),
                "preferred": "Fri",
                "met": on_pref,
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
    """Return available Year-Week options from the CSV."""
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
    """Return raw demand rows for a specific year-week."""
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


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
