"""
Production Scheduling Demo – Flask Backend
Exposes a single POST /solve endpoint that accepts demand data + parameters
and returns an optimized weekly schedule.
"""

from flask import Flask, request, jsonify, send_from_directory
from ortools.sat.python import cp_model
import os

app = Flask(__name__, static_folder="static")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ND = len(DAYS)


def solve_schedule(demand_rows, alpha=1, beta=1,
                   small_model_threshold=5, time_limit_sec=10):
    """
    demand_rows: list of dicts with keys 'model', 'line', 'qty'
    Returns dict with detail, model_day, line_day tables.
    """
    # Build aggregated demand q[(m,s)]
    q = {}
    for row in demand_rows:
        m = str(row["model"]).strip()
        s = str(row["line"]).strip()
        qty = int(row.get("qty", 1))
        if qty <= 0:
            continue
        key = (m, s)
        q[key] = q.get(key, 0) + qty

    if not q:
        raise ValueError("No valid demand rows provided.")

    models = sorted(set(m for m, s in q))
    lines  = sorted(set(s for m, s in q))

    Qm = {m: sum(q.get((m, s), 0) for s in lines) for m in models}
    Qs = {s: sum(q.get((m, s), 0) for m in models) for s in lines}

    # ---------- CP-SAT Model ----------
    cp = cp_model.CpModel()

    x = {}
    for m in models:
        for s in lines:
            q_ms = q.get((m, s), 0)
            if q_ms == 0:
                continue
            for d in range(ND):
                x[(m, s, d)] = cp.new_int_var(0, q_ms, f"x[{m},{s},{d}]")

    y = {}
    z = {}
    for m in models:
        for d in range(ND):
            terms = [x[(m, s, d)] for s in lines if (m, s, d) in x]
            y[(m, d)] = cp.new_int_var(0, Qm[m], f"y[{m},{d}]")
            cp.add(y[(m, d)] == sum(terms)) if terms else cp.add(y[(m, d)] == 0)

    for s in lines:
        for d in range(ND):
            terms = [x[(m, s, d)] for m in models if (m, s, d) in x]
            z[(s, d)] = cp.new_int_var(0, Qs[s], f"z[{s},{d}]")
            cp.add(z[(s, d)] == sum(terms)) if terms else cp.add(z[(s, d)] == 0)

    # (A) Meet weekly demand
    for m in models:
        for s in lines:
            q_ms = q.get((m, s), 0)
            if q_ms == 0:
                continue
            cp.add(sum(x[(m, s, d)] for d in range(ND)) == q_ms)

    # (B) Small model spread rule
    for m in models:
        if Qm[m] < small_model_threshold:
            lo = Qm[m] // ND
            hi = (Qm[m] + ND - 1) // ND
            for d in range(ND):
                cp.add(y[(m, d)] >= lo)
                cp.add(y[(m, d)] <= hi)

    # Balance objective
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

    cp.minimize(alpha * sum(u.values()) + beta * sum(v.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 4

    status = solver.solve(cp)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found. Try relaxing constraints or increasing time limit.")

    status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

    # Build output
    detail = []
    for (m, s, d), var in x.items():
        val = solver.value(var)
        if val > 0:
            detail.append({"model": m, "line": s, "day": DAYS[d], "day_idx": d, "qty": val})

    detail.sort(key=lambda r: (r["day_idx"], r["line"], r["model"]))

    # Model/day summary
    model_day = {}
    for row in detail:
        key = (row["model"], row["day"])
        model_day[key] = model_day.get(key, 0) + row["qty"]
    model_day_list = [{"model": m, "day": d, "qty": v} for (m, d), v in model_day.items()]

    # Line/day summary
    line_day = {}
    for row in detail:
        key = (row["line"], row["day"])
        line_day[key] = line_day.get(key, 0) + row["qty"]
    line_day_list = [{"line": s, "day": d, "qty": v} for (s, d), v in line_day.items()]

    # Objective value
    obj = solver.objective_value

    return {
        "status": status_str,
        "objective": obj,
        "detail": detail,
        "model_day": model_day_list,
        "line_day": line_day_list,
        "models": models,
        "lines": lines,
        "Qm": Qm,
        "Qs": Qs,
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/solve", methods=["POST"])
def solve():
    try:
        body = request.get_json(force=True)
        demand = body.get("demand", [])
        alpha = float(body.get("alpha", 1))
        beta  = float(body.get("beta", 1))
        threshold = int(body.get("threshold", 5))
        time_limit = int(body.get("time_limit", 10))

        result = solve_schedule(demand, alpha=alpha, beta=beta,
                                small_model_threshold=threshold,
                                time_limit_sec=time_limit)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5050)
