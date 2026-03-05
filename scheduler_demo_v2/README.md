# Production Scheduler – Interactive Demo

An AI-powered weekly production scheduling demo built for C-level presentations.
Uses Google OR-Tools CP-SAT constraint programming to optimally distribute production
demand across models, lines, and days.

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

### 3. Open in browser

```
http://localhost:5050
```

---

## How to Use the Demo

| Section | What to Do |
|---|---|
| **Weekly Demand** | Edit model names, line names, and quantities. Add/remove rows freely. |
| **Alpha (α)** | Drag slider to increase/decrease model-balance priority (0 = ignore, 10 = strict). |
| **Beta (β)** | Drag slider to increase/decrease line-balance priority. |
| **Spread Threshold** | Models with total qty below this are forced to spread evenly across all 5 days. |
| **Time Limit** | How long the solver runs — longer may yield a better (lower objective) solution. |
| **Run Optimizer** | Click to solve. Results appear with animated charts and a schedule heatmap. |

---

## What the Optimizer Does

The solver distributes weekly production demand across **5 working days (Mon–Fri)**
while minimizing two types of imbalance:

- **Model balance** (α): Each model's daily output should be as close to `Total/5` as possible.
- **Line balance** (β): Each production line's daily load should be as close to `Total/5` as possible.

The **Spread Threshold** enforces that low-volume models (qty < threshold) must produce
at least `floor(Q/5)` and at most `ceil(Q/5)` units per day.

---

## Output Panels

| Panel | Description |
|---|---|
| KPI Cards | Total units, model count, lines used, solver status & objective |
| Daily Production by Model | Stacked bar chart — units per model per day |
| Daily Load by Production Line | Grouped bar chart — workload per line per day |
| Model Balance Score | Bar chart showing max deviation from ideal daily average |
| Weekly Volume by Model | Doughnut chart — share of total production per model |
| Schedule Heatmap | Color-coded grid: model × day, intensity = volume |
| Full Schedule Detail | Row-level table: day, model, line, units |

---

## File Structure

```
scheduler_demo/
├── app.py              ← Flask backend + CP-SAT solver
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── static/
    └── index.html      ← Interactive frontend (Chart.js, no build step)
```
