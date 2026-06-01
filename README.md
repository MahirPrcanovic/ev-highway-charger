# EV Highway Charging Queue Simulation

This project models congestion at highway fast-charging stations during peak demand periods.
The workflow combines:
- ODE-based nonlinear battery charging with SciPy
- Poisson vehicle arrivals (Monte Carlo with SimPy)
- Scenario comparison for 2, 4, and 6 chargers
- Statistical metrics and cost-benefit what-if analysis

## Mathematical Model

Charging dynamics are modeled as:

$$
\frac{dQ}{dt} = k(Q_{max} - Q)
$$

where:
- `Q(t)` is battery energy state
- `Q_max` is battery capacity
- `k` is charging rate constant

The queueing side is simulated as an M/G/c-like system with stochastic arrivals and nonlinear charging service time.

## Project Structure

- `src/ev_charging/config.py`: simulation and analysis configuration dataclasses
- `src/ev_charging/charging.py`: SciPy ODE charging model
- `src/ev_charging/station.py`: SimPy charging-station queue simulation
- `src/ev_charging/analysis.py`: statistical and cost-benefit analysis
- `src/ev_charging/visualization.py`: plotting utilities
- `src/ev_charging/runner.py`: end-to-end experiment orchestration
- `scripts/run_experiments.py`: CLI entrypoint

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Step-by-Step (Fresh Clone, Windows)

Use these commands if you just cloned the repository and want to run the project for the first time.

1. Clone and enter the project:

```powershell
git clone https://github.com/MahirPrcanovic/ev-highway-charger.git
cd ev-highway-charger
```

2. Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

If `py` is not available, use:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Run a quick validation (faster):

```powershell
python scripts/run_experiments.py --hours 12 --arrival-rate 9.5 --replications 20 --scenarios 2 4 6
```

6. Run the full experiment (report-ready):

```powershell
python scripts/run_experiments.py --hours 12 --arrival-rate 9.5 --replications 120 --scenarios 2 4 6
```

7. Open generated outputs in the `outputs/` folder.

## Updating After New Git Changes

When you pull new code:

```powershell
git pull
.\.venv\Scripts\Activate
pip install -r requirements.txt
python scripts/run_experiments.py --hours 12 --arrival-rate 9.5 --replications 120 --scenarios 2 4 6
```

## Troubleshooting

- If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

- If you see missing package errors, run:

```powershell
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_experiments.py --hours 12 --arrival-rate 9.5 --replications 120 --scenarios 2 4 6
```

Optional parameters:
- `--charger-cost`: cost-benefit baseline per added charger (EUR)
- `--output-dir`: output folder for CSV and figures

## Output Artifacts

After execution, generated files are in `outputs/`:
- `tables/vehicle_records.csv`
- `tables/scenario_summary.csv`
- `tables/cost_benefit.csv`
- `figures/charging_curve.png`
- `figures/wait_distribution.png`
- `figures/wait_summary.png`
- `figures/cost_benefit.png`
- `results_summary.md`

## Interpretation Guide

Primary KPIs:
- `avg_wait_min`: average waiting time in queue
- `avg_max_wait_min`: expected peak waiting time
- `avg_p95_wait_min`: tail-risk delay indicator
- `avg_utilization`: charger occupancy ratio

Cost-benefit columns:
- `delta_avg_wait_min`: saved average minutes vs previous scenario
- `relative_wait_reduction_pct`: percentage reduction in average wait
- `minutes_saved_per_1k_eur`: marginal efficiency of investment
