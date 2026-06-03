"""CLI entrypoint for EV charging queue simulation project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ev_charging.config import AnalysisConfig, ChargingConfig, SimulationConfig
from ev_charging.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configurable experiments."""

    parser = argparse.ArgumentParser(description="EV highway charging queue simulation")
    parser.add_argument("--hours", type=float, default=12.0, help="Simulation horizon in hours")
    parser.add_argument("--arrival-rate", type=float, default=9.5, help="Average vehicle arrivals per hour")
    parser.add_argument("--replications", type=int, default=120, help="Monte Carlo replication count")
    parser.add_argument(
        "--warmup-method",
        type=str,
        choices=["fixed", "welch"],
        default="welch",
        help="Warm-up strategy: fixed cutoff or Welch-based estimation",
    )
    parser.add_argument("--warmup-minutes", type=float, default=60.0, help="Warm-up period excluded from KPI analysis")
    parser.add_argument("--welch-bin-minutes", type=float, default=10.0, help="Bin size in minutes for Welch warm-up estimation")
    parser.add_argument("--welch-smoothing-bins", type=int, default=3, help="Rolling-average window (bins) for Welch curve")
    parser.add_argument("--welch-stability-bins", type=int, default=3, help="Consecutive stable bins required for warm-up detection")
    parser.add_argument("--welch-tolerance", type=float, default=0.05, help="Relative tolerance around steady-state for Welch")
    parser.add_argument("--metamodel-test-size", type=float, default=0.25, help="Test split ratio for metamodel evaluation")
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=200,
        help="Number of trees for RandomForest metamodel",
    )
    parser.add_argument(
        "--sensitivity-replications",
        type=int,
        default=60,
        help="Monte Carlo replications per sensitivity configuration",
    )
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2, 4, 6], help="Charger scenarios")
    parser.add_argument(
        "--sensitivity-arrival-rates",
        type=float,
        nargs="*",
        default=[7.0, 9.5, 12.0],
        help="Arrival-rate levels for sensitivity analysis; pass no values to disable",
    )
    parser.add_argument(
        "--charger-cost",
        type=float,
        default=42000.0,
        help="Cost per additional fast charger in EUR",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for CSV tables and figures",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable step-by-step progress messages during simulation",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end simulation study and print key metrics."""

    args = parse_args()

    simulation_config = SimulationConfig(
        simulation_hours=args.hours,
        arrival_rate_per_hour=args.arrival_rate,
    )
    charging_config = ChargingConfig()
    analysis_config = AnalysisConfig(
        replications=args.replications,
        warmup_method=args.warmup_method,
        warm_up_minutes=args.warmup_minutes,
        welch_bin_minutes=args.welch_bin_minutes,
        welch_smoothing_bins=args.welch_smoothing_bins,
        welch_stability_bins=args.welch_stability_bins,
        welch_relative_tolerance=args.welch_tolerance,
        metamodel_test_size=args.metamodel_test_size,
        random_forest_estimators=args.rf_estimators,
        sensitivity_replications=args.sensitivity_replications,
        infrastructure_cost_per_charger_eur=args.charger_cost,
    )

    runner = ExperimentRunner(
        simulation_config=simulation_config,
        charging_config=charging_config,
        analysis_config=analysis_config,
    )
    outputs = runner.run(
        scenarios=args.scenarios,
        output_root=args.output_dir,
        sensitivity_arrival_rates=args.sensitivity_arrival_rates,
        verbose=not args.quiet,
    )

    summary = outputs["summary"]
    cost_benefit = outputs["cost_benefit"]
    metamodel_metrics = outputs["metamodel_metrics"]
    warm_up_minutes = outputs["warm_up_minutes"].iloc[0]["warm_up_minutes"]
    sensitivity_summary = outputs["sensitivity_summary"]

    print(f"=== Warm-up ({args.warmup_method}) ===")
    print(f"Estimated/applied warm-up minutes: {warm_up_minutes:.2f}")
    print("=== Scenario Summary ===")
    print(summary[["chargers", "avg_wait_min", "avg_max_wait_min", "avg_p95_wait_min", "avg_utilization"]])
    print("\n=== Cost-Benefit ===")
    print(cost_benefit[["chargers", "added_chargers", "delta_avg_wait_min", "relative_wait_reduction_pct"]])
    print("\n=== Metamodel Metrics ===")
    print(metamodel_metrics)
    if not sensitivity_summary.empty:
        print("\n=== Sensitivity Snapshot ===")
        print(sensitivity_summary[["arrival_rate_per_hour", "chargers", "avg_wait_min", "avg_utilization"]])


if __name__ == "__main__":
    main()
