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
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2, 4, 6], help="Charger scenarios")
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
        infrastructure_cost_per_charger_eur=args.charger_cost,
    )

    runner = ExperimentRunner(
        simulation_config=simulation_config,
        charging_config=charging_config,
        analysis_config=analysis_config,
    )
    outputs = runner.run(scenarios=args.scenarios, output_root=args.output_dir)

    summary = outputs["summary"]
    cost_benefit = outputs["cost_benefit"]

    print("=== Scenario Summary ===")
    print(summary[["chargers", "avg_wait_min", "avg_max_wait_min", "avg_p95_wait_min", "avg_utilization"]])
    print("\n=== Cost-Benefit ===")
    print(cost_benefit[["chargers", "added_chargers", "delta_avg_wait_min", "relative_wait_reduction_pct"]])


if __name__ == "__main__":
    main()
