"""High-level experiment orchestration and artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .analysis import ResultAnalyzer
from .charging import ChargingModel
from .config import AnalysisConfig, ChargingConfig, SimulationConfig
from .station import ChargingStationSimulator
from .visualization import ResultVisualizer


@dataclass(slots=True)
class ExperimentRunner:
    """Coordinates simulation, analysis, visualization, and exported artifacts."""

    simulation_config: SimulationConfig
    charging_config: ChargingConfig
    analysis_config: AnalysisConfig
    base_seed: int = 2026

    def _run_scenario(self, chargers: int) -> pd.DataFrame:
        """Execute Monte Carlo replications for one charger scenario."""

        frames: list[pd.DataFrame] = []
        for replication_id in range(self.analysis_config.replications):
            simulator = ChargingStationSimulator(
                simulation_config=self.simulation_config,
                charging_config=self.charging_config,
                chargers=chargers,
                seed=self.base_seed + chargers * 10000 + replication_id,
                replication_id=replication_id,
            )
            rep_df = simulator.run()
            if not rep_df.empty:
                frames.append(rep_df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _save_markdown_summary(summary: pd.DataFrame, cost_benefit: pd.DataFrame, output_file: Path) -> None:
        """Write concise textual summary suitable for inclusion in a paper appendix."""

        lines: list[str] = []
        lines.append("# Simulation Findings\n")
        lines.append("## Queue Metrics by Scenario\n")
        lines.append(summary.to_markdown(index=False))
        lines.append("\n## Cost-Benefit What-If\n")
        lines.append(cost_benefit.to_markdown(index=False))
        output_file.write_text("\n".join(lines), encoding="utf-8")

    def run(self, scenarios: list[int], output_root: Path) -> dict[str, pd.DataFrame]:
        """Run all scenarios and generate tabular and graphical outputs."""

        output_root.mkdir(parents=True, exist_ok=True)
        table_dir = output_root / "tables"
        fig_dir = output_root / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        all_records: list[pd.DataFrame] = []
        for chargers in scenarios:
            scenario_df = self._run_scenario(chargers)
            if scenario_df.empty:
                continue
            all_records.append(scenario_df)

        if not all_records:
            raise RuntimeError("No simulation records were produced")

        records = pd.concat(all_records, ignore_index=True)
        summary = ResultAnalyzer.scenario_summary(
            records=records,
            simulation_minutes=self.simulation_config.simulation_minutes,
        )
        cost_benefit = ResultAnalyzer.cost_benefit(
            summary=summary,
            charger_cost_eur=self.analysis_config.infrastructure_cost_per_charger_eur,
        )

        records.to_csv(table_dir / "vehicle_records.csv", index=False)
        summary.to_csv(table_dir / "scenario_summary.csv", index=False)
        cost_benefit.to_csv(table_dir / "cost_benefit.csv", index=False)

        visualizer = ResultVisualizer()
        visualizer.plot_wait_distribution(records, fig_dir / "wait_distribution.png")
        visualizer.plot_summary_wait(summary, fig_dir / "wait_summary.png")
        visualizer.plot_cost_benefit(cost_benefit, fig_dir / "cost_benefit.png")
        visualizer.plot_charging_curve(ChargingModel(self.charging_config), fig_dir / "charging_curve.png")

        self._save_markdown_summary(summary, cost_benefit, output_root / "results_summary.md")

        return {
            "records": records,
            "summary": summary,
            "cost_benefit": cost_benefit,
        }
