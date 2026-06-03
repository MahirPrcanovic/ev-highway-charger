"""High-level experiment orchestration and artifact generation."""

from __future__ import annotations

from dataclasses import replace
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

    @staticmethod
    def _log(verbose: bool, message: str) -> None:
        """Print progress messages when verbose mode is enabled."""

        if verbose:
            print(message, flush=True)

    def _run_scenario(
        self,
        chargers: int,
        simulation_config: SimulationConfig,
        seed_offset: int = 0,
        verbose: bool = False,
        tag: str = "base",
    ) -> pd.DataFrame:
        """Execute Monte Carlo replications for one charger scenario."""

        frames: list[pd.DataFrame] = []
        total = self.analysis_config.replications
        report_every = max(1, total // 5)
        self._log(verbose, f"[simulate:{tag}] chargers={chargers} replications={total}")
        for replication_id in range(self.analysis_config.replications):
            simulator = ChargingStationSimulator(
                simulation_config=simulation_config,
                charging_config=self.charging_config,
                chargers=chargers,
                seed=self.base_seed + seed_offset + chargers * 10000 + replication_id,
                replication_id=replication_id,
            )
            rep_df = simulator.run()
            if not rep_df.empty:
                frames.append(rep_df)
            done = replication_id + 1
            if done == 1 or done == total or done % report_every == 0:
                self._log(verbose, f"  -> {tag} chargers={chargers}: {done}/{total} done")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _run_sensitivity(self, scenarios: list[int], arrival_rates: list[float], verbose: bool = False) -> pd.DataFrame:
        """Run sensitivity study over multiple arrival-rate levels."""

        all_frames: list[pd.DataFrame] = []
        for idx, arrival_rate in enumerate(arrival_rates):
            self._log(verbose, f"[sensitivity] arrival_rate={arrival_rate} vehicles/hour")
            sensitivity_config = replace(self.simulation_config, arrival_rate_per_hour=arrival_rate)
            for chargers in scenarios:
                frames: list[pd.DataFrame] = []
                total = self.analysis_config.sensitivity_replications
                report_every = max(1, total // 5)
                for replication_id in range(self.analysis_config.sensitivity_replications):
                    simulator = ChargingStationSimulator(
                        simulation_config=sensitivity_config,
                        charging_config=self.charging_config,
                        chargers=chargers,
                        seed=self.base_seed + 200000 + idx * 50000 + chargers * 10000 + replication_id,
                        replication_id=replication_id,
                    )
                    rep_df = simulator.run()
                    if not rep_df.empty:
                        frames.append(rep_df)
                    done = replication_id + 1
                    if done == 1 or done == total or done % report_every == 0:
                        self._log(
                            verbose,
                            f"  -> sensitivity rate={arrival_rate}, chargers={chargers}: {done}/{total} done",
                        )

                if not frames:
                    continue

                scenario_records = pd.concat(frames, ignore_index=True)
                scenario_records["arrival_rate_per_hour"] = arrival_rate
                all_frames.append(scenario_records)

        if not all_frames:
            return pd.DataFrame()
        return pd.concat(all_frames, ignore_index=True)

    @staticmethod
    def _save_markdown_summary(
        summary: pd.DataFrame,
        cost_benefit: pd.DataFrame,
        metamodel_metrics: pd.DataFrame,
        warmup_method: str,
        warm_up_minutes: float,
        output_file: Path,
    ) -> None:
        """Write concise textual summary suitable for inclusion in a paper appendix."""

        def table_as_markdown(df: pd.DataFrame) -> str:
            """Return markdown table, with plain-text fallback if tabulate is unavailable."""

            try:
                return df.to_markdown(index=False)
            except ImportError:
                return "```text\n" + df.to_string(index=False) + "\n```"

        lines: list[str] = []
        lines.append("# Simulation Findings\n")
        lines.append(f"Warm-up method: {warmup_method}.\n")
        lines.append(f"Warm-up period excluded from KPI analysis: {warm_up_minutes:.1f} minutes.\n")
        lines.append("## Queue Metrics by Scenario\n")
        lines.append(table_as_markdown(summary))
        lines.append("\n## Cost-Benefit What-If\n")
        lines.append(table_as_markdown(cost_benefit))
        lines.append("\n## Metamodel Quality\n")
        lines.append(table_as_markdown(metamodel_metrics))
        output_file.write_text("\n".join(lines), encoding="utf-8")

    def run(
        self,
        scenarios: list[int],
        output_root: Path,
        sensitivity_arrival_rates: list[float] | None = None,
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Run all scenarios and generate tabular and graphical outputs."""

        self._log(verbose, "[step] Initializing output directories")
        output_root.mkdir(parents=True, exist_ok=True)
        table_dir = output_root / "tables"
        fig_dir = output_root / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        self._log(verbose, "[step] Running base Monte Carlo scenarios")
        all_records: list[pd.DataFrame] = []
        for chargers in scenarios:
            scenario_df = self._run_scenario(
                chargers,
                simulation_config=self.simulation_config,
                verbose=verbose,
                tag="base",
            )
            if scenario_df.empty:
                continue
            all_records.append(scenario_df)

        if not all_records:
            raise RuntimeError("No simulation records were produced")

        records = pd.concat(all_records, ignore_index=True)
        warmup_profile = pd.DataFrame()
        warm_up_minutes = self.analysis_config.warm_up_minutes
        if self.analysis_config.warmup_method == "welch":
            self._log(verbose, "[step] Estimating warm-up period with Welch method")
            warm_up_minutes, warmup_profile = ResultAnalyzer.estimate_warmup_welch(
                records=records,
                simulation_minutes=self.simulation_config.simulation_minutes,
                bin_minutes=self.analysis_config.welch_bin_minutes,
                smoothing_bins=self.analysis_config.welch_smoothing_bins,
                stability_bins=self.analysis_config.welch_stability_bins,
                relative_tolerance=self.analysis_config.welch_relative_tolerance,
            )
            self._log(verbose, f"  -> Welch estimated warm-up: {warm_up_minutes:.2f} minutes")

        self._log(verbose, "[step] Computing KPI summaries and metamodels")
        records_analysis = ResultAnalyzer.exclude_warmup(records, warm_up_minutes)

        if records_analysis.empty:
            raise RuntimeError("Warm-up exclusion removed all records; reduce warm_up_minutes")

        summary = ResultAnalyzer.scenario_summary(
            records=records_analysis,
            simulation_minutes=self.simulation_config.simulation_minutes,
        )
        cost_benefit = ResultAnalyzer.cost_benefit(
            summary=summary,
            charger_cost_eur=self.analysis_config.infrastructure_cost_per_charger_eur,
        )
        replication_metrics = ResultAnalyzer.replication_metrics(records_analysis)
        metamodel_metrics, metamodel_predictions = ResultAnalyzer.fit_wait_metamodel(
            replication_metrics=replication_metrics,
            scenario_summary=summary,
            test_size=self.analysis_config.metamodel_test_size,
            random_forest_estimators=self.analysis_config.random_forest_estimators,
        )

        self._log(verbose, "[step] Saving tables")
        records.to_csv(table_dir / "vehicle_records.csv", index=False)
        records_analysis.to_csv(table_dir / "vehicle_records_analysis.csv", index=False)
        summary.to_csv(table_dir / "scenario_summary.csv", index=False)
        cost_benefit.to_csv(table_dir / "cost_benefit.csv", index=False)
        replication_metrics.to_csv(table_dir / "replication_metrics.csv", index=False)
        metamodel_metrics.to_csv(table_dir / "metamodel_metrics.csv", index=False)
        metamodel_predictions.to_csv(table_dir / "metamodel_predictions.csv", index=False)
        welch_profile_path = table_dir / "welch_warmup_profile.csv"
        if not warmup_profile.empty:
            warmup_profile.to_csv(welch_profile_path, index=False)
        elif welch_profile_path.exists():
            welch_profile_path.unlink()

        sensitivity_summary = pd.DataFrame()
        if sensitivity_arrival_rates:
            self._log(verbose, "[step] Running sensitivity analysis")
            sensitivity_records = self._run_sensitivity(scenarios, sensitivity_arrival_rates, verbose=verbose)
            sensitivity_records = ResultAnalyzer.exclude_warmup(sensitivity_records, warm_up_minutes)
            sensitivity_summary = ResultAnalyzer.sensitivity_summary(
                records=sensitivity_records,
                simulation_minutes=self.simulation_config.simulation_minutes,
            )
            if not sensitivity_records.empty:
                sensitivity_records.to_csv(table_dir / "sensitivity_records.csv", index=False)
            if not sensitivity_summary.empty:
                sensitivity_summary.to_csv(table_dir / "sensitivity_summary.csv", index=False)

        self._log(verbose, "[step] Generating figures")
        visualizer = ResultVisualizer()
        visualizer.plot_wait_distribution(records_analysis, fig_dir / "wait_distribution.png")
        visualizer.plot_summary_wait(summary, fig_dir / "wait_summary.png")
        visualizer.plot_cost_benefit(cost_benefit, fig_dir / "cost_benefit.png")
        visualizer.plot_charging_curve(ChargingModel(self.charging_config), fig_dir / "charging_curve.png")
        visualizer.plot_metamodel_fit(metamodel_predictions, fig_dir / "metamodel_fit.png")
        if not sensitivity_summary.empty:
            visualizer.plot_sensitivity_heatmap(sensitivity_summary, fig_dir / "sensitivity_heatmap.png")

        self._log(verbose, "[step] Writing markdown summary")
        self._save_markdown_summary(
            summary,
            cost_benefit,
            metamodel_metrics,
            self.analysis_config.warmup_method,
            warm_up_minutes,
            output_root / "results_summary.md",
        )
        self._log(verbose, "[done] Simulation pipeline completed")

        return {
            "records": records,
            "records_analysis": records_analysis,
            "summary": summary,
            "cost_benefit": cost_benefit,
            "metamodel_metrics": metamodel_metrics,
            "metamodel_predictions": metamodel_predictions,
            "warm_up_minutes": pd.DataFrame([{"warm_up_minutes": warm_up_minutes}]),
            "warmup_profile": warmup_profile,
            "sensitivity_summary": sensitivity_summary,
        }
