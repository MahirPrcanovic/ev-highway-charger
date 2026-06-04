"""High-level experiment orchestration and artifact generation."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

    def _generate_metamodel_training_data(self, verbose: bool = False) -> pd.DataFrame:
        """Sample random combinations from 5D parameter space for metamodel training.

        Samples arrival_rate, chargers, k, q_max_kwh, and max_initial_soc uniformly
        and runs a small number of replications per combination to estimate mean wait time.
        """
        rng = np.random.default_rng(self.base_seed + 999999)
        n_samples = self.analysis_config.metamodel_n_samples
        n_reps = self.analysis_config.metamodel_reps_per_sample

        arrival_rates_sample = rng.uniform(5.0, 15.0, size=n_samples)
        chargers_sample = rng.integers(2, 7, size=n_samples)
        k_sample = rng.uniform(0.02, 0.10, size=n_samples)
        q_max_sample = rng.uniform(50.0, 100.0, size=n_samples)
        max_soc_sample = rng.uniform(0.30, 0.65, size=n_samples)

        rows: list[dict[str, float]] = []
        report_every = max(1, n_samples // 5)
        self._log(verbose, f"[metamodel] Generating {n_samples} training samples ({n_reps} reps each)")

        for i in range(n_samples):
            sim_cfg = replace(
                self.simulation_config,
                arrival_rate_per_hour=float(arrival_rates_sample[i]),
                max_initial_soc=float(max_soc_sample[i]),
            )
            chg_cfg = replace(
                self.charging_config,
                charge_rate_constant_per_minute=float(k_sample[i]),
                default_battery_capacity_kwh=float(q_max_sample[i]),
            )
            wait_times: list[float] = []
            for rep_id in range(n_reps):
                simulator = ChargingStationSimulator(
                    simulation_config=sim_cfg,
                    charging_config=chg_cfg,
                    chargers=int(chargers_sample[i]),
                    seed=self.base_seed + 500000 + i * 100 + rep_id,
                    replication_id=rep_id,
                )
                df = simulator.run()
                if not df.empty:
                    wait_times.append(float(df["wait_time_min"].mean()))

            if wait_times:
                rows.append({
                    "arrival_rate": float(arrival_rates_sample[i]),
                    "chargers": float(chargers_sample[i]),
                    "inv_chargers": 1.0 / float(chargers_sample[i]),
                    "k": float(k_sample[i]),
                    "q_max_kwh": float(q_max_sample[i]),
                    "max_initial_soc": float(max_soc_sample[i]),
                    "mean_wait_min": float(np.mean(wait_times)),
                })

            done = i + 1
            if done == 1 or done == n_samples or done % report_every == 0:
                self._log(verbose, f"  -> metamodel training: {done}/{n_samples} samples done")

        return pd.DataFrame(rows)

    def _run_sensitivity_2d(
        self,
        arrival_rates: list[float],
        k_values: list[float],
        chargers: int,
        warm_up_minutes: float,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Run 2D sensitivity grid over arrival_rate × charging-rate k with fixed charger count.

        Args:
            arrival_rates: Arrival rate levels to sweep [vehicles/hour].
            k_values: Charging rate constant levels to sweep [1/min].
            chargers: Fixed charger count for the 2D grid.
            warm_up_minutes: Warm-up period excluded from wait-time calculation.
        """
        all_rows: list[dict[str, float]] = []
        n_reps = self.analysis_config.sensitivity_replications
        total = len(arrival_rates) * len(k_values)
        combo = 0

        for lam_idx, lam in enumerate(arrival_rates):
            for k_idx, k in enumerate(k_values):
                combo += 1
                self._log(verbose, f"[sensitivity_2d] ({combo}/{total}) λ={lam}, k={k:.4f}")
                sim_cfg = replace(self.simulation_config, arrival_rate_per_hour=lam)
                chg_cfg = replace(self.charging_config, charge_rate_constant_per_minute=k)

                wait_times: list[float] = []
                for rep_id in range(n_reps):
                    simulator = ChargingStationSimulator(
                        simulation_config=sim_cfg,
                        charging_config=chg_cfg,
                        chargers=chargers,
                        seed=self.base_seed + 700000 + lam_idx * 50000 + k_idx * 1000 + rep_id,
                        replication_id=rep_id,
                    )
                    df = simulator.run()
                    if not df.empty:
                        post = ResultAnalyzer.exclude_warmup(df, warm_up_minutes)
                        if not post.empty:
                            wait_times.append(float(post["wait_time_min"].mean()))

                if wait_times:
                    all_rows.append({
                        "arrival_rate_per_hour": lam,
                        "k_value": k,
                        "avg_wait_min": float(np.mean(wait_times)),
                        "chargers": float(chargers),
                    })

        return pd.DataFrame(all_rows)

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

        # Determine recommended number of replications from pilot data
        pilot_wait = replication_metrics["mean_wait_min"].values
        n_recommended = ResultAnalyzer.determine_n_replications(
            pilot_data=pilot_wait,
            target_half_width=2.0,
        )
        self._log(verbose, f"  -> Recommended replications for CI half-width ≤2 min: {n_recommended}")
        n_reps_df = pd.DataFrame([{
            "actual_replications": int(replication_metrics["replication_id"].nunique()),
            "recommended_replications": n_recommended,
            "target_half_width_min": 2.0,
            "confidence": 0.95,
        }])

        # Generate rich multi-dimensional training data for metamodel
        self._log(verbose, "[step] Generating metamodel training data (5D parameter space)")
        metamodel_training_data = self._generate_metamodel_training_data(verbose=verbose)

        metamodel_metrics, metamodel_predictions, feature_importance = ResultAnalyzer.fit_wait_metamodel(
            replication_metrics=replication_metrics,
            scenario_summary=summary,
            test_size=self.analysis_config.metamodel_test_size,
            random_forest_estimators=self.analysis_config.random_forest_estimators,
            training_data=metamodel_training_data if not metamodel_training_data.empty else None,
        )

        self._log(verbose, "[step] Saving tables")
        records.to_csv(table_dir / "vehicle_records.csv", index=False)
        records_analysis.to_csv(table_dir / "vehicle_records_analysis.csv", index=False)
        summary.to_csv(table_dir / "scenario_summary.csv", index=False)
        cost_benefit.to_csv(table_dir / "cost_benefit.csv", index=False)
        replication_metrics.to_csv(table_dir / "replication_metrics.csv", index=False)
        metamodel_metrics.to_csv(table_dir / "metamodel_metrics.csv", index=False)
        metamodel_predictions.to_csv(table_dir / "metamodel_predictions.csv", index=False)
        metamodel_training_data.to_csv(table_dir / "metamodel_training_data.csv", index=False)
        feature_importance.to_csv(table_dir / "feature_importance.csv", index=False)
        n_reps_df.to_csv(table_dir / "n_replications_recommendation.csv", index=False)
        welch_profile_path = table_dir / "welch_warmup_profile.csv"
        if not warmup_profile.empty:
            warmup_profile.to_csv(welch_profile_path, index=False)
        elif welch_profile_path.exists():
            welch_profile_path.unlink()

        sensitivity_summary = pd.DataFrame()
        sensitivity_2d = pd.DataFrame()
        if sensitivity_arrival_rates:
            self._log(verbose, "[step] Running 1D sensitivity analysis (arrival rate)")
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

            # 2D sensitivity: arrival_rate × k
            self._log(verbose, "[step] Running 2D sensitivity analysis (arrival rate × k)")
            mid_chargers = scenarios[len(scenarios) // 2]
            sensitivity_2d = self._run_sensitivity_2d(
                arrival_rates=sensitivity_arrival_rates,
                k_values=self.analysis_config.k_sensitivity_values,
                chargers=mid_chargers,
                warm_up_minutes=warm_up_minutes,
                verbose=verbose,
            )
            if not sensitivity_2d.empty:
                sensitivity_2d.to_csv(table_dir / "sensitivity_2d_summary.csv", index=False)

        self._log(verbose, "[step] Generating figures")
        visualizer = ResultVisualizer()
        visualizer.plot_wait_distribution(records_analysis, fig_dir / "wait_distribution.png")
        visualizer.plot_summary_wait(summary, fig_dir / "wait_summary.png")
        visualizer.plot_cost_benefit(cost_benefit, fig_dir / "cost_benefit.png")
        visualizer.plot_charging_curve(ChargingModel(self.charging_config), fig_dir / "charging_curve.png")
        visualizer.plot_metamodel_fit(metamodel_predictions, fig_dir / "metamodel_fit.png")
        visualizer.plot_feature_importance(feature_importance, fig_dir / "feature_importance.png")
        visualizer.plot_replication_convergence(replication_metrics, fig_dir / "replication_convergence.png")
        if not sensitivity_summary.empty:
            visualizer.plot_sensitivity_heatmap(sensitivity_summary, fig_dir / "sensitivity_heatmap.png")
        if not sensitivity_2d.empty:
            visualizer.plot_sensitivity_heatmap_2d(sensitivity_2d, fig_dir / "sensitivity_heatmap_2d.png")

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
            "feature_importance": feature_importance,
            "n_replications_recommendation": n_reps_df,
            "warm_up_minutes": pd.DataFrame([{"warm_up_minutes": warm_up_minutes}]),
            "warmup_profile": warmup_profile,
            "sensitivity_summary": sensitivity_summary,
            "sensitivity_2d": sensitivity_2d,
        }
