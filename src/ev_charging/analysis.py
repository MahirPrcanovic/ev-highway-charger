"""Statistical post-processing and cost-benefit calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ResultAnalyzer:
    """Transforms raw simulation output into scenario-level decision metrics."""

    @staticmethod
    def replication_metrics(records: pd.DataFrame) -> pd.DataFrame:
        """Compute per-replication queue indicators for each charger scenario."""

        if records.empty:
            raise ValueError("records DataFrame is empty")

        grouped = records.groupby(["chargers", "replication_id"], as_index=False)
        rep = grouped.agg(
            mean_wait_min=("wait_time_min", "mean"),
            max_wait_min=("wait_time_min", "max"),
            p95_wait_min=("wait_time_min", lambda x: np.percentile(x, 95)),
            mean_charge_min=("charge_time_min", "mean"),
            served_vehicles=("vehicle_id", "count"),
        )
        return rep

    @staticmethod
    def scenario_summary(records: pd.DataFrame, simulation_minutes: float) -> pd.DataFrame:
        """Aggregate Monte Carlo outputs into means and confidence intervals."""

        rep = ResultAnalyzer.replication_metrics(records)

        scenario = rep.groupby("chargers", as_index=False).agg(
            avg_wait_min=("mean_wait_min", "mean"),
            avg_max_wait_min=("max_wait_min", "mean"),
            avg_p95_wait_min=("p95_wait_min", "mean"),
            avg_served_vehicles=("served_vehicles", "mean"),
            std_wait_min=("mean_wait_min", "std"),
        )

        rep_count = rep.groupby("chargers")["replication_id"].nunique().rename("replications")
        scenario = scenario.merge(rep_count, on="chargers", how="left")

        scenario["wait_ci95_halfwidth"] = 1.96 * (
            scenario["std_wait_min"] / np.sqrt(scenario["replications"].clip(lower=1))
        )
        scenario["wait_ci95_low"] = scenario["avg_wait_min"] - scenario["wait_ci95_halfwidth"]
        scenario["wait_ci95_high"] = scenario["avg_wait_min"] + scenario["wait_ci95_halfwidth"]

        total_charge_by_rep = records.groupby(["chargers", "replication_id"], as_index=False)[
            "charge_time_min"
        ].sum()
        utilization = total_charge_by_rep.groupby("chargers", as_index=False)["charge_time_min"].mean()
        utilization = utilization.rename(columns={"charge_time_min": "avg_total_charge_time_min"})

        scenario = scenario.merge(utilization, on="chargers", how="left")
        scenario["avg_utilization"] = scenario["avg_total_charge_time_min"] / (
            scenario["chargers"] * simulation_minutes
        )

        return scenario.sort_values("chargers").reset_index(drop=True)

    @staticmethod
    def cost_benefit(summary: pd.DataFrame, charger_cost_eur: float) -> pd.DataFrame:
        """Build what-if economics table for incremental charger additions."""

        sorted_summary = summary.sort_values("chargers").reset_index(drop=True).copy()
        sorted_summary["added_chargers"] = sorted_summary["chargers"].diff().fillna(0).astype(int)
        sorted_summary["delta_avg_wait_min"] = sorted_summary["avg_wait_min"].shift(1) - sorted_summary["avg_wait_min"]
        sorted_summary["delta_avg_max_wait_min"] = (
            sorted_summary["avg_max_wait_min"].shift(1) - sorted_summary["avg_max_wait_min"]
        )

        sorted_summary["relative_wait_reduction_pct"] = (
            sorted_summary["delta_avg_wait_min"] / sorted_summary["avg_wait_min"].shift(1) * 100.0
        )
        sorted_summary["incremental_cost_eur"] = sorted_summary["added_chargers"] * charger_cost_eur
        sorted_summary["minutes_saved_per_1k_eur"] = (
            sorted_summary["delta_avg_wait_min"] / (sorted_summary["incremental_cost_eur"] / 1000.0)
        )

        return sorted_summary.fillna(0.0)
