"""Statistical post-processing and cost-benefit calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class ResultAnalyzer:
    """Transforms raw simulation output into scenario-level decision metrics."""

    @staticmethod
    def _ci95_halfwidth_t(std: pd.Series, n: pd.Series) -> pd.Series:
        """Compute 95% CI half-width using Student-t distribution."""

        n_safe = n.clip(lower=2).astype(float)
        t_crit = pd.Series(stats.t.ppf(0.975, n_safe - 1), index=n.index)
        return t_crit * (std / np.sqrt(n_safe))

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
    def exclude_warmup(records: pd.DataFrame, warm_up_minutes: float) -> pd.DataFrame:
        """Exclude vehicles that arrived during warm-up period."""

        if warm_up_minutes <= 0:
            return records.copy()
        return records.loc[records["arrival_time_min"] >= warm_up_minutes].reset_index(drop=True)

    @staticmethod
    def estimate_warmup_welch(
        records: pd.DataFrame,
        simulation_minutes: float,
        bin_minutes: float,
        smoothing_bins: int,
        stability_bins: int,
        relative_tolerance: float,
    ) -> tuple[float, pd.DataFrame]:
        """Estimate warm-up period using Welch-style moving-average stabilization."""

        if records.empty:
            return 0.0, pd.DataFrame()

        working = records.copy()
        working["time_bin_start"] = (working["arrival_time_min"] // bin_minutes) * bin_minutes
        working = working.loc[working["time_bin_start"] < simulation_minutes]

        profile = (
            working.groupby(["chargers", "replication_id", "time_bin_start"], as_index=False)["wait_time_min"]
            .mean()
            .rename(columns={"wait_time_min": "mean_wait_bin"})
        )
        profile = (
            profile.groupby(["chargers", "time_bin_start"], as_index=False)["mean_wait_bin"]
            .mean()
            .rename(columns={"mean_wait_bin": "mean_wait"})
        )
        profile["smooth_wait"] = (
            profile.sort_values(["chargers", "time_bin_start"])
            .groupby("chargers")["mean_wait"]
            .transform(lambda s: s.rolling(window=smoothing_bins, min_periods=1).mean())
        )

        per_charger_warmup: list[dict[str, float]] = []
        for chargers, group in profile.groupby("chargers"):
            seq = group.sort_values("time_bin_start").reset_index(drop=True)
            if seq.empty:
                per_charger_warmup.append({"chargers": float(chargers), "warmup_minutes": 0.0})
                continue

            tail_count = min(stability_bins, len(seq))
            target_level = float(seq["smooth_wait"].tail(tail_count).mean())
            tolerance_abs = relative_tolerance * max(abs(target_level), 1.0)

            warmup = float(seq["time_bin_start"].iloc[0])
            for idx in range(max(0, len(seq) - stability_bins + 1)):
                window = seq.iloc[idx : idx + stability_bins]
                within = (window["smooth_wait"] - target_level).abs() <= tolerance_abs
                if bool(within.all()):
                    warmup = float(window["time_bin_start"].iloc[0])
                    break

            per_charger_warmup.append({"chargers": float(chargers), "warmup_minutes": warmup})

        warmup_df = pd.DataFrame(per_charger_warmup).sort_values("chargers").reset_index(drop=True)
        estimated = float(warmup_df["warmup_minutes"].max()) if not warmup_df.empty else 0.0
        return estimated, warmup_df

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

        scenario["wait_ci95_halfwidth"] = ResultAnalyzer._ci95_halfwidth_t(
            scenario["std_wait_min"],
            scenario["replications"],
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

    @staticmethod
    def sensitivity_summary(records: pd.DataFrame, simulation_minutes: float) -> pd.DataFrame:
        """Aggregate metrics for sensitivity runs across arrival-rate scenarios."""

        if records.empty:
            return pd.DataFrame()

        rep = records.groupby(["arrival_rate_per_hour", "chargers", "replication_id"], as_index=False).agg(
            mean_wait_min=("wait_time_min", "mean"),
            max_wait_min=("wait_time_min", "max"),
            served_vehicles=("vehicle_id", "count"),
        )

        scenario = rep.groupby(["arrival_rate_per_hour", "chargers"], as_index=False).agg(
            avg_wait_min=("mean_wait_min", "mean"),
            avg_max_wait_min=("max_wait_min", "mean"),
            avg_served_vehicles=("served_vehicles", "mean"),
            std_wait_min=("mean_wait_min", "std"),
            replications=("replication_id", "nunique"),
        )

        scenario["wait_ci95_halfwidth"] = ResultAnalyzer._ci95_halfwidth_t(
            scenario["std_wait_min"],
            scenario["replications"],
        )
        scenario["wait_ci95_low"] = scenario["avg_wait_min"] - scenario["wait_ci95_halfwidth"]
        scenario["wait_ci95_high"] = scenario["avg_wait_min"] + scenario["wait_ci95_halfwidth"]

        total_charge = records.groupby(["arrival_rate_per_hour", "chargers", "replication_id"], as_index=False)[
            "charge_time_min"
        ].sum()
        util = total_charge.groupby(["arrival_rate_per_hour", "chargers"], as_index=False)["charge_time_min"].mean()
        util = util.rename(columns={"charge_time_min": "avg_total_charge_time_min"})
        scenario = scenario.merge(util, on=["arrival_rate_per_hour", "chargers"], how="left")
        scenario["avg_utilization"] = scenario["avg_total_charge_time_min"] / (
            scenario["chargers"] * simulation_minutes
        )

        return scenario.sort_values(["arrival_rate_per_hour", "chargers"]).reset_index(drop=True)

    @staticmethod
    def fit_wait_metamodel(
        replication_metrics: pd.DataFrame,
        scenario_summary: pd.DataFrame,
        test_size: float,
        random_forest_estimators: int,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit linear and random-forest metamodels and report R2/MAE."""

        if replication_metrics.empty:
            raise ValueError("replication_metrics DataFrame is empty")

        features = pd.DataFrame(
            {
                "chargers": replication_metrics["chargers"].astype(float),
                "inv_chargers": 1.0 / replication_metrics["chargers"].astype(float),
            }
        )
        target = replication_metrics["mean_wait_min"].astype(float)

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )

        models: dict[str, object] = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=random_forest_estimators,
                random_state=random_state,
            ),
        }

        pred_features = pd.DataFrame(
            {
                "chargers": scenario_summary["chargers"].astype(float),
                "inv_chargers": 1.0 / scenario_summary["chargers"].astype(float),
            }
        )

        metric_rows: list[dict[str, float | str]] = []
        prediction_frames: list[pd.DataFrame] = []
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred_test = model.predict(x_test)
            metric_rows.extend(
                [
                    {
                        "model": model_name,
                        "metric": "R2",
                        "value": float(r2_score(y_test, y_pred_test)),
                    },
                    {
                        "model": model_name,
                        "metric": "MAE",
                        "value": float(mean_absolute_error(y_test, y_pred_test)),
                    },
                ]
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "chargers": scenario_summary["chargers"],
                        "observed_avg_wait_min": scenario_summary["avg_wait_min"],
                        "predicted_avg_wait_min": model.predict(pred_features),
                    }
                )
            )

        metrics = pd.DataFrame(metric_rows)
        predictions = pd.concat(prediction_frames, ignore_index=True)

        return metrics, predictions
