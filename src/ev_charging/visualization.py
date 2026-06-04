"""Plotting utilities for simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .charging import ChargingModel


class ResultVisualizer:
    """Produces publication-ready figures for the EV queueing study."""

    def __init__(self) -> None:
        sns.set_theme(style="whitegrid", context="talk")

    def plot_wait_distribution(self, records: pd.DataFrame, output_path: Path) -> None:
        """Create distribution chart of waiting times by charger scenario."""

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=records, x="chargers", y="wait_time_min", ax=ax, color="#56B4E9", showfliers=False)
        ax.set_title("Queue Waiting Time by Number of Chargers")
        ax.set_xlabel("Chargers")
        ax.set_ylabel("Waiting Time [min]")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_summary_wait(self, summary: pd.DataFrame, output_path: Path) -> None:
        """Plot average waiting and max waiting trends across scenarios."""

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(summary["chargers"], summary["avg_wait_min"], marker="o", label="Avg wait")
        ax.fill_between(
            summary["chargers"],
            summary["wait_ci95_low"],
            summary["wait_ci95_high"],
            alpha=0.2,
            label="95% CI",
        )
        ax.plot(summary["chargers"], summary["avg_max_wait_min"], marker="s", label="Avg max wait")
        ax.set_title("Queue Delay Metrics vs Charger Count")
        ax.set_xlabel("Chargers")
        ax.set_ylabel("Time [min]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_cost_benefit(self, cost_benefit: pd.DataFrame, output_path: Path) -> None:
        """Plot marginal waiting-time reduction per charger addition."""

        fig, ax = plt.subplots(figsize=(10, 6))
        filtered = cost_benefit[cost_benefit["added_chargers"] > 0]
        ax.bar(filtered["chargers"].astype(str), filtered["delta_avg_wait_min"], color="#009E73")
        ax.set_title("Marginal Reduction of Average Wait")
        ax.set_xlabel("Scenario (Total Chargers)")
        ax.set_ylabel("Minutes Saved vs Previous Scenario")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_charging_curve(self, charging_model: ChargingModel, output_path: Path) -> None:
        """Visualize nonlinear battery charging profile implied by ODE."""

        timeline, soc = charging_model.charging_profile(initial_soc=0.2, duration_minutes=80)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timeline, soc * 100.0, linewidth=2.5, color="#D55E00")
        ax.set_title("Nonlinear EV Charging Curve")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("State of Charge [%]")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_metamodel_fit(self, metamodel_predictions: pd.DataFrame, output_path: Path) -> None:
        """Plot observed versus metamodel-predicted average waiting times."""

        fig, ax = plt.subplots(figsize=(10, 6))
        observed = (
            metamodel_predictions[["chargers", "observed_avg_wait_min"]]
            .drop_duplicates(subset=["chargers"])
            .sort_values("chargers")
        )
        ax.plot(
            observed["chargers"],
            observed["observed_avg_wait_min"],
            marker="o",
            linewidth=2,
            label="Observed",
        )
        for model_name, group in metamodel_predictions.groupby("model"):
            sorted_group = group.sort_values("chargers")
            ax.plot(
                sorted_group["chargers"],
                sorted_group["predicted_avg_wait_min"],
                marker="s",
                linewidth=2,
                linestyle="--",
                label=f"{model_name} prediction",
            )
        ax.set_title("Metamodel Fit: Average Wait vs Chargers")
        ax.set_xlabel("Chargers")
        ax.set_ylabel("Average Waiting Time [min]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_sensitivity_heatmap(self, sensitivity_summary: pd.DataFrame, output_path: Path) -> None:
        """Visualize sensitivity of average waiting time to arrival-rate and capacity changes."""

        pivot = sensitivity_summary.pivot(
            index="arrival_rate_per_hour",
            columns="chargers",
            values="avg_wait_min",
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title("Sensitivity Heatmap: Average Wait [min]")
        ax.set_xlabel("Chargers")
        ax.set_ylabel("Arrival rate [vehicles/hour]")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_sensitivity_heatmap_2d(self, sensitivity_2d: pd.DataFrame, output_path: Path) -> None:
        """2D heatmap: arrival_rate × charging-rate k → average waiting time."""

        pivot = sensitivity_2d.pivot(
            index="arrival_rate_per_hour",
            columns="k_value",
            values="avg_wait_min",
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title("2D Sensitivity: Avg Wait [min] — Arrival Rate × Charging Rate k")
        ax.set_xlabel("Charging rate constant k [1/min]")
        ax.set_ylabel("Arrival rate [vehicles/hour]")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_feature_importance(self, importance_df: pd.DataFrame, output_path: Path) -> None:
        """Horizontal bar chart of Random Forest feature importances."""

        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_df = importance_df.sort_values("importance", ascending=True)
        ax.barh(sorted_df["feature"], sorted_df["importance"], color="#0072B2")
        ax.set_title("Random Forest — Feature Importances")
        ax.set_xlabel("Mean Importance")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    def plot_replication_convergence(self, replication_metrics: pd.DataFrame, output_path: Path) -> None:
        """Cumulative mean of average wait time across replications per charger scenario."""

        fig, ax = plt.subplots(figsize=(10, 6))
        for chargers, group in replication_metrics.groupby("chargers"):
            vals = group.sort_values("replication_id")["mean_wait_min"].values
            cumulative_mean = np.cumsum(vals) / np.arange(1, len(vals) + 1)
            ax.plot(range(1, len(cumulative_mean) + 1), cumulative_mean, label=f"{chargers} chargers")
        ax.set_title("Replication Convergence: Cumulative Mean Wait Time")
        ax.set_xlabel("Number of Replications")
        ax.set_ylabel("Cumulative Mean Wait [min]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
