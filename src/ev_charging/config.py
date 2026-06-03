"""Configuration objects for EV charging simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChargingConfig:
    """Physical and numerical parameters for the charging model."""

    default_battery_capacity_kwh: float = 75.0
    charge_rate_constant_per_minute: float = 0.055
    max_solver_time_minutes: float = 180.0

    def __post_init__(self) -> None:
        if self.default_battery_capacity_kwh <= 0:
            raise ValueError("default_battery_capacity_kwh must be positive")
        if self.charge_rate_constant_per_minute <= 0:
            raise ValueError("charge_rate_constant_per_minute must be positive")
        if self.max_solver_time_minutes <= 0:
            raise ValueError("max_solver_time_minutes must be positive")


@dataclass(slots=True)
class SimulationConfig:
    """Operational parameters of the station and arrival process."""

    simulation_hours: float = 12.0
    arrival_rate_per_hour: float = 9.5
    min_initial_soc: float = 0.08
    max_initial_soc: float = 0.65
    min_target_soc: float = 0.75
    max_target_soc: float = 0.90

    def __post_init__(self) -> None:
        if self.simulation_hours <= 0:
            raise ValueError("simulation_hours must be positive")
        if self.arrival_rate_per_hour <= 0:
            raise ValueError("arrival_rate_per_hour must be positive")
        if not 0 <= self.min_initial_soc < self.max_initial_soc < 1:
            raise ValueError("initial SoC bounds must satisfy 0 <= min < max < 1")
        if not 0 < self.min_target_soc < self.max_target_soc <= 1:
            raise ValueError("target SoC bounds must satisfy 0 < min < max <= 1")
        if self.min_target_soc <= self.min_initial_soc:
            raise ValueError("target SoC should be above initial SoC")

    @property
    def simulation_minutes(self) -> float:
        """Return simulation horizon in minutes."""

        return self.simulation_hours * 60.0


@dataclass(slots=True)
class AnalysisConfig:
    """Parameters used for statistical analysis and what-if economics."""

    replications: int = 120
    warmup_method: str = "welch"
    warm_up_minutes: float = 60.0
    welch_bin_minutes: float = 10.0
    welch_smoothing_bins: int = 3
    welch_stability_bins: int = 3
    welch_relative_tolerance: float = 0.05
    metamodel_test_size: float = 0.25
    random_forest_estimators: int = 200
    sensitivity_replications: int = 60
    infrastructure_cost_per_charger_eur: float = 42000.0

    def __post_init__(self) -> None:
        if self.replications < 2:
            raise ValueError("replications must be at least 2 for Monte Carlo analysis")
        if self.warmup_method not in {"fixed", "welch"}:
            raise ValueError("warmup_method must be either 'fixed' or 'welch'")
        if self.warm_up_minutes < 0:
            raise ValueError("warm_up_minutes cannot be negative")
        if self.welch_bin_minutes <= 0:
            raise ValueError("welch_bin_minutes must be positive")
        if self.welch_smoothing_bins < 1:
            raise ValueError("welch_smoothing_bins must be at least 1")
        if self.welch_stability_bins < 1:
            raise ValueError("welch_stability_bins must be at least 1")
        if not 0 < self.welch_relative_tolerance < 1:
            raise ValueError("welch_relative_tolerance must be in (0, 1)")
        if not 0 < self.metamodel_test_size < 1:
            raise ValueError("metamodel_test_size must be in (0, 1)")
        if self.random_forest_estimators < 10:
            raise ValueError("random_forest_estimators must be at least 10")
        if self.sensitivity_replications < 2:
            raise ValueError("sensitivity_replications must be at least 2")
        if self.infrastructure_cost_per_charger_eur <= 0:
            raise ValueError("infrastructure_cost_per_charger_eur must be positive")
