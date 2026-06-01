"""Battery charging model solved through SciPy ODE integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .config import ChargingConfig


@dataclass(slots=True)
class ChargingModel:
    """Encapsulates ODE-based charging-time estimation for EV batteries."""

    config: ChargingConfig

    @staticmethod
    def _ode_rhs(_: float, q: np.ndarray, q_max: float, k: float) -> np.ndarray:
        """Right-hand side for dQ/dt = k(Qmax - Q)."""

        return np.array([k * (q_max - q[0])], dtype=float)

    @staticmethod
    def _analytical_time_minutes(q0: float, q_target: float, q_max: float, k: float) -> float:
        """Closed-form time for first-order charging model, used as fallback."""

        if q_target <= q0:
            return 0.0
        ratio = (q_max - q_target) / (q_max - q0)
        ratio = float(np.clip(ratio, 1e-10, 1.0))
        return float(-np.log(ratio) / k)

    def charging_time_minutes(
        self,
        initial_soc: float,
        target_soc: float,
        battery_capacity_kwh: float | None = None,
    ) -> float:
        """Estimate charging duration from initial SoC to target SoC in minutes."""

        if not 0 <= initial_soc < 1:
            raise ValueError("initial_soc must be in [0, 1)")
        if not 0 < target_soc <= 1:
            raise ValueError("target_soc must be in (0, 1]")
        if target_soc <= initial_soc:
            return 0.0

        q_max = battery_capacity_kwh or self.config.default_battery_capacity_kwh
        q0 = initial_soc * q_max
        q_target = target_soc * q_max
        k = self.config.charge_rate_constant_per_minute

        def reach_target_event(_: float, q: np.ndarray) -> float:
            return q[0] - q_target

        reach_target_event.terminal = True
        reach_target_event.direction = 1

        result = solve_ivp(
            fun=lambda t, y: self._ode_rhs(t, y, q_max=q_max, k=k),
            t_span=(0.0, self.config.max_solver_time_minutes),
            y0=np.array([q0], dtype=float),
            events=reach_target_event,
            max_step=0.5,
            method="RK45",
        )

        if result.success and result.t_events[0].size > 0:
            return float(result.t_events[0][0])

        return self._analytical_time_minutes(q0=q0, q_target=q_target, q_max=q_max, k=k)

    def charging_profile(
        self,
        initial_soc: float,
        duration_minutes: float,
        points: int = 250,
        battery_capacity_kwh: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate state-of-charge profile over time for visualization."""

        q_max = battery_capacity_kwh or self.config.default_battery_capacity_kwh
        q0 = initial_soc * q_max
        k = self.config.charge_rate_constant_per_minute

        timeline = np.linspace(0.0, max(duration_minutes, 1e-6), points)
        soc = 1.0 - ((q_max - q0) / q_max) * np.exp(-k * timeline)
        return timeline, soc
