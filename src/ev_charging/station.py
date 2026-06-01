"""SimPy-based queue simulation for highway EV charging stations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

from .charging import ChargingModel
from .config import ChargingConfig, SimulationConfig
from .models import VehicleRecord


@dataclass(slots=True)
class ChargingStationSimulator:
    """Runs one Monte Carlo replication for a fixed number of chargers."""

    simulation_config: SimulationConfig
    charging_config: ChargingConfig
    chargers: int
    seed: int
    replication_id: int
    _records: list[VehicleRecord] = field(default_factory=list)
    _rng: np.random.Generator = field(init=False)
    _vehicle_id: int = field(init=False)
    _charging_model: ChargingModel = field(init=False)

    def __post_init__(self) -> None:
        if self.chargers <= 0:
            raise ValueError("chargers must be positive")
        self._rng = np.random.default_rng(self.seed)
        self._vehicle_id = 0
        self._charging_model = ChargingModel(self.charging_config)

    def _sample_initial_soc(self) -> float:
        """Sample initial state of charge from scaled beta distribution."""

        raw = self._rng.beta(2.2, 4.8)
        low = self.simulation_config.min_initial_soc
        high = self.simulation_config.max_initial_soc
        return float(low + raw * (high - low))

    def _sample_target_soc(self, initial_soc: float) -> float:
        """Sample target SoC while enforcing minimum session energy gain."""

        low = max(self.simulation_config.min_target_soc, initial_soc + 0.08)
        high = self.simulation_config.max_target_soc
        if low >= high:
            low = min(high - 0.01, initial_soc + 0.02)
        return float(self._rng.uniform(low, high))

    def _interarrival_minutes(self) -> float:
        """Generate exponential interarrival time from Poisson process rate."""

        rate_per_min = self.simulation_config.arrival_rate_per_hour / 60.0
        return float(self._rng.exponential(1.0 / rate_per_min))

    def _vehicle_process(self, env: simpy.Environment, chargers_resource: simpy.Resource) -> simpy.events.Event:
        """Vehicle lifecycle from arrival to departure."""

        self._vehicle_id += 1
        vehicle_id = self._vehicle_id

        arrival = float(env.now)
        initial_soc = self._sample_initial_soc()
        target_soc = self._sample_target_soc(initial_soc)
        charge_time = self._charging_model.charging_time_minutes(
            initial_soc=initial_soc,
            target_soc=target_soc,
        )

        with chargers_resource.request() as req:
            yield req
            start = float(env.now)
            wait = start - arrival
            yield env.timeout(charge_time)
            departure = float(env.now)

        self._records.append(
            VehicleRecord(
                vehicle_id=vehicle_id,
                replication_id=self.replication_id,
                chargers=self.chargers,
                arrival_time_min=arrival,
                service_start_min=start,
                departure_time_min=departure,
                initial_soc=initial_soc,
                target_soc=target_soc,
                charge_time_min=charge_time,
                wait_time_min=wait,
            )
        )

    def _arrival_generator(self, env: simpy.Environment, chargers_resource: simpy.Resource) -> simpy.events.Event:
        """Create arrivals until simulation horizon is reached."""

        while env.now < self.simulation_config.simulation_minutes:
            yield env.timeout(self._interarrival_minutes())
            if env.now <= self.simulation_config.simulation_minutes:
                env.process(self._vehicle_process(env, chargers_resource))

    def run(self) -> pd.DataFrame:
        """Execute one replication and return vehicle-level records."""

        env = simpy.Environment()
        chargers_resource = simpy.Resource(env, capacity=self.chargers)
        env.process(self._arrival_generator(env, chargers_resource))
        env.run(until=self.simulation_config.simulation_minutes)

        return pd.DataFrame([record.to_dict() for record in self._records])
