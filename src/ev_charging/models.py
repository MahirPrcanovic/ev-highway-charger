"""Domain models for simulation data exchange."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class VehicleRecord:
    """Single vehicle trajectory through queueing and charging process."""

    vehicle_id: int
    replication_id: int
    chargers: int
    arrival_time_min: float
    service_start_min: float
    departure_time_min: float
    initial_soc: float
    target_soc: float
    charge_time_min: float
    wait_time_min: float

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to plain dictionary for DataFrame creation."""

        return asdict(self)
