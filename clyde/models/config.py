"""ShockConfig, ShockDelta, SimulationWorld."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clyde.models.actors import Actor
from clyde.models.networks import NetworkBundle
from clyde.models.reporting import HistoricalAnalog
from clyde.models.time import TimeHorizon


VALID_SCOPES: frozenset[str] = frozenset({"micro", "sectoral", "macro", "cross_border"})


@dataclass
class ShockConfig:
    """Complete specification for initializing a simulation."""

    shock_type: str
    severity: float
    scope: str
    duration_steps: int
    geography: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    initial_contact_actors: list[str] = field(default_factory=list)
    agent_counts: dict[str, int] = field(default_factory=dict)
    behavioral_overrides: dict[str, Any] = field(default_factory=dict)
    time_horizon: TimeHorizon = field(default_factory=lambda: TimeHorizon(steps=0, step_unit="day"))
    ensemble_seed: int = 0
    historical_analogs: list[HistoricalAnalog] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"ShockConfig.scope must be one of {sorted(VALID_SCOPES)}, got {self.scope!r}")
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"ShockConfig.severity must be in [0, 1], got {self.severity}")
        if self.duration_steps < 0:
            raise ValueError(f"ShockConfig.duration_steps must be >= 0, got {self.duration_steps}")

    def to_dict(self) -> dict:
        return {
            "shock_type": self.shock_type,
            "severity": self.severity,
            "scope": self.scope,
            "duration_steps": self.duration_steps,
            "geography": list(self.geography),
            "sectors": list(self.sectors),
            "initial_contact_actors": list(self.initial_contact_actors),
            "agent_counts": {k: int(v) for k, v in self.agent_counts.items()},
            "behavioral_overrides": dict(self.behavioral_overrides),
            "time_horizon": self.time_horizon.to_dict(),
            "ensemble_seed": self.ensemble_seed,
            "historical_analogs": [a.to_dict() for a in self.historical_analogs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShockConfig":
        return cls(
            shock_type=data["shock_type"],
            severity=float(data["severity"]),
            scope=data["scope"],
            duration_steps=int(data["duration_steps"]),
            geography=list(data.get("geography", [])),
            sectors=list(data.get("sectors", [])),
            initial_contact_actors=list(data.get("initial_contact_actors", [])),
            agent_counts={k: int(v) for k, v in data.get("agent_counts", {}).items()},
            behavioral_overrides=dict(data.get("behavioral_overrides", {})),
            time_horizon=TimeHorizon.from_dict(data.get("time_horizon", {"steps": 0, "step_unit": "day"})),
            ensemble_seed=int(data.get("ensemble_seed", 0)),
            historical_analogs=[HistoricalAnalog.from_dict(a) for a in data.get("historical_analogs", [])],
        )


@dataclass
class ShockDelta:
    """Intervention override for branch creation via God's Eye Console."""

    intervention_step: int
    param_overrides: dict[str, Any] = field(default_factory=dict)
    new_events: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "intervention_step": self.intervention_step,
            "param_overrides": dict(self.param_overrides),
            "new_events": list(self.new_events),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShockDelta":
        return cls(
            intervention_step=int(data["intervention_step"]),
            param_overrides=dict(data.get("param_overrides", {})),
            new_events=list(data.get("new_events", [])),
            description=data.get("description", ""),
        )


@dataclass
class SimulationWorld:
    """Fully resolved world ready for simulation. No LLM needed beyond this."""

    config: ShockConfig
    actors: list[Actor] = field(default_factory=list)
    networks: NetworkBundle = field(default_factory=NetworkBundle)
    prior_library_version: str = ""

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "actors": [a.to_dict() for a in self.actors],
            "networks": self.networks.to_dict(),
            "prior_library_version": self.prior_library_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationWorld":
        return cls(
            config=ShockConfig.from_dict(data["config"]),
            actors=[Actor.from_dict(a) for a in data.get("actors", [])],
            networks=NetworkBundle.from_dict(data.get("networks", {})),
            prior_library_version=data.get("prior_library_version", ""),
        )
