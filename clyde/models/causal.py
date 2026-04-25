"""Causal event and causal chain dataclasses with JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalEvent:
    """Single actor-to-actor shock transmission link."""

    step: int
    source_actor_id: str
    target_actor_id: str
    channel: str
    variable_affected: str
    magnitude: float
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "source_actor_id": self.source_actor_id,
            "target_actor_id": self.target_actor_id,
            "channel": self.channel,
            "variable_affected": self.variable_affected,
            "magnitude": self.magnitude,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CausalEvent":
        return cls(
            step=int(data["step"]),
            source_actor_id=data["source_actor_id"],
            target_actor_id=data["target_actor_id"],
            channel=data["channel"],
            variable_affected=data["variable_affected"],
            magnitude=float(data["magnitude"]),
            description=data.get("description", ""),
        )


@dataclass
class CausalChain:
    """Ordered sequence of actor-to-actor shock transmissions."""

    chain_id: str
    events: list[CausalEvent] = field(default_factory=list)
    origin_shock: str = ""
    total_magnitude: float = 0.0

    def serialize(self) -> dict:
        """Serialize to JSON-compatible dict. Keys are emitted in a stable order."""
        return {
            "chain_id": self.chain_id,
            "origin_shock": self.origin_shock,
            "total_magnitude": self.total_magnitude,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "CausalChain":
        return cls(
            chain_id=data["chain_id"],
            origin_shock=data.get("origin_shock", ""),
            total_magnitude=float(data.get("total_magnitude", 0.0)),
            events=[CausalEvent.from_dict(e) for e in data.get("events", [])],
        )
