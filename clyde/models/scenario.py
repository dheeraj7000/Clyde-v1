"""Scenario: a fully serializable simulation specification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from clyde.models.actors import Actor
from clyde.models.config import ShockConfig
from clyde.models.networks import NetworkBundle


@dataclass
class Scenario:
    """Complete simulation specification — serializable and shareable.

    Note: ``overrides`` and ``metadata`` must contain only JSON-serializable
    values if you want full round-trip equality.
    """

    scenario_id: str
    description: str
    config: ShockConfig
    actors: list[Actor] = field(default_factory=list)
    networks: NetworkBundle = field(default_factory=NetworkBundle)
    prior_library_version: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict:
        """Serialize to a JSON-compatible dict with stable field order."""
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "config": self.config.to_dict(),
            "actors": [a.to_dict() for a in self.actors],
            "networks": self.networks.to_dict(),
            "prior_library_version": self.prior_library_version,
            "overrides": dict(self.overrides),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Scenario":
        return cls(
            scenario_id=data["scenario_id"],
            description=data.get("description", ""),
            config=ShockConfig.from_dict(data["config"]),
            actors=[Actor.from_dict(a) for a in data.get("actors", [])],
            networks=NetworkBundle.from_dict(data.get("networks", {})),
            prior_library_version=data.get("prior_library_version", ""),
            overrides=dict(data.get("overrides", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def pretty_print(self) -> str:
        """Produce a human-readable representation of the scenario."""
        c = self.config
        lines: list[str] = []
        lines.append(f"Scenario: {self.scenario_id}")
        lines.append(f"  Description: {self.description}")
        lines.append("  Shock:")
        lines.append(f"    type = {c.shock_type}")
        lines.append(f"    severity = {c.severity:.3f}")
        lines.append(f"    scope = {c.scope}")
        lines.append(f"    duration_steps = {c.duration_steps}")
        if c.geography:
            lines.append(f"    geography = {', '.join(c.geography)}")
        if c.sectors:
            lines.append(f"    sectors = {', '.join(c.sectors)}")
        if c.initial_contact_actors:
            lines.append(
                f"    initial_contact_actors = {', '.join(c.initial_contact_actors)}"
            )
        lines.append(f"  Time horizon: {c.time_horizon.steps} {c.time_horizon.step_unit}(s)")
        lines.append(f"  Ensemble seed: {c.ensemble_seed}")
        if c.agent_counts:
            lines.append("  Agent counts:")
            for k in sorted(c.agent_counts):
                lines.append(f"    {k}: {c.agent_counts[k]}")
        lines.append(f"  Actors: {len(self.actors)}")
        lines.append(
            "  Networks: "
            f"labor_market={len(self.networks.labor_market.edges)} edges, "
            f"supply_chain={len(self.networks.supply_chain.edges)} edges, "
            f"interbank={len(self.networks.interbank.edges)} edges"
        )
        lines.append(f"  Prior library: {self.prior_library_version or '(unset)'}")
        if c.historical_analogs:
            lines.append("  Historical analogs:")
            for a in c.historical_analogs:
                lines.append(
                    f"    - {a.event_name} ({a.year}): similarity={a.similarity_score:.2f}"
                )
        if self.overrides:
            lines.append("  Overrides:")
            for k in sorted(self.overrides):
                lines.append(f"    {k} = {self.overrides[k]!r}")
        if self.metadata:
            lines.append("  Metadata:")
            for k in sorted(self.metadata):
                lines.append(f"    {k} = {self.metadata[k]!r}")
        return "\n".join(lines)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.serialize(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        return cls.deserialize(json.loads(text))
