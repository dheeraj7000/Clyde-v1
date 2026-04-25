"""Input-layer dataclasses: parser results, ambiguities, documents, actor hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from clyde.models.time import TimeHorizon


@dataclass
class Document:
    """A piece of ingested content (PDF/MD/TXT) with extracted text."""

    path: str
    content: str
    format: str  # 'pdf' | 'md' | 'txt'
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path, content: str, fmt: str, metadata: dict | None = None) -> "Document":
        return cls(path=str(path), content=content, format=fmt, metadata=metadata or {})


@dataclass
class Ambiguity:
    field: str
    description: str
    options: list[str] | None = None
    resolved: bool = False
    resolution: str | None = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "description": self.description,
            "options": list(self.options) if self.options is not None else None,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Ambiguity":
        opts = data.get("options")
        return cls(
            field=data["field"],
            description=data["description"],
            options=list(opts) if opts is not None else None,
            resolved=bool(data.get("resolved", False)),
            resolution=data.get("resolution"),
        )


@dataclass
class ActorHint:
    """Hint extracted by the parser about which actor types are likely relevant."""

    actor_type: str
    count_estimate: int | None = None
    description: str = ""


@dataclass
class ShockParams:
    """Raw shock parameters extracted from NL input (pre-KnowledgeGraph)."""

    shock_type: str = ""
    severity: float = 0.0
    scope: str = "micro"
    duration_steps: int = 0
    initial_contact_actors: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    triggering_event: str
    geographies: list[str] = field(default_factory=list)
    markets: list[str] = field(default_factory=list)
    shock_params: ShockParams = field(default_factory=ShockParams)
    time_horizon: TimeHorizon = field(default_factory=lambda: TimeHorizon(steps=0, step_unit="day"))
    ambiguities: list[Ambiguity] = field(default_factory=list)
    actor_hints: list[ActorHint] = field(default_factory=list)
