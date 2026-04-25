"""Reporting-side dataclasses: citations, divergence maps, historical analogs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    title: str
    authors: list[str]
    year: int
    source: str
    url: str | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "source": self.source,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        return cls(
            title=data["title"],
            authors=list(data["authors"]),
            year=int(data["year"]),
            source=data["source"],
            url=data.get("url"),
        )


@dataclass
class HistoricalAnalog:
    event_name: str
    year: int
    similarity_score: float
    param_adjustments: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "event_name": self.event_name,
            "year": self.year,
            "similarity_score": self.similarity_score,
            "param_adjustments": dict(self.param_adjustments),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalAnalog":
        return cls(
            event_name=data["event_name"],
            year=int(data["year"]),
            similarity_score=float(data["similarity_score"]),
            param_adjustments={k: float(v) for k, v in data.get("param_adjustments", {}).items()},
            source=data.get("source", ""),
        )


@dataclass
class DivergenceVariable:
    name: str
    sensitivity: float
    current_uncertainty: float
    monitoring_indicator: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sensitivity": self.sensitivity,
            "current_uncertainty": self.current_uncertainty,
            "monitoring_indicator": self.monitoring_indicator,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DivergenceVariable":
        return cls(
            name=data["name"],
            sensitivity=float(data["sensitivity"]),
            current_uncertainty=float(data["current_uncertainty"]),
            monitoring_indicator=data["monitoring_indicator"],
        )


@dataclass
class DivergenceMap:
    variables: list[DivergenceVariable] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"variables": [v.to_dict() for v in self.variables]}

    @classmethod
    def from_dict(cls, data: dict) -> "DivergenceMap":
        return cls(variables=[DivergenceVariable.from_dict(v) for v in data.get("variables", [])])
