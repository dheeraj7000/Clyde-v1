"""Network topology dataclasses: labor market, supply chain, interbank."""

from __future__ import annotations

from dataclasses import dataclass, field

Edge = tuple[str, str, float]


def _serialize_edges(edges: list[Edge]) -> list[list]:
    return [[e[0], e[1], float(e[2])] for e in edges]


def _deserialize_edges(data: list) -> list[Edge]:
    return [(str(e[0]), str(e[1]), float(e[2])) for e in data]


@dataclass
class BipartiteGraph:
    """Labor market: household ↔ firm edges."""

    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"edges": _serialize_edges(self.edges)}

    @classmethod
    def from_dict(cls, data: dict) -> "BipartiteGraph":
        return cls(edges=_deserialize_edges(data.get("edges", [])))


@dataclass
class DirectedGraph:
    """Supply chain: firm → firm and firm → household."""

    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"edges": _serialize_edges(self.edges)}

    @classmethod
    def from_dict(cls, data: dict) -> "DirectedGraph":
        return cls(edges=_deserialize_edges(data.get("edges", [])))


@dataclass
class ScaleFreeGraph:
    """Interbank lending network."""

    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"edges": _serialize_edges(self.edges)}

    @classmethod
    def from_dict(cls, data: dict) -> "ScaleFreeGraph":
        return cls(edges=_deserialize_edges(data.get("edges", [])))


@dataclass
class NetworkBundle:
    labor_market: BipartiteGraph = field(default_factory=BipartiteGraph)
    supply_chain: DirectedGraph = field(default_factory=DirectedGraph)
    interbank: ScaleFreeGraph = field(default_factory=ScaleFreeGraph)

    def to_dict(self) -> dict:
        return {
            "labor_market": self.labor_market.to_dict(),
            "supply_chain": self.supply_chain.to_dict(),
            "interbank": self.interbank.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkBundle":
        return cls(
            labor_market=BipartiteGraph.from_dict(data.get("labor_market", {})),
            supply_chain=DirectedGraph.from_dict(data.get("supply_chain", {})),
            interbank=ScaleFreeGraph.from_dict(data.get("interbank", {})),
        )
