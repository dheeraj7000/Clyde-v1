"""Constructs the three network structures used by the Propagation Engine.

Topology constraints enforced here match Property 3:

- Labor market is **bipartite**: every edge connects a household to a firm.
- Supply chain is **directed**: every edge is firm → firm or firm → household.
- Interbank is bank-to-bank only (a scale-free graph built via preferential
  attachment).

The builder accepts ``list[Actor]`` inputs and filters by ``actor_type``, so
callers don't have to pre-partition. Determinism is controlled by a seeded
``random.Random`` instance — same seed, same graph.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clyde.models.actors import Actor
from clyde.models.enums import ActorType
from clyde.models.networks import BipartiteGraph, DirectedGraph, Edge, ScaleFreeGraph


DEFAULT_EMPLOYMENT_RATE = 0.94
DEFAULT_SUPPLIERS_PER_FIRM_RANGE = (1, 4)
DEFAULT_CUSTOMERS_PER_HOUSEHOLD_RANGE = (1, 3)
DEFAULT_INTERBANK_M = 2  # preferential-attachment degree per new node


@dataclass(frozen=True)
class NetworkBuildConfig:
    employment_rate: float = DEFAULT_EMPLOYMENT_RATE
    suppliers_per_firm_range: tuple[int, int] = DEFAULT_SUPPLIERS_PER_FIRM_RANGE
    customers_per_household_range: tuple[int, int] = DEFAULT_CUSTOMERS_PER_HOUSEHOLD_RANGE
    interbank_m: int = DEFAULT_INTERBANK_M


def _filter(actors: list[Actor], actor_type: ActorType) -> list[Actor]:
    return [a for a in actors if a.actor_type == actor_type]


class NetworkBuilder:
    """Constructs the labor/supply/interbank networks for a SimulationWorld."""

    def __init__(
        self,
        config: NetworkBuildConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or NetworkBuildConfig()
        self.rng = rng or random.Random()

    # ----- Labor market (bipartite) -----------------------------------------

    def build_labor_market(
        self,
        households: list[Actor],
        firms: list[Actor],
    ) -> BipartiteGraph:
        """Match each household to at most one firm via an employment edge."""
        hs = _filter(households, ActorType.HOUSEHOLD)
        fs = _filter(firms, ActorType.FIRM)
        if not hs or not fs:
            return BipartiteGraph(edges=[])

        edges: list[Edge] = []
        rate = self.config.employment_rate
        for h in hs:
            if self.rng.random() < rate:
                firm = self.rng.choice(fs)
                edges.append((h.id, firm.id, 1.0))
        return BipartiteGraph(edges=edges)

    # ----- Supply chain (directed) ------------------------------------------

    def build_supply_chain(
        self,
        firms: list[Actor],
        households: list[Actor],
    ) -> DirectedGraph:
        """Directed edges: firm → firm (supply) and firm → household (sales)."""
        fs = _filter(firms, ActorType.FIRM)
        hs = _filter(households, ActorType.HOUSEHOLD)
        edges: list[Edge] = []

        # Firm-to-firm supply edges
        if len(fs) >= 2:
            lo, hi = self.config.suppliers_per_firm_range
            lo = max(0, lo)
            hi = max(lo, hi)
            for buyer in fs:
                candidates = [f for f in fs if f.id != buyer.id]
                if not candidates:
                    continue
                count = self.rng.randint(lo, min(hi, len(candidates)))
                if count <= 0:
                    continue
                suppliers = self.rng.sample(candidates, count)
                for sup in suppliers:
                    weight = round(self.rng.uniform(0.1, 1.0), 4)
                    edges.append((sup.id, buyer.id, weight))

        # Firm-to-household consumer sales
        if fs and hs:
            lo, hi = self.config.customers_per_household_range
            lo = max(0, lo)
            hi = max(lo, hi)
            for h in hs:
                count = self.rng.randint(lo, min(hi, len(fs)))
                if count <= 0:
                    continue
                sellers = self.rng.sample(fs, count)
                for seller in sellers:
                    weight = round(self.rng.uniform(0.1, 1.0), 4)
                    edges.append((seller.id, h.id, weight))

        return DirectedGraph(edges=edges)

    # ----- Interbank (scale-free, Barabási–Albert) --------------------------

    def build_interbank(self, banks: list[Actor]) -> ScaleFreeGraph:
        """Barabási–Albert preferential attachment over bank actors."""
        bs = _filter(banks, ActorType.BANK)
        if len(bs) < 2:
            return ScaleFreeGraph(edges=[])

        m = max(1, min(self.config.interbank_m, len(bs) - 1))
        edges: list[Edge] = []
        degrees: dict[str, int] = {b.id: 0 for b in bs}

        # Seed: m+1 fully-connected initial nodes (or just the first two).
        seed_count = min(m + 1, len(bs))
        seed_nodes = bs[:seed_count]
        for i in range(len(seed_nodes)):
            for j in range(i + 1, len(seed_nodes)):
                a, b = seed_nodes[i].id, seed_nodes[j].id
                w = round(self.rng.uniform(0.1, 1.0), 4)
                edges.append((a, b, w))
                degrees[a] += 1
                degrees[b] += 1

        # Add remaining banks, each attaching to m existing nodes preferentially.
        for new_bank in bs[seed_count:]:
            existing = [b for b in bs if b.id != new_bank.id and (degrees[b.id] > 0 or b in seed_nodes)]
            # Deduplicate while preserving order
            seen: set[str] = set()
            pool: list[Actor] = []
            for b in existing:
                if b.id != new_bank.id and b.id not in seen:
                    pool.append(b)
                    seen.add(b.id)
            if not pool:
                pool = [b for b in bs if b.id != new_bank.id]
            chosen: set[str] = set()
            attempts = 0
            while len(chosen) < m and attempts < 10 * m:
                weights = [max(1, degrees[b.id]) for b in pool]
                pick = self.rng.choices(pool, weights=weights, k=1)[0]
                if pick.id not in chosen:
                    chosen.add(pick.id)
                    w = round(self.rng.uniform(0.1, 1.0), 4)
                    edges.append((pick.id, new_bank.id, w))
                    degrees[pick.id] += 1
                    degrees[new_bank.id] += 1
                attempts += 1

        return ScaleFreeGraph(edges=edges)
