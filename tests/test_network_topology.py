# Feature: clyde-economic-simulator, Property 3: Network Topology Constraints
"""Property-based tests for the three network topologies built by NetworkBuilder.

- Labor market: bipartite (household ↔ firm only).
- Supply chain: directed (firm → {firm, household}); no household origins, no self-loops.
- Interbank: bank ↔ bank only; no self-loops; non-empty when ≥ 2 banks exist.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clyde.models.actors import Actor
from clyde.models.enums import ActorType
from clyde.models.networks import BipartiteGraph, DirectedGraph, ScaleFreeGraph
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary


_LIB = PriorLibrary()

_PREFIX_BY_TYPE: dict[ActorType, str] = {
    ActorType.HOUSEHOLD: "h",
    ActorType.FIRM: "f",
    ActorType.BANK: "b",
    ActorType.CENTRAL_BANK: "cb",
}


def _make_actor(atype: ActorType, idx: int) -> Actor:
    params = _LIB.get_params(atype)
    return Actor(id=f"{_PREFIX_BY_TYPE[atype]}{idx}", actor_type=atype, params=params)


@st.composite
def actors(draw, atype: ActorType, min_size: int, max_size: int) -> list[Actor]:
    """Draw a list of ``Actor`` instances of ``atype`` with unique ids."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [_make_actor(atype, i) for i in range(n)]


_SEEDS = st.integers(min_value=0, max_value=2**31 - 1)


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    households=actors(ActorType.HOUSEHOLD, 0, 10),
    firms=actors(ActorType.FIRM, 0, 10),
    seed=_SEEDS,
)
def test_labor_market_is_bipartite(
    households: list[Actor], firms: list[Actor], seed: int
) -> None:
    """Every labor-market edge connects a household id to a firm id."""
    builder = NetworkBuilder(rng=random.Random(seed))
    graph = builder.build_labor_market(households, firms)
    assert isinstance(graph, BipartiteGraph)

    household_ids = {a.id for a in households}
    firm_ids = {a.id for a in firms}

    for s, t, w in graph.edges:
        assert s in household_ids, f"source {s!r} is not a household id"
        assert t in firm_ids, f"target {t!r} is not a firm id"
        # Bipartite: never household↔household or firm↔firm.
        assert s not in firm_ids
        assert t not in household_ids
        assert isinstance(w, float)


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    firms=actors(ActorType.FIRM, 0, 10),
    households=actors(ActorType.HOUSEHOLD, 0, 10),
    seed=_SEEDS,
)
def test_supply_chain_is_directed_firm_to_firm_or_household(
    firms: list[Actor], households: list[Actor], seed: int
) -> None:
    """Supply-chain edges originate at firms and terminate at firms or households."""
    builder = NetworkBuilder(rng=random.Random(seed))
    graph = builder.build_supply_chain(firms, households)
    assert isinstance(graph, DirectedGraph)

    firm_ids = {a.id for a in firms}
    household_ids = {a.id for a in households}

    for s, t, w in graph.edges:
        assert s in firm_ids, f"source {s!r} is not a firm id (no household origins)"
        assert s not in household_ids, "household cannot be a supply-chain source"
        assert t in firm_ids or t in household_ids, (
            f"target {t!r} must be firm or household"
        )
        assert s != t, "no self-loops in supply chain"
        assert isinstance(w, float)


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(banks=actors(ActorType.BANK, 0, 10), seed=_SEEDS)
def test_interbank_is_bank_to_bank_only(banks: list[Actor], seed: int) -> None:
    """Every interbank edge connects two distinct bank ids."""
    builder = NetworkBuilder(rng=random.Random(seed))
    graph = builder.build_interbank(banks)
    assert isinstance(graph, ScaleFreeGraph)

    bank_ids = {a.id for a in banks}

    for s, t, w in graph.edges:
        assert s in bank_ids, f"source {s!r} is not a bank id"
        assert t in bank_ids, f"target {t!r} is not a bank id"
        assert s != t, "no self-loops in interbank graph"
        assert isinstance(w, float)

    if len(banks) >= 2:
        assert len(graph.edges) >= 1, (
            "interbank graph must be non-empty when ≥ 2 banks are provided"
        )


def test_empty_inputs_yield_empty_graphs_of_correct_type() -> None:
    """With no actors, each builder returns the correct empty graph type."""
    builder = NetworkBuilder(rng=random.Random(0))

    labor = builder.build_labor_market([], [])
    supply = builder.build_supply_chain([], [])
    interbank = builder.build_interbank([])

    assert isinstance(labor, BipartiteGraph)
    assert isinstance(supply, DirectedGraph)
    assert isinstance(interbank, ScaleFreeGraph)
    assert labor.edges == []
    assert supply.edges == []
    assert interbank.edges == []
