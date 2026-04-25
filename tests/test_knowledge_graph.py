"""Tests for :mod:`clyde.setup.knowledge_graph`.

Covers:

* :meth:`KnowledgeGraph.build_from_documents` against a deterministic
  :class:`MockLLMClient` -- different responses per document, source
  tagging, relation persistence.
* :meth:`KnowledgeGraph.query` -- substring match across multiple
  entities; non-matching entries excluded.
* :meth:`KnowledgeGraph.merge_sources` -- conflict semantics
  (Requirement 16.5: NL/document disagreement is surfaced, not silently
  resolved) and the non-conflicting union path.
* :meth:`KnowledgeGraph.extract_shock_config` -- parse-result-driven
  path and graph-fallback path; agent count defaulting.
* Artifact round-trip + listing helpers.
* Smoke test wiring the graph output through
  :class:`EconomicWorldFactory.build_world`.
"""

from __future__ import annotations

import pytest

from clyde.llm import MockLLMClient
from clyde.models.input import (
    ActorHint,
    Document,
    ParseResult,
    ShockParams,
)
from clyde.models.time import TimeHorizon
from clyde.setup.knowledge_graph import (
    Conflict,
    Entity,
    GraphRelation,
    KnowledgeGraph,
    SimulationArtifact,
)
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary
from clyde.setup.world_factory import EconomicWorldFactory


# ---------------------------------------------------------------- Fixtures


def _doc(path: str, content: str = "irrelevant") -> Document:
    return Document(path=path, content=content, format="md", metadata={})


def _parse_result_basic() -> ParseResult:
    return ParseResult(
        triggering_event="oil_shock",
        geographies=["US", "EU"],
        markets=["energy"],
        shock_params=ShockParams(
            shock_type="oil_price_spike",
            severity=0.6,
            scope="macro",
            duration_steps=12,
            initial_contact_actors=["firm_0001"],
        ),
        time_horizon=TimeHorizon(steps=12, step_unit="week"),
        actor_hints=[
            ActorHint(actor_type="household", count_estimate=40, description=""),
            ActorHint(actor_type="firm", count_estimate=8, description=""),
            ActorHint(actor_type="bank", count_estimate=2, description=""),
            ActorHint(actor_type="central_bank", count_estimate=1, description=""),
        ],
    )


# ---------------------------------------------------------------- Tests


@pytest.mark.asyncio
async def test_build_from_documents_extracts_entities_and_relations_per_doc():
    """LLM responses per doc are merged; sources are tagged with the doc path."""
    payload_a = {
        "entities": [
            {"id": "USA", "type": "geography", "name": "United States", "attributes": {"region": "north_america"}},
            {"id": "OIL", "type": "market", "name": "Oil Market", "attributes": {}},
        ],
        "relations": [
            {"source_id": "USA", "target_id": "OIL", "rel_type": "regulates", "weight": 0.5},
        ],
    }
    payload_b = {
        "entities": [
            {"id": "FED", "type": "actor", "name": "Federal Reserve", "attributes": {}},
        ],
        "relations": [
            {"source_id": "FED", "target_id": "USA", "rel_type": "located_in", "weight": 1.0},
        ],
    }
    client = MockLLMClient(responses=[payload_a, payload_b])
    kg = KnowledgeGraph(llm_client=client)

    docs = [_doc("/seed/a.md"), _doc("/seed/b.md")]
    await kg.build_from_documents(docs)

    ids = {e.id for e in kg.list_entities()}
    assert ids == {"USA", "OIL", "FED"}

    usa = next(e for e in kg.list_entities() if e.id == "USA")
    assert usa.sources == ["/seed/a.md"]

    fed = next(e for e in kg.list_entities() if e.id == "FED")
    assert fed.sources == ["/seed/b.md"]

    # Relations are stored with the originating document as their source.
    rels = kg.relations
    assert len(rels) == 2
    rel_pairs = {(r.source_id, r.target_id, r.rel_type) for r in rels}
    assert ("USA", "OIL", "regulates") in rel_pairs
    assert ("FED", "USA", "located_in") in rel_pairs
    rel_a = next(r for r in rels if r.source_id == "USA")
    assert rel_a.sources == ["/seed/a.md"]


@pytest.mark.asyncio
async def test_build_from_documents_merges_parse_result_entities():
    """ParseResult-derived NL entities are added alongside doc entities."""
    payload = {
        "entities": [
            {"id": "OIL", "type": "market", "name": "Oil Market", "attributes": {}},
        ],
        "relations": [],
    }
    client = MockLLMClient(responses=[payload])
    kg = KnowledgeGraph(llm_client=client)

    parse_result = _parse_result_basic()
    await kg.build_from_documents([_doc("/seed/a.md")], parse_result)

    geos = {e.name for e in kg.list_entities(type="geography")}
    assert geos == {"US", "EU"}

    markets = {e.name for e in kg.list_entities(type="market")}
    # NL "energy" + doc "Oil Market" both present.
    assert {"energy", "Oil Market"} <= markets

    actors = {e.name for e in kg.list_entities(type="actor")}
    assert {"household", "firm", "bank", "central_bank"} <= actors

    shocks = kg.list_entities(type="shock")
    assert len(shocks) == 1
    assert shocks[0].attributes["scope"] == "macro"


def test_query_substring_matches_name_case_insensitive():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="ind:headline", type="indicator", name="Headline Inflation", attributes={}, sources=[]))
    kg.add_entity(Entity(id="ind:rate", type="indicator", name="Inflation Rate", attributes={}, sources=[]))
    kg.add_entity(Entity(id="geo:US", type="geography", name="United States", attributes={}, sources=[]))

    nodes = kg.query("inflation")
    names = {n.entity.name for n in nodes}
    assert names == {"Headline Inflation", "Inflation Rate"}

    # Non-matching entity is excluded.
    assert "United States" not in names


def test_query_returns_inbound_and_outbound_edges():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="A", type="actor", name="Alpha", attributes={}, sources=[]))
    kg.add_entity(Entity(id="B", type="actor", name="Beta", attributes={}, sources=[]))
    kg.add_relation(GraphRelation(source_id="A", target_id="B", rel_type="affects", weight=1.0, sources=[]))
    kg.add_relation(GraphRelation(source_id="B", target_id="A", rel_type="lends_to", weight=2.0, sources=[]))

    nodes = kg.query("alpha")
    assert len(nodes) == 1
    node = nodes[0]
    out_pairs = {(r.source_id, r.target_id) for r in node.outbound}
    in_pairs = {(r.source_id, r.target_id) for r in node.inbound}
    assert out_pairs == {("A", "B")}
    assert in_pairs == {("B", "A")}


def test_query_empty_string_returns_no_results():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="A", type="actor", name="Alpha", attributes={}, sources=[]))
    assert kg.query("") == []


def test_merge_sources_emits_conflict_for_name_mismatch():
    kg = KnowledgeGraph()
    nl = Entity(id="USA", type="geography", name="United States", attributes={}, sources=["nl_input"])
    doc = Entity(id="USA", type="geography", name="United States of America", attributes={}, sources=["/seed/a.md"])

    conflicts = kg.merge_sources([nl], [doc])

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.field == "name"
    assert c.nl_value == "United States"
    assert c.doc_value == "United States of America"

    # The conflicting entity is NOT added.
    assert kg.list_entities() == []


def test_merge_sources_attribute_conflict():
    kg = KnowledgeGraph()
    nl = Entity(
        id="shock:oil",
        type="shock",
        name="oil",
        attributes={"severity": 0.5, "scope": "macro"},
        sources=["nl_input"],
    )
    doc = Entity(
        id="shock:oil",
        type="shock",
        name="oil",
        attributes={"severity": 0.9, "scope": "macro"},
        sources=["/seed/a.md"],
    )

    conflicts = kg.merge_sources([nl], [doc])

    assert any(c.field == "attributes.severity" for c in conflicts)
    assert kg.list_entities() == []


def test_merge_sources_non_conflicting_overlap_unions_sources():
    kg = KnowledgeGraph()
    nl = Entity(
        id="USA",
        type="geography",
        name="United States",
        attributes={"region": "north_america"},
        sources=["nl_input"],
    )
    doc = Entity(
        id="USA",
        type="geography",
        name="United States",
        attributes={"timezone": "UTC-5"},
        sources=["/seed/a.md"],
    )

    conflicts = kg.merge_sources([nl], [doc])

    assert conflicts == []
    entities = kg.list_entities()
    assert len(entities) == 1
    merged = entities[0]
    assert merged.id == "USA"
    assert set(merged.sources) == {"nl_input", "/seed/a.md"}
    # Both attributes preserved.
    assert merged.attributes == {"region": "north_america", "timezone": "UTC-5"}


def test_merge_sources_disjoint_lists_add_both_sides():
    kg = KnowledgeGraph()
    nl = Entity(id="A", type="actor", name="Alpha", attributes={}, sources=["nl_input"])
    doc = Entity(id="B", type="actor", name="Beta", attributes={}, sources=["/seed/a.md"])

    conflicts = kg.merge_sources([nl], [doc])

    assert conflicts == []
    ids = {e.id for e in kg.list_entities()}
    assert ids == {"A", "B"}


def test_extract_shock_config_from_parse_result():
    kg = KnowledgeGraph()
    pr = _parse_result_basic()

    cfg = kg.extract_shock_config(pr)

    assert cfg.shock_type == "oil_price_spike"
    assert cfg.severity == pytest.approx(0.6)
    assert cfg.scope == "macro"
    assert cfg.duration_steps == 12
    assert cfg.time_horizon == TimeHorizon(steps=12, step_unit="week")
    assert "US" in cfg.geography and "EU" in cfg.geography
    assert "energy" in cfg.sectors
    assert cfg.initial_contact_actors == ["firm_0001"]
    # Agent counts come from actor hints.
    assert cfg.agent_counts == {
        "household": 40,
        "firm": 8,
        "bank": 2,
        "central_bank": 1,
    }


def test_extract_shock_config_from_graph_shock_entity_when_no_parse_result():
    kg = KnowledgeGraph()
    kg.add_entity(
        Entity(
            id="shock:credit_crunch",
            type="shock",
            name="credit_crunch",
            attributes={
                "shock_type": "credit_crunch",
                "severity": 0.4,
                "scope": "sectoral",
                "duration_steps": 8,
            },
            sources=["/seed/a.md"],
        )
    )
    kg.add_entity(
        Entity(id="geo:US", type="geography", name="US", attributes={}, sources=[])
    )
    kg.add_entity(
        Entity(id="mkt:finance", type="market", name="finance", attributes={}, sources=[])
    )

    cfg = kg.extract_shock_config()

    assert cfg.shock_type == "credit_crunch"
    assert cfg.severity == pytest.approx(0.4)
    assert cfg.scope == "sectoral"
    assert cfg.duration_steps == 8
    assert cfg.time_horizon.steps == 8
    assert "US" in cfg.geography
    assert "finance" in cfg.sectors
    # No actor hints -> default mixed economy.
    assert cfg.agent_counts == {
        "household": 50,
        "firm": 10,
        "bank": 3,
        "central_bank": 1,
    }


def test_extract_shock_config_validates_via_post_init():
    """A bad scope on a shock entity falls back rather than raising."""
    kg = KnowledgeGraph()
    kg.add_entity(
        Entity(
            id="shock:bad",
            type="shock",
            name="bad",
            attributes={"severity": 0.1, "scope": "not-a-scope", "duration_steps": 1},
            sources=[],
        )
    )
    # Should fall back to "micro" when scope is not in VALID_SCOPES.
    cfg = kg.extract_shock_config()
    assert cfg.scope == "micro"


def test_store_and_get_artifact_roundtrip():
    kg = KnowledgeGraph()
    art = SimulationArtifact(
        artifact_id="trajectory:p50",
        kind="trajectory_summary",
        payload={"gdp": [100, 99, 98]},
        refs=["geo:US"],
    )

    kg.store_simulation_artifact(art)
    fetched = kg.get_artifact("trajectory:p50")
    assert fetched is not None
    assert fetched.artifact_id == "trajectory:p50"
    assert fetched.kind == "trajectory_summary"
    assert fetched.payload == {"gdp": [100, 99, 98]}
    assert fetched.refs == ["geo:US"]

    # Mutating the top-level refs list on the returned copy does not mutate
    # the stored artifact (the storage method shallow-copies refs / payload).
    fetched.refs.append("mutated")
    again = kg.get_artifact("trajectory:p50")
    assert again is not None
    assert again.refs == ["geo:US"]


def test_get_artifact_missing_returns_none():
    kg = KnowledgeGraph()
    assert kg.get_artifact("nonexistent") is None


def test_list_entities_filters_by_type():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(id="A", type="actor", name="A", attributes={}, sources=[]))
    kg.add_entity(Entity(id="B", type="actor", name="B", attributes={}, sources=[]))
    kg.add_entity(Entity(id="G", type="geography", name="US", attributes={}, sources=[]))

    actors = kg.list_entities(type="actor")
    assert {e.id for e in actors} == {"A", "B"}

    geos = kg.list_entities(type="geography")
    assert {e.id for e in geos} == {"G"}

    # Unfiltered returns everything.
    assert {e.id for e in kg.list_entities()} == {"A", "B", "G"}


@pytest.mark.asyncio
async def test_build_from_documents_requires_llm_client():
    kg = KnowledgeGraph()
    with pytest.raises(RuntimeError, match="LLMClient"):
        await kg.build_from_documents([_doc("/seed/a.md")])


@pytest.mark.asyncio
async def test_build_from_documents_skips_failed_extractions():
    """A bad document does not abort processing of the remaining docs."""
    good_payload = {
        "entities": [
            {"id": "OIL", "type": "market", "name": "Oil", "attributes": {}},
        ],
        "relations": [],
    }

    # Bad first response (not a dict), good second response.
    client = MockLLMClient(responses=["not-json", good_payload])
    kg = KnowledgeGraph(llm_client=client)

    await kg.build_from_documents([_doc("/seed/a.md"), _doc("/seed/b.md")])

    # First doc's bad response was skipped; second doc's entity is present.
    ids = {e.id for e in kg.list_entities()}
    assert ids == {"OIL"}


@pytest.mark.asyncio
async def test_build_from_documents_records_conflicts_on_graph():
    nl_pr = ParseResult(
        triggering_event="x",
        geographies=["US"],
        markets=[],
        shock_params=ShockParams(),
        time_horizon=TimeHorizon(steps=0, step_unit="day"),
        actor_hints=[],
    )
    payload = {
        "entities": [
            # NL says name="US"; doc says name="United States" -> conflict.
            {"id": "geography:US", "type": "geography", "name": "United States", "attributes": {}},
        ],
        "relations": [],
    }
    client = MockLLMClient(responses=[payload])
    kg = KnowledgeGraph(llm_client=client)

    await kg.build_from_documents([_doc("/seed/a.md")], nl_pr)

    conflicts = kg.conflicts()
    assert any(c.field == "name" for c in conflicts)
    # Conflicting entity was NOT added.
    assert kg.list_entities(type="geography") == []


# ---------------------------------------------------------------- Integration


@pytest.mark.asyncio
async def test_integration_kg_to_world_factory_smoke():
    """Build a small KG, extract a ShockConfig, hand to EconomicWorldFactory."""
    payload = {
        "entities": [
            {"id": "geo:US", "type": "geography", "name": "US", "attributes": {}},
            {"id": "mkt:tech", "type": "market", "name": "tech", "attributes": {}},
            {
                "id": "shock:rate_hike",
                "type": "shock",
                "name": "rate_hike",
                "attributes": {
                    "shock_type": "rate_hike",
                    "severity": 0.3,
                    "scope": "macro",
                    "duration_steps": 5,
                },
            },
        ],
        "relations": [
            {"source_id": "mkt:tech", "target_id": "geo:US", "rel_type": "located_in", "weight": 1.0},
        ],
    }
    client = MockLLMClient(responses=[payload])
    kg = KnowledgeGraph(llm_client=client)

    await kg.build_from_documents([_doc("/seed/a.md")])

    cfg = kg.extract_shock_config()
    assert cfg.shock_type == "rate_hike"
    assert cfg.scope == "macro"
    assert "US" in cfg.geography
    assert "tech" in cfg.sectors

    factory = EconomicWorldFactory(
        network_builder=NetworkBuilder(),
        rng_seed=42,
    )
    world = factory.build_world(cfg, PriorLibrary())

    # Smoke: world has actors of every type from the default agent_counts.
    actor_types = {a.actor_type.value for a in world.actors}
    assert {"household", "firm", "bank", "central_bank"} <= actor_types
    assert world.config.shock_type == "rate_hike"
    assert world.prior_library_version  # non-empty
