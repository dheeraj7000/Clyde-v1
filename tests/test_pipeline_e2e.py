"""End-to-end pipeline integration tests (Task 18.2, Requirements 15.1/15.2/15.4).

These tests exercise the full :class:`ClydePipeline` flow with a
:class:`MockLLMClient` that routes per-subsystem on the message content.
The tests intentionally use a tiny ensemble (run_count=4) and a short
horizon (4 steps) to keep runtime well under a second per test.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from clyde.llm import LLMMessage, LLMResponse, MockLLMClient
from clyde.persistence.db import SimulationDB
from clyde.pipeline import ClydePipeline, PipelineConfig
from clyde.reporting import ProvenanceAnnotation


# ---------------------------------------------------------------------------
# LLM router
# ---------------------------------------------------------------------------


def _parser_payload(*, shock_type: str = "rate_hike", scope: str = "macro") -> dict:
    """Well-formed ParseResult JSON for the ScenarioParser."""
    return {
        "triggering_event": "Federal Reserve rate hike announcement",
        "geographies": ["US"],
        "markets": ["finance", "consumer"],
        "shock_params": {
            "shock_type": shock_type,
            "severity": 0.40,
            "scope": scope,
            "duration_steps": 4,
            "initial_contact_actors": ["central_bank_0000"],
        },
        "time_horizon": {"steps": 4, "step_unit": "quarter"},
        "ambiguities": [],
        "actor_hints": [
            {"actor_type": "household", "count_estimate": 30, "description": "US households"},
            {"actor_type": "firm", "count_estimate": 6, "description": "US firms"},
            {"actor_type": "bank", "count_estimate": 2, "description": "US commercial banks"},
            {"actor_type": "central_bank", "count_estimate": 1, "description": "Federal Reserve"},
        ],
    }


def _kg_payload() -> dict:
    """Minimal KnowledgeGraph extraction payload — one entity, no relations."""
    return {
        "entities": [
            {
                "id": "policy:fed_rate_hike",
                "type": "policy",
                "name": "Federal Reserve rate hike",
                "attributes": {"basis_points": 50},
            }
        ],
        "relations": [],
    }


def _gods_eye_payload() -> dict:
    """A minimal ShockDelta JSON for fork_branch."""
    return {
        "intervention_step": 2,
        "param_overrides": {"severity": 0.20},
        "new_events": [],
        "description": "Cut rates by 75bp at step 2.",
    }


def _is_parser_request(messages: list[LLMMessage]) -> bool:
    if not messages:
        return False
    return "scenario parser" in messages[0].content.lower()


def _is_kg_request(messages: list[LLMMessage]) -> bool:
    if not messages:
        return False
    return "economic-ontology extractor" in messages[0].content.lower()


def _is_gods_eye_request(messages: list[LLMMessage]) -> bool:
    if not messages:
        return False
    return "god's eye console" in messages[0].content.lower()


def _make_router(*, shock_type: str = "rate_hike", scope: str = "macro"):
    """Return a router callable that dispatches by system prompt content."""

    def _router(messages: list[LLMMessage]):
        if _is_parser_request(messages):
            return _parser_payload(shock_type=shock_type, scope=scope)
        if _is_kg_request(messages):
            return _kg_payload()
        if _is_gods_eye_request(messages):
            return _gods_eye_payload()
        # ReportAgent prose calls (complete) — return a short stylistic line.
        return LLMResponse(
            content="Generic placeholder narrative — no numeric content here.",
            model="mock-1",
        )

    return _router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        run_count=4,
        ensemble_seed=7,
        max_workers=1,  # serial path — keeps tests fast & avoids pool spinup.
        db_path=tmp_path / "pipeline.sqlite",
        use_analogs=True,
        rng_seed=42,
    )


@pytest.fixture()
def llm() -> MockLLMClient:
    return MockLLMClient(router=_make_router())


# ---------------------------------------------------------------------------
# 1. End-to-end happy path
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end_happy_path(cfg: PipelineConfig, llm: MockLLMClient) -> None:
    description = (
        "A 50bp central bank rate hike in the United States affecting "
        "financial and consumer markets."
    )
    pipeline = ClydePipeline(llm, config=cfg)
    try:
        result = asyncio.run(pipeline.run(description))
    finally:
        pipeline.close()

    # Parse result reflects the routed payload.
    assert result.parse_result.shock_params.shock_type == "rate_hike"
    assert "US" in result.parse_result.geographies

    # ShockConfig promoted from KG + parse result.
    assert result.shock_config.scope == "macro"
    assert result.shock_config.shock_type == "rate_hike"
    assert result.shock_config.ensemble_seed == cfg.ensemble_seed

    # Ensemble sized to config.
    assert len(result.ensemble.trajectories) == cfg.run_count

    # Report carries the same scenario id and at least one section per the
    # canonical structure (Outcome Range / Causal Pathways / Divergence /
    # Uncertainty).
    assert result.report.scenario_id == result.scenario_id
    assert len(result.report.sections) >= 4

    # Property 14: every provenance annotation is well-formed.
    for prov in result.report.provenance:
        assert isinstance(prov, ProvenanceAnnotation)
        assert prov.claim
        assert prov.source_type in {"simulation_db", "knowledge_graph"}
        assert prov.source_ref
        assert prov.query_used


# ---------------------------------------------------------------------------
# 2. DB persistence end-to-end
# ---------------------------------------------------------------------------


def test_pipeline_persists_runs_and_events(
    cfg: PipelineConfig, llm: MockLLMClient
) -> None:
    description = "Banking-system stress in the US affecting credit markets."
    pipeline = ClydePipeline(llm, config=cfg)
    try:
        result = asyncio.run(pipeline.run(description))
    finally:
        pipeline.close()

    # Re-open the DB and verify rows are queryable.
    db = SimulationDB(result.db_path)
    try:
        first = result.ensemble.trajectories[0]
        run = db.get_run(first.run_id)
        assert run is not None
        assert run["scenario_id"] == result.scenario_id

        traj = db.get_trajectory(first.run_id)
        assert len(traj) > 0
        # Steps match the world's time horizon.
        assert len(traj) == result.world.config.time_horizon.steps

        # Causal events are queryable per run_id (the table may legitimately
        # be empty when no propagation thresholds were exceeded). This call
        # exercises the get_causal_events path and asserts it returns a
        # list (never ``None``).
        for t in result.ensemble.trajectories:
            evs = db.get_causal_events(t.run_id)
            assert isinstance(evs, list)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Branch forking
# ---------------------------------------------------------------------------


def test_pipeline_fork_branch_re_simulates_from_step_zero(
    cfg: PipelineConfig, llm: MockLLMClient
) -> None:
    description = "A rate hike scenario for branch testing."
    pipeline = ClydePipeline(llm, config=cfg)
    try:
        result = asyncio.run(pipeline.run(description))
        fork = asyncio.run(
            pipeline.fork_branch(result, "Cut rates by 75bp at step 2.")
        )
    finally:
        pipeline.close()

    # Branch shock_config diverges from the base after delta application.
    # The mock God's Eye payload sets severity=0.20, base was 0.40.
    assert fork.merged_config.severity != result.shock_config.severity
    assert fork.merged_config.severity == pytest.approx(0.20)

    # Same ensemble size as base.
    assert len(fork.ensemble.trajectories) == cfg.run_count

    # Each branch trajectory has a full re-simulation: time_horizon.steps step metrics.
    horizon_steps = result.world.config.time_horizon.steps
    for traj in fork.ensemble.trajectories:
        assert len(traj.steps) == horizon_steps

    # The branch is queryable in the DB (insert_branch happened inline).
    db = SimulationDB(result.db_path)
    try:
        row = db.get_branch(fork.branch_id)
        assert row is not None
        assert row["parent_scenario_id"] == result.scenario_id
    finally:
        db.close()

    # Branches list on the result was mutated as documented.
    assert fork in result.branches


# ---------------------------------------------------------------------------
# 4. LLM boundary holds at runtime
# ---------------------------------------------------------------------------


def test_pipeline_does_not_inject_llm_into_simulation(
    cfg: PipelineConfig, llm: MockLLMClient
) -> None:
    """Sanity check: the simulation subsystem holds none of our LLM client.

    Static analysis (tests/test_llm_boundary.py) already enforces zero
    imports of clyde.llm under clyde/simulation/. This runtime check just
    confirms the pipeline didn't try to *inject* the client through a side
    door (e.g. by setting an attribute on the controller or engine).
    """
    pipeline = ClydePipeline(llm, config=cfg)
    try:
        asyncio.run(pipeline.run("A short rate hike test."))

        # Walk the controller + engine instance dicts; assert no attribute
        # value is the LLM client (or anything that quacks like one).
        ctrl = pipeline._controller
        assert ctrl is not None

        for obj in (ctrl, ctrl._engine):
            for value in vars(obj).values():
                assert value is not llm, (
                    f"{type(obj).__name__} unexpectedly references the LLM client"
                )
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# 5. Document-augmented run
# ---------------------------------------------------------------------------


def test_pipeline_with_document_populates_knowledge_graph(
    cfg: PipelineConfig, llm: MockLLMClient, tmp_path: Path
) -> None:
    doc_path = tmp_path / "rate_hike_brief.txt"
    doc_path.write_text(
        "The Federal Reserve raised the federal funds rate by 50 basis "
        "points to combat inflation in financial and consumer markets.",
        encoding="utf-8",
    )

    pipeline = ClydePipeline(llm, config=cfg)
    try:
        result = asyncio.run(
            pipeline.run("A US rate hike scenario.", document_paths=[doc_path])
        )
    finally:
        pipeline.close()

    entities = result.knowledge_graph.list_entities()
    assert len(entities) >= 1, "KG should have at least one entity"

    # The doc-derived policy entity routed via the mock should land in the KG.
    ids = {e.id for e in entities}
    assert "policy:fed_rate_hike" in ids


# ---------------------------------------------------------------------------
# 6. No documents = no fabrication
# ---------------------------------------------------------------------------


def test_pipeline_without_documents_keeps_provenance_clean(
    cfg: PipelineConfig, llm: MockLLMClient
) -> None:
    description = "A small US rate hike scenario with no supporting documents."
    pipeline = ClydePipeline(llm, config=cfg)
    try:
        result = asyncio.run(pipeline.run(description))
    finally:
        pipeline.close()

    # Provenance only ever points at simulation_db or knowledge_graph
    # (never an LLM-fabricated source).
    allowed = {"simulation_db", "knowledge_graph"}
    for prov in result.report.provenance:
        assert prov.source_type in allowed, (
            f"unexpected provenance source_type: {prov.source_type!r}"
        )

    # The KG may still contain NL-derived entities even with no documents.
    nl_entities = result.knowledge_graph.list_entities()
    # Geographies + markets + shock entity should all be present from the
    # parse result.
    types_seen = {e.type for e in nl_entities}
    assert "geography" in types_seen
    assert "market" in types_seen


# ---------------------------------------------------------------------------
# 7. Module-level re-exports
# ---------------------------------------------------------------------------


def test_top_level_reexports() -> None:
    import clyde

    assert hasattr(clyde, "ClydePipeline")
    assert hasattr(clyde, "PipelineConfig")
    assert hasattr(clyde, "PipelineResult")
    assert clyde.__version__
