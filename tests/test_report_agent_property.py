"""Property-based tests for the ReportAgent (Property 14).

Feature: clyde-economic-simulator
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from clyde.llm import MockLLMClient
from clyde.models import (
    CausalChain,
    CausalEvent,
    DivergenceMap,
    DivergenceVariable,
    PathBundle,
    ShockConfig,
    StepMetrics,
    TimeHorizon,
)
from clyde.persistence.db import SimulationDB
from clyde.reporting import (
    NarrativeReport,
    ProvenanceAnnotation,
    ReportAgent,
    SynthesisResult,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def step_metrics_strategy(draw, step: int) -> StepMetrics:
    return StepMetrics(
        step=step,
        gdp_index=draw(st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False)),
        inflation_rate=draw(st.floats(min_value=-0.05, max_value=0.20, allow_nan=False, allow_infinity=False)),
        unemployment_rate=draw(st.floats(min_value=0.02, max_value=0.30, allow_nan=False, allow_infinity=False)),
        gini_coefficient=draw(st.floats(min_value=0.2, max_value=0.7, allow_nan=False, allow_infinity=False)),
        credit_tightening_index=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        firm_bankruptcy_count=draw(st.integers(min_value=0, max_value=500)),
        bank_stress_index=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        consumer_confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        interbank_freeze=draw(st.booleans()),
    )


@st.composite
def path_bundle_strategy(draw, n_steps: int) -> PathBundle:
    central = [draw(step_metrics_strategy(step=i)) for i in range(n_steps)]
    optimistic = [draw(step_metrics_strategy(step=i)) for i in range(n_steps)]
    pessimistic = [draw(step_metrics_strategy(step=i)) for i in range(n_steps)]
    tail_upper = [draw(step_metrics_strategy(step=i)) for i in range(n_steps)]
    tail_lower = [draw(step_metrics_strategy(step=i)) for i in range(n_steps)]
    return PathBundle(
        central=central,
        optimistic=optimistic,
        pessimistic=pessimistic,
        tail_upper=tail_upper,
        tail_lower=tail_lower,
    )


_DIV_METRIC_NAMES = (
    "gdp_index",
    "inflation_rate",
    "unemployment_rate",
    "gini_coefficient",
    "credit_tightening_index",
    "firm_bankruptcy_count",
    "bank_stress_index",
    "consumer_confidence",
    "interbank_freeze",
)


@st.composite
def divergence_map_strategy(draw) -> DivergenceMap:
    n = draw(st.integers(min_value=1, max_value=4))
    metric_names = draw(
        st.lists(
            st.sampled_from(_DIV_METRIC_NAMES),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    variables = [
        DivergenceVariable(
            name=name,
            sensitivity=draw(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)),
            current_uncertainty=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            monitoring_indicator=f"watch_{name}",
        )
        for name in metric_names
    ]
    return DivergenceMap(variables=variables)


@st.composite
def causal_chain_strategy(draw, max_events: int = 4) -> CausalChain:
    n_events = draw(st.integers(min_value=1, max_value=max_events))
    actor_pool = ["A1", "A2", "A3", "A4"]
    events: list[CausalEvent] = []
    for i in range(n_events):
        events.append(
            CausalEvent(
                step=i,
                source_actor_id=draw(st.sampled_from(actor_pool)),
                target_actor_id=draw(st.sampled_from(actor_pool)),
                channel=draw(st.sampled_from(["supply_chain", "credit", "labor", "confidence"])),
                variable_affected=draw(st.sampled_from(["gdp_index", "credit_tightening_index"])),
                magnitude=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            )
        )
    chain_id = f"chain_{draw(st.integers(min_value=0, max_value=10**9))}"
    return CausalChain(
        chain_id=chain_id,
        events=events,
        origin_shock="bank_run",
        total_magnitude=sum(abs(e.magnitude) for e in events),
    )


@st.composite
def synthesis_strategy(draw) -> SynthesisResult:
    n_steps = draw(st.integers(min_value=1, max_value=10))
    paths = draw(path_bundle_strategy(n_steps=n_steps))
    dm = draw(divergence_map_strategy())
    n_chains = draw(st.integers(min_value=0, max_value=4))
    chains = [draw(causal_chain_strategy()) for _ in range(n_chains)]
    config = ShockConfig(
        shock_type="bank_run",
        severity=0.5,
        scope="macro",
        duration_steps=n_steps,
        time_horizon=TimeHorizon(steps=n_steps, step_unit="day"),
    )
    return SynthesisResult(
        scenario_id="scenario-1",
        config=config,
        paths=paths,
        divergence_map=dm,
        causal_chains=chains,
        metric_selections=[],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_db(db: SimulationDB, run_id: str, scenario_id: str, n_steps: int) -> None:
    cfg = ShockConfig(
        shock_type="bank_run",
        severity=0.5,
        scope="macro",
        duration_steps=n_steps,
        time_horizon=TimeHorizon(steps=n_steps, step_unit="day"),
    )
    db.insert_run(run_id=run_id, scenario_id=scenario_id, seed=0, config=cfg)
    if n_steps:
        steps = [
            StepMetrics(
                step=i,
                gdp_index=1.0,
                inflation_rate=0.02,
                unemployment_rate=0.05,
                gini_coefficient=0.4,
                credit_tightening_index=0.1,
                firm_bankruptcy_count=10,
                bank_stress_index=0.2,
                consumer_confidence=0.6,
                interbank_freeze=False,
            )
            for i in range(n_steps)
        ]
        db.insert_step_metrics_batch(run_id, steps)


def _router_for_section(messages):
    """Return canned prose tagged with the section heading from the user msg."""
    user_msg = next((m for m in messages if m.role == "user"), None)
    heading = "section"
    if user_msg is not None:
        first = user_msg.content.splitlines()[0] if user_msg.content else ""
        if first.startswith("Section: "):
            heading = first[len("Section: "):]
    return f"prose for {heading}"


# ---------------------------------------------------------------------------
# Property 14: Report Provenance Completeness
# ---------------------------------------------------------------------------


# Feature: clyde-economic-simulator, Property 14: Report Provenance Completeness
@pytest.mark.property
@given(synthesis=synthesis_strategy())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.function_scoped_fixture],
)
def test_report_provenance_completeness(synthesis: SynthesisResult, tmp_path_factory) -> None:
    """Every claim in any generated NarrativeReport has a complete provenance."""
    db_dir = tmp_path_factory.mktemp("clyde_report_property")
    db = SimulationDB(db_dir / "sim.sqlite")
    try:
        run_id = "run-prop-1"
        _seed_db(db, run_id=run_id, scenario_id=synthesis.scenario_id, n_steps=len(synthesis.paths.central))

        agent = ReportAgent(
            llm_client=MockLLMClient(router=_router_for_section),
            db=db,
        )
        report: NarrativeReport = asyncio.run(
            agent.generate_report(synthesis, ensemble_run_ids=[run_id])
        )

        assert isinstance(report, NarrativeReport)
        assert report.scenario_id == synthesis.scenario_id

        # Every section's provenance, and the flat provenance, is well-formed.
        def _check(p: ProvenanceAnnotation) -> None:
            assert p.claim, "claim must be non-empty"
            assert p.source_type in {"simulation_db", "knowledge_graph"}, (
                f"source_type must be one of the allowed values, got {p.source_type!r}"
            )
            assert p.source_ref, "source_ref must be non-empty"
            assert p.query_used, "query_used must be non-empty"

        for section in report.sections:
            for prov in section.provenance:
                _check(prov)
        for prov in report.provenance:
            _check(prov)

        # Flat provenance equals the union of section provenance.
        per_section = [p for s in report.sections for p in s.provenance]
        assert report.provenance == per_section
    finally:
        db.close()
