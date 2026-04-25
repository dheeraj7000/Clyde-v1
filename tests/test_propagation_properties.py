"""Property-based tests for the rule-based Propagation Engine.

Covers three design properties:

- Property 7: Simulation Step Count
- Property 10: Deterministic Reproducibility
- Property 18: Core Metrics Completeness
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from clyde.models.causal import CausalEvent
from clyde.models.config import ShockConfig, VALID_SCOPES
from clyde.models.enums import ActorType
from clyde.models.metrics import CORE_METRIC_NAMES, StepMetrics, TrajectoryResult
from clyde.models.time import TimeHorizon, VALID_STEP_UNITS
from clyde.persistence.db import SimulationDB
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary
from clyde.setup.world_factory import EconomicWorldFactory
from clyde.simulation import PropagationEngine


# ---------------------------------------------------------------------------
# Strategies — build small, fast worlds.
# ---------------------------------------------------------------------------


def _minimal_config(steps: int, step_unit: str, severity: float, seed: int) -> ShockConfig:
    # Small counts keep each simulation < 20ms on average.
    return ShockConfig(
        shock_type="synthetic",
        severity=severity,
        scope="macro",
        duration_steps=steps,
        agent_counts={
            ActorType.HOUSEHOLD.value: 2,
            ActorType.FIRM.value: 2,
            ActorType.BANK.value: 2,
            ActorType.CENTRAL_BANK.value: 1,
        },
        time_horizon=TimeHorizon(steps=steps, step_unit=step_unit),
        ensemble_seed=seed,
        initial_contact_actors=["firm_0000"],
    )


def _make_factory(seed: int) -> EconomicWorldFactory:
    nb = NetworkBuilder(rng=random.Random(seed))
    return EconomicWorldFactory(network_builder=nb, rng_seed=seed)


# ---------------------------------------------------------------------------
# Property 7: Simulation Step Count
# Feature: clyde-economic-simulator, Property 7: Simulation Step Count
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    steps=st.integers(min_value=1, max_value=25),
    step_unit=st.sampled_from(sorted(VALID_STEP_UNITS)),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_7_simulation_step_count(
    steps: int, step_unit: str, severity: float, seed: int
) -> None:
    cfg = _minimal_config(steps, step_unit, severity, seed)
    factory = _make_factory(seed)
    world = factory.build_world(cfg, PriorLibrary())
    engine = PropagationEngine()
    traj = engine.run(world, seed=seed, run_id="prop7")

    assert isinstance(traj, TrajectoryResult)
    assert len(traj.steps) == steps, f"expected {steps} StepMetrics, got {len(traj.steps)}"
    for i, m in enumerate(traj.steps):
        assert m.step == i, f"StepMetrics at position {i} has step={m.step}"


# ---------------------------------------------------------------------------
# Property 10: Deterministic Reproducibility
# Feature: clyde-economic-simulator, Property 10: Deterministic Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    steps=st.integers(min_value=1, max_value=10),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_10_deterministic_reproducibility(
    steps: int, severity: float, seed: int
) -> None:
    cfg = _minimal_config(steps, "day", severity, seed)
    # Use the SAME factory seed for both worlds so they're structurally
    # identical; then run the engine twice with the same run seed.
    world_a = _make_factory(seed).build_world(cfg, PriorLibrary())
    world_b = _make_factory(seed).build_world(cfg, PriorLibrary())

    engine = PropagationEngine()
    run_a = engine.run(world_a, seed=seed, run_id="det_a")
    run_b = engine.run(world_b, seed=seed, run_id="det_b")

    assert len(run_a.steps) == len(run_b.steps) == steps
    for sa, sb in zip(run_a.steps, run_b.steps):
        assert sa.step == sb.step
        for name in CORE_METRIC_NAMES:
            va, vb = getattr(sa, name), getattr(sb, name)
            assert va == vb, f"metric {name} diverged at step {sa.step}: {va} vs {vb}"
        assert sa.custom_metrics == sb.custom_metrics

    # Causal events must match exactly, field by field, in order.
    assert len(run_a.causal_events) == len(run_b.causal_events)
    for ea, eb in zip(run_a.causal_events, run_b.causal_events):
        assert isinstance(ea, CausalEvent) and isinstance(eb, CausalEvent)
        assert ea.to_dict() == eb.to_dict()


# ---------------------------------------------------------------------------
# Property 18: Core Metrics Completeness
# Feature: clyde-economic-simulator, Property 18: Core Metrics Completeness
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    steps=st.integers(min_value=1, max_value=12),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_18_core_metrics_completeness(
    steps: int, severity: float, seed: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    cfg = _minimal_config(steps, "day", severity, seed)
    world = _make_factory(seed).build_world(cfg, PriorLibrary())

    # Persist to a fresh sqlite DB so we can verify round-trip.
    tmp_dir: Path = tmp_path_factory.mktemp(f"prop18_{seed}_{steps}")
    db_path = tmp_dir / "sim.db"
    engine = PropagationEngine()
    with SimulationDB(db_path) as db:
        traj = engine.run(world, seed=seed, run_id="prop18", db=db)
        # Reload the trajectory from the DB and verify equivalence.
        reloaded = db.get_trajectory("prop18")

    # 1) every StepMetrics has non-None, finite core values.
    assert len(traj.steps) == steps
    for m in traj.steps:
        assert isinstance(m, StepMetrics)
        for name in CORE_METRIC_NAMES:
            value = getattr(m, name)
            assert value is not None, f"metric {name} is None at step {m.step}"
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                assert math.isfinite(value), f"metric {name} not finite at step {m.step}"
        assert isinstance(m.custom_metrics, dict)

    # 2) DB reproduces the trajectory step-for-step.
    assert len(reloaded) == steps
    for orig, round_trip in zip(traj.steps, reloaded):
        assert orig.to_dict() == round_trip.to_dict()


# ---------------------------------------------------------------------------
# Boundary test: simulation package must not import any LLM client.
# This is cheap to run and guards the isolation commitment in the design doc.
# ---------------------------------------------------------------------------


def test_simulation_has_no_llm_imports() -> None:
    import clyde.simulation.propagation as prop_mod

    src = Path(prop_mod.__file__).read_text()
    # Forbid top-level LLM SDK / internal LLM module imports.
    for forbidden in ("clyde.llm", "anthropic", "openai"):
        assert f"import {forbidden}" not in src, f"forbidden import of {forbidden}"
        assert f"from {forbidden}" not in src, f"forbidden from-import of {forbidden}"
