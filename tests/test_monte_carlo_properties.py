"""Property-based tests for the Monte Carlo Controller (task 7).

Covers three design properties:

- Property 8: Ensemble Run Distinctness
- Property 9: Ensemble Trajectory Count
- Property 13: Branch Re-simulation Integrity
"""

from __future__ import annotations

import random
from dataclasses import asdict, fields, is_dataclass

import pytest
from hypothesis import given, settings, strategies as st

from clyde.models.config import ShockConfig, ShockDelta, VALID_SCOPES
from clyde.models.enums import ActorType
from clyde.models.metrics import EnsembleResult
from clyde.models.time import TimeHorizon
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary
from clyde.setup.world_factory import EconomicWorldFactory
from clyde.simulation.monte_carlo import (
    BranchResult,
    MonteCarloController,
    _jitter_world_impl,
)


# ---------------------------------------------------------------------------
# Helpers — small worlds keep each example fast.
# ---------------------------------------------------------------------------


def _minimal_config(
    *,
    steps: int = 3,
    severity: float = 0.3,
    seed: int = 0,
) -> ShockConfig:
    return ShockConfig(
        shock_type="synthetic",
        severity=severity,
        scope="macro",
        duration_steps=steps,
        agent_counts={
            ActorType.HOUSEHOLD.value: 3,
            ActorType.FIRM.value: 2,
            ActorType.BANK.value: 2,
            ActorType.CENTRAL_BANK.value: 1,
        },
        time_horizon=TimeHorizon(steps=steps, step_unit="day"),
        ensemble_seed=seed,
        initial_contact_actors=["firm_0000"],
    )


def _make_world(seed: int, steps: int = 3, severity: float = 0.3):
    cfg = _minimal_config(steps=steps, severity=severity, seed=seed)
    nb = NetworkBuilder(rng=random.Random(seed))
    factory = EconomicWorldFactory(network_builder=nb, rng_seed=seed)
    return factory.build_world(cfg, PriorLibrary())


def _params_signature(world) -> dict[tuple[str, str], float]:
    """Flatten (actor.id, param_name) → float across the world."""
    sig: dict[tuple[str, str], float] = {}
    for a in world.actors:
        if not is_dataclass(a.params):
            continue
        for f in fields(a.params):
            v = getattr(a.params, f.name)
            if isinstance(v, (int, float)):
                sig[(a.id, f.name)] = float(v)
    # Severity is also part of the per-run parameter set.
    sig[("__config__", "severity")] = float(world.config.severity)
    return sig


# ---------------------------------------------------------------------------
# Property 8: Ensemble Run Distinctness
# Feature: clyde-economic-simulator, Property 8: Ensemble Run Distinctness
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=10, deadline=None)
@given(
    run_count=st.integers(min_value=6, max_value=12),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_8_ensemble_run_distinctness(run_count: int, seed: int) -> None:
    world = _make_world(seed)
    mc = MonteCarloController(param_jitter=0.10)

    # Seeds must be pairwise distinct.
    seeds = mc._generate_seeds(world.config.ensemble_seed, run_count)
    assert len(seeds) == run_count
    assert len(set(seeds)) == run_count, f"duplicate seeds: {seeds}"

    # Build a jittered world per run and confirm the parameter signatures
    # are all pairwise distinct.
    signatures: list[dict[tuple[str, str], float]] = []
    for s in seeds:
        jw = _jitter_world_impl(world, seed=s, jitter=0.10)
        signatures.append(_params_signature(jw))

    # Use a hashable representation (frozenset of items) for set membership.
    frozen = [frozenset(sig.items()) for sig in signatures]
    assert len(set(frozen)) == run_count, (
        "expected each run's parameter set to be distinct"
    )


# ---------------------------------------------------------------------------
# Property 9: Ensemble Trajectory Count
# Feature: clyde-economic-simulator, Property 9: Ensemble Trajectory Count
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=10, deadline=None)
@given(
    run_count=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_9_ensemble_trajectory_count(run_count: int, seed: int) -> None:
    world = _make_world(seed)
    mc = MonteCarloController()
    # Use the serial path here: stable across CI containers and avoids
    # ProcessPoolExecutor startup overhead per Hypothesis example.
    result = mc.run_ensemble(world, run_count=run_count, parallel=False)

    assert isinstance(result, EnsembleResult)
    assert len(result.trajectories) == run_count
    assert result.run_count == run_count
    # Seeds inside the result must also be pairwise distinct.
    seeds = [t.seed for t in result.trajectories]
    assert len(set(seeds)) == run_count


# ---------------------------------------------------------------------------
# Property 13: Branch Re-simulation Integrity
# Feature: clyde-economic-simulator, Property 13: Branch Re-simulation Integrity
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    steps=st.integers(min_value=2, max_value=5),
    intervention_offset=st.integers(min_value=0, max_value=4),
    delta_severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    new_event=st.text(min_size=0, max_size=8),
    extra_key=st.sampled_from(["custom_a", "custom_b", "custom_c"]),
    extra_val=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
)
def test_property_13_branch_resimulation_integrity(
    seed: int,
    steps: int,
    intervention_offset: int,
    delta_severity: float,
    new_event: str,
    extra_key: str,
    extra_val: float,
) -> None:
    world = _make_world(seed, steps=steps, severity=0.4)
    base_config = world.config
    intervention_step = min(intervention_offset, steps - 1)

    delta = ShockDelta(
        intervention_step=intervention_step,
        param_overrides={
            "severity": delta_severity,
            extra_key: extra_val,
        },
        new_events=[new_event] if new_event else [],
        description="hypothesis-generated intervention",
    )

    merged = MonteCarloController.merge_delta(base_config, delta)

    # Top-level severity overridden, all other top-level fields preserved.
    assert merged.severity == pytest.approx(delta_severity)
    assert merged.shock_type == base_config.shock_type
    assert merged.scope == base_config.scope
    assert merged.duration_steps == base_config.duration_steps
    assert merged.geography == base_config.geography
    assert merged.sectors == base_config.sectors
    assert merged.initial_contact_actors == base_config.initial_contact_actors
    assert merged.agent_counts == base_config.agent_counts
    assert merged.time_horizon == base_config.time_horizon
    assert merged.ensemble_seed == base_config.ensemble_seed

    # Behavioral overrides include the non-top-level delta keys.
    assert merged.behavioral_overrides.get(extra_key) == extra_val
    if new_event:
        assert new_event in merged.behavioral_overrides.get("new_events", [])

    # Now actually re-simulate (small run count to keep this fast).
    mc = MonteCarloController(param_jitter=0.05)
    branch = mc.fork_branch(
        base_world=world,
        delta=delta,
        run_count=2,
        parent_scenario_id="prop13_parent",
        parallel=False,
    )

    assert isinstance(branch, BranchResult)
    assert branch.parent_scenario_id == "prop13_parent"
    assert branch.delta is delta
    assert branch.merged_config.severity == pytest.approx(delta_severity)
    assert len(branch.ensemble.trajectories) == 2
    # Each trajectory must run the *full* time horizon — re-simulation, not
    # a perturbation from `intervention_step`.
    for traj in branch.ensemble.trajectories:
        assert len(traj.steps) == base_config.time_horizon.steps
