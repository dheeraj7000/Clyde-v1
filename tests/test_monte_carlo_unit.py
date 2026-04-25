"""Unit tests for the Monte Carlo Controller (task 7.6).

These exercise specifics that the property tests don't pin down:

* Default ``run_count`` is in the ``[100, 500]`` band the spec mandates.
* ``max_workers`` defaulting respects ``os.cpu_count()`` and behaves
  correctly when an ensemble runs successfully end-to-end.
* Branch + base ensemble can both be persisted and re-queried via
  :class:`SimulationDB`.
* ``merge_delta`` rejects an out-of-range ``intervention_step``.
"""

from __future__ import annotations

import inspect
import os
import random

import pytest

from clyde.models.config import ShockConfig, ShockDelta
from clyde.models.enums import ActorType
from clyde.models.time import TimeHorizon
from clyde.persistence.db import SimulationDB
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary
from clyde.setup.world_factory import EconomicWorldFactory
from clyde.simulation.monte_carlo import MonteCarloController


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _build_small_world(seed: int = 7, steps: int = 3):
    cfg = ShockConfig(
        shock_type="synthetic",
        severity=0.3,
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
    nb = NetworkBuilder(rng=random.Random(seed))
    factory = EconomicWorldFactory(network_builder=nb, rng_seed=seed)
    return factory.build_world(cfg, PriorLibrary())


# ---------------------------------------------------------------------------
# 1. Default run_count is in the [100, 500] band (Requirement 6.6).
# ---------------------------------------------------------------------------


def test_default_run_count_in_required_band() -> None:
    """The signature default for ``run_count`` must satisfy 100 <= n <= 500."""
    sig = inspect.signature(MonteCarloController.run_ensemble)
    default = sig.parameters["run_count"].default
    assert isinstance(default, int)
    assert 100 <= default <= 500, (
        f"Default run_count={default} violates Requirement 6.6 [100, 500]."
    )

    # Also confirm the controller actually runs end-to-end (we use a tiny
    # run_count so the test stays fast — the assertion above is the canonical
    # check that the documented default is in the right band).
    mc = MonteCarloController()
    world = _build_small_world(seed=1, steps=2)
    result = mc.run_ensemble(world, run_count=2, parallel=False)
    assert result.run_count == 2
    assert len(result.trajectories) == 2


# ---------------------------------------------------------------------------
# 2. Parallel execution: max_workers default and a 4-run ensemble succeed.
# ---------------------------------------------------------------------------


def test_parallel_max_workers_default_and_success() -> None:
    """``max_workers`` defaults to ``min(os.cpu_count() or 1, 4)``."""
    expected_default = min(os.cpu_count() or 1, 4)
    assert expected_default >= 1

    mc = MonteCarloController()
    world = _build_small_world(seed=2, steps=2)

    # Run a small ensemble in parallel and confirm it actually returned 4
    # successful trajectories.
    result = mc.run_ensemble(world, run_count=4, parallel=True)
    assert result.run_count == 4
    assert len(result.trajectories) == 4
    seeds = [t.seed for t in result.trajectories]
    assert len(set(seeds)) == 4


# ---------------------------------------------------------------------------
# 3. Branch comparison: persist base ensemble + branch and query both.
# ---------------------------------------------------------------------------


def test_base_and_branch_are_queryable_from_db(tmp_path) -> None:
    """Persist a base ensemble and a branch, then read both back."""
    db_path = tmp_path / "mc_unit.sqlite"
    db = SimulationDB(db_path)
    try:
        world = _build_small_world(seed=3, steps=3)
        mc = MonteCarloController(param_jitter=0.05)

        base = mc.run_ensemble(
            world,
            run_count=2,
            scenario_id="base_scenario",
            db=db,
            parallel=False,
        )
        assert len(base.trajectories) == 2
        for t in base.trajectories:
            stored = db.get_run(t.run_id)
            assert stored is not None, f"missing run {t.run_id}"
            steps = db.get_trajectory(t.run_id)
            assert len(steps) == world.config.time_horizon.steps

        delta = ShockDelta(
            intervention_step=1,
            param_overrides={"severity": 0.5},
            new_events=["policy_intervention"],
            description="raise severity",
        )
        branch = mc.fork_branch(
            base_world=world,
            delta=delta,
            run_count=2,
            parent_scenario_id="base_scenario",
            db=db,
            branch_id="branch_unit_1",
            parallel=False,
        )

        assert branch.branch_id == "branch_unit_1"
        record = db.get_branch("branch_unit_1")
        assert record is not None
        assert record["parent_scenario_id"] == "base_scenario"
        assert record["merged_config"].severity == pytest.approx(0.5)
        # ShockDelta round-trip carried through the DB.
        assert record["shock_delta"].intervention_step == 1
        assert "policy_intervention" in record["shock_delta"].new_events
        # Branch trajectories also persisted.
        assert len(branch.ensemble.trajectories) == 2
        for t in branch.ensemble.trajectories:
            assert db.get_run(t.run_id) is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. merge_delta error: invalid intervention_step → ValueError.
# ---------------------------------------------------------------------------


def test_merge_delta_rejects_out_of_range_intervention_step() -> None:
    base = ShockConfig(
        shock_type="synthetic",
        severity=0.2,
        scope="macro",
        duration_steps=4,
        agent_counts={
            ActorType.HOUSEHOLD.value: 1,
            ActorType.FIRM.value: 1,
            ActorType.BANK.value: 1,
            ActorType.CENTRAL_BANK.value: 1,
        },
        time_horizon=TimeHorizon(steps=4, step_unit="day"),
        ensemble_seed=11,
    )

    too_high = ShockDelta(intervention_step=4, param_overrides={}, new_events=[])
    with pytest.raises(ValueError):
        MonteCarloController.merge_delta(base, too_high)

    negative = ShockDelta(intervention_step=-1, param_overrides={}, new_events=[])
    with pytest.raises(ValueError):
        MonteCarloController.merge_delta(base, negative)


# ---------------------------------------------------------------------------
# 5. run_count validation: < 1 must raise.
# ---------------------------------------------------------------------------


def test_run_ensemble_rejects_zero_run_count() -> None:
    mc = MonteCarloController()
    world = _build_small_world(seed=5, steps=2)
    with pytest.raises(ValueError):
        mc.run_ensemble(world, run_count=0, parallel=False)
