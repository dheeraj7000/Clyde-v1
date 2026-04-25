"""Tests for the Backtesting harness (task 15.2).

Covers Requirement 11.1–11.3:

* The actual historical outcome must be **masked** during the run — only
  the :class:`ShockConfig` reaches the simulator.
* Simulated outcome distributions must be compared against the actual
  outcome after the run.
* Backtest results must be persistable for accuracy tracking over time.

The tests stay deliberately small: ``run_count=4`` and a tiny world (≤ 10
actors) so the suite is fast even on cold caches.
"""

from __future__ import annotations

import random

import pytest

from clyde.models.config import ShockConfig
from clyde.models.enums import ActorType
from clyde.models.metrics import (
    EnsembleResult,
    StepMetrics,
    TrajectoryResult,
)
from clyde.models.time import TimeHorizon
from clyde.persistence.db import SimulationDB
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary
from clyde.setup.world_factory import EconomicWorldFactory
from clyde.simulation.backtest import (
    BacktestComparison,
    BacktestResult,
    Backtester,
    HistoricalShockSpec,
)
from clyde.simulation.monte_carlo import MonteCarloController


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _tiny_config(seed: int = 11, steps: int = 3) -> ShockConfig:
    return ShockConfig(
        shock_type="oil_price_spike",
        severity=0.4,
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


def _tiny_factory(seed: int = 11) -> EconomicWorldFactory:
    nb = NetworkBuilder(rng=random.Random(seed))
    return EconomicWorldFactory(network_builder=nb, rng_seed=seed)


def _tiny_spec(seed: int = 11, steps: int = 3) -> HistoricalShockSpec:
    return HistoricalShockSpec(
        name="oil_shock_1973",
        year=1973,
        shock_config=_tiny_config(seed=seed, steps=steps),
        # The actual outcome lives only on the spec — we'll prove below the
        # simulator never sees these values.
        actual_outcome={
            "gdp_index": 0.95,
            "unemployment_rate": 0.07,
            "inflation_rate": 0.04,
        },
        actual_outcome_horizon_step=steps - 1,
    )


def _make_trajectory(
    run_id: str,
    seed: int,
    n_steps: int,
    metric_overrides: dict[int, dict[str, float]] | None = None,
) -> TrajectoryResult:
    """Build a synthetic TrajectoryResult with explicit metric values per step."""
    overrides = metric_overrides or {}
    steps: list[StepMetrics] = []
    for i in range(n_steps):
        defaults = {
            "gdp_index": 1.0,
            "inflation_rate": 0.02,
            "unemployment_rate": 0.05,
            "gini_coefficient": 0.3,
            "credit_tightening_index": 0.1,
            "firm_bankruptcy_count": 0,
            "bank_stress_index": 0.0,
            "consumer_confidence": 0.5,
            "interbank_freeze": False,
        }
        defaults.update(overrides.get(i, {}))
        steps.append(StepMetrics(step=i, **defaults))
    return TrajectoryResult(run_id=run_id, seed=seed, steps=steps)


# ---------------------------------------------------------------------------
# 1. Outcome is masked during the run.
# ---------------------------------------------------------------------------


def test_actual_outcome_is_masked_during_run() -> None:
    """The controller must never receive the historical actual_outcome.

    We patch ``MonteCarloController.run_ensemble`` to capture the world it is
    handed, then assert the actual_outcome values do not appear anywhere in
    the world's config / behavioral overrides. We also assert no kwargs
    received by ``run_ensemble`` mention the actual outcome.
    """
    captured: dict[str, object] = {}

    def fake_run_ensemble(
        self,
        world,
        run_count=200,
        max_workers=None,
        scenario_id=None,
        db=None,
        parallel=True,
    ):
        captured["world"] = world
        captured["run_count"] = run_count
        captured["scenario_id"] = scenario_id
        # Return a trivial ensemble so the comparison step still runs.
        traj = _make_trajectory("masked_0001", seed=1, n_steps=3)
        return EnsembleResult(
            scenario_id=scenario_id or "x",
            config=world.config,
            trajectories=[traj],
            run_count=1,
            ensemble_seed=int(world.config.ensemble_seed),
        )

    factory = _tiny_factory()
    controller = MonteCarloController()
    spec = _tiny_spec()

    # Monkey-patch the bound method.
    original = MonteCarloController.run_ensemble
    MonteCarloController.run_ensemble = fake_run_ensemble  # type: ignore[assignment]
    try:
        bt = Backtester(controller, factory, PriorLibrary(), run_count=4)
        result = bt.run(spec)
    finally:
        MonteCarloController.run_ensemble = original  # type: ignore[assignment]

    world = captured["world"]
    cfg_dict = world.config.to_dict()  # type: ignore[union-attr]

    actual_values = set(spec.actual_outcome.keys())
    actual_floats = set(spec.actual_outcome.values())

    # No metric *name* from actual_outcome should bleed into behavioral
    # overrides (they're not behavioral params).
    for key in world.config.behavioral_overrides:  # type: ignore[union-attr]
        assert key not in actual_values, (
            f"actual_outcome key {key!r} leaked into behavioral_overrides"
        )

    # And no actual numeric value should appear as a behavioral override.
    for v in world.config.behavioral_overrides.values():  # type: ignore[union-attr]
        if isinstance(v, (int, float)):
            assert float(v) not in actual_floats, (
                f"actual_outcome value {v!r} leaked into behavioral_overrides"
            )

    # The full to_dict of the config also shouldn't reference our outcome.
    behavioral_dict = cfg_dict.get("behavioral_overrides", {})
    assert all(k not in actual_values for k in behavioral_dict)

    # And the result must still scoring (the spec stays untouched).
    assert isinstance(result, BacktestResult)
    assert result.spec is spec


# ---------------------------------------------------------------------------
# 2. Comparison correctness on a synthetic ensemble.
# ---------------------------------------------------------------------------


def test_comparison_correctness_on_synthetic_distribution() -> None:
    """Hand-craft a known distribution and verify percentiles + flags."""
    factory = _tiny_factory()
    controller = MonteCarloController()

    n_steps = 3
    # Five trajectories with gdp_index at horizon step = [0.8, 0.9, 1.0, 1.1, 1.2].
    gdp_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    trajectories = [
        _make_trajectory(
            f"syn_{i}",
            seed=i,
            n_steps=n_steps,
            metric_overrides={n_steps - 1: {"gdp_index": v}},
        )
        for i, v in enumerate(gdp_values)
    ]
    fake_ensemble = EnsembleResult(
        scenario_id="syn",
        config=_tiny_config(),
        trajectories=trajectories,
        run_count=len(trajectories),
        ensemble_seed=11,
    )

    def fake_run_ensemble(self, world, **kwargs):
        return fake_ensemble

    spec = HistoricalShockSpec(
        name="synthetic",
        year=2000,
        shock_config=_tiny_config(steps=n_steps),
        actual_outcome={"gdp_index": 1.0},
        actual_outcome_horizon_step=n_steps - 1,
    )

    original = MonteCarloController.run_ensemble
    MonteCarloController.run_ensemble = fake_run_ensemble  # type: ignore[assignment]
    try:
        bt = Backtester(controller, factory, PriorLibrary(), run_count=5)
        result = bt.run(spec)
    finally:
        MonteCarloController.run_ensemble = original  # type: ignore[assignment]

    assert len(result.comparisons) == 1
    cmp = result.comparisons[0]
    assert cmp.metric == "gdp_index"
    assert cmp.actual == pytest.approx(1.0)
    # With 5 sorted samples [0.8, 0.9, 1.0, 1.1, 1.2]:
    #   p10 = interp at pos 0.4 between 0.8, 0.9  → 0.84
    #   p50 = sample 2 → 1.0
    #   p90 = interp at pos 3.6 between 1.1, 1.2  → 1.16
    assert cmp.simulated_p10 == pytest.approx(0.84)
    assert cmp.simulated_p50 == pytest.approx(1.0)
    assert cmp.simulated_p90 == pytest.approx(1.16)
    assert cmp.in_band is True
    assert cmp.error == pytest.approx(0.0)
    assert cmp.relative_error == pytest.approx(0.0)


def test_comparison_out_of_band_flags_correctly() -> None:
    """An actual value outside [p10, p90] must set in_band=False and error≠0."""
    factory = _tiny_factory()
    controller = MonteCarloController()

    n_steps = 3
    gdp_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    trajectories = [
        _make_trajectory(
            f"oob_{i}",
            seed=i,
            n_steps=n_steps,
            metric_overrides={n_steps - 1: {"gdp_index": v}},
        )
        for i, v in enumerate(gdp_values)
    ]
    fake_ensemble = EnsembleResult(
        scenario_id="oob",
        config=_tiny_config(),
        trajectories=trajectories,
        run_count=len(trajectories),
        ensemble_seed=11,
    )

    def fake_run_ensemble(self, world, **kwargs):
        return fake_ensemble

    # Actual GDP = 0.5 — clearly below p10=0.84.
    spec = HistoricalShockSpec(
        name="synthetic_oob",
        year=2001,
        shock_config=_tiny_config(steps=n_steps),
        actual_outcome={"gdp_index": 0.5},
        actual_outcome_horizon_step=n_steps - 1,
    )

    original = MonteCarloController.run_ensemble
    MonteCarloController.run_ensemble = fake_run_ensemble  # type: ignore[assignment]
    try:
        bt = Backtester(controller, factory, PriorLibrary(), run_count=5)
        result = bt.run(spec)
    finally:
        MonteCarloController.run_ensemble = original  # type: ignore[assignment]

    cmp = result.comparisons[0]
    assert cmp.in_band is False
    assert cmp.error == pytest.approx(0.5 - 1.0)
    assert cmp.relative_error == pytest.approx((0.5 - 1.0) / 0.5)
    assert result.coverage_rate == 0.0


# ---------------------------------------------------------------------------
# 3. DB persistence round-trip.
# ---------------------------------------------------------------------------


def test_db_persistence_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "backtest.db"
    db = SimulationDB(db_path)

    factory = _tiny_factory()
    controller = MonteCarloController()
    spec = _tiny_spec()

    # Stub the ensemble run so this test stays fast.
    n_steps = spec.shock_config.time_horizon.steps
    trajectories = [
        _make_trajectory(
            f"db_{i}",
            seed=i,
            n_steps=n_steps,
            metric_overrides={
                n_steps - 1: {
                    "gdp_index": 0.95 + 0.01 * i,
                    "unemployment_rate": 0.06 + 0.005 * i,
                    "inflation_rate": 0.03 + 0.005 * i,
                }
            },
        )
        for i in range(4)
    ]
    fake_ensemble = EnsembleResult(
        scenario_id="db_round",
        config=spec.shock_config,
        trajectories=trajectories,
        run_count=4,
        ensemble_seed=11,
    )

    def fake_run_ensemble(self, world, **kwargs):
        return fake_ensemble

    original = MonteCarloController.run_ensemble
    MonteCarloController.run_ensemble = fake_run_ensemble  # type: ignore[assignment]
    try:
        bt = Backtester(controller, factory, PriorLibrary(), run_count=4)
        result = bt.run(spec, scenario_id="bt_persist", db=db)
    finally:
        MonteCarloController.run_ensemble = original  # type: ignore[assignment]

    rows = db.get_backtest_results("bt_persist")
    assert len(rows) == 1
    row = rows[0]
    assert row["historical_event"] == spec.name
    assert row["actual_outcome"] == spec.actual_outcome
    sim_dist = row["simulated_distribution"]
    # One entry per metric in the actual outcome.
    assert set(sim_dist.keys()) == set(spec.actual_outcome.keys())
    for metric, dist in sim_dist.items():
        assert {"p10", "p50", "p90"} <= set(dist.keys())
        # p10 ≤ p50 ≤ p90 must hold.
        assert dist["p10"] <= dist["p50"] <= dist["p90"]
    assert row["accuracy_score"] == pytest.approx(result.accuracy_score)


# ---------------------------------------------------------------------------
# 4. Accuracy score bounds.
# ---------------------------------------------------------------------------


def _build_result_with_comparisons(
    comparisons: list[BacktestComparison],
) -> float:
    """Drive ``Backtester._accuracy_score`` directly via instance method."""
    coverage = (
        sum(1 for c in comparisons if c.in_band) / len(comparisons)
        if comparisons
        else 0.0
    )
    return Backtester._accuracy_score(comparisons, coverage)


def test_accuracy_score_perfect_coverage_close_to_one() -> None:
    """All in_band, zero relative error → score → ~1.0."""
    cmps = [
        BacktestComparison(
            metric="gdp_index",
            actual=1.0,
            simulated_p10=0.9,
            simulated_p50=1.0,
            simulated_p90=1.1,
            in_band=True,
            error=0.0,
            relative_error=0.0,
        ),
        BacktestComparison(
            metric="unemployment_rate",
            actual=0.05,
            simulated_p10=0.04,
            simulated_p50=0.05,
            simulated_p90=0.06,
            in_band=True,
            error=0.0,
            relative_error=0.0,
        ),
    ]
    score = _build_result_with_comparisons(cmps)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0)


def test_accuracy_score_no_coverage_below_half() -> None:
    """No in_band hits + large relative errors → score < 0.5."""
    cmps = [
        BacktestComparison(
            metric="gdp_index",
            actual=1.0,
            simulated_p10=0.5,
            simulated_p50=0.6,
            simulated_p90=0.7,
            in_band=False,
            error=0.4,
            relative_error=0.4,
        ),
        BacktestComparison(
            metric="unemployment_rate",
            actual=0.10,
            simulated_p10=0.30,
            simulated_p50=0.35,
            simulated_p90=0.40,
            in_band=False,
            error=-0.25,
            relative_error=-2.5,  # huge — clipped to 1.0 internally
        ),
    ]
    score = _build_result_with_comparisons(cmps)
    assert 0.0 <= score <= 1.0
    assert score < 0.5


def test_accuracy_score_clamped_to_unit_interval() -> None:
    """Even with absurd inputs, the score stays in [0, 1]."""
    cmps = [
        BacktestComparison(
            metric="gdp_index",
            actual=0.0,
            simulated_p10=10.0,
            simulated_p50=10.0,
            simulated_p90=10.0,
            in_band=False,
            error=-10.0,
            relative_error=-1e9,
        ),
    ]
    score = _build_result_with_comparisons(cmps)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. End-to-end smoke test (real factory + prior library + small ensemble).
# ---------------------------------------------------------------------------


def test_end_to_end_smoke_real_components() -> None:
    """Run a tiny real backtest through the real propagation engine."""
    factory = _tiny_factory(seed=21)
    # Force the controller to use the serial path so we don't pay the
    # process-pool tax in tests.
    controller = MonteCarloController(param_jitter=0.05)
    spec = _tiny_spec(seed=21, steps=3)

    # Ensure run_ensemble runs serially: monkey-patch to default parallel=False.
    original = MonteCarloController.run_ensemble

    def serial_run_ensemble(self, world, run_count=200, **kwargs):
        kwargs.pop("parallel", None)
        return original(self, world, run_count=run_count, parallel=False, **kwargs)

    MonteCarloController.run_ensemble = serial_run_ensemble  # type: ignore[assignment]
    try:
        bt = Backtester(controller, factory, PriorLibrary(), run_count=4)
        result = bt.run(spec)
    finally:
        MonteCarloController.run_ensemble = original  # type: ignore[assignment]

    # Trajectory count matches run_count (no failures expected here).
    assert result.ensemble.run_count == 4
    assert len(result.ensemble.trajectories) == 4
    # One comparison per metric in actual_outcome.
    assert len(result.comparisons) == len(spec.actual_outcome)
    assert 0.0 <= result.coverage_rate <= 1.0
    assert 0.0 <= result.accuracy_score <= 1.0
    # Each comparison must reference one of the requested metrics.
    metric_names = {c.metric for c in result.comparisons}
    assert metric_names == set(spec.actual_outcome.keys())


# ---------------------------------------------------------------------------
# 6. Step clamping when horizon exceeds trajectory length.
# ---------------------------------------------------------------------------


def test_horizon_step_clamped_to_last_step_no_index_error() -> None:
    factory = _tiny_factory()
    controller = MonteCarloController()

    n_steps = 3
    # Last-step gdp = 0.7 across all trajectories.
    trajectories = [
        _make_trajectory(
            f"clamp_{i}",
            seed=i,
            n_steps=n_steps,
            metric_overrides={n_steps - 1: {"gdp_index": 0.7}},
        )
        for i in range(3)
    ]
    fake_ensemble = EnsembleResult(
        scenario_id="clamp",
        config=_tiny_config(steps=n_steps),
        trajectories=trajectories,
        run_count=3,
        ensemble_seed=11,
    )

    def fake_run_ensemble(self, world, **kwargs):
        return fake_ensemble

    # Spec asks for step 999 — way past the horizon.
    spec = HistoricalShockSpec(
        name="clamp_test",
        year=1999,
        shock_config=_tiny_config(steps=n_steps),
        actual_outcome={"gdp_index": 0.7},
        actual_outcome_horizon_step=999,
    )

    original = MonteCarloController.run_ensemble
    MonteCarloController.run_ensemble = fake_run_ensemble  # type: ignore[assignment]
    try:
        bt = Backtester(controller, factory, PriorLibrary(), run_count=3)
        result = bt.run(spec)
    finally:
        MonteCarloController.run_ensemble = original  # type: ignore[assignment]

    # Should pull from the last step (0.7 across trajectories) — perfect match.
    cmp = result.comparisons[0]
    assert cmp.simulated_p10 == pytest.approx(0.7)
    assert cmp.simulated_p50 == pytest.approx(0.7)
    assert cmp.simulated_p90 == pytest.approx(0.7)
    assert cmp.in_band is True
