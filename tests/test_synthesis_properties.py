"""Property-based tests for the SynthesisEngine (Properties 11, 12, 15).

Feature: clyde-economic-simulator
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from clyde.models import (
    CORE_METRIC_NAMES,
    CausalEvent,
    EnsembleResult,
    ShockConfig,
    StepMetrics,
    TimeHorizon,
    TrajectoryResult,
)
from clyde.synthesis import SynthesisEngine
from clyde.synthesis.engine import _LOWER_IS_BETTER, _BOOL_METRICS


# ---------------------------------------------------------------------------
# Strategy helpers
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
def trajectory_strategy(
    draw,
    n_steps: int,
    run_id: str,
    actor_ids: list[str] | None = None,
    max_events: int = 0,
) -> TrajectoryResult:
    steps = [draw(step_metrics_strategy(step=i)) for i in range(n_steps)]
    causal_events: list[CausalEvent] = []
    if actor_ids and max_events > 0:
        n_events = draw(st.integers(min_value=0, max_value=max_events))
        for _ in range(n_events):
            src = draw(st.sampled_from(actor_ids))
            tgt = draw(st.sampled_from(actor_ids))
            channel = draw(st.sampled_from(["supply_chain", "credit", "labor", "confidence"]))
            variable = draw(st.sampled_from(["gdp_index", "unemployment_rate", "credit_tightening_index"]))
            magnitude = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
            step = draw(st.integers(min_value=0, max_value=max(0, n_steps - 1)))
            causal_events.append(
                CausalEvent(
                    step=step,
                    source_actor_id=src,
                    target_actor_id=tgt,
                    channel=channel,
                    variable_affected=variable,
                    magnitude=magnitude,
                )
            )
    return TrajectoryResult(
        run_id=run_id,
        seed=draw(st.integers(min_value=0, max_value=10_000)),
        steps=steps,
        causal_events=causal_events,
    )


@st.composite
def ensemble_strategy(
    draw,
    min_trajs: int = 20,
    max_trajs: int = 100,
    min_steps: int = 1,
    max_steps: int = 20,
    with_events: bool = False,
) -> EnsembleResult:
    n_trajs = draw(st.integers(min_value=min_trajs, max_value=max_trajs))
    n_steps = draw(st.integers(min_value=min_steps, max_value=max_steps))

    config = ShockConfig(
        shock_type="bank_run",
        severity=0.5,
        scope="macro",
        duration_steps=n_steps,
        time_horizon=TimeHorizon(steps=n_steps, step_unit="day"),
    )

    actor_ids = ["A1", "A2", "A3", "A4"] if with_events else None
    max_events = 20 if with_events else 0

    trajectories = [
        draw(
            trajectory_strategy(
                n_steps=n_steps,
                run_id=f"r{i}",
                actor_ids=actor_ids,
                max_events=max_events,
            )
        )
        for i in range(n_trajs)
    ]

    return EnsembleResult(
        scenario_id="s1",
        config=config,
        trajectories=trajectories,
        run_count=n_trajs,
        ensemble_seed=42,
    )


# ---------------------------------------------------------------------------
# Property 11: Percentile Band Ordering
# ---------------------------------------------------------------------------


_FP_TOL = 1e-9


def _worse_equal(metric: str, a, b) -> bool:
    """Return True iff `a` is >= `b` in the "worse" direction of `metric`.

    i.e. a is at least as bad as b. Tolerates tiny FP drift from percentile
    interpolation (e.g. p50 vs p10 on a uniform-ish sample may differ by 1 ULP).
    """
    if metric in _BOOL_METRICS:
        # Worse = True for interbank_freeze.
        return bool(a) >= bool(b)
    af, bf = float(a), float(b)
    if metric in _LOWER_IS_BETTER:
        return af >= bf - _FP_TOL
    # higher_is_better: worse means smaller.
    return af <= bf + _FP_TOL


# Feature: clyde-economic-simulator, Property 11: Percentile Band Ordering
@pytest.mark.property
@given(ensemble=ensemble_strategy(min_trajs=20, max_trajs=40, min_steps=1, max_steps=5))
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_percentile_band_ordering(ensemble: EnsembleResult) -> None:
    engine = SynthesisEngine()
    bundle = engine.compute_paths(ensemble)

    assert len(bundle.central) == len(ensemble.trajectories[0].steps)
    assert len(bundle.optimistic) == len(bundle.central)
    assert len(bundle.pessimistic) == len(bundle.central)
    assert len(bundle.tail_upper) == len(bundle.central)
    assert len(bundle.tail_lower) == len(bundle.central)

    for i in range(len(bundle.central)):
        for metric in CORE_METRIC_NAMES:
            c = getattr(bundle.central[i], metric)
            opt = getattr(bundle.optimistic[i], metric)
            pes = getattr(bundle.pessimistic[i], metric)
            tu = getattr(bundle.tail_upper[i], metric)
            tl = getattr(bundle.tail_lower[i], metric)

            # Outcome-based ordering: worst (tail_lower) >= pessimistic >= central >= optimistic >= tail_upper.
            # For bool interbank_freeze: worse = True. tail_lower should be at least as bad as pessimistic, etc.
            # But wait: tail_lower = "most extreme worse", tail_upper = "most extreme better".
            # Let's use the outcome-semantic: tail_lower is worst extreme. For the bool case the spec says
            # tail_upper >= optimistic >= central >= pessimistic >= tail_lower where "worse" = True.
            # Build an outcome-neutral assertion: pessimistic is worse-or-equal-to central; optimistic is
            # better-or-equal-to central; tail_lower is worse-or-equal-to pessimistic; tail_upper is
            # better-or-equal-to optimistic.
            assert _worse_equal(metric, pes, c), (
                f"metric={metric} step={i}: pessimistic={pes} is not worse-than-or-equal-to central={c}"
            )
            assert _worse_equal(metric, c, opt), (
                f"metric={metric} step={i}: central={c} is not worse-than-or-equal-to optimistic={opt}"
            )
            assert _worse_equal(metric, tl, pes), (
                f"metric={metric} step={i}: tail_lower={tl} is not worse-than-or-equal-to pessimistic={pes}"
            )
            assert _worse_equal(metric, opt, tu), (
                f"metric={metric} step={i}: optimistic={opt} is not worse-than-or-equal-to tail_upper={tu}"
            )


# ---------------------------------------------------------------------------
# Property 12: Divergence Map Completeness and Watchlist Derivation
# ---------------------------------------------------------------------------


def _make_varied_ensemble(n_trajs: int = 30, n_steps: int = 5) -> EnsembleResult:
    """Build an ensemble with obvious variance across trajectories."""
    import random

    rng = random.Random(12345)
    config = ShockConfig(
        shock_type="bank_run",
        severity=0.5,
        scope="macro",
        duration_steps=n_steps,
        time_horizon=TimeHorizon(steps=n_steps, step_unit="day"),
    )
    trajectories: list[TrajectoryResult] = []
    for i in range(n_trajs):
        steps = [
            StepMetrics(
                step=s,
                gdp_index=1.0 + rng.uniform(-0.2, 0.2),
                inflation_rate=0.02 + rng.uniform(-0.02, 0.05),
                unemployment_rate=0.05 + rng.uniform(-0.02, 0.1),
                gini_coefficient=0.4 + rng.uniform(-0.05, 0.1),
                credit_tightening_index=rng.uniform(0.0, 0.5),
                firm_bankruptcy_count=rng.randint(0, 50),
                bank_stress_index=rng.uniform(0.0, 0.7),
                consumer_confidence=rng.uniform(0.3, 0.9),
                interbank_freeze=rng.random() < 0.3,
            )
            for s in range(n_steps)
        ]
        trajectories.append(TrajectoryResult(run_id=f"r{i}", seed=i, steps=steps))
    return EnsembleResult(
        scenario_id="s1",
        config=config,
        trajectories=trajectories,
        run_count=n_trajs,
        ensemble_seed=12345,
    )


# Feature: clyde-economic-simulator, Property 12: Divergence Map Completeness and Watchlist Derivation
@pytest.mark.property
@given(
    n_trajs=st.integers(min_value=20, max_value=60),
    n_steps=st.integers(min_value=3, max_value=10),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_divergence_map_completeness(n_trajs: int, n_steps: int) -> None:
    ensemble = _make_varied_ensemble(n_trajs=n_trajs, n_steps=n_steps)
    engine = SynthesisEngine()
    dm = engine.compute_divergence_map(ensemble)

    assert dm.variables, "DivergenceMap should be non-empty for varied ensemble"
    for var in dm.variables:
        assert var.sensitivity > 0.0, f"{var.name} has non-positive sensitivity"
        assert var.monitoring_indicator, f"{var.name} has empty monitoring indicator"

    watchlist = engine.indicator_watchlist(dm)
    expected = [v.monitoring_indicator for v in dm.variables]
    assert watchlist == expected


# Feature: clyde-economic-simulator, Property 12: Divergence Map (identical-trajectory case)
@pytest.mark.property
def test_divergence_map_empty_for_identical_trajectories() -> None:
    """If all trajectories are identical, the divergence map should be empty."""
    config = ShockConfig(
        shock_type="bank_run",
        severity=0.5,
        scope="macro",
        duration_steps=5,
        time_horizon=TimeHorizon(steps=5, step_unit="day"),
    )
    step_template = [
        StepMetrics(
            step=s,
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
        for s in range(5)
    ]
    trajectories = [
        TrajectoryResult(run_id=f"r{i}", seed=i, steps=list(step_template)) for i in range(30)
    ]
    ensemble = EnsembleResult(
        scenario_id="s1",
        config=config,
        trajectories=trajectories,
        run_count=30,
        ensemble_seed=0,
    )
    engine = SynthesisEngine()
    dm = engine.compute_divergence_map(ensemble)
    assert dm.variables == []


# ---------------------------------------------------------------------------
# Property 15: Causal Chain Ordering
# ---------------------------------------------------------------------------


# Feature: clyde-economic-simulator, Property 15: Causal Chain Ordering
@pytest.mark.property
@given(
    ensemble=ensemble_strategy(
        min_trajs=5, max_trajs=30, min_steps=1, max_steps=10, with_events=True
    )
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_causal_chain_ordering(ensemble: EnsembleResult) -> None:
    engine = SynthesisEngine()

    # Collect every actor id that appears in the ensemble's causal events.
    known_ids: set[str] = set()
    for traj in ensemble.trajectories:
        for ev in traj.causal_events:
            known_ids.add(ev.source_actor_id)
            known_ids.add(ev.target_actor_id)

    chains = engine.detect_causal_chains(ensemble)
    for chain in chains:
        # Non-decreasing by step.
        steps_seq = [ev.step for ev in chain.events]
        assert steps_seq == sorted(steps_seq), f"chain {chain.chain_id} events not ordered by step"
        # Every actor id references an id known to appear in the input events.
        for ev in chain.events:
            assert ev.source_actor_id in known_ids, (
                f"chain {chain.chain_id} has unknown source actor {ev.source_actor_id}"
            )
            assert ev.target_actor_id in known_ids, (
                f"chain {chain.chain_id} has unknown target actor {ev.target_actor_id}"
            )
