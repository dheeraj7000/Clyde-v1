"""Backtesting harness (task 15.2 / Requirement 11.1–11.3).

A backtest replays a known historical shock through the rule-based simulator
with the actual outcome **masked** (the simulator only sees the
:class:`~clyde.models.config.ShockConfig`), then compares the resulting
ensemble distribution against the recorded outcome.

Design commitments mirrored from ``.kiro/specs/clyde-economic-simulator/design.md``:

* The simulator phase is LLM-free — this module imports nothing from
  ``clyde.llm``. The boundary check in ``tests/test_llm_boundary.py`` keeps us
  honest.
* The historical outcome is *never* fed into the simulation. We carry it on
  :class:`HistoricalShockSpec` purely as ground truth for the post-run
  comparison; the world built and handed to the
  :class:`~clyde.simulation.monte_carlo.MonteCarloController` is constructed
  exclusively from ``spec.shock_config``.
* The setup-phase classes (``EconomicWorldFactory`` / ``PriorLibrary``) are
  typed loosely as :class:`Any` so this module does not import the setup
  phase at all — keeps the simulation package free of upward dependencies.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from typing import Any

from clyde.models.config import ShockConfig
from clyde.models.metrics import (
    CORE_METRIC_NAMES,
    EnsembleResult,
    StepMetrics,
)
from clyde.persistence.db import SimulationDB
from clyde.simulation.monte_carlo import MonteCarloController


# ---------------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalShockSpec:
    """A reproducible historical shock with masked outcome for backtesting.

    Only :attr:`shock_config` is fed into the simulator. :attr:`actual_outcome`
    and :attr:`actual_outcome_horizon_step` are used **after** the run for
    comparison; they never reach the simulation engine.
    """

    name: str
    year: int
    shock_config: ShockConfig
    actual_outcome: dict[str, float]
    actual_outcome_horizon_step: int


@dataclass
class BacktestComparison:
    """Per-metric comparison of the simulated distribution vs the actual."""

    metric: str
    actual: float
    simulated_p10: float
    simulated_p50: float
    simulated_p90: float
    in_band: bool
    error: float
    relative_error: float


@dataclass
class BacktestResult:
    """Result bundle for a single backtest invocation."""

    backtest_id: str
    spec: HistoricalShockSpec
    ensemble: EnsembleResult
    comparisons: list[BacktestComparison] = field(default_factory=list)
    accuracy_score: float = 0.0
    coverage_rate: float = 0.0


# ---------------------------------------------------------------------------
# Backtester.
# ---------------------------------------------------------------------------


class Backtester:
    """Runs a single historical shock through the simulator and scores it.

    Usage::

        bt = Backtester(controller, factory, prior_library, run_count=50)
        result = bt.run(spec)
    """

    def __init__(
        self,
        controller: MonteCarloController,
        world_factory: Any,
        prior_library: Any,
        *,
        run_count: int = 50,
    ) -> None:
        if run_count < 1:
            raise ValueError(f"run_count must be >= 1, got {run_count}")
        self._controller = controller
        self._world_factory = world_factory
        self._prior_library = prior_library
        self._run_count = int(run_count)

    # ---------------------------------------------------------------- public

    def run(
        self,
        spec: HistoricalShockSpec,
        scenario_id: str | None = None,
        db: SimulationDB | None = None,
    ) -> BacktestResult:
        """Execute the backtest.

        The ground-truth outcome is *masked* — only :attr:`HistoricalShockSpec.shock_config`
        flows into the world factory and the controller. The post-run
        comparison is done locally against ``spec.actual_outcome``.
        """
        backtest_id = f"backtest_{spec.name}_{uuid.uuid4().hex[:8]}"
        sid = scenario_id or backtest_id

        # 1. Build the world from the *config only* — actual outcome stays
        #    behind on ``spec`` and never enters world construction.
        world = self._world_factory.build_world(
            spec.shock_config, self._prior_library
        )

        # 2. Run the ensemble. The controller has no way to peek at
        #    ``spec.actual_outcome`` — we never pass it.
        ensemble = self._controller.run_ensemble(
            world,
            run_count=self._run_count,
            scenario_id=sid,
        )

        # 3. Compute per-metric comparisons against the (now visible) actual.
        comparisons = self._compare(spec, ensemble)
        coverage_rate = (
            sum(1 for c in comparisons if c.in_band) / len(comparisons)
            if comparisons
            else 0.0
        )
        accuracy_score = self._accuracy_score(comparisons, coverage_rate)

        # 4. Optional persistence.
        if db is not None:
            sim_dist = {
                c.metric: {
                    "p10": c.simulated_p10,
                    "p50": c.simulated_p50,
                    "p90": c.simulated_p90,
                }
                for c in comparisons
            }
            db.insert_backtest_result(
                backtest_id=backtest_id,
                scenario_id=sid,
                historical_event=spec.name,
                actual_outcome=dict(spec.actual_outcome),
                simulated_distribution=sim_dist,
                accuracy_score=accuracy_score,
            )

        return BacktestResult(
            backtest_id=backtest_id,
            spec=spec,
            ensemble=ensemble,
            comparisons=comparisons,
            accuracy_score=accuracy_score,
            coverage_rate=coverage_rate,
        )

    # --------------------------------------------------------------- helpers

    def _compare(
        self,
        spec: HistoricalShockSpec,
        ensemble: EnsembleResult,
    ) -> list[BacktestComparison]:
        """Build one :class:`BacktestComparison` per metric in the spec."""
        comparisons: list[BacktestComparison] = []
        for metric, actual in spec.actual_outcome.items():
            if metric not in CORE_METRIC_NAMES:
                # Skip unknown metrics rather than crash — the spec keys are
                # human-authored and should be tolerant.
                continue
            samples = self._gather_samples(
                ensemble, metric, spec.actual_outcome_horizon_step
            )
            if not samples:
                # No trajectories at all — record a degenerate comparison so
                # the caller still gets a row per requested metric.
                comparisons.append(
                    BacktestComparison(
                        metric=metric,
                        actual=float(actual),
                        simulated_p10=0.0,
                        simulated_p50=0.0,
                        simulated_p90=0.0,
                        in_band=False,
                        error=float(actual),
                        relative_error=float(actual)
                        / max(abs(float(actual)), 1e-9),
                    )
                )
                continue

            p10, p50, p90 = _percentiles(samples)
            in_band = p10 <= float(actual) <= p90
            error = float(actual) - p50
            rel_error = error / max(abs(float(actual)), 1e-9)
            comparisons.append(
                BacktestComparison(
                    metric=metric,
                    actual=float(actual),
                    simulated_p10=p10,
                    simulated_p50=p50,
                    simulated_p90=p90,
                    in_band=in_band,
                    error=error,
                    relative_error=rel_error,
                )
            )
        return comparisons

    @staticmethod
    def _gather_samples(
        ensemble: EnsembleResult,
        metric: str,
        horizon_step: int,
    ) -> list[float]:
        """Pull metric values at ``horizon_step`` (clamped) across trajectories."""
        samples: list[float] = []
        for traj in ensemble.trajectories:
            if not traj.steps:
                continue
            # Clamp the step into the trajectory's range so a slightly
            # mis-aligned spec doesn't blow up.
            idx = max(0, min(horizon_step, len(traj.steps) - 1))
            sm: StepMetrics = traj.steps[idx]
            value = getattr(sm, metric, None)
            if value is None:
                # Possibly a custom metric — support that path too.
                value = sm.custom_metrics.get(metric)
            if value is None:
                continue
            samples.append(float(value))
        return samples

    @staticmethod
    def _accuracy_score(
        comparisons: list[BacktestComparison],
        coverage_rate: float,
    ) -> float:
        """Blend coverage and median error into a [0, 1] accuracy score.

        ``score = 0.6 * coverage + 0.4 * (1 - clip(mean(|relative_error|), 0, 1))``
        """
        if not comparisons:
            return 0.0
        mean_abs_rel = statistics.fmean(
            min(1.0, max(0.0, abs(c.relative_error))) for c in comparisons
        )
        raw = 0.6 * coverage_rate + 0.4 * (1.0 - mean_abs_rel)
        return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Small numeric helpers.
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return ``(p10, p50, p90)`` using linear interpolation.

    We implement this directly (rather than leaning on ``statistics.quantiles``)
    so the behaviour is well-defined for small ``len(values)`` — the backtest
    harness regularly runs with ``run_count`` in the tens.
    """
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    return _quantile(s, 0.10), _quantile(s, 0.50), _quantile(s, 0.90)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile on a sorted vector."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


__all__ = [
    "Backtester",
    "BacktestComparison",
    "BacktestResult",
    "HistoricalShockSpec",
]
