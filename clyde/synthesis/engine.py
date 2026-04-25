"""SynthesisEngine: percentile bands, divergence maps, causal chains, metric selection."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from clyde.models.causal import CausalChain, CausalEvent
from clyde.models.config import ShockConfig
from clyde.models.metrics import (
    CORE_METRIC_NAMES,
    EnsembleResult,
    PathBundle,
    StepMetrics,
    TrajectoryResult,
)
from clyde.models.reporting import DivergenceMap, DivergenceVariable

# --------------------------------------------------------------------------------------
# Directionality + monitoring indicators
# --------------------------------------------------------------------------------------

_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "inflation_rate",
        "unemployment_rate",
        "gini_coefficient",
        "credit_tightening_index",
        "firm_bankruptcy_count",
        "bank_stress_index",
        "interbank_freeze",
    }
)
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"gdp_index", "consumer_confidence"})

_MONITORING_INDICATORS: dict[str, str] = {
    "gdp_index": "quarterly_gdp_growth",
    "inflation_rate": "cpi_yoy",
    "unemployment_rate": "u3_unemployment",
    "gini_coefficient": "household_income_gini",
    "credit_tightening_index": "senior_loan_officer_survey",
    "firm_bankruptcy_count": "chapter_11_filings",
    "bank_stress_index": "bank_stress_test_scores",
    "consumer_confidence": "michigan_consumer_sentiment",
    "interbank_freeze": "libor_ois_spread",
}

_INT_METRICS: frozenset[str] = frozenset({"firm_bankruptcy_count"})
_BOOL_METRICS: frozenset[str] = frozenset({"interbank_freeze"})


@dataclass
class MetricSelection:
    metric: str
    reason: str


# --------------------------------------------------------------------------------------
# Percentile helper
# --------------------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear interpolation percentile on a pre-sorted list.

    ``pct`` is a value in [0, 100].
    """
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


# --------------------------------------------------------------------------------------
# SynthesisEngine
# --------------------------------------------------------------------------------------


class SynthesisEngine:
    """Extracts distributional outputs from ensemble results."""

    # ---- Directionality ------------------------------------------------------------

    def _metric_direction(self, name: str) -> Literal["higher_is_better", "lower_is_better"]:
        if name in _HIGHER_IS_BETTER:
            return "higher_is_better"
        if name in _LOWER_IS_BETTER:
            return "lower_is_better"
        # Default to higher_is_better for unknown metrics (safe fallback).
        return "higher_is_better"

    # ---- Path computation ----------------------------------------------------------

    def compute_paths(self, ensemble: EnsembleResult) -> PathBundle:
        """Per-step, per-metric percentile bands across all trajectories."""
        trajectories = ensemble.trajectories
        if not trajectories:
            return PathBundle()

        # Number of steps = min length across trajectories (should all be equal in practice).
        n_steps = min(len(t.steps) for t in trajectories)
        if n_steps == 0:
            return PathBundle()

        central: list[StepMetrics] = []
        optimistic: list[StepMetrics] = []
        pessimistic: list[StepMetrics] = []
        tail_upper: list[StepMetrics] = []
        tail_lower: list[StepMetrics] = []

        for step_idx in range(n_steps):
            # Use the step number from the first trajectory's StepMetrics at that index.
            step_number = trajectories[0].steps[step_idx].step

            bands: dict[str, dict[str, float | int | bool]] = {
                "central": {},
                "optimistic": {},
                "pessimistic": {},
                "tail_upper": {},
                "tail_lower": {},
            }

            for metric in CORE_METRIC_NAMES:
                values = [getattr(t.steps[step_idx], metric) for t in trajectories]

                if metric in _BOOL_METRICS:
                    # Worse = True for interbank_freeze. Use fraction-True thresholds that
                    # mirror the numeric percentile directionality (lower_is_better).
                    bool_values = [bool(v) for v in values]
                    p = sum(1 for v in bool_values if v) / len(bool_values)
                    # central = majority (p >= 0.5)
                    bands["central"][metric] = p >= 0.5
                    # pessimistic (worse-than-central band) = True iff p >= 0.10
                    bands["pessimistic"][metric] = p >= 0.10
                    # optimistic (better-than-central band) = True iff p >= 0.90
                    bands["optimistic"][metric] = p >= 0.90
                    # tail_lower (most extreme worse) = True iff p >= 0.03
                    bands["tail_lower"][metric] = p >= 0.03
                    # tail_upper (most extreme better) = True iff p >= 0.97
                    bands["tail_upper"][metric] = p >= 0.97
                    continue

                numeric_values = sorted(float(v) for v in values)
                p03 = _percentile(numeric_values, 3.0)
                p10 = _percentile(numeric_values, 10.0)
                p50 = _percentile(numeric_values, 50.0)
                p90 = _percentile(numeric_values, 90.0)
                p97 = _percentile(numeric_values, 97.0)

                direction = self._metric_direction(metric)
                if direction == "higher_is_better":
                    c_val = p50
                    opt_val = p90
                    pes_val = p10
                    tu_val = p97
                    tl_val = p03
                else:
                    # lower_is_better: invert so optimistic = low, pessimistic = high
                    c_val = p50
                    opt_val = p10
                    pes_val = p90
                    tu_val = p03  # most extreme "better" side
                    tl_val = p97  # most extreme "worse" side

                if metric in _INT_METRICS:
                    c_val = int(round(c_val))
                    opt_val = int(round(opt_val))
                    pes_val = int(round(pes_val))
                    tu_val = int(round(tu_val))
                    tl_val = int(round(tl_val))

                bands["central"][metric] = c_val
                bands["optimistic"][metric] = opt_val
                bands["pessimistic"][metric] = pes_val
                bands["tail_upper"][metric] = tu_val
                bands["tail_lower"][metric] = tl_val

            central.append(_make_step_metrics(step_number, bands["central"]))
            optimistic.append(_make_step_metrics(step_number, bands["optimistic"]))
            pessimistic.append(_make_step_metrics(step_number, bands["pessimistic"]))
            tail_upper.append(_make_step_metrics(step_number, bands["tail_upper"]))
            tail_lower.append(_make_step_metrics(step_number, bands["tail_lower"]))

        return PathBundle(
            central=central,
            optimistic=optimistic,
            pessimistic=pessimistic,
            tail_upper=tail_upper,
            tail_lower=tail_lower,
        )

    # ---- Divergence map ------------------------------------------------------------

    def compute_divergence_map(
        self, ensemble: EnsembleResult, top_k: int = 5
    ) -> DivergenceMap:
        """Identify the metrics that drive outcome variance the most."""
        trajectories = ensemble.trajectories
        if not trajectories:
            return DivergenceMap(variables=[])

        n_steps = min(len(t.steps) for t in trajectories)
        if n_steps == 0 or len(trajectories) < 2:
            return DivergenceMap(variables=[])

        # Per-metric: compute (avg cross-traj variance over steps, final-step stdev/mean).
        scored: list[tuple[str, float, float]] = []  # (metric, sensitivity, uncertainty)
        for metric in CORE_METRIC_NAMES:
            if metric in _BOOL_METRICS:
                numeric_rows: list[list[float]] = [
                    [float(bool(getattr(t.steps[s], metric))) for t in trajectories]
                    for s in range(n_steps)
                ]
            else:
                numeric_rows = [
                    [float(getattr(t.steps[s], metric)) for t in trajectories]
                    for s in range(n_steps)
                ]

            # sensitivity = mean cross-trajectory variance across steps
            per_step_var = [statistics.pvariance(row) for row in numeric_rows]
            sensitivity = statistics.fmean(per_step_var) if per_step_var else 0.0

            # uncertainty = CV at final step (clamped to [0, 1])
            final_row = numeric_rows[-1]
            mean_final = statistics.fmean(final_row) if final_row else 0.0
            stdev_final = statistics.pstdev(final_row) if len(final_row) > 1 else 0.0
            if abs(mean_final) < 1e-12:
                # If mean is ~0, uncertainty = stdev itself, clamped.
                uncertainty = min(max(stdev_final, 0.0), 1.0)
            else:
                uncertainty = min(max(stdev_final / abs(mean_final), 0.0), 1.0)

            scored.append((metric, sensitivity, uncertainty))

        # Drop zero-variance metrics, rank by sensitivity desc, keep top-k.
        filtered = [row for row in scored if row[1] > 0.0]
        filtered.sort(key=lambda r: r[1], reverse=True)
        filtered = filtered[:top_k]

        variables = [
            DivergenceVariable(
                name=name,
                sensitivity=sensitivity,
                current_uncertainty=uncertainty,
                monitoring_indicator=_MONITORING_INDICATORS[name],
            )
            for (name, sensitivity, uncertainty) in filtered
        ]
        return DivergenceMap(variables=variables)

    # ---- Indicator watchlist -------------------------------------------------------

    def indicator_watchlist(self, divergence_map: DivergenceMap) -> list[str]:
        """Return the monitoring_indicator values from the divergence map in order."""
        return [v.monitoring_indicator for v in divergence_map.variables]

    # ---- Causal chains -------------------------------------------------------------

    def detect_causal_chains(self, ensemble: EnsembleResult) -> list[CausalChain]:
        """Collapse per-trajectory event sequences into canonical causal chains."""
        origin_shock = ""
        if isinstance(ensemble.config, ShockConfig):
            origin_shock = ensemble.config.shock_type

        # Gather the set of known actor IDs across all trajectories' causal_events.
        known_actor_ids: set[str] = set()
        for traj in ensemble.trajectories:
            for ev in traj.causal_events:
                if isinstance(ev, CausalEvent):
                    known_actor_ids.add(ev.source_actor_id)
                    known_actor_ids.add(ev.target_actor_id)

        # Group trajectories by their (source, target, channel) pattern signature.
        # Represent each trajectory's pattern as a tuple of (source, target, channel).
        pattern_to_trajs: dict[tuple[tuple[str, str, str], ...], list[list[CausalEvent]]] = {}
        for traj in ensemble.trajectories:
            events = [ev for ev in traj.causal_events if isinstance(ev, CausalEvent)]
            if not events:
                continue
            # Sort events by step (non-decreasing) for canonical ordering.
            sorted_events = sorted(events, key=lambda e: e.step)
            signature = tuple(
                (ev.source_actor_id, ev.target_actor_id, ev.channel) for ev in sorted_events
            )
            pattern_to_trajs.setdefault(signature, []).append(sorted_events)

        chains: list[CausalChain] = []
        for signature, traj_event_lists in pattern_to_trajs.items():
            # Representative: the trajectory's events with the most common full-content pattern.
            # All trajectories in this bucket share the same signature but may differ in magnitudes.
            # Pick the representative as the event list with the median-ish magnitudes — simplest is first.
            # Deduplicate: if events share identical (step, source, target, channel, variable_affected),
            # keep the first.
            representative = traj_event_lists[0]

            # Deduplicate within representative based on the full tuple.
            seen: set[tuple] = set()
            dedup_events: list[CausalEvent] = []
            for ev in representative:
                key = (
                    ev.step,
                    ev.source_actor_id,
                    ev.target_actor_id,
                    ev.channel,
                    ev.variable_affected,
                )
                if key in seen:
                    continue
                seen.add(key)
                dedup_events.append(ev)

            # Ensure ordering guarantee (defensive — already sorted).
            dedup_events.sort(key=lambda e: e.step)

            # Chain id: stable hash of the signature.
            sig_str = "|".join(f"{s}->{t}:{c}" for (s, t, c) in signature)
            chain_id = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:12]

            total_magnitude = sum(abs(ev.magnitude) for ev in dedup_events)

            # Filter out events referencing unknown actors (defensive — should
            # not happen by construction, but avoids crashing synthesis).
            dedup_events = [
                ev for ev in dedup_events
                if ev.source_actor_id in known_actor_ids
                and ev.target_actor_id in known_actor_ids
            ]
            if not dedup_events:
                continue

            chains.append(
                CausalChain(
                    chain_id=chain_id,
                    events=dedup_events,
                    origin_shock=origin_shock,
                    total_magnitude=total_magnitude,
                )
            )

        # Sort chains by frequency of pattern (most common first) for determinism.
        # Build a mapping from chain_id to the original (non-deduplicated) signature
        # so frequency lookup matches the pattern_to_trajs keys.
        chain_id_to_sig: dict[str, tuple[tuple[str, str, str], ...]] = {}
        for signature, traj_event_lists in pattern_to_trajs.items():
            representative = traj_event_lists[0]
            sig_str = "|".join(f"{s}->{t}:{c}" for (s, t, c) in signature)
            cid = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:12]
            chain_id_to_sig[cid] = signature

        freq = Counter(
            sig
            for sig, trajs in pattern_to_trajs.items()
            for _ in trajs
        )
        chains.sort(key=lambda ch: (-freq.get(chain_id_to_sig.get(ch.chain_id, ()), 0), ch.chain_id))
        return chains

    # ---- Metric selection ----------------------------------------------------------

    def select_metrics(
        self, scenario: object, ensemble: EnsembleResult
    ) -> list[MetricSelection]:
        """Select situation-relevant metrics for reporting.

        Always include divergence-map variables plus the 3 metrics with the largest
        step-0 vs step-N delta in the central path.
        """
        divergence_map = self.compute_divergence_map(ensemble)
        bundle = self.compute_paths(ensemble)

        selections: list[MetricSelection] = []
        seen: set[str] = set()

        # Divergence-map driven selections.
        for var in divergence_map.variables:
            if var.name in seen:
                continue
            seen.add(var.name)
            selections.append(
                MetricSelection(metric=var.name, reason="high cross-run variance")
            )

        # Step-0 vs step-N delta selections (top 3 by |delta|).
        if bundle.central:
            first = bundle.central[0]
            last = bundle.central[-1]
            deltas: list[tuple[str, float]] = []
            for metric in CORE_METRIC_NAMES:
                if metric in _BOOL_METRICS:
                    d = float(bool(getattr(last, metric))) - float(bool(getattr(first, metric)))
                else:
                    d = float(getattr(last, metric)) - float(getattr(first, metric))
                deltas.append((metric, abs(d)))
            deltas.sort(key=lambda r: r[1], reverse=True)
            for metric, _ in deltas[:3]:
                if metric in seen:
                    continue
                seen.add(metric)
                selections.append(
                    MetricSelection(metric=metric, reason="large end-to-end movement")
                )

        return selections


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _make_step_metrics(step: int, values: dict[str, float | int | bool]) -> StepMetrics:
    """Build a StepMetrics from a dict of metric-name -> value with correct types."""
    return StepMetrics(
        step=step,
        gdp_index=float(values["gdp_index"]),
        inflation_rate=float(values["inflation_rate"]),
        unemployment_rate=float(values["unemployment_rate"]),
        gini_coefficient=float(values["gini_coefficient"]),
        credit_tightening_index=float(values["credit_tightening_index"]),
        firm_bankruptcy_count=int(values["firm_bankruptcy_count"]),
        bank_stress_index=float(values["bank_stress_index"]),
        consumer_confidence=float(values["consumer_confidence"]),
        interbank_freeze=bool(values["interbank_freeze"]),
    )


__all__ = ["SynthesisEngine", "MetricSelection"]
