"""Monte Carlo Controller (task 7).

Orchestrates parallel ensembles of :class:`PropagationEngine` runs and
provides branch (intervention) re-simulation. Strictly LLM-free — this
module sits inside ``clyde.simulation`` and is covered by the static
LLM-boundary test.

Responsibilities (Requirements 6.1–6.7, 8.1–8.7):

* Generate per-run deterministic seeds from a single ensemble seed.
* Build a *jittered* :class:`SimulationWorld` per run so each trajectory
  uses a distinct, sampled combination of parameter values across the four
  axes called out by the design: behavioral response strength, timing,
  shock severity, and contagion thresholds.
* Run the ensemble in parallel via :class:`ProcessPoolExecutor`. A
  module-level worker entry point keeps everything picklable.
* Tolerate worker crashes: failed runs are logged + warned about, the
  successful trajectories are still returned in an :class:`EnsembleResult`.
* Fork branches by merging a :class:`ShockDelta` into a base
  :class:`ShockConfig` and re-simulating from step 0 (NEVER mutating
  running state).
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

from clyde.models.actors import Actor
from clyde.models.config import ShockConfig, ShockDelta, SimulationWorld
from clyde.models.metrics import EnsembleResult, TrajectoryResult
from clyde.persistence.db import SimulationDB
from clyde.simulation.propagation import PropagationEngine

logger = logging.getLogger(__name__)


# Knuth multiplicative hash constant (golden-ratio integer). Used to derive
# per-run seeds from the ensemble seed; cheap, dependency-free, and stable
# across Python versions / processes.
_KNUTH_MULTIPLIER = 2654435761
_UINT32_MASK = 0xFFFFFFFF

# Default run count must fall in the [100, 500] range from Requirement 6.6.
_DEFAULT_RUN_COUNT = 200


# ---------------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class BranchResult:
    """Output of :meth:`MonteCarloController.fork_branch`.

    Records the parent scenario id, the delta that was applied, the merged
    :class:`ShockConfig` that drove re-simulation, and the resulting
    ensemble. Persisted via :meth:`SimulationDB.insert_branch` when a
    database is supplied.
    """

    branch_id: str
    parent_scenario_id: str
    delta: ShockDelta
    merged_config: ShockConfig
    ensemble: EnsembleResult


# ---------------------------------------------------------------------------
# Worker entry point.
#
# This function MUST be at module top-level so that ``ProcessPoolExecutor``
# can pickle a reference to it. Workers re-import this module on spawn.
# ---------------------------------------------------------------------------


def _run_one(args: tuple[dict, int, int, str, float]) -> dict:
    """Worker entry point: rebuild a world, jitter it, run one trajectory.

    Inputs are tuple-packed so the executor can pickle them once. The
    payload is:

    ``(world_dict, seed, run_idx, run_id_prefix, jitter)``

    Returns a serialisable dict with either ``"ok": True`` and the
    trajectory dict, or ``"ok": False`` and an error string. We round-trip
    through ``dict`` (via ``SimulationWorld.to_dict``) to avoid pickling
    issues with closures, locks, or RNG state.
    """
    world_dict, seed, run_idx, run_id_prefix, jitter = args
    try:
        world = SimulationWorld.from_dict(world_dict)
        jittered = _jitter_world_impl(world, seed=seed, jitter=jitter)
        engine = PropagationEngine()
        run_id = f"{run_id_prefix}_{run_idx:04d}"
        trajectory = engine.run(jittered, seed=seed, run_id=run_id, db=None)
        return {
            "ok": True,
            "run_idx": run_idx,
            "trajectory": _trajectory_to_payload(trajectory),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ok": False,
            "run_idx": run_idx,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _trajectory_to_payload(traj: TrajectoryResult) -> dict:
    """Convert a TrajectoryResult to a picklable dict (steps + events)."""
    return {
        "run_id": traj.run_id,
        "seed": traj.seed,
        "steps": [s.to_dict() for s in traj.steps],
        "causal_events": [
            {
                "step": ev.step,
                "source_actor_id": ev.source_actor_id,
                "target_actor_id": ev.target_actor_id,
                "channel": ev.channel,
                "variable_affected": ev.variable_affected,
                "magnitude": ev.magnitude,
                "description": ev.description,
            }
            for ev in traj.causal_events
        ],
        "final_state_ref": traj.final_state_ref,
    }


def _payload_to_trajectory(payload: dict) -> TrajectoryResult:
    from clyde.models.causal import CausalEvent
    from clyde.models.metrics import StepMetrics

    return TrajectoryResult(
        run_id=payload["run_id"],
        seed=int(payload["seed"]),
        steps=[StepMetrics.from_dict(s) for s in payload["steps"]],
        causal_events=[
            CausalEvent(
                step=int(e["step"]),
                source_actor_id=e["source_actor_id"],
                target_actor_id=e["target_actor_id"],
                channel=e["channel"],
                variable_affected=e["variable_affected"],
                magnitude=float(e["magnitude"]),
                description=e.get("description", ""),
            )
            for e in payload["causal_events"]
        ],
        final_state_ref=payload.get("final_state_ref"),
    )


# ---------------------------------------------------------------------------
# Jitter helpers (also used by tests directly).
# ---------------------------------------------------------------------------


def _jitter_world_impl(
    base_world: SimulationWorld,
    *,
    seed: int,
    jitter: float,
) -> SimulationWorld:
    """Return a fresh :class:`SimulationWorld` with multiplicatively jittered
    actor params and a (deterministically) jittered shock severity.

    The four axes called out by Requirement 6.1 are perturbed:

    * **behavioral response strength** — every actor param times
      ``(1 + uniform(-jitter, jitter))``.
    * **timing** — small offset added to firing/hiring threshold params for
      firms (kept inside the param's natural range).
    * **shock severity** — ``shock_config.severity * (1 + uniform(...))``,
      clamped to ``[0, 1]``.
    * **contagion thresholds** — bank ``reserve_threshold`` /
      ``credit_approval_floor`` are part of the multiplicative jitter pass
      above; we do not perturb them a second time.

    The returned world reuses the base world's networks & actor identities
    — only ``params`` (immutable dataclasses) and the embedded ``config``
    are replaced.
    """
    rng = random.Random(seed)

    new_actors: list[Actor] = []
    for a in base_world.actors:
        new_params = _jitter_params(a.params, rng=rng, jitter=jitter)
        new_actors.append(replace(a, params=new_params, state={}))

    # Severity jitter (clamped). Small probability we land at exactly the
    # boundary — that's still legal for ShockConfig.
    base_cfg = base_world.config
    sev = base_cfg.severity * (1.0 + rng.uniform(-jitter, jitter))
    sev = max(0.0, min(1.0, sev))
    new_cfg = replace(base_cfg, severity=sev)

    return SimulationWorld(
        config=new_cfg,
        actors=new_actors,
        networks=base_world.networks,
        prior_library_version=base_world.prior_library_version,
    )


def _jitter_params(params: Any, *, rng: random.Random, jitter: float) -> Any:
    """Multiplicative jitter on every numeric field of a params dataclass.

    Plus a small additive 'timing' offset on firm hiring/firing thresholds
    so the *timing* axis is varied independently of pure response strength.
    Negative or out-of-range outcomes are clamped to a small positive value
    where appropriate.
    """
    from dataclasses import fields, replace as dc_replace

    new_kwargs: dict[str, Any] = {}
    timing_fields = {"firing_threshold", "hiring_elasticity"}
    for f in fields(params):
        cur = getattr(params, f.name)
        if not isinstance(cur, (int, float)):
            new_kwargs[f.name] = cur
            continue
        # Multiplicative noise.
        scaled = float(cur) * (1.0 + rng.uniform(-jitter, jitter))
        # Timing axis: extra small additive offset on hiring/firing.
        if f.name in timing_fields:
            scaled = scaled + rng.uniform(-jitter, jitter) * 0.05
        # Keep params strictly positive — most of the prior library values
        # are rates / weights that go ill-defined at zero or negative.
        if cur > 0 and scaled <= 0:
            scaled = max(1e-6, abs(scaled))
        new_kwargs[f.name] = scaled
    return dc_replace(params, **new_kwargs)


# ---------------------------------------------------------------------------
# Controller.
# ---------------------------------------------------------------------------


class MonteCarloController:
    """Run an ensemble of :class:`PropagationEngine` simulations in parallel."""

    def __init__(
        self,
        engine: PropagationEngine | None = None,
        max_workers: int | None = None,
        param_jitter: float = 0.10,
    ) -> None:
        # The engine is held for the serial path / single-shot calls. The
        # parallel path constructs a fresh engine inside each worker (engines
        # are stateless so there's no behavioural difference).
        self._engine = engine if engine is not None else PropagationEngine()
        self._default_max_workers = max_workers
        if not (0.0 <= param_jitter <= 1.0):
            raise ValueError(
                f"param_jitter must be in [0, 1], got {param_jitter}"
            )
        self._jitter = float(param_jitter)

    # ------------------------------------------------------------------ API

    def run_ensemble(
        self,
        world: SimulationWorld,
        run_count: int = _DEFAULT_RUN_COUNT,
        max_workers: int | None = None,
        scenario_id: str | None = None,
        db: SimulationDB | None = None,
        parallel: bool = True,
    ) -> EnsembleResult:
        """Run ``run_count`` simulations and return their aggregated result.

        Parameters
        ----------
        world:
            Base :class:`SimulationWorld`. Each run jitters a fresh copy.
        run_count:
            Number of simulations. Must be >= 1.
        max_workers:
            Override the default worker pool size. ``None`` ⇒ instance
            default ⇒ ``min(os.cpu_count() or 1, 4)``.
        scenario_id:
            Identifier for the ensemble. Falls back to a stable string built
            from the ensemble seed if not supplied.
        db:
            Optional persistence handle. When provided, every successful run
            is persisted (run record + step metrics + causal events).
        parallel:
            If False, runs serially — useful in environments where the
            process pool is unreliable (e.g. some CI containers).
        """
        if run_count < 1:
            raise ValueError(f"run_count must be >= 1, got {run_count}")

        ensemble_seed = int(world.config.ensemble_seed)
        seeds = self._generate_seeds(ensemble_seed, run_count)
        scenario = scenario_id or f"ensemble_{ensemble_seed}"

        trajectories: list[TrajectoryResult] = []
        failed: list[tuple[int, str]] = []

        if parallel:
            trajectories, failed = self._run_parallel(
                world=world,
                seeds=seeds,
                run_id_prefix=scenario,
                max_workers=max_workers,
            )
        else:
            trajectories, failed = self._run_serial(
                world=world,
                seeds=seeds,
                run_id_prefix=scenario,
            )

        # Persist successful trajectories if a DB was supplied. We do this in
        # the orchestrator process (not in the worker) so that the DB
        # connection — which is not picklable — never crosses a process
        # boundary.
        if db is not None:
            for traj in trajectories:
                db.insert_run(
                    run_id=traj.run_id,
                    scenario_id=scenario,
                    seed=traj.seed,
                    config=world.config,
                    status="completed",
                )
                db.insert_step_metrics_batch(traj.run_id, traj.steps)
                for ev in traj.causal_events:
                    db.insert_causal_event(traj.run_id, ev)

        # Surface worker crashes loudly but don't take down the ensemble.
        if failed:
            ids = ", ".join(f"#{idx}: {err}" for idx, err in failed)
            msg = f"{len(failed)} of {run_count} runs failed: {ids}"
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return EnsembleResult(
            scenario_id=scenario,
            config=world.config,
            trajectories=trajectories,
            run_count=len(trajectories),
            ensemble_seed=ensemble_seed,
        )

    def fork_branch(
        self,
        base_world: SimulationWorld,
        delta: ShockDelta,
        run_count: int = _DEFAULT_RUN_COUNT,
        parent_scenario_id: str = "",
        db: SimulationDB | None = None,
        branch_id: str | None = None,
        parallel: bool = True,
    ) -> BranchResult:
        """Re-simulate from step 0 with ``delta`` merged into the base config.

        We never mutate running simulation state (Requirement 8.1, 8.2). The
        merged config drives a fresh ensemble; the resulting trajectories are
        wrapped in a :class:`BranchResult` and (optionally) recorded in the
        ``branches`` table.
        """
        merged_config = self.merge_delta(base_world.config, delta)

        # Build a fresh world: keep the base actors and networks, swap in the
        # merged config. The actors carry their own params so the merged
        # behavioral_overrides flow through ``run_ensemble``'s jitter step.
        forked_world = SimulationWorld(
            config=merged_config,
            actors=list(base_world.actors),
            networks=base_world.networks,
            prior_library_version=base_world.prior_library_version,
        )

        bid = branch_id or f"branch_{merged_config.ensemble_seed}_{delta.intervention_step}"
        ensemble = self.run_ensemble(
            world=forked_world,
            run_count=run_count,
            scenario_id=bid,
            db=db,
            parallel=parallel,
        )

        if db is not None:
            db.insert_branch(
                branch_id=bid,
                parent_scenario_id=parent_scenario_id,
                shock_delta=delta,
                merged_config=merged_config,
            )

        return BranchResult(
            branch_id=bid,
            parent_scenario_id=parent_scenario_id,
            delta=delta,
            merged_config=merged_config,
            ensemble=ensemble,
        )

    @staticmethod
    def merge_delta(base_config: ShockConfig, delta: ShockDelta) -> ShockConfig:
        """Merge a :class:`ShockDelta` into a :class:`ShockConfig`.

        Rules:

        * The delta's ``intervention_step`` is validated against the base
          config's time horizon. Out-of-range ⇒ ``ValueError``.
        * Top-level keys in ``param_overrides`` (``severity``,
          ``duration_steps``, ``shock_type``, ``scope``) override the
          corresponding fields of ``ShockConfig`` directly. Validation runs
          when the new ``ShockConfig`` is constructed (e.g. invalid ``scope``
          will raise from ``ShockConfig.__post_init__``).
        * All remaining ``param_overrides`` keys are merged into
          ``behavioral_overrides`` (delta wins on conflict).
        * ``new_events`` is appended into
          ``behavioral_overrides["new_events"]`` (created if absent), as a
          *list*, preserving any existing list-typed value.
        """
        n_steps = int(base_config.time_horizon.steps)
        if delta.intervention_step < 0 or (
            n_steps > 0 and delta.intervention_step > n_steps - 1
        ):
            raise ValueError(
                f"ShockDelta.intervention_step={delta.intervention_step} is "
                f"outside the base config's [0, {n_steps - 1}] range."
            )

        top_level_keys = {"severity", "duration_steps", "shock_type", "scope"}
        top_level: dict[str, Any] = {}
        behavioral_extra: dict[str, Any] = {}
        for k, v in delta.param_overrides.items():
            if k in top_level_keys:
                top_level[k] = v
            else:
                behavioral_extra[k] = v

        # Behavioral overrides — start with the base copy, layer the delta on
        # top so the delta wins on conflicting keys.
        merged_behavioral = dict(base_config.behavioral_overrides)
        merged_behavioral.update(behavioral_extra)

        # Append new_events into behavioral_overrides["new_events"] as a
        # list, preserving any existing list value.
        if delta.new_events:
            existing = merged_behavioral.get("new_events")
            if isinstance(existing, list):
                merged_behavioral["new_events"] = list(existing) + list(delta.new_events)
            else:
                merged_behavioral["new_events"] = list(delta.new_events)

        # Use ``replace`` so all other fields (geography, sectors, agent
        # counts, time horizon, ensemble_seed, historical_analogs) survive
        # untouched.
        merged = replace(
            base_config,
            shock_type=top_level.get("shock_type", base_config.shock_type),
            severity=float(top_level["severity"])
            if "severity" in top_level
            else base_config.severity,
            scope=top_level.get("scope", base_config.scope),
            duration_steps=int(top_level["duration_steps"])
            if "duration_steps" in top_level
            else base_config.duration_steps,
            geography=list(base_config.geography),
            sectors=list(base_config.sectors),
            initial_contact_actors=list(base_config.initial_contact_actors),
            agent_counts=dict(base_config.agent_counts),
            behavioral_overrides=merged_behavioral,
            historical_analogs=list(base_config.historical_analogs),
        )
        return merged

    # ----------------------------------------------------------- internals

    def _generate_seeds(self, ensemble_seed: int, run_count: int) -> list[int]:
        """Deterministic per-run seed derivation from the master seed.

        We multiply the master seed by Knuth's golden-ratio constant and XOR
        with the run index, then mask to 32-bit positive. Collisions are
        astronomically unlikely (run_count is in the hundreds, the space is
        2**32) but the spec asks us to detect and re-derive — we do so by
        bumping the index until uniqueness holds.
        """
        seeds: list[int] = []
        used: set[int] = set()
        for i in range(run_count):
            j = i
            while True:
                seed = (ensemble_seed * _KNUTH_MULTIPLIER) ^ j
                seed &= _UINT32_MASK
                if seed not in used:
                    used.add(seed)
                    seeds.append(seed)
                    break
                # Collision: bump j by a large prime so we don't immediately
                # collide again (and remain deterministic).
                j += 0x9E3779B1
        return seeds

    def _run_serial(
        self,
        world: SimulationWorld,
        seeds: list[int],
        run_id_prefix: str,
    ) -> tuple[list[TrajectoryResult], list[tuple[int, str]]]:
        trajectories: list[TrajectoryResult] = []
        failed: list[tuple[int, str]] = []
        for idx, seed in enumerate(seeds):
            try:
                jittered = _jitter_world_impl(world, seed=seed, jitter=self._jitter)
                run_id = f"{run_id_prefix}_{idx:04d}"
                traj = self._engine.run(jittered, seed=seed, run_id=run_id, db=None)
                trajectories.append(traj)
            except Exception as exc:
                failed.append((idx, f"{type(exc).__name__}: {exc}"))
        return trajectories, failed

    def _run_parallel(
        self,
        world: SimulationWorld,
        seeds: list[int],
        run_id_prefix: str,
        max_workers: int | None,
    ) -> tuple[list[TrajectoryResult], list[tuple[int, str]]]:
        chosen_workers = max_workers
        if chosen_workers is None:
            chosen_workers = self._default_max_workers
        if chosen_workers is None:
            chosen_workers = min(os.cpu_count() or 1, 4)
        chosen_workers = max(1, int(chosen_workers))

        # Cheap escape hatch: a run_count of 1 doesn't need a process pool.
        if len(seeds) == 1 or chosen_workers == 1:
            return self._run_serial(world, seeds, run_id_prefix)

        world_dict = world.to_dict()
        payloads: list[tuple[dict, int, int, str, float]] = [
            (world_dict, seed, idx, run_id_prefix, self._jitter)
            for idx, seed in enumerate(seeds)
        ]

        trajectories_by_idx: dict[int, TrajectoryResult] = {}
        failed: list[tuple[int, str]] = []

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=chosen_workers
            ) as pool:
                future_to_idx = {
                    pool.submit(_run_one, p): p[2] for p in payloads
                }
                for fut in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        failed.append((idx, f"{type(exc).__name__}: {exc}"))
                        continue
                    if result.get("ok"):
                        trajectories_by_idx[idx] = _payload_to_trajectory(
                            result["trajectory"]
                        )
                    else:
                        failed.append((idx, result.get("error", "unknown")))
        except concurrent.futures.process.BrokenProcessPool as exc:
            # Pool blew up mid-flight: fall back to serial so the caller
            # still gets a usable EnsembleResult.
            logger.warning(
                "ProcessPoolExecutor broke (%s); falling back to serial path.",
                exc,
            )
            return self._run_serial(world, seeds, run_id_prefix)

        # Preserve seed order (deterministic w.r.t. the seed list, not the
        # order in which futures completed).
        ordered = [trajectories_by_idx[i] for i in sorted(trajectories_by_idx)]
        return ordered, failed


__all__ = [
    "BranchResult",
    "MonteCarloController",
]
