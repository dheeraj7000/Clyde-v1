"""In-memory job store + background pipeline runner.

A *job* here is a thin handle around an asyncio task that drives the
``ClydePipeline`` end-to-end. Jobs and their forked branches live in a
single process-wide :class:`JobStore` keyed by UUID. The store enforces a
soft retention policy: when more than :data:`MAX_JOBS` jobs accumulate, the
oldest are evicted.

Progress is reported via a :class:`ProgressTracker` that the runner threads
through to the pipeline-phase wrappers. We avoid touching
``clyde/pipeline.py`` by wrapping the pipeline phases at the
``ClydePipeline.run`` level — the runner just calls ``pipeline.run()`` and
brackets it with progress markers. (The pipeline does not yield mid-flight,
so progress jumps in coarse increments.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    """Return ``True`` when ``CLYDE_DEBUG`` is set to a truthy value."""
    val = os.environ.get("CLYDE_DEBUG", "").strip().lower()
    return val in {"1", "true", "yes", "on"}

from clyde.models.causal import CausalChain
from clyde.models.metrics import StepMetrics
from clyde.pipeline import ClydePipeline, PipelineConfig, PipelineResult
from clyde.reporting import NarrativeReport
from clyde.simulation import BranchResult


MAX_JOBS: int = 32


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


@dataclass
class Progress:
    stage: str = "queued"
    percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "percent": float(self.percent)}


class ProgressTracker:
    """Mutable progress holder shared with the running task."""

    def __init__(self) -> None:
        self._progress = Progress()

    @property
    def progress(self) -> Progress:
        return self._progress

    def set(self, stage: str, percent: float) -> None:
        self._progress = Progress(stage=stage, percent=float(percent))


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


@dataclass
class BranchJob:
    branch_id: str
    status: str = "pending"
    progress: Progress = field(default_factory=Progress)
    result: dict[str, Any] | None = None
    error: str | None = None
    details: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task[Any] | None = None
    branch_result: BranchResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "status": self.status,
            "progress": self.progress.to_dict(),
            "result": self.result,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class Job:
    job_id: str
    status: str = "pending"
    progress: Progress = field(default_factory=Progress)
    result: dict[str, Any] | None = None
    error: str | None = None
    details: str | None = None
    pipeline: ClydePipeline | None = None
    pipeline_result: PipelineResult | None = None
    branches: dict[str, BranchJob] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress.to_dict(),
            "result": self.result,
            "error": self.error,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class JobStore:
    """Process-wide job + branch store with soft retention."""

    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._max = max_jobs
        self._lock = asyncio.Lock()

    def _evict_if_needed(self) -> None:
        while len(self._jobs) > self._max:
            jid, evicted = self._jobs.popitem(last=False)
            # Best-effort cleanup: close DB if the pipeline still holds one.
            try:
                if evicted.pipeline is not None:
                    evicted.pipeline.close()
            except Exception:
                pass
            # Cancel the pipeline driver task and any in-flight branch tasks
            # so background coroutines don't outlive their job handle.
            try:
                if evicted.task is not None and not evicted.task.done():
                    evicted.task.cancel()
            except Exception:
                pass
            for branch in evicted.branches.values():
                try:
                    if branch.task is not None and not branch.task.done():
                        branch.task.cancel()
                except Exception:
                    pass

    async def create_job_async(self) -> Job:
        async with self._lock:
            job = Job(job_id=str(uuid.uuid4()))
            self._jobs[job.job_id] = job
            self._evict_if_needed()
            return job

    def create_job(self) -> Job:
        """Non-async create for backwards compatibility (tests, sync callers)."""
        job = Job(job_id=str(uuid.uuid4()))
        self._jobs[job.job_id] = job
        self._evict_if_needed()
        return job

    async def get_async(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all_ids(self) -> list[str]:
        return list(self._jobs.keys())

    def __len__(self) -> int:
        return len(self._jobs)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _step_to_dict(step: StepMetrics) -> dict[str, Any]:
    return step.to_dict()


def _path_to_list(steps: list[StepMetrics]) -> list[dict[str, Any]]:
    return [_step_to_dict(s) for s in steps]


def _serialize_parse_result(result: PipelineResult) -> dict[str, Any]:
    pr = result.parse_result
    return {
        "triggering_event": pr.triggering_event,
        "geographies": list(pr.geographies),
        "markets": list(pr.markets),
        "ambiguities": [a.to_dict() for a in pr.ambiguities],
        "shock_params": {
            "shock_type": pr.shock_params.shock_type,
            "severity": pr.shock_params.severity,
            "scope": pr.shock_params.scope,
            "duration_steps": pr.shock_params.duration_steps,
            "initial_contact_actors": list(pr.shock_params.initial_contact_actors),
        },
        "time_horizon": {
            "steps": pr.time_horizon.steps,
            "step_unit": pr.time_horizon.step_unit,
        },
    }


def _serialize_paths(result: PipelineResult) -> dict[str, Any]:
    paths = result.synthesis.paths
    return {
        "metrics": [
            "gdp_index",
            "inflation_rate",
            "unemployment_rate",
            "gini_coefficient",
            "credit_tightening_index",
            "firm_bankruptcy_count",
            "bank_stress_index",
            "consumer_confidence",
            "interbank_freeze",
        ],
        "central": _path_to_list(paths.central),
        "optimistic": _path_to_list(paths.optimistic),
        "pessimistic": _path_to_list(paths.pessimistic),
        "tail_upper": _path_to_list(paths.tail_upper),
        "tail_lower": _path_to_list(paths.tail_lower),
    }


def _serialize_divergence(result: PipelineResult) -> dict[str, Any]:
    return result.synthesis.divergence_map.to_dict()


def _serialize_network(result: PipelineResult) -> dict[str, Any]:
    """Serialize actors + network edges for the graph visualization."""
    nodes = []
    initial_contacts = set(result.shock_config.initial_contact_actors or [])
    for a in result.world.actors:
        from dataclasses import asdict
        node: dict[str, Any] = {
            "id": a.id,
            "type": a.actor_type.value,
            "shocked": a.id in initial_contacts,
            "params": asdict(a.params),
            "relationships": [r.to_dict() for r in a.relationships],
        }
        nodes.append(node)
    edges = []
    nets = result.world.networks
    for src, tgt, w in nets.labor_market.edges:
        edges.append({"source": src, "target": tgt, "type": "employment", "weight": w})
    for src, tgt, w in nets.supply_chain.edges:
        edges.append({"source": src, "target": tgt, "type": "supply", "weight": w})
    for src, tgt, w in nets.interbank.edges:
        edges.append({"source": src, "target": tgt, "type": "interbank", "weight": w})
    return {"nodes": nodes, "edges": edges}


def _serialize_causal_chains(chains: list[CausalChain]) -> list[dict[str, Any]]:
    return [c.serialize() for c in chains]


def _serialize_report(report: NarrativeReport) -> dict[str, Any]:
    sections = []
    for sec in report.sections:
        sections.append(
            {
                "heading": sec.heading,
                "body": sec.body,
                "flags": list(sec.flags),
                "provenance": [
                    {
                        "claim": p.claim,
                        "source_type": p.source_type,
                        "source_ref": p.source_ref,
                        "query_used": p.query_used,
                    }
                    for p in sec.provenance
                ],
            }
        )
    return {
        "scenario_id": report.scenario_id,
        "uncertainty_flags": list(report.uncertainty_flags),
        "sections": sections,
        "provenance_count": len(report.provenance),
    }


def serialize_pipeline_result(result: PipelineResult) -> dict[str, Any]:
    """Convert a :class:`PipelineResult` to a JSON-friendly dict."""
    from clyde.synthesis import SynthesisEngine

    watchlist = SynthesisEngine().indicator_watchlist(result.synthesis.divergence_map)
    return {
        "scenario_id": result.scenario_id,
        "shock_config": result.shock_config.to_dict(),
        "parse_result": _serialize_parse_result(result),
        "paths": _serialize_paths(result),
        "divergence": _serialize_divergence(result),
        "watchlist": watchlist,
        "causal_chains": _serialize_causal_chains(result.synthesis.causal_chains),
        "report": _serialize_report(result.report),
        "branches": [serialize_branch_result(b) for b in result.branches],
        "network": _serialize_network(result),
        "personas": [p.to_dict() for p in (result.personas or [])],
        "influence_config": result.influence_config.to_dict() if hasattr(result.influence_config, 'to_dict') and result.influence_config else {},
        "influence_reasoning": result.influence_reasoning or "",
    }


def serialize_branch_result(branch: BranchResult) -> dict[str, Any]:
    """Compact JSON view of a forked branch — paths + divergence, no report."""
    from clyde.synthesis import SynthesisEngine

    engine = SynthesisEngine()
    paths = engine.compute_paths(branch.ensemble)
    divergence = engine.compute_divergence_map(branch.ensemble)
    chains = engine.detect_causal_chains(branch.ensemble)

    return {
        "branch_id": branch.branch_id,
        "parent_scenario_id": branch.parent_scenario_id,
        "merged_config": branch.merged_config.to_dict(),
        "delta": branch.delta.to_dict(),
        "paths": {
            "metrics": [
                "gdp_index",
                "inflation_rate",
                "unemployment_rate",
                "gini_coefficient",
                "credit_tightening_index",
                "firm_bankruptcy_count",
                "bank_stress_index",
                "consumer_confidence",
                "interbank_freeze",
            ],
            "central": _path_to_list(paths.central),
            "optimistic": _path_to_list(paths.optimistic),
            "pessimistic": _path_to_list(paths.pessimistic),
            "tail_upper": _path_to_list(paths.tail_upper),
            "tail_lower": _path_to_list(paths.tail_lower),
        },
        "divergence": divergence.to_dict(),
        "causal_chains": [c.serialize() for c in chains],
    }


# ---------------------------------------------------------------------------
# Background runners
# ---------------------------------------------------------------------------


# Stage / percent waypoints for the run pipeline. The runner emits a stage
# at start, then bumps to "completed" on success.
_RUN_START_STAGE = ("parsing", 5.0)
_RUN_COMPLETE_STAGE = ("completed", 100.0)


def _stage_progress(job: Job, stage: str, percent: float) -> None:
    job.progress = Progress(stage=stage, percent=percent)
    job.updated_at = time.time()


def _branch_stage_progress(branch: BranchJob, stage: str, percent: float) -> None:
    branch.progress = Progress(stage=stage, percent=percent)
    branch.updated_at = time.time()


# Type alias for pipeline factory injection (allows test override).
PipelineFactory = Callable[[PipelineConfig], ClydePipeline]


async def run_pipeline_job(
    job: Job,
    *,
    description: str,
    config: PipelineConfig,
    pipeline_factory: PipelineFactory,
    horizon_steps: int | None = None,
) -> None:
    """Drive a single pipeline run in the background.

    All exceptions are captured onto the job — this coroutine never raises.
    """
    job.status = "running"
    _stage_progress(job, *_RUN_START_STAGE)

    timer_handles: list[Any] = []
    try:
        pipeline = pipeline_factory(config)
        job.pipeline = pipeline

        # Coarse progress: we can't actually slice ClydePipeline.run without
        # modifying it, so we set "running ensemble" right before the call
        # and bump after it returns.
        _stage_progress(job, "building world", 25.0)

        def _bump(stage: str, percent: float) -> None:
            # No-op once the job has reached a terminal status; protects
            # against a still-pending timer overwriting the final state.
            if job.status in ("completed", "failed"):
                return
            _stage_progress(job, stage, percent)

        async def _runner() -> PipelineResult:
            # Mid-flight bumps via call_later — purely advisory. Capture
            # the handles so we can cancel any still-pending timers before
            # they clobber the terminal progress state.
            loop = asyncio.get_running_loop()
            timer_handles.append(loop.call_later(0.05, lambda: _bump("simulating ensemble", 35.0)))
            timer_handles.append(loop.call_later(0.5, lambda: _bump("synthesizing", 80.0)))
            timer_handles.append(loop.call_later(0.85, lambda: _bump("divergence map", 87.0)))
            timer_handles.append(loop.call_later(1.0, lambda: _bump("generating report", 92.0)))
            return await pipeline.run(description)

        result = await _runner()

        # Cancel any still-pending advisory timers so they don't fire
        # after we set the terminal state.
        for h in timer_handles:
            try:
                h.cancel()
            except Exception:
                pass

        # If the caller asked for a horizon override, we honour it post-hoc by
        # truncating the path bundles. The world's actual time_horizon is set
        # from the parse result during `pipeline.run`, so we can't override it
        # in advance without modifying pipeline.py — but we can clamp the
        # outputs when the caller is asking for a smaller view.
        if horizon_steps is not None and horizon_steps > 0:
            _truncate_paths_in_place(result, horizon_steps)

        job.pipeline_result = result
        job.result = serialize_pipeline_result(result)
        job.status = "completed"
        _stage_progress(job, *_RUN_COMPLETE_STAGE)
    except Exception as exc:
        for h in timer_handles:
            try:
                h.cancel()
            except Exception:
                pass
        logger.exception("pipeline run failed")
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.details = traceback.format_exc() if _debug_enabled() else None
        _stage_progress(job, "failed", 100.0)


def _truncate_paths_in_place(result: PipelineResult, horizon_steps: int) -> None:
    """Clip each path band to ``horizon_steps`` entries (best-effort)."""
    paths = result.synthesis.paths
    for band_name in ("central", "optimistic", "pessimistic", "tail_upper", "tail_lower"):
        steps = getattr(paths, band_name)
        if len(steps) > horizon_steps:
            setattr(paths, band_name, steps[:horizon_steps])


async def run_branch_job(
    job: Job,
    branch: BranchJob,
    *,
    injection_text: str,
) -> None:
    """Fork a branch off ``job``'s pipeline + result."""
    branch.status = "running"
    _branch_stage_progress(branch, "parsing injection", 10.0)

    try:
        pipeline = job.pipeline
        result = job.pipeline_result
        if pipeline is None or result is None:
            raise RuntimeError("Parent job has no live pipeline; cannot fork.")

        _branch_stage_progress(branch, "running branch ensemble", 50.0)
        branch_result = await pipeline.fork_branch(result, injection_text)
        branch.branch_result = branch_result
        payload = serialize_branch_result(branch_result)
        # Surface the API-level branch id (the UUID we minted at request
        # time) so the client can correlate the polling URL with the
        # response body. Preserve the simulator-internal id under a
        # secondary key for traceability.
        payload["internal_branch_id"] = payload.get("branch_id")
        payload["branch_id"] = branch.branch_id
        branch.result = payload
        branch.status = "completed"
        _branch_stage_progress(branch, "completed", 100.0)
    except Exception as exc:
        logger.exception("branch run failed")
        branch.status = "failed"
        branch.error = f"{type(exc).__name__}: {exc}"
        branch.details = traceback.format_exc() if _debug_enabled() else None
        _branch_stage_progress(branch, "failed", 100.0)


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def default_pipeline_factory_for(
    *,
    provider: str = "auto",
    model: str | None = None,
) -> PipelineFactory:
    """Return a factory that builds a real :class:`ClydePipeline`."""
    from clyde.llm import make_llm_client

    def _factory(cfg: PipelineConfig) -> ClydePipeline:
        client = make_llm_client(provider, model=model)  # type: ignore[arg-type]
        return ClydePipeline(client, config=cfg)

    return _factory


__all__ = [
    "MAX_JOBS",
    "BranchJob",
    "Job",
    "JobStore",
    "PipelineFactory",
    "Progress",
    "ProgressTracker",
    "default_pipeline_factory_for",
    "run_branch_job",
    "run_pipeline_job",
    "serialize_branch_result",
    "serialize_pipeline_result",
]
