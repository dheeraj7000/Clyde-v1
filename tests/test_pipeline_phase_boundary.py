"""Runtime contract tests for Requirements 15.1, 15.2, 15.4 / Task 18.

The static import-analysis test in ``tests/test_llm_boundary.py`` proves that
``clyde.simulation`` never imports an LLM client. These tests are the
*runtime* analogue: they patch the ``MockLLMClient`` to record every call,
then assert the call log shows LLM activity exclusively during the *setup*
and *reporting* phases — never during ``MonteCarloController.run_ensemble``
or ``PropagationEngine.run``.

This closes Req 15 end-to-end: the architectural split holds in practice,
not just at import time.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import pytest

from clyde.llm import LLMMessage, LLMResponse, MockLLMClient
from clyde.pipeline import ClydePipeline, PipelineConfig
from clyde.simulation import MonteCarloController, PropagationEngine


# Re-use the same router shape from the existing E2E test so we get
# well-formed responses for parser / KG / God's Eye / report polish.
def _parser_payload(*, shock_type: str = "rate_hike", scope: str = "macro") -> dict:
    return {
        "triggering_event": "Fed rate hike",
        "geographies": ["US"],
        "markets": ["finance", "consumer"],
        "shock_params": {
            "shock_type": shock_type,
            "severity": 0.40,
            "scope": scope,
            "duration_steps": 4,
            "initial_contact_actors": ["central_bank_0000"],
        },
        "time_horizon": {"steps": 4, "step_unit": "quarter"},
        "ambiguities": [],
        "actor_hints": [
            {"actor_type": "household", "count_estimate": 20, "description": ""},
            {"actor_type": "firm", "count_estimate": 4, "description": ""},
            {"actor_type": "bank", "count_estimate": 2, "description": ""},
            {"actor_type": "central_bank", "count_estimate": 1, "description": ""},
        ],
    }


def _make_router() -> Callable[[list[LLMMessage]], object]:
    def _router(messages: list[LLMMessage]):
        head = messages[0].content.lower() if messages else ""
        if "scenario parser" in head:
            return _parser_payload()
        if "economic-ontology extractor" in head:
            return {"entities": [], "relations": []}
        if "god's eye console" in head:
            return {
                "intervention_step": 2,
                "param_overrides": {"severity": 0.20},
                "new_events": [],
                "description": "Cut rates",
            }
        return LLMResponse(content="placeholder prose", model="mock-1")

    return _router


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        run_count=3,
        ensemble_seed=5,
        max_workers=1,
        db_path=tmp_path / "phase.sqlite",
        use_analogs=True,
        rng_seed=11,
    )


@pytest.mark.asyncio
async def test_llm_is_quiet_during_simulation_phase(cfg: PipelineConfig) -> None:
    """Req 15.3, 15.4: once setup hands off, simulation must not call the LLM.

    We instrument both the MonteCarloController and the PropagationEngine to
    snapshot the LLM call-log size at entry and exit. The simulation phase
    must produce zero new LLM calls between any matched (enter, exit) pair.
    """
    llm = MockLLMClient(router=_make_router())
    pipeline = ClydePipeline(llm_client=llm, config=cfg)
    snapshots: list[tuple[str, int]] = []

    original_run_ensemble = MonteCarloController.run_ensemble

    def _instrumented_ensemble(self, *args, **kwargs):
        snapshots.append(("ensemble_enter", len(llm.call_log)))
        try:
            return original_run_ensemble(self, *args, **kwargs)
        finally:
            snapshots.append(("ensemble_exit", len(llm.call_log)))

    original_engine_run = PropagationEngine.run

    def _instrumented_engine(self, *args, **kwargs):
        snapshots.append(("engine_enter", len(llm.call_log)))
        try:
            return original_engine_run(self, *args, **kwargs)
        finally:
            snapshots.append(("engine_exit", len(llm.call_log)))

    assert len(llm.call_log) == 0, "MockLLMClient should be empty before pipeline.run"

    MonteCarloController.run_ensemble = _instrumented_ensemble  # type: ignore[method-assign]
    PropagationEngine.run = _instrumented_engine  # type: ignore[method-assign]
    try:
        result = await pipeline.run("A 50bp Fed rate hike in the US.")
    finally:
        MonteCarloController.run_ensemble = original_run_ensemble  # type: ignore[method-assign]
        PropagationEngine.run = original_engine_run  # type: ignore[method-assign]

    assert len(llm.call_log) > 0, "Pipeline must invoke the LLM at least once for setup/report"
    assert snapshots, "Simulation phase did not run"

    # The Monte Carlo controller wraps multiple PropagationEngine runs, so
    # snapshots interleave (ensemble_enter, engine_enter, engine_exit, ...,
    # ensemble_exit). Use a stack to match each *_enter with its closing
    # *_exit and assert zero LLM calls inside every matched span.
    stack: list[tuple[str, int]] = []
    for label, count in snapshots:
        prefix, kind = label.rsplit("_", 1)
        if kind == "enter":
            stack.append((prefix, count))
            continue
        assert stack, f"unmatched exit: {label}"
        open_prefix, entry_count = stack.pop()
        assert open_prefix == prefix, f"mismatched span: enter={open_prefix} exit={prefix}"
        assert count == entry_count, (
            f"LLM called inside {prefix}: Δ={count - entry_count} (expected 0)"
        )
    assert not stack, f"unmatched enters: {stack}"

    # Sanity: result is still well-formed.
    assert len(result.ensemble.trajectories) == cfg.run_count
    assert result.report.sections, "Report must have produced sections"


@pytest.mark.asyncio
async def test_simulation_components_carry_no_llm_reference(cfg: PipelineConfig) -> None:
    """Req 15.1, 15.2: simulation-phase components never receive the LLM client.

    Walks every attribute reachable from the pipeline's MonteCarloController
    and PropagationEngine instances and asserts that ``MockLLMClient`` is
    nowhere in their object graph.
    """
    llm = MockLLMClient(router=_make_router())
    pipeline = ClydePipeline(llm_client=llm, config=cfg)
    await pipeline.run("Brief test scenario.")

    seen: set[int] = set()
    targets: list[object] = []

    def _walk(obj: object, depth: int = 0) -> None:
        if depth > 4:
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        targets.append(obj)
        for attr in vars(obj).values() if hasattr(obj, "__dict__") else ():
            if isinstance(attr, (str, int, float, bool, bytes)) or attr is None:
                continue
            try:
                _walk(attr, depth + 1)
            except Exception:
                continue

    # Resolve the actual component instances from the pipeline.
    controller = pipeline._controller  # type: ignore[attr-defined]
    engine = controller._engine  # type: ignore[attr-defined]
    _walk(controller)
    _walk(engine)
    assert not any(isinstance(t, MockLLMClient) for t in targets), (
        "Simulation-phase component holds a reference to the LLM client"
    )


@pytest.mark.asyncio
async def test_setup_handoff_is_fully_resolved(cfg: PipelineConfig) -> None:
    """Req 15.4: setup phase produces a fully-resolved ShockConfig + actor list.

    After ``pipeline.run`` returns, the SimulationWorld used by the controller
    must have every actor's params populated. We verify that the static
    invariant from Property 4 still holds at runtime end-to-end.
    """
    from clyde.models.actors import REQUIRED_PARAM_FIELDS

    llm = MockLLMClient(router=_make_router())
    pipeline = ClydePipeline(llm_client=llm, config=cfg)
    result = await pipeline.run("Brief test scenario for handoff resolution.")

    assert result.world.actors, "Pipeline produced no actors"
    for actor in result.world.actors:
        required = REQUIRED_PARAM_FIELDS[actor.actor_type]
        for name in required:
            value = getattr(actor.params, name, None)
            assert value is not None, (
                f"Actor {actor.id} ({actor.actor_type.value}) missing param {name!r}"
            )

    # The handoff config must satisfy ShockConfig validation (post_init has
    # already run by this point — re-construction proves no later mutation
    # broke the invariant).
    from clyde.models.config import ShockConfig

    rebuilt = ShockConfig.from_dict(result.shock_config.to_dict())
    assert rebuilt.scope == result.shock_config.scope
    assert 0.0 <= rebuilt.severity <= 1.0


@pytest.mark.asyncio
async def test_branch_fork_does_not_leak_llm_into_simulation(
    cfg: PipelineConfig,
) -> None:
    """Req 15.3 holds for branch re-simulation too.

    Forking a branch invokes the LLM (God's Eye) for parsing the injection,
    then runs another ensemble. The ensemble call must not invoke the LLM.
    """
    llm = MockLLMClient(router=_make_router())
    pipeline = ClydePipeline(llm_client=llm, config=cfg)
    base = await pipeline.run("A 50bp Fed rate hike in the US.")

    snapshots: list[tuple[str, int]] = []
    original_run_ensemble = MonteCarloController.run_ensemble

    def _instrumented(self, *args, **kwargs):
        snapshots.append(("enter", len(llm.call_log)))
        try:
            return original_run_ensemble(self, *args, **kwargs)
        finally:
            snapshots.append(("exit", len(llm.call_log)))

    MonteCarloController.run_ensemble = _instrumented  # type: ignore[method-assign]
    try:
        await pipeline.fork_branch(base, "Cut rates by 75bp at step 2.")
    finally:
        MonteCarloController.run_ensemble = original_run_ensemble  # type: ignore[method-assign]

    assert snapshots, "Branch fork did not invoke run_ensemble"
    pairs = list(zip(snapshots[::2], snapshots[1::2]))
    for (_, enter_count), (_, exit_count) in pairs:
        assert exit_count == enter_count, (
            f"Branch ensemble triggered an LLM call: Δ={exit_count - enter_count}"
        )
