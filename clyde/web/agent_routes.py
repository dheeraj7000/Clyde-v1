"""Agent simulation logic (MiroFish-style). Called from server routes."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from clyde.web.jobs import Job, JobStore


class InjectRequest(BaseModel):
    description: str = Field(..., min_length=3)


def _get_job(store: JobStore, job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


def _get_sim(job: Job, sim_id: str) -> dict:
    sims = getattr(job, "agent_sim", {})
    data = sims.get(sim_id)
    if data is None:
        raise HTTPException(404, f"Sim {sim_id} not found")
    return data


async def start_agent_sim(job_id: str, store: JobStore) -> dict:
    job = _get_job(store, job_id)
    if job.status != "completed" or job.pipeline_result is None:
        raise HTTPException(409, "Run must be completed first")

    from clyde.setup.agent_sim import AgentSimEngine, AgentSimConfig
    from clyde.llm import make_llm_client

    result = job.pipeline_result
    llm = make_llm_client("auto")
    engine = AgentSimEngine(llm)
    personas = [p.to_dict() if hasattr(p, "to_dict") else p for p in (result.personas or [])]
    sim_state = engine.init_state(
        world=result.world,
        shock_config=result.shock_config,
        personas=personas,
        config=AgentSimConfig(total_rounds=result.shock_config.time_horizon.steps or 12),
    )

    if not hasattr(job, "agent_sim"):
        job.agent_sim = {}  # type: ignore[attr-defined]
    sim_id = str(uuid.uuid4())[:8]
    job.agent_sim[sim_id] = {"engine": engine, "state": sim_state}  # type: ignore[attr-defined]

    return {
        "sim_id": sim_id,
        "total_rounds": sim_state.config.total_rounds,
        "actors": len(sim_state.world.actors),
        "status": sim_state.status,
    }


async def run_agent_round(job_id: str, sim_id: str, store: JobStore) -> dict:
    job = _get_job(store, job_id)
    sim_data = _get_sim(job, sim_id)
    engine = sim_data["engine"]
    state = sim_data["state"]
    if state.status == "completed":
        raise HTTPException(409, "Simulation already completed")
    state.status = "running"
    result = await engine.run_round(state)
    return result.to_dict()


async def inject_agent_event(job_id: str, sim_id: str, description: str, store: JobStore) -> dict:
    job = _get_job(store, job_id)
    sim_data = _get_sim(job, sim_id)
    engine = sim_data["engine"]
    state = sim_data["state"]
    engine.inject_event(state, description)
    return {"injected": True, "current_round": state.current_round}


async def get_agent_sim_state(job_id: str, sim_id: str, store: JobStore) -> dict:
    job = _get_job(store, job_id)
    sim_data = _get_sim(job, sim_id)
    state = sim_data["state"]
    return {
        "sim_id": sim_id,
        "status": state.status,
        "current_round": state.current_round,
        "total_rounds": state.config.total_rounds,
        "rounds_completed": len(state.rounds),
        "actor_count": len(state.world.actors),
        "memories": {aid: m.to_dict() for aid, m in list(state.memories.items())[:10]},
        "rounds": [r.to_dict() for r in state.rounds[-3:]],
    }
