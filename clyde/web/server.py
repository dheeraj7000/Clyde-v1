"""FastAPI surface for the Clyde economic simulator.

The server is intentionally thin: it accepts a description, kicks off a
background :class:`asyncio.Task` running the pipeline, and exposes polling
+ branch endpoints. All persistent state lives in a single
:class:`JobStore` attached to ``app.state.jobs``.

Tests can override the LLM / pipeline factory via FastAPI's
``dependency_overrides`` mechanism — see ``get_pipeline_factory`` below.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from clyde.llm.factory import available_providers, resolve_provider
from clyde.pipeline import PipelineConfig
from clyde.web import agent_routes as _ar
from clyde.web.agent_routes import InjectRequest as _InjectReq
from clyde.web.jobs import (
    BranchJob,
    Job,
    JobStore,
    PipelineFactory,
    default_pipeline_factory_for,
    run_branch_job,
    run_pipeline_job,
)
from clyde.web.schemas import (
    BranchRequest,
    HealthResponse,
    JobAccepted,
    JobStatus,
    RunRequest,
    SampleScenario,
)


# ---------------------------------------------------------------------------
# Sample scenarios (hand-crafted demo seeds)
# ---------------------------------------------------------------------------


_SAMPLE_SCENARIOS: list[dict[str, str]] = [
    {
        "name": "Fed Rate Hike",
        "description": (
            "The Federal Reserve raises the federal funds rate by 75 basis "
            "points to combat persistent inflation, tightening credit "
            "conditions across US banks, firms, and households."
        ),
    },
    {
        "name": "Banking Crisis Contagion",
        "description": (
            "A regional US bank fails after a run on uninsured deposits, "
            "triggering interbank lending stress and tighter credit standards "
            "for small businesses across the country."
        ),
    },
    {
        "name": "Oil Supply Shock",
        "description": (
            "An OPEC+ production cut pushes Brent crude prices up 30% over "
            "two months, squeezing margins for energy-intensive manufacturers "
            "in Europe and lifting headline inflation."
        ),
    },
    {
        "name": "Housing Market Correction",
        "description": (
            "US housing prices decline 15% over six months as mortgage rates "
            "climb and inventory expands, eroding household wealth and "
            "consumer confidence in discretionary sectors."
        ),
    },
    {
        "name": "Cross-Border Trade Tariff",
        "description": (
            "The US imposes a 25% tariff on imported semiconductors from Asia, "
            "disrupting electronics supply chains and forcing firms to raise "
            "prices or absorb margin compression."
        ),
    },
]


# ---------------------------------------------------------------------------
# Dependency hooks (overridable in tests)
# ---------------------------------------------------------------------------


def get_job_store(app: FastAPI) -> JobStore:
    return app.state.jobs  # type: ignore[no-any-return]


def get_pipeline_factory(provider: str = "auto", model: str | None = None) -> PipelineFactory:
    """Default factory hook — tests override this via ``app.dependency_overrides``."""
    return default_pipeline_factory_for(provider=provider, model=model)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _static_dir() -> Path:
    return Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Construct the FastAPI app + register routes."""
    app = FastAPI(title="Clyde", version="0.1.0")
    app.state.jobs = JobStore()

    # --- CORS -----------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Static frontend ------------------------------------------------
    static_dir = _static_dir()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- Routes ---------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        index = static_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>Clyde</h1><p>Frontend not built yet. POST /api/runs to use the API.</p>",
            status_code=200,
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        avail = available_providers()
        try:
            resolved = resolve_provider()
        except Exception:
            resolved = "mock"
        return HealthResponse(
            status="ok",
            provider=resolved,
            providers_available=avail,
            model=None,
        )

    @app.get("/api/scenarios/sample", response_model=list[SampleScenario])
    async def sample_scenarios() -> list[SampleScenario]:
        return [SampleScenario(**s) for s in _SAMPLE_SCENARIOS]

    @app.post("/api/runs", response_model=JobAccepted)
    async def create_run(req: RunRequest) -> JobAccepted:
        store: JobStore = app.state.jobs
        job = await store.create_job_async()

        cfg = PipelineConfig(
            run_count=req.run_count,
            ensemble_seed=req.ensemble_seed,
            rng_seed=req.rng_seed,
            use_analogs=req.use_analogs,
            max_workers=1,
        )

        # Resolve the pipeline factory: prefer a test override on the app
        # (``app.state.pipeline_factory_override``), fall back to the real
        # provider-driven factory.
        override = getattr(app.state, "pipeline_factory_override", None)
        if override is not None:
            factory: PipelineFactory = override
        else:
            factory = default_pipeline_factory_for(
                provider=req.provider, model=req.model
            )

        async def _go() -> None:
            await run_pipeline_job(
                job,
                description=req.description,
                config=cfg,
                pipeline_factory=factory,
                horizon_steps=req.horizon_steps,
            )

        job.task = asyncio.create_task(_go())
        return JobAccepted(job_id=job.job_id, status="pending")

    @app.get("/api/runs/{job_id}", response_model=JobStatus)
    async def get_run(job_id: str) -> JobStatus:
        store: JobStore = app.state.jobs
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return _job_to_status(job)

    @app.post("/api/runs/{job_id}/branches", response_model=JobAccepted)
    async def create_branch(job_id: str, req: BranchRequest) -> JobAccepted:
        store: JobStore = app.state.jobs
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if job.status != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Cannot fork a branch off a job in status {job.status!r}",
            )

        branch = BranchJob(branch_id=str(uuid.uuid4()))
        job.branches[branch.branch_id] = branch

        async def _go() -> None:
            await run_branch_job(job, branch, injection_text=req.injection_text)

        branch.task = asyncio.create_task(_go())
        return JobAccepted(branch_id=branch.branch_id, status="pending")

    @app.get(
        "/api/runs/{job_id}/branches/{branch_id}",
        response_model=JobStatus,
    )
    async def get_branch(job_id: str, branch_id: str) -> JobStatus:
        store: JobStore = app.state.jobs
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        branch = job.branches.get(branch_id)
        if branch is None:
            raise HTTPException(
                status_code=404, detail=f"Branch {branch_id} not found"
            )
        return JobStatus(
            job_id=branch.branch_id,
            status=branch.status,  # type: ignore[arg-type]
            progress=branch.progress.to_dict(),  # type: ignore[arg-type]
            result=branch.result,
            error=branch.error,
            details=branch.details,
        )

    # --- Agent Simulation routes (MiroFish-style) --------------------------

    @app.post("/api/runs/{job_id}/agent-sim/start")
    async def start_agent_sim(job_id: str) -> dict:
        return await _ar.start_agent_sim(job_id, store=app.state.jobs)

    @app.post("/api/runs/{job_id}/agent-sim/{sim_id}/round")
    async def run_agent_round(job_id: str, sim_id: str) -> dict:
        return await _ar.run_agent_round(job_id, sim_id, store=app.state.jobs)

    @app.post("/api/runs/{job_id}/agent-sim/{sim_id}/inject")
    async def inject_agent_event(job_id: str, sim_id: str, req: _InjectReq) -> dict:
        return await _ar.inject_agent_event(job_id, sim_id, req.description, store=app.state.jobs)

    @app.get("/api/runs/{job_id}/agent-sim/{sim_id}/state")
    async def get_agent_sim_state(job_id: str, sim_id: str) -> dict:
        return await _ar.get_agent_sim_state(job_id, sim_id, store=app.state.jobs)

    # ──────────────────────────────────────────────────────────────────
    # /api/tts — ElevenLabs text-to-speech proxy.
    # The browser POSTs {text, voice_id?, model_id?} and we forward to
    # ElevenLabs with the API key from CLYDE_ELEVENLABS_KEY env. Audio
    # streams back as audio/mpeg so the SPA can <audio src=blob-url>.
    # Keeping the key server-side avoids leaking it to clients.
    # ──────────────────────────────────────────────────────────────────
    @app.post("/api/tts")
    async def tts(payload: dict = Body(...)) -> Response:
        api_key = os.getenv("CLYDE_ELEVENLABS_KEY") or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="ElevenLabs key not configured (set CLYDE_ELEVENLABS_KEY)")
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        # Cap at ~3500 chars so a malformed report can't burn the quota.
        if len(text) > 3500:
            text = text[:3500]
        voice_id = payload.get("voice_id") or "JBFqnCBsd6RMkjVDRZzb"  # "George" — clear, neutral
        model_id = payload.get("model_id") or "eleven_turbo_v2_5"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        body = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.7, "style": 0.2},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    json=body,
                    headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ElevenLabs request failed: {exc}") from exc
        if resp.status_code != 200:
            # Surface the ElevenLabs error for debuggability instead of raw 502.
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:400])
        return Response(content=resp.content, media_type="audio/mpeg")

    return app


def _job_to_status(job: Job) -> JobStatus:
    return JobStatus(
        job_id=job.job_id,
        status=job.status,  # type: ignore[arg-type]
        progress=job.progress.to_dict(),  # type: ignore[arg-type]
        result=job.result,
        error=job.error,
        details=job.details,
    )


# Module-level app instance for ``uvicorn clyde.web.server:app``.
app = create_app()


__all__ = ["app", "create_app", "get_job_store", "get_pipeline_factory"]
