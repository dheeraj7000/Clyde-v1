"""Pydantic request/response schemas for the Clyde HTTP API.

These models are deliberately permissive on the response side (``Any``-typed
result payloads) so the serialiser in :mod:`clyde.web.jobs` can emit the
full PipelineResult without fighting the schema. Request bodies, on the
other hand, get strict validation: too-short or out-of-range inputs surface
as 422s before any pipeline work is kicked off.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Body for ``POST /api/runs``."""

    description: str = Field(..., min_length=5, max_length=5000)
    run_count: int = Field(default=50, ge=1, le=500)
    horizon_steps: int | None = Field(default=None, ge=1, le=500)
    provider: Literal["auto", "openrouter", "cerebras", "mock"] = "auto"
    model: str | None = None
    rng_seed: int = 0
    ensemble_seed: int = 0
    use_analogs: bool = True


class BranchRequest(BaseModel):
    """Body for ``POST /api/runs/{job_id}/branches``."""

    injection_text: str = Field(..., min_length=5, max_length=2000)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class JobAccepted(BaseModel):
    """Returned when a run or branch is accepted."""

    job_id: str | None = None
    branch_id: str | None = None
    status: str


class ProgressInfo(BaseModel):
    stage: str
    percent: float


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: ProgressInfo
    result: dict[str, Any] | None = None
    error: str | None = None
    details: str | None = None


class BranchStatus(BaseModel):
    branch_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: ProgressInfo
    result: dict[str, Any] | None = None
    error: str | None = None
    details: str | None = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    providers_available: dict[str, bool]
    model: str | None = None


class SampleScenario(BaseModel):
    name: str
    description: str


__all__ = [
    "RunRequest",
    "BranchRequest",
    "JobAccepted",
    "ProgressInfo",
    "JobStatus",
    "BranchStatus",
    "HealthResponse",
    "SampleScenario",
]
