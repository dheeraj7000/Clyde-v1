"""HTTP-layer tests for the Clyde FastAPI surface.

These tests use FastAPI's :class:`TestClient` (sync). The pipeline factory
is overridden to inject a :class:`MockLLMClient` that routes per-subsystem
on the message content — same pattern as ``tests/test_pipeline_e2e.py``.

We use ``run_count=4`` and short-horizon mock payloads so each test
finishes in well under a second.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from clyde.llm import LLMMessage, LLMResponse, MockLLMClient
from clyde.pipeline import ClydePipeline, PipelineConfig
from clyde.web.server import create_app


# ---------------------------------------------------------------------------
# Mock LLM router (mirrors tests/test_pipeline_e2e.py)
# ---------------------------------------------------------------------------


def _parser_payload() -> dict:
    return {
        "triggering_event": "Federal Reserve rate hike announcement",
        "geographies": ["US"],
        "markets": ["finance", "consumer"],
        "shock_params": {
            "shock_type": "rate_hike",
            "severity": 0.40,
            "scope": "macro",
            "duration_steps": 4,
            "initial_contact_actors": ["central_bank_0000"],
        },
        "time_horizon": {"steps": 4, "step_unit": "quarter"},
        "ambiguities": [],
        "actor_hints": [
            {"actor_type": "household", "count_estimate": 30, "description": "US households"},
            {"actor_type": "firm", "count_estimate": 6, "description": "US firms"},
            {"actor_type": "bank", "count_estimate": 2, "description": "US commercial banks"},
            {"actor_type": "central_bank", "count_estimate": 1, "description": "Federal Reserve"},
        ],
    }


def _kg_payload() -> dict:
    return {
        "entities": [
            {
                "id": "policy:fed_rate_hike",
                "type": "policy",
                "name": "Federal Reserve rate hike",
                "attributes": {"basis_points": 50},
            }
        ],
        "relations": [],
    }


def _gods_eye_payload() -> dict:
    return {
        "intervention_step": 2,
        "param_overrides": {"severity": 0.20},
        "new_events": [],
        "description": "Cut rates by 75bp at step 2.",
    }


def _router(messages: list[LLMMessage]):
    if messages and "scenario parser" in messages[0].content.lower():
        return _parser_payload()
    if messages and "economic-ontology extractor" in messages[0].content.lower():
        return _kg_payload()
    if messages and "god's eye console" in messages[0].content.lower():
        return _gods_eye_payload()
    return LLMResponse(content="Generic placeholder narrative.", model="mock-1")


def _mock_pipeline_factory(cfg: PipelineConfig) -> ClydePipeline:
    return ClydePipeline(MockLLMClient(router=_router), config=cfg)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    # Inject the mock pipeline factory so no real LLM is reached.
    app.state.pipeline_factory_override = _mock_pipeline_factory
    return TestClient(app)


def _wait_for_status(
    client: TestClient,
    url: str,
    target: str = "completed",
    timeout: float = 30.0,
) -> dict:
    """Poll ``url`` until the response ``status`` reaches ``target``."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(url)
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in {target, "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for {url} to reach status={target!r}; last={last}"
    )


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["providers_available"], dict)
    assert "mock" in body["providers_available"]
    assert body["provider"] in {"openrouter", "cerebras", "mock"}


# ---------------------------------------------------------------------------
# 2. Run + poll happy path
# ---------------------------------------------------------------------------


def test_create_run_and_poll(client: TestClient) -> None:
    payload = {
        "description": "A 50bp Federal Reserve rate hike in the US.",
        "run_count": 4,
        "provider": "mock",
        "rng_seed": 1,
        "ensemble_seed": 7,
        "use_analogs": True,
    }
    resp = client.post("/api/runs", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] == "pending"
    assert isinstance(job_id, str) and len(job_id) > 0

    final = _wait_for_status(client, f"/api/runs/{job_id}")
    assert final["status"] == "completed", final
    result = final["result"]
    assert result is not None

    # Top-level keys.
    for key in (
        "scenario_id",
        "shock_config",
        "parse_result",
        "paths",
        "divergence",
        "watchlist",
        "causal_chains",
        "report",
        "branches",
    ):
        assert key in result, f"missing top-level key {key!r}"

    # Central path has horizon_steps entries (parser routed payload sets steps=4).
    central = result["paths"]["central"]
    assert len(central) == 4
    # Each step is a StepMetrics dict.
    assert "gdp_index" in central[0]
    assert "step" in central[0]


# ---------------------------------------------------------------------------
# 3. Validation: too-short description -> 422
# ---------------------------------------------------------------------------


def test_create_run_rejects_short_description(client: TestClient) -> None:
    resp = client.post("/api/runs", json={"description": "abc", "run_count": 4})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. Branch fork
# ---------------------------------------------------------------------------


def test_branch_fork(client: TestClient) -> None:
    resp = client.post(
        "/api/runs",
        json={
            "description": "A US rate hike for branch testing.",
            "run_count": 4,
            "provider": "mock",
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    final = _wait_for_status(client, f"/api/runs/{job_id}")
    assert final["status"] == "completed"

    bresp = client.post(
        f"/api/runs/{job_id}/branches",
        json={"injection_text": "Cut rates by 75bp at step 2."},
    )
    assert bresp.status_code == 200, bresp.text
    branch_id = bresp.json()["branch_id"]
    assert isinstance(branch_id, str)

    bfinal = _wait_for_status(
        client, f"/api/runs/{job_id}/branches/{branch_id}"
    )
    assert bfinal["status"] == "completed", bfinal
    bresult = bfinal["result"]
    assert bresult is not None
    assert bresult["branch_id"] == branch_id
    assert "merged_config" in bresult
    assert "paths" in bresult
    assert "divergence" in bresult
    # Branch summary should NOT include the heavyweight report.
    assert "report" not in bresult


# ---------------------------------------------------------------------------
# 5. Sample scenarios
# ---------------------------------------------------------------------------


def test_sample_scenarios(client: TestClient) -> None:
    resp = client.get("/api/scenarios/sample")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) >= 3
    for scen in body:
        assert "name" in scen and scen["name"]
        assert "description" in scen and scen["description"]


# ---------------------------------------------------------------------------
# 6. CORS
# ---------------------------------------------------------------------------


def test_cors_preflight(client: TestClient) -> None:
    resp = client.options(
        "/api/runs",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in resp.headers.keys()}


# ---------------------------------------------------------------------------
# 7. Job not found
# ---------------------------------------------------------------------------


def test_job_not_found(client: TestClient) -> None:
    resp = client.get("/api/runs/nonexistent-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. Branch on missing job
# ---------------------------------------------------------------------------


def test_branch_on_missing_job(client: TestClient) -> None:
    resp = client.post(
        "/api/runs/missing/branches",
        json={"injection_text": "Some intervention text."},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. Root serves friendly HTML stub
# ---------------------------------------------------------------------------


def test_root_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Clyde" in resp.text
