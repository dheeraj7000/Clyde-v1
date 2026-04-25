"""Default router payloads for Demo Mode (no API key configured).

Mirrors the per-subsystem payload pattern used in
``tests/test_web_api.py`` so that a bare :class:`MockLLMClient` can run
the full pipeline end-to-end without operator-supplied credentials.
"""

from __future__ import annotations

from clyde.llm.client import LLMMessage, LLMResponse


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


def demo_router(messages: list[LLMMessage]):
    """Route a system prompt to a sensible canned payload."""
    if messages and "scenario parser" in messages[0].content.lower():
        return _parser_payload()
    if messages and "economic-ontology extractor" in messages[0].content.lower():
        return _kg_payload()
    if messages and "god's eye console" in messages[0].content.lower():
        return _gods_eye_payload()
    return LLMResponse(content="Generic placeholder narrative.", model="mock-1")


__all__ = ["demo_router"]
