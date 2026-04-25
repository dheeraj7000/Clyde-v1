"""Tests for :mod:`clyde.setup.gods_eye`.

Covers both the pure :meth:`GodsEyeConsole.apply_delta` merge logic and
the asynchronous :meth:`GodsEyeConsole.parse_injection` LLM-driven parse,
plus an end-to-end "parse then apply" round-trip.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from clyde.llm import LLMMessage, MockLLMClient
from clyde.models.config import ShockConfig, ShockDelta
from clyde.models.scenario import Scenario
from clyde.models.time import TimeHorizon
from clyde.setup.gods_eye import AMBIGUITY_KEY, GodsEyeConsole


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_base_config(
    *,
    behavioral_overrides: dict[str, Any] | None = None,
    severity: float = 0.5,
    duration: int = 10,
    horizon_steps: int = 12,
) -> ShockConfig:
    return ShockConfig(
        shock_type="rate_hike",
        severity=severity,
        scope="macro",
        duration_steps=duration,
        geography=["US"],
        sectors=["banking"],
        initial_contact_actors=["bank_001"],
        agent_counts={"bank": 5, "household": 100},
        behavioral_overrides=dict(behavioral_overrides) if behavioral_overrides else {},
        time_horizon=TimeHorizon(steps=horizon_steps, step_unit="day"),
        ensemble_seed=42,
    )


def _make_base_scenario(config: ShockConfig | None = None) -> Scenario:
    return Scenario(
        scenario_id="scn-test-001",
        description="baseline rate-hike scenario",
        config=config if config is not None else _make_base_config(),
    )


# ---------------------------------------------------------------------------
# apply_delta -- pure unit tests
# ---------------------------------------------------------------------------


def test_apply_delta_promotes_severity_top_level():
    base = _make_base_config(severity=0.4)
    delta = ShockDelta(
        intervention_step=3,
        param_overrides={"severity": 0.8},
    )

    result = GodsEyeConsole.apply_delta(base, delta)

    assert result.severity == pytest.approx(0.8)
    # Other top-level fields untouched.
    assert result.shock_type == base.shock_type
    assert result.scope == base.scope
    assert result.duration_steps == base.duration_steps
    assert result.time_horizon == base.time_horizon
    # Promoted key MUST NOT also show up in behavioral_overrides.
    assert "severity" not in result.behavioral_overrides


def test_apply_delta_promotes_shock_type_keeps_other_overrides_in_behavioral():
    base = _make_base_config()
    delta = ShockDelta(
        intervention_step=2,
        param_overrides={
            "shock_type": "currency_crisis",
            "fiscal_stimulus_pct": 0.05,
        },
    )

    result = GodsEyeConsole.apply_delta(base, delta)

    # shock_type promoted to top-level.
    assert result.shock_type == "currency_crisis"
    assert "shock_type" not in result.behavioral_overrides
    # Non-special override stays in behavioral_overrides.
    assert result.behavioral_overrides["fiscal_stimulus_pct"] == pytest.approx(0.05)


def test_apply_delta_appends_new_events_to_behavioral_overrides():
    # No pre-existing new_events.
    base = _make_base_config()
    delta = ShockDelta(
        intervention_step=1,
        param_overrides={},
        new_events=["A", "B"],
    )
    result = GodsEyeConsole.apply_delta(base, delta)
    assert result.behavioral_overrides["new_events"] == ["A", "B"]

    # Pre-existing new_events should be concatenated, not replaced.
    base2 = _make_base_config(behavioral_overrides={"new_events": ["pre1", "pre2"]})
    delta2 = ShockDelta(
        intervention_step=1,
        param_overrides={},
        new_events=["A", "B"],
    )
    result2 = GodsEyeConsole.apply_delta(base2, delta2)
    assert result2.behavioral_overrides["new_events"] == ["pre1", "pre2", "A", "B"]


def test_apply_delta_does_not_mutate_inputs():
    base = _make_base_config(behavioral_overrides={"x": 1, "new_events": ["e0"]})
    delta = ShockDelta(
        intervention_step=4,
        param_overrides={"severity": 0.7, "fiscal_stimulus_pct": 0.02},
        new_events=["e1"],
        description="raise rates",
    )

    base_snapshot = copy.deepcopy(base.to_dict())
    delta_snapshot = copy.deepcopy(delta.to_dict())

    _ = GodsEyeConsole.apply_delta(base, delta)

    assert base.to_dict() == base_snapshot
    assert delta.to_dict() == delta_snapshot


def test_apply_delta_invalid_severity_raises_value_error():
    base = _make_base_config()
    delta = ShockDelta(intervention_step=0, param_overrides={"severity": 2.0})
    with pytest.raises(ValueError):
        GodsEyeConsole.apply_delta(base, delta)


def test_apply_delta_strips_ambiguity_key_from_behavioral_overrides():
    base = _make_base_config()
    delta = ShockDelta(
        intervention_step=0,
        param_overrides={
            "fiscal_stimulus_pct": 0.01,
            AMBIGUITY_KEY: [{"field": "x", "description": "y", "options": None}],
        },
    )
    result = GodsEyeConsole.apply_delta(base, delta)
    assert AMBIGUITY_KEY not in result.behavioral_overrides
    assert result.behavioral_overrides["fiscal_stimulus_pct"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# parse_injection -- LLM-mediated tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_injection_well_formed_response():
    scenario = _make_base_scenario()
    payload = {
        "intervention_step": 5,
        "param_overrides": {"severity": 0.7, "shock_type": "rate_hike"},
        "new_events": ["central_bank_emergency_meeting"],
        "description": "Central bank hikes rates by 50bp at step 5",
        # No "ambiguities" key -> fully resolved.
    }
    mock = MockLLMClient(responses=[payload])
    console = GodsEyeConsole(mock, model="gpt-test")

    delta = await console.parse_injection(
        "Central bank hikes rates by 50bp at step 5", scenario
    )

    assert isinstance(delta, ShockDelta)
    assert delta.intervention_step == 5
    assert delta.param_overrides["severity"] == pytest.approx(0.7)
    assert delta.param_overrides["shock_type"] == "rate_hike"
    assert delta.new_events == ["central_bank_emergency_meeting"]
    assert delta.description == "Central bank hikes rates by 50bp at step 5"
    # No ambiguities expected.
    assert AMBIGUITY_KEY not in delta.param_overrides

    # Inspect call_log: messages must mention the injection text and the
    # scenario id.
    log = mock.call_log
    assert len(log) == 1
    method_name, messages, kwargs = log[0]
    assert method_name == "complete_json"
    assert kwargs["model"] == "gpt-test"
    joined = "\n".join(m.content for m in messages)
    assert "Central bank hikes rates by 50bp at step 5" in joined
    assert "scn-test-001" in joined
    # Must include both a system and a user message.
    roles = [m.role for m in messages]
    assert "system" in roles and "user" in roles


@pytest.mark.asyncio
async def test_parse_injection_clips_out_of_range_intervention_step():
    scenario = _make_base_scenario(_make_base_config(horizon_steps=10))
    # Horizon = 10 -> valid range [0, 9]. Ask for step 99.
    payload = {
        "intervention_step": 99,
        "param_overrides": {"severity": 0.6},
        "new_events": [],
        "description": "very late intervention",
    }
    mock = MockLLMClient(responses=[payload])
    console = GodsEyeConsole(mock)

    delta = await console.parse_injection("very late intervention", scenario)

    assert delta.intervention_step == 9  # clipped to max valid step
    assert AMBIGUITY_KEY in delta.param_overrides
    ambs = delta.param_overrides[AMBIGUITY_KEY]
    assert isinstance(ambs, list) and len(ambs) >= 1
    assert any(a.get("field") == "intervention_step" for a in ambs)
    # The original out-of-range value should be referenced.
    assert any("99" in (a.get("description") or "") for a in ambs)


@pytest.mark.asyncio
async def test_parse_injection_missing_intervention_step_defaults_to_zero():
    scenario = _make_base_scenario()
    payload = {
        # intervention_step deliberately omitted
        "param_overrides": {"severity": 0.6},
        "new_events": [],
        "description": "no step given",
    }
    mock = MockLLMClient(responses=[payload])
    console = GodsEyeConsole(mock)

    delta = await console.parse_injection("no step given", scenario)

    assert delta.intervention_step == 0
    assert AMBIGUITY_KEY in delta.param_overrides
    fields = [a.get("field") for a in delta.param_overrides[AMBIGUITY_KEY]]
    assert "intervention_step" in fields


@pytest.mark.asyncio
async def test_parse_injection_retries_on_malformed_json_then_fails():
    scenario = _make_base_scenario()
    # Three malformed JSON strings -- all attempts will fail.
    mock = MockLLMClient(
        responses=["not json", "still not json", "{unclosed"]
    )
    console = GodsEyeConsole(
        mock,
        model="gpt-test",
        max_retries=3,
        backoff_base_s=0.0,  # avoid sleep delays in tests
    )

    with pytest.raises(RuntimeError) as excinfo:
        await console.parse_injection("anything", scenario)

    assert "god's eye parsing failed" in str(excinfo.value).lower()
    assert "3 retries" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    # Verify we actually tried max_retries times.
    assert len(mock.call_log) == 3


@pytest.mark.asyncio
async def test_parse_injection_retries_then_succeeds():
    """A failure followed by a good response should produce a clean delta."""
    scenario = _make_base_scenario()
    good_payload = {
        "intervention_step": 2,
        "param_overrides": {"severity": 0.5},
        "new_events": [],
        "description": "second-try success",
    }
    mock = MockLLMClient(responses=["garbage", good_payload])
    console = GodsEyeConsole(
        mock, max_retries=3, backoff_base_s=0.0
    )

    delta = await console.parse_injection("second-try success", scenario)

    assert delta.intervention_step == 2
    assert delta.description == "second-try success"
    assert AMBIGUITY_KEY not in delta.param_overrides
    assert len(mock.call_log) == 2


@pytest.mark.asyncio
async def test_parse_injection_with_ambiguities_in_response_smuggles_them():
    scenario = _make_base_scenario()
    payload = {
        "intervention_step": 4,
        "param_overrides": {"severity": 0.6},
        "new_events": [],
        "description": "ambiguous one",
        "ambiguities": [
            {
                "field": "scope",
                "description": "macro vs sectoral unclear",
                "options": ["macro", "sectoral"],
            }
        ],
    }
    mock = MockLLMClient(responses=[payload])
    console = GodsEyeConsole(mock)

    delta = await console.parse_injection("ambiguous one", scenario)

    assert AMBIGUITY_KEY in delta.param_overrides
    ambs = delta.param_overrides[AMBIGUITY_KEY]
    assert len(ambs) == 1
    assert ambs[0]["field"] == "scope"
    assert ambs[0]["options"] == ["macro", "sectoral"]


# ---------------------------------------------------------------------------
# End-to-end: parse_injection + apply_delta
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_parse_then_apply_produces_valid_config():
    scenario = _make_base_scenario(
        _make_base_config(
            severity=0.3,
            duration=8,
            horizon_steps=20,
            behavioral_overrides={"existing_param": "keep_me"},
        )
    )
    payload = {
        "intervention_step": 5,
        "param_overrides": {
            "severity": 0.85,                # promoted to top level
            "shock_type": "credit_crunch",   # promoted to top level
            "fiscal_stimulus_pct": 0.04,     # stays in behavioral_overrides
        },
        "new_events": ["fed_emergency_meeting"],
        "description": "credit crunch + stimulus at step 5",
    }
    mock = MockLLMClient(responses=[payload])
    console = GodsEyeConsole(mock)

    delta = await console.parse_injection(
        "credit crunch + stimulus at step 5", scenario
    )
    merged = GodsEyeConsole.apply_delta(scenario.config, delta)

    # Top-level promotions.
    assert merged.shock_type == "credit_crunch"
    assert merged.severity == pytest.approx(0.85)
    # Untouched top-level fields.
    assert merged.scope == "macro"
    assert merged.duration_steps == 8
    # Behavioral merge: existing param preserved, new override added,
    # promoted keys absent, ambiguity key absent.
    assert merged.behavioral_overrides["existing_param"] == "keep_me"
    assert merged.behavioral_overrides["fiscal_stimulus_pct"] == pytest.approx(0.04)
    assert "severity" not in merged.behavioral_overrides
    assert "shock_type" not in merged.behavioral_overrides
    assert AMBIGUITY_KEY not in merged.behavioral_overrides
    # new_events appended.
    assert merged.behavioral_overrides["new_events"] == ["fed_emergency_meeting"]
    # And the merged config is valid (re-running __post_init__ via to_dict
    # round-trip is a cheap integrity check).
    round_trip = ShockConfig.from_dict(merged.to_dict())
    assert round_trip.shock_type == "credit_crunch"
    assert round_trip.severity == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Misc: constructor validation
# ---------------------------------------------------------------------------


def test_console_rejects_zero_max_retries():
    with pytest.raises(ValueError):
        GodsEyeConsole(MockLLMClient(), max_retries=0)


def test_console_rejects_negative_backoff():
    with pytest.raises(ValueError):
        GodsEyeConsole(MockLLMClient(), backoff_base_s=-0.1)


# ---------------------------------------------------------------------------
# Sanity: AMBIGUITY_KEY is the documented sentinel
# ---------------------------------------------------------------------------


def test_ambiguity_key_constant_value():
    """The convention is part of the public contract."""
    assert AMBIGUITY_KEY == "_ambiguities"


# ---------------------------------------------------------------------------
# LLM message-structure check: snapshot includes config dict, not just id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_injection_user_message_contains_config_snapshot():
    scenario = _make_base_scenario(
        _make_base_config(severity=0.42, duration=7)
    )
    payload = {
        "intervention_step": 1,
        "param_overrides": {},
        "new_events": [],
        "description": "noop",
    }
    captured: dict[str, list[LLMMessage]] = {}

    def router(messages: list[LLMMessage]):
        captured["msgs"] = list(messages)
        return payload

    mock = MockLLMClient(router=router)
    console = GodsEyeConsole(mock)
    await console.parse_injection("noop", scenario)

    msgs = captured["msgs"]
    user_msg = next(m for m in msgs if m.role == "user")
    # The base ShockConfig.to_dict() should be embedded somewhere in the
    # user payload -- check a few characteristic substrings.
    assert "shock_type" in user_msg.content
    assert "rate_hike" in user_msg.content
    assert "0.42" in user_msg.content
    assert "scn-test-001" in user_msg.content
