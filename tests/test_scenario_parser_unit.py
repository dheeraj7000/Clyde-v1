"""Curated unit tests for the ScenarioParser.

Five hand-written scenarios spanning the four scopes (micro, sectoral, macro,
cross_border) plus error-handling tests for empty input, retry / timeout
semantics, and validator clamps. Covers Task 13.4.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from clyde.llm import LLMMessage, MockLLMClient
from clyde.models.config import VALID_SCOPES
from clyde.models.input import Document
from clyde.models.time import VALID_STEP_UNITS
from clyde.setup.parser import ParsingTimeoutError, ScenarioParser


# ---------- Canonical "good" LLM payloads -----------------------------------


def _payload(
    *,
    triggering_event: str,
    scope: str,
    shock_type: str = "generic_shock",
    severity: float = 0.5,
    duration_steps: int = 5,
    geographies: list[str] | None = None,
    markets: list[str] | None = None,
    steps: int = 10,
    step_unit: str = "week",
    ambiguities: list[dict[str, Any]] | None = None,
    actor_hints: list[dict[str, Any]] | None = None,
    initial_actors: list[str] | None = None,
) -> dict:
    return {
        "triggering_event": triggering_event,
        "geographies": geographies or [],
        "markets": markets or [],
        "shock_params": {
            "shock_type": shock_type,
            "severity": severity,
            "scope": scope,
            "duration_steps": duration_steps,
            "initial_contact_actors": initial_actors or [],
        },
        "time_horizon": {"steps": steps, "step_unit": step_unit},
        "ambiguities": ambiguities or [],
        "actor_hints": actor_hints or [],
    }


# ---------- Curated scenarios (one per scope + extra micro) -----------------


_CURATED_SCENARIOS: list[tuple[str, str, str, dict]] = [
    (
        "micro_household_layoff",
        "A household in Detroit loses its primary income after a layoff at a manufacturing plant.",
        "micro",
        _payload(
            triggering_event="household_layoff",
            scope="micro",
            shock_type="income_loss",
            severity=0.6,
            duration_steps=4,
            geographies=["US-MI"],
            markets=["labor"],
            steps=12,
            step_unit="week",
            actor_hints=[
                {"actor_type": "household", "count_estimate": 1, "description": "affected family"},
                {"actor_type": "firm", "count_estimate": 1, "description": "manufacturing employer"},
            ],
        ),
    ),
    (
        "sectoral_oil_price_shock",
        "Oil prices spike 40% across the US energy sector after a refinery outage.",
        "sectoral",
        _payload(
            triggering_event="oil_price_spike",
            scope="sectoral",
            shock_type="commodity_price_shock",
            severity=0.4,
            duration_steps=8,
            geographies=["US"],
            markets=["energy"],
            steps=26,
            step_unit="week",
        ),
    ),
    (
        "macro_rate_hike",
        "The Federal Reserve raises rates by 75bps to combat persistent inflation.",
        "macro",
        _payload(
            triggering_event="federal_funds_rate_hike",
            scope="macro",
            shock_type="monetary_policy_tightening",
            severity=0.3,
            duration_steps=12,
            geographies=["US"],
            markets=["credit", "housing", "labor"],
            steps=8,
            step_unit="quarter",
        ),
    ),
    (
        "cross_border_trade_war",
        "China imposes tariffs of 25% on US semiconductor exports and the US retaliates.",
        "cross_border",
        _payload(
            triggering_event="bilateral_tariff_escalation",
            scope="cross_border",
            shock_type="trade_war",
            severity=0.7,
            duration_steps=24,
            geographies=["US", "CN"],
            markets=["semiconductor", "trade"],
            steps=12,
            step_unit="month",
        ),
    ),
    (
        "micro_small_business_loan_denial",
        "A small bakery in Atlanta is denied a working-capital loan and faces a cashflow crunch.",
        "micro",
        _payload(
            triggering_event="loan_denial",
            scope="micro",
            shock_type="credit_constraint",
            severity=0.5,
            duration_steps=6,
            geographies=["US-GA"],
            markets=["small_business_credit"],
            steps=10,
            step_unit="week",
        ),
    ),
]


@pytest.mark.parametrize(
    "name,description,expected_scope,payload",
    _CURATED_SCENARIOS,
    ids=[s[0] for s in _CURATED_SCENARIOS],
)
@pytest.mark.asyncio
async def test_parse_curated_scenarios(name, description, expected_scope, payload):
    """Parser accepts a well-formed LLM payload across all four scopes."""
    mock = MockLLMClient(responses=[payload])
    parser = ScenarioParser(mock)

    result = await parser.parse(description)

    assert result.triggering_event == payload["triggering_event"]
    assert result.shock_params.scope == expected_scope
    assert result.shock_params.scope in VALID_SCOPES
    assert result.time_horizon.step_unit in VALID_STEP_UNITS
    # All curated payloads are clean -> no validator-emitted ambiguities.
    assert all(not a.resolved for a in result.ambiguities)
    # Mock was called exactly once.
    assert len(mock.call_log) == 1
    method, _messages, _kwargs = mock.call_log[0]
    assert method == "complete_json"


# ---------- Empty / nonsensical input ---------------------------------------


@pytest.mark.asyncio
async def test_empty_input_short_circuits_without_calling_llm():
    """Empty input ⇒ all-ambiguity ParseResult, LLM never called."""
    mock = MockLLMClient(responses=[])  # would IndexError if called
    parser = ScenarioParser(mock)

    result = await parser.parse("   ")

    assert len(mock.call_log) == 0
    assert result.triggering_event == ""
    # Every structured field flagged.
    flagged = {a.field for a in result.ambiguities}
    expected_flagged = {
        "triggering_event",
        "geographies",
        "markets",
        "shock_params.shock_type",
        "shock_params.severity",
        "shock_params.scope",
        "shock_params.duration_steps",
        "shock_params.initial_contact_actors",
        "time_horizon.steps",
        "time_horizon.step_unit",
    }
    assert expected_flagged.issubset(flagged)


@pytest.mark.asyncio
async def test_short_description_short_circuits():
    """Sub-5-character input also short-circuits."""
    mock = MockLLMClient(responses=[])
    parser = ScenarioParser(mock)
    result = await parser.parse("a")
    assert len(mock.call_log) == 0
    assert len(result.ambiguities) >= 10


# ---------- Retry semantics --------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_triggers_retry_then_succeeds():
    """Invalid JSON from the LLM triggers a retry; verify retry count via call_log."""
    good = _payload(triggering_event="oil_price_spike", scope="sectoral")

    # First response is a non-JSON string -> MockLLMClient.complete_json raises
    # ValueError. Second response is the good payload.
    mock = MockLLMClient(responses=["not-json-at-all", good])
    parser = ScenarioParser(mock, max_retries=3, backoff_base_s=0.0)

    result = await parser.parse("Oil prices spike across the US energy sector.")

    assert result.triggering_event == "oil_price_spike"
    assert len(mock.call_log) == 2  # one failed attempt + one successful retry


@pytest.mark.asyncio
async def test_all_retries_fail_raises_parsing_timeout_error():
    """After all retries fail, ParsingTimeoutError is raised."""
    mock = MockLLMClient(responses=["bad1", "bad2", "bad3"])
    parser = ScenarioParser(mock, max_retries=3, backoff_base_s=0.0)

    with pytest.raises(ParsingTimeoutError) as excinfo:
        await parser.parse("Oil prices spike across the US energy sector.")

    assert excinfo.value.cause is not None
    assert len(mock.call_log) == 3


@pytest.mark.asyncio
async def test_timeout_triggers_retry():
    """asyncio.TimeoutError from the LLM is retried."""

    call_counter = {"n": 0}
    good = _payload(triggering_event="evt", scope="micro")

    async def slow_then_fast(messages):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            await asyncio.sleep(1.0)  # will exceed the parser's timeout
        return good

    class StubLLM:
        async def complete(self, *args, **kwargs):  # pragma: no cover - unused
            raise NotImplementedError

        async def complete_json(self, messages, **kwargs):
            return await slow_then_fast(messages)

    parser = ScenarioParser(
        StubLLM(),
        max_retries=3,
        backoff_base_s=0.0,
        request_timeout_s=0.05,
    )
    result = await parser.parse("A meaningful scenario description.")
    assert result.triggering_event == "evt"
    assert call_counter["n"] == 2


# ---------- Validator clamps -------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_scope_is_flagged_and_clamped():
    """LLM returns scope='global' (invalid) → ambiguity flagged, value clamped to 'sectoral'."""
    bad = _payload(triggering_event="evt", scope="global")
    # Mutation: override the scope field directly to a bad value.
    bad["shock_params"]["scope"] = "global"

    mock = MockLLMClient(responses=[bad])
    parser = ScenarioParser(mock)
    result = await parser.parse("A meaningful scenario description.")

    assert result.shock_params.scope == "sectoral"  # clamp
    flagged_fields = {a.field: a for a in result.ambiguities}
    assert "shock_params.scope" in flagged_fields
    amb = flagged_fields["shock_params.scope"]
    assert amb.description == "global"  # offending value preserved


@pytest.mark.asyncio
async def test_invalid_step_unit_is_flagged_and_clamped():
    """LLM returns step_unit='hour' → ambiguity flagged, clamped to 'week'."""
    bad = _payload(triggering_event="evt", scope="sectoral", step_unit="hour")

    mock = MockLLMClient(responses=[bad])
    parser = ScenarioParser(mock)
    result = await parser.parse("A meaningful scenario description.")

    assert result.time_horizon.step_unit == "week"
    flagged_fields = {a.field for a in result.ambiguities}
    assert "time_horizon.step_unit" in flagged_fields


@pytest.mark.asyncio
async def test_severity_out_of_range_is_flagged():
    """Severity outside [0, 1] is flagged."""
    bad = _payload(triggering_event="evt", scope="micro", severity=1.5)

    mock = MockLLMClient(responses=[bad])
    parser = ScenarioParser(mock)
    result = await parser.parse("A meaningful scenario description.")

    flagged = {a.field for a in result.ambiguities}
    assert "shock_params.severity" in flagged
    assert 0.0 <= result.shock_params.severity <= 1.0


@pytest.mark.asyncio
async def test_negative_steps_is_flagged():
    """Negative time_horizon.steps is flagged and clamped to 0."""
    bad = _payload(triggering_event="evt", scope="micro", steps=-3)

    mock = MockLLMClient(responses=[bad])
    parser = ScenarioParser(mock)
    result = await parser.parse("A meaningful scenario description.")

    flagged = {a.field for a in result.ambiguities}
    assert "time_horizon.steps" in flagged
    assert result.time_horizon.steps == 0


# ---------- Document inclusion ----------------------------------------------


@pytest.mark.asyncio
async def test_documents_are_included_in_user_message():
    """Document content is injected into the LLM user message."""
    payload = _payload(triggering_event="evt", scope="sectoral")
    mock = MockLLMClient(responses=[payload])
    parser = ScenarioParser(mock)

    docs = [
        Document(
            path="/tmp/report.md",
            content="# Refinery outage\nOutage took out 200kbpd capacity for 10 days.",
            format="md",
        ),
        Document(
            path="/tmp/data.txt",
            content="weekly_oil_price=85.42",
            format="txt",
        ),
    ]
    await parser.parse("Oil prices spike after a refinery outage.", documents=docs)

    # Inspect call_log[0][1] -> the messages list.
    method, messages, _kwargs = mock.call_log[0]
    assert method == "complete_json"
    user_message = next(m for m in messages if m.role == "user")
    assert "Refinery outage" in user_message.content
    assert "weekly_oil_price=85.42" in user_message.content
    assert "/tmp/report.md" in user_message.content


@pytest.mark.asyncio
async def test_long_documents_are_truncated_to_2000_chars():
    """Each document excerpt should be capped at 2000 chars."""
    payload = _payload(triggering_event="evt", scope="sectoral")
    mock = MockLLMClient(responses=[payload])
    parser = ScenarioParser(mock)

    huge = "X" * 5000 + "TAIL_MARKER"
    doc = Document(path="/tmp/huge.txt", content=huge, format="txt")
    await parser.parse("A meaningful scenario description.", documents=[doc])

    _method, messages, _kwargs = mock.call_log[0]
    user_message = next(m for m in messages if m.role == "user")
    assert "TAIL_MARKER" not in user_message.content
    # The 'X' run length present in the message must not exceed 2000.
    assert "X" * 2001 not in user_message.content
    assert "X" * 2000 in user_message.content


# ---------- LLM-supplied ambiguities passed through -------------------------


@pytest.mark.asyncio
async def test_llm_supplied_ambiguities_are_preserved():
    """Ambiguities the LLM returns are included in the ParseResult."""
    payload = _payload(
        triggering_event="evt",
        scope="sectoral",
        ambiguities=[
            {
                "field": "geographies",
                "description": "Did you mean US or US-CA?",
                "options": ["US", "US-CA"],
            }
        ],
    )
    mock = MockLLMClient(responses=[payload])
    parser = ScenarioParser(mock)
    result = await parser.parse("A meaningful scenario description.")

    fields = {a.field for a in result.ambiguities}
    assert "geographies" in fields
    geo_amb = next(a for a in result.ambiguities if a.field == "geographies")
    assert geo_amb.options == ["US", "US-CA"]


# ---------- resolve_ambiguities() returns a NEW result ----------------------


@pytest.mark.asyncio
async def test_resolve_ambiguities_does_not_mutate_input():
    """resolve_ambiguities returns a new ParseResult; input untouched."""
    bad = _payload(triggering_event="", scope="sectoral")
    mock = MockLLMClient(responses=[bad])
    parser = ScenarioParser(mock)

    result = await parser.parse("A meaningful scenario description.")
    assert any(a.field == "triggering_event" for a in result.ambiguities)

    new_result = await parser.resolve_ambiguities(
        result, {"triggering_event": "user_supplied_event"}
    )
    # Input ambiguity must still be unresolved.
    orig = next(a for a in result.ambiguities if a.field == "triggering_event")
    assert orig.resolved is False
    # Output reflects the resolution.
    assert new_result.triggering_event == "user_supplied_event"
    new_amb = next(a for a in new_result.ambiguities if a.field == "triggering_event")
    assert new_amb.resolved is True
    assert new_amb.resolution == "user_supplied_event"


@pytest.mark.asyncio
async def test_resolve_ambiguities_ignores_unflagged_fields():
    """Resolutions for fields that aren't flagged are silently ignored."""
    payload = _payload(triggering_event="evt", scope="sectoral")
    mock = MockLLMClient(responses=[payload])
    parser = ScenarioParser(mock)

    result = await parser.parse("A meaningful scenario description.")
    # No ambiguities flagged for this clean payload.
    new_result = await parser.resolve_ambiguities(
        result, {"triggering_event": "should_be_ignored"}
    )
    # Field is unchanged because it was never flagged.
    assert new_result.triggering_event == "evt"


@pytest.mark.asyncio
async def test_resolve_ambiguities_coerces_list_from_csv():
    """List fields are parsed from comma-separated user strings."""
    bad = _payload(triggering_event="evt", scope="sectoral")
    # Force the geographies field into the ambiguity list manually via a flagged LLM ambiguity.
    bad["ambiguities"] = [
        {"field": "geographies", "description": "?", "options": None}
    ]
    mock = MockLLMClient(responses=[bad])
    parser = ScenarioParser(mock)
    result = await parser.parse("A meaningful scenario description.")

    new_result = await parser.resolve_ambiguities(
        result, {"geographies": "US, EU, UK"}
    )
    assert new_result.geographies == ["US", "EU", "UK"]
