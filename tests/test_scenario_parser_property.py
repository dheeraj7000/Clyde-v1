"""Property-based tests for the ScenarioParser.

# Feature: clyde-economic-simulator, Property 1: Ambiguity Resolution Merge
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings, strategies as st

from clyde.llm import MockLLMClient
from clyde.models.input import Ambiguity, ParseResult, ShockParams
from clyde.models.time import TimeHorizon
from clyde.setup.parser import ScenarioParser


# ---------- Field pool -------------------------------------------------------

# Fixed pool of dotted field paths supported by resolve_ambiguities. Each entry
# pairs a field path with a Hypothesis strategy that produces a *valid* user-
# resolution string for that field, and a getter that extracts the resulting
# Python value from a ParseResult so we can assert on it.

_TRIGGERING_EVENT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip() != "")

_SEVERITY = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_STEPS = st.integers(min_value=0, max_value=365)

_GEO_TOKEN = st.sampled_from(["US", "EU", "UK", "JP", "CN", "BR", "IN", "MX"])
_MARKET_TOKEN = st.sampled_from(
    ["energy", "tech", "auto", "housing", "ag", "finance", "retail"]
)


def _csv_strategy(token: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.lists(token, min_size=1, max_size=4, unique=True).map(", ".join)


# Map: field_path -> (resolution_str_strategy, getter(parse_result) -> value,
#                    expected_coercer(resolution_str) -> value)
_FIELD_CONFIG: dict[str, tuple[st.SearchStrategy[str], callable, callable]] = {
    "triggering_event": (
        _TRIGGERING_EVENT,
        lambda pr: pr.triggering_event,
        lambda s: s,
    ),
    "shock_params.severity": (
        _SEVERITY.map(lambda v: f"{v:.4f}"),
        lambda pr: pr.shock_params.severity,
        lambda s: float(s),
    ),
    "time_horizon.steps": (
        _STEPS.map(str),
        lambda pr: pr.time_horizon.steps,
        lambda s: int(s),
    ),
    "geographies": (
        _csv_strategy(_GEO_TOKEN),
        lambda pr: pr.geographies,
        lambda s: [t.strip() for t in s.split(",") if t.strip()],
    ),
    "markets": (
        _csv_strategy(_MARKET_TOKEN),
        lambda pr: pr.markets,
        lambda s: [t.strip() for t in s.split(",") if t.strip()],
    ),
}

_FIELD_POOL = list(_FIELD_CONFIG.keys())


# ---------- Strategies -------------------------------------------------------


@st.composite
def parse_result_with_ambiguities(draw):
    """Build a ParseResult with N ambiguities (1..6) and resolutions for 0..N."""
    n = draw(st.integers(min_value=1, max_value=min(6, len(_FIELD_POOL))))
    fields = draw(
        st.lists(
            st.sampled_from(_FIELD_POOL),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )

    ambiguities = [
        Ambiguity(
            field=field,
            description=f"flagged: {field}",
            options=None,
        )
        for field in fields
    ]

    parse_result = ParseResult(
        triggering_event="initial",
        geographies=["XX"],
        markets=["initial-market"],
        shock_params=ShockParams(
            shock_type="synthetic",
            severity=0.5,
            scope="sectoral",
            duration_steps=1,
            initial_contact_actors=[],
        ),
        time_horizon=TimeHorizon(steps=10, step_unit="day"),
        ambiguities=ambiguities,
        actor_hints=[],
    )

    # For each flagged field, decide whether to resolve it and pick a value.
    resolutions: dict[str, str] = {}
    for field in fields:
        include = draw(st.booleans())
        if not include:
            continue
        value_strategy = _FIELD_CONFIG[field][0]
        value = draw(value_strategy)
        resolutions[field] = value

    return parse_result, resolutions


# ---------- The property -----------------------------------------------------


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=40, deadline=None)
@given(parse_result_with_ambiguities())
async def test_property_1_ambiguity_resolution_merge(case):
    """Property 1: resolve_ambiguities applies user values to flagged fields.

    For every key in `resolutions` matching an ambiguity field:
      * the resulting ParseResult's field reflects the resolved value
        (with type coercion correct);
      * the matching Ambiguity has resolved=True and resolution=<user string>.
    Ambiguities not in `resolutions` pass through unmodified.
    """
    parse_result, resolutions = case

    # Snapshot input identity so we can verify the parser deep-copied it.
    original_ambiguities = [
        Ambiguity(
            field=a.field,
            description=a.description,
            options=list(a.options) if a.options is not None else None,
            resolved=a.resolved,
            resolution=a.resolution,
        )
        for a in parse_result.ambiguities
    ]

    parser = ScenarioParser(MockLLMClient())
    result = await parser.resolve_ambiguities(parse_result, resolutions)

    # 1) Every resolved field reflects the user value with proper coercion.
    for field_path, raw_value in resolutions.items():
        getter = _FIELD_CONFIG[field_path][1]
        coercer = _FIELD_CONFIG[field_path][2]
        actual = getter(result)
        expected = coercer(raw_value)
        assert actual == expected, (
            f"Field {field_path!r} expected {expected!r}, got {actual!r} "
            f"(raw resolution={raw_value!r})"
        )

    # 2) Matching ambiguities are marked resolved with the user's string.
    result_amb_by_field = {a.field: a for a in result.ambiguities}
    for field_path, raw_value in resolutions.items():
        amb = result_amb_by_field[field_path]
        assert amb.resolved is True, f"Ambiguity for {field_path} not marked resolved"
        assert amb.resolution == str(raw_value), (
            f"Ambiguity for {field_path}: resolution={amb.resolution!r}, "
            f"expected {str(raw_value)!r}"
        )

    # 3) Ambiguities NOT in resolutions pass through unmodified.
    for orig in original_ambiguities:
        if orig.field in resolutions:
            continue
        amb = result_amb_by_field[orig.field]
        assert amb.resolved is False, (
            f"Unresolved ambiguity {orig.field} should remain resolved=False"
        )
        assert amb.resolution is None, (
            f"Unresolved ambiguity {orig.field} should remain resolution=None"
        )
        assert amb.description == orig.description
        assert amb.options == orig.options

    # 4) Input was not mutated.
    for orig, current in zip(original_ambiguities, parse_result.ambiguities):
        assert current.resolved == orig.resolved
        assert current.resolution == orig.resolution
