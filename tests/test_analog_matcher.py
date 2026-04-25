"""Tests for clyde.setup.analog_matcher (Task 15.1, Requirements 10.1-10.3)."""

from __future__ import annotations

import copy

import pytest
from hypothesis import given, settings, strategies as st

from clyde.models.config import ShockConfig
from clyde.models.input import ParseResult, ShockParams
from clyde.models.reporting import HistoricalAnalog
from clyde.setup.analog_matcher import (
    AnalogDisclosure,
    AnalogMatcher,
    HistoricalEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def matcher() -> AnalogMatcher:
    return AnalogMatcher()


def _volcker_shock() -> ShockConfig:
    return ShockConfig(
        shock_type="rate_hike",
        severity=0.7,
        scope="macro",
        duration_steps=24,
        geography=["US"],
        sectors=["banking", "housing"],
    )


def _gfc_shock() -> ShockConfig:
    return ShockConfig(
        shock_type="banking_crisis",
        severity=0.9,
        scope="macro",
        duration_steps=18,
        geography=["US"],
        sectors=["banking", "real_estate", "finance"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_corpus_is_non_trivial(matcher: AnalogMatcher) -> None:
    corpus = matcher.corpus
    assert len(corpus) >= 12
    for ev in corpus:
        assert isinstance(ev, HistoricalEvent)
        assert ev.name and ev.year > 0
        assert ev.shock_type
        assert ev.scope in {"micro", "sectoral", "macro", "cross_border"}
        assert 0.0 <= ev.severity <= 1.0
        assert ev.duration_months > 0
        assert ev.geographies
        assert ev.sectors
        assert ev.keywords
        assert ev.param_adjustments
        assert ev.source


def test_volcker_case(matcher: AnalogMatcher) -> None:
    analogs = matcher.match(shock_config=_volcker_shock())
    assert analogs, "expected at least one analog for a rate-hike macro shock"
    top = analogs[0]
    assert top.event_name == "Volcker Disinflation"
    assert top.year == 1981
    assert top.similarity_score > 0.6
    # param_adjustments should be propagated.
    assert top.param_adjustments
    assert any("elasticity" in k or "growth" in k or "lag" in k for k in top.param_adjustments)


def test_gfc_case(matcher: AnalogMatcher) -> None:
    analogs = matcher.match(shock_config=_gfc_shock())
    assert analogs
    names = [a.event_name for a in analogs]
    assert "2008 Global Financial Crisis" in names
    # And it should rank first since it matches shock_type, scope, geography, sectors.
    assert analogs[0].event_name == "2008 Global Financial Crisis"
    assert analogs[0].similarity_score > 0.6


def test_no_false_positives_for_off_domain(matcher: AnalogMatcher) -> None:
    bogus = ShockConfig(
        shock_type="alien_invasion",
        severity=0.5,
        scope="micro",
        duration_steps=1,
        geography=["mars"],
        sectors=["xenobiology"],
    )
    analogs = matcher.match(shock_config=bogus)
    # Either no matches at all, or matches all below the default threshold ceiling.
    assert all(a.similarity_score < 0.5 for a in analogs)


def test_top_k_is_bounded() -> None:
    m = AnalogMatcher(top_k=3, min_similarity=0.0)
    analogs = m.match(
        shock_config=ShockConfig(
            shock_type="banking_crisis",
            severity=0.6,
            scope="macro",
            duration_steps=10,
            geography=["US", "EU", "global"],
            sectors=["banking", "finance", "real_estate"],
        )
    )
    assert len(analogs) <= 3


def test_disclosure_contents(matcher: AnalogMatcher) -> None:
    shock = _gfc_shock()
    analogs = matcher.match(shock_config=shock)
    assert analogs
    disclosures = matcher.disclose(analogs, shock_config=shock)
    assert len(disclosures) == len(analogs)
    for d in disclosures:
        assert isinstance(d, AnalogDisclosure)
        assert d.rationale.strip()
        assert d.affected_params, "expected affected_params to be non-empty"
        assert set(d.shifted_ranges.keys()) == set(d.affected_params)
        for name, (lo, hi) in d.shifted_ranges.items():
            assert lo <= hi


def test_range_combination_widens(matcher: AnalogMatcher) -> None:
    # Pick two analogs from the corpus that share a parameter.
    shock = ShockConfig(
        shock_type="banking_crisis",
        severity=0.8,
        scope="macro",
        duration_steps=12,
        geography=["US", "EU"],
        sectors=["banking", "finance"],
    )
    analogs = matcher.match(shock_config=shock)
    # Find two analogs that share at least one param key.
    pair = None
    for i in range(len(analogs)):
        for j in range(i + 1, len(analogs)):
            shared = set(analogs[i].param_adjustments) & set(analogs[j].param_adjustments)
            if shared:
                pair = (analogs[i], analogs[j], next(iter(shared)))
                break
        if pair:
            break
    assert pair is not None, "expected at least two analogs to share a param"
    a, b, key = pair

    base = {key: 1.0}
    range_a = matcher.informed_param_ranges([a], base)
    range_b = matcher.informed_param_ranges([b], base)
    range_both = matcher.informed_param_ranges([a, b], base)

    width_a = range_a[key][1] - range_a[key][0]
    width_b = range_b[key][1] - range_b[key][0]
    width_both = range_both[key][1] - range_both[key][0]

    assert width_both >= width_a
    assert width_both >= width_b
    # Strictly wider unless the two analogs happen to produce identical ranges.
    if range_a[key] != range_b[key]:
        assert width_both > min(width_a, width_b)


def test_no_mutation_of_inputs(matcher: AnalogMatcher) -> None:
    shock = _gfc_shock()
    shock_snapshot = copy.deepcopy(shock.to_dict())
    corpus_snapshot = matcher.corpus  # this is a copy itself
    corpus_repr_before = [
        (ev.name, ev.year, dict(ev.param_adjustments)) for ev in corpus_snapshot
    ]

    analogs = matcher.match(shock_config=shock)
    matcher.disclose(analogs, shock_config=shock)
    matcher.informed_param_ranges(analogs, base_params={"npl_tightening_elasticity": 1.5})

    # Original ShockConfig untouched.
    assert shock.to_dict() == shock_snapshot

    # Corpus untouched.
    corpus_repr_after = [
        (ev.name, ev.year, dict(ev.param_adjustments)) for ev in matcher.corpus
    ]
    assert corpus_repr_before == corpus_repr_after

    # Mutating the returned analog list should not affect the matcher.
    if analogs:
        analogs[0].param_adjustments["__bogus__"] = 999.0
        for ev in matcher.corpus:
            assert "__bogus__" not in ev.param_adjustments


def test_parse_result_input(matcher: AnalogMatcher) -> None:
    parse = ParseResult(
        triggering_event="oil price shock from OPEC embargo",
        geographies=["US", "EU"],
        markets=["energy"],
        shock_params=ShockParams(
            shock_type="supply_disruption",
            severity=0.8,
            scope="macro",
            duration_steps=24,
        ),
    )
    analogs = matcher.match(parse_result=parse)
    assert analogs
    names = [a.event_name for a in analogs]
    assert "1973 Oil Shock" in names


# ---------------------------------------------------------------------------
# Hypothesis property: keyword-overlap symmetry & bounded similarity
# ---------------------------------------------------------------------------


_KEYWORD_POOL = [
    "oil", "energy", "banking", "credit", "sovereign", "currency", "tariff",
    "supply", "chain", "labor", "rate", "inflation", "crisis", "policy",
    "trade", "uncertainty", "default", "deposit",
]

_GEO_POOL = ["US", "EU", "UK", "China", "Japan", "Greece", "Argentina", "global"]
_SECTOR_POOL = ["banking", "energy", "manufacturing", "finance", "trade", "services"]


@settings(max_examples=30, deadline=None)
@given(
    keywords=st.lists(st.sampled_from(_KEYWORD_POOL), min_size=0, max_size=6, unique=True),
    geos=st.lists(st.sampled_from(_GEO_POOL), min_size=0, max_size=4, unique=True),
    sectors=st.lists(st.sampled_from(_SECTOR_POOL), min_size=0, max_size=4, unique=True),
    shock_type=st.sampled_from(
        ["banking_crisis", "rate_hike", "supply_disruption", "currency_crisis", "tariff"]
    ),
    scope=st.sampled_from(["micro", "sectoral", "macro", "cross_border"]),
)
def test_similarity_bounded_and_symmetric_in_keywords(
    keywords: list[str], geos: list[str], sectors: list[str], shock_type: str, scope: str
) -> None:
    matcher = AnalogMatcher(min_similarity=0.0, top_k=20)
    shock = ShockConfig(
        shock_type=shock_type,
        severity=0.5,
        scope=scope,
        duration_steps=12,
        geography=list(geos),
        sectors=list(sectors),
    )
    a1 = matcher.match(shock_config=shock, keywords=list(keywords))
    a2 = matcher.match(shock_config=shock, keywords=list(reversed(keywords)))

    # Bounded.
    for a in a1 + a2:
        assert 0.0 <= a.similarity_score <= 1.0

    # Order-independence in keyword input -> identical scores per event.
    by_event_1 = {(a.event_name, a.year): a.similarity_score for a in a1}
    by_event_2 = {(a.event_name, a.year): a.similarity_score for a in a2}
    assert by_event_1 == by_event_2
