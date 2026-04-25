# Feature: clyde-economic-simulator, Property 17: Scenario Serialization Round-Trip
"""Property-based tests for Scenario serialize/deserialize round-trips."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clyde.models import (
    PARAMS_CLASS_BY_TYPE,
    RELATIONSHIP_TYPES,
    REQUIRED_PARAM_FIELDS,
    Actor,
    ActorType,
    BipartiteGraph,
    DirectedGraph,
    HistoricalAnalog,
    NetworkBundle,
    Relationship,
    Scenario,
    ScaleFreeGraph,
    ShockConfig,
    TimeHorizon,
    VALID_SCOPES,
    VALID_STEP_UNITS,
)


_ID_ALPHABET = st.characters(categories=["Ll", "Lu", "Nd"])
_SHORT_ID = st.text(min_size=1, max_size=8, alphabet=_ID_ALPHABET)
_SAFE_TEXT = st.text(
    min_size=0,
    max_size=16,
    alphabet=st.characters(categories=["Ll", "Lu", "Nd", "Zs", "Pc"]),
)
_NONNEG_FLOAT = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
)
_SMALL_FLOAT = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)
_SCOPES = st.sampled_from(sorted(VALID_SCOPES))
_STEP_UNITS = st.sampled_from(sorted(VALID_STEP_UNITS))
_ACTOR_TYPES = st.sampled_from(list(ActorType))
_REL_TYPES = st.sampled_from(sorted(RELATIONSHIP_TYPES))


# -- Time horizon -------------------------------------------------------------


@st.composite
def time_horizons(draw) -> TimeHorizon:
    return TimeHorizon(
        steps=draw(st.integers(min_value=0, max_value=1000)),
        step_unit=draw(_STEP_UNITS),
    )


# -- Historical analog --------------------------------------------------------


@st.composite
def historical_analogs(draw) -> HistoricalAnalog:
    return HistoricalAnalog(
        event_name=draw(_SAFE_TEXT),
        year=draw(st.integers(min_value=1800, max_value=2100)),
        similarity_score=draw(_NONNEG_FLOAT),
        param_adjustments=draw(
            st.dictionaries(_SHORT_ID, _SMALL_FLOAT, max_size=3)
        ),
        source=draw(_SAFE_TEXT),
    )


# -- ShockConfig --------------------------------------------------------------


@st.composite
def shock_configs(draw) -> ShockConfig:
    return ShockConfig(
        shock_type=draw(_SHORT_ID),
        severity=draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        scope=draw(_SCOPES),
        duration_steps=draw(st.integers(min_value=0, max_value=500)),
        geography=draw(st.lists(_SHORT_ID, max_size=3)),
        sectors=draw(st.lists(_SHORT_ID, max_size=3)),
        initial_contact_actors=draw(st.lists(_SHORT_ID, max_size=3)),
        agent_counts=draw(
            st.dictionaries(
                _SHORT_ID, st.integers(min_value=0, max_value=100), max_size=3
            )
        ),
        behavioral_overrides={},
        time_horizon=draw(time_horizons()),
        ensemble_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        historical_analogs=draw(st.lists(historical_analogs(), max_size=3)),
    )


# -- Actors -------------------------------------------------------------------


@st.composite
def actor_params(draw, actor_type: ActorType):
    params_cls = PARAMS_CLASS_BY_TYPE[actor_type]
    fields = REQUIRED_PARAM_FIELDS[actor_type]
    kwargs = {name: draw(_NONNEG_FLOAT) for name in fields}
    return params_cls(**kwargs)


@st.composite
def actors(draw, actor_id: str) -> Actor:
    actor_type = draw(_ACTOR_TYPES)
    params = draw(actor_params(actor_type))
    # Keep state JSON-safe: floats only (Actor.from_dict coerces numeric values).
    state = draw(st.dictionaries(_SHORT_ID, _SMALL_FLOAT, max_size=3))
    return Actor(
        id=actor_id,
        actor_type=actor_type,
        params=params,
        state=state,
        relationships=[],
    )


@st.composite
def unique_actor_list(draw) -> list[Actor]:
    ids = draw(st.lists(_SHORT_ID, min_size=0, max_size=5, unique=True))
    return [draw(actors(actor_id=aid)) for aid in ids]


@st.composite
def relationships_for(draw, actor_ids: list[str]) -> list[Relationship]:
    if not actor_ids:
        return []
    n = draw(st.integers(min_value=0, max_value=3))
    rels: list[Relationship] = []
    for _ in range(n):
        src = draw(st.sampled_from(actor_ids))
        tgt = draw(st.sampled_from(actor_ids))
        rels.append(
            Relationship(
                source_id=src,
                target_id=tgt,
                rel_type=draw(_REL_TYPES),
                weight=draw(_SMALL_FLOAT),
            )
        )
    return rels


# -- Networks -----------------------------------------------------------------


@st.composite
def edge_list(draw) -> list[tuple[str, str, float]]:
    return draw(
        st.lists(
            st.tuples(_SHORT_ID, _SHORT_ID, _SMALL_FLOAT),
            min_size=0,
            max_size=10,
        )
    )


@st.composite
def network_bundles(draw) -> NetworkBundle:
    return NetworkBundle(
        labor_market=BipartiteGraph(edges=draw(edge_list())),
        supply_chain=DirectedGraph(edges=draw(edge_list())),
        interbank=ScaleFreeGraph(edges=draw(edge_list())),
    )


# -- Scenario -----------------------------------------------------------------


# Restrict overrides/metadata to simple JSON-safe scalar values.
_JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    _SMALL_FLOAT,
    _SAFE_TEXT,
)


@st.composite
def scenarios(draw) -> Scenario:
    actor_list = draw(unique_actor_list())
    actor_ids = [a.id for a in actor_list]
    # Attach random relationships referencing existing actor IDs.
    for actor in actor_list:
        actor.relationships = draw(relationships_for(actor_ids))
    return Scenario(
        scenario_id=draw(_SHORT_ID),
        description=draw(_SAFE_TEXT),
        config=draw(shock_configs()),
        actors=actor_list,
        networks=draw(network_bundles()),
        prior_library_version=draw(_SAFE_TEXT),
        overrides=draw(st.dictionaries(_SHORT_ID, _JSON_SCALAR, max_size=3)),
        metadata=draw(st.dictionaries(_SHORT_ID, _JSON_SCALAR, max_size=3)),
    )


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(scenario=scenarios())
def test_scenario_serialize_roundtrip(scenario: Scenario) -> None:
    """serialize(deserialize(serialize(scenario))) == serialize(scenario)."""
    first = scenario.serialize()
    restored = Scenario.deserialize(first)
    second = restored.serialize()
    assert second == first


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(scenario=scenarios())
def test_scenario_json_roundtrip(scenario: Scenario) -> None:
    """to_json / from_json preserve the serialized form."""
    text = scenario.to_json()
    restored = Scenario.from_json(text)
    assert restored.serialize() == scenario.serialize()
    # And the JSON itself must be valid and equivalent.
    assert json.loads(text) == scenario.serialize()
