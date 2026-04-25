# Feature: clyde-economic-simulator, Property 16: CausalChain Serialization Round-Trip
"""Property-based tests for CausalChain serialize/deserialize round-trips."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clyde.models import CausalChain, CausalEvent


_ID_ALPHABET = st.characters(categories=["Ll", "Lu", "Nd"])
_SHORT_ID = st.text(min_size=1, max_size=8, alphabet=_ID_ALPHABET)
_SAFE_TEXT = st.text(
    min_size=0,
    max_size=16,
    alphabet=st.characters(categories=["Ll", "Lu", "Nd", "Zs", "Pc"]),
)
_SAFE_FLOAT = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def causal_events(draw) -> CausalEvent:
    return CausalEvent(
        step=draw(st.integers(min_value=0, max_value=10_000)),
        source_actor_id=draw(_SHORT_ID),
        target_actor_id=draw(_SHORT_ID),
        channel=draw(_SHORT_ID),
        variable_affected=draw(_SHORT_ID),
        magnitude=draw(_SAFE_FLOAT),
        description=draw(_SAFE_TEXT),
    )


@st.composite
def causal_chains(draw) -> CausalChain:
    return CausalChain(
        chain_id=draw(_SHORT_ID),
        origin_shock=draw(_SAFE_TEXT),
        total_magnitude=draw(_SAFE_FLOAT),
        events=draw(st.lists(causal_events(), min_size=0, max_size=20)),
    )


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(chain=causal_chains())
def test_causal_chain_serialize_roundtrip(chain: CausalChain) -> None:
    """serialize(deserialize(serialize(chain))) == serialize(chain)."""
    first = chain.serialize()
    restored = CausalChain.deserialize(first)
    second = restored.serialize()
    assert second == first


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(chain=causal_chains())
def test_causal_chain_deserialize_matches_input_dict(chain: CausalChain) -> None:
    """deserialize(d).serialize() == d for any d produced by serialize()."""
    data = chain.serialize()
    restored = CausalChain.deserialize(data)
    assert isinstance(restored, CausalChain)
    assert restored.serialize() == data
