# Feature: clyde-economic-simulator, Property 5: Prior Library Citability
"""Exhaustive tests that every PriorLibrary entry is complete and citable.

Every parameter surfaced by ``iter_params()`` must carry a Citation with
non-empty title/authors/source and a plausible year. The library must also
cover every field required to construct each actor type.
"""

from __future__ import annotations

import datetime
import math

import pytest

from clyde.models.actors import REQUIRED_PARAM_FIELDS
from clyde.models.enums import ActorType
from clyde.models.reporting import Citation
from clyde.setup.prior_library import PriorLibrary


@pytest.mark.property
def test_prior_library_is_fully_citable_and_complete() -> None:
    """Every entry has a well-formed Citation; library covers all required fields."""
    lib = PriorLibrary()

    version = lib.version()
    assert isinstance(version, str)
    assert version, "PriorLibrary.version() must be non-empty"

    max_year = datetime.date.today().year + 1

    entries = list(lib.iter_params())
    assert entries, "iter_params() must yield at least one entry"

    seen_keys: set[tuple[ActorType, str]] = set()

    for actor_type, name, value, citation in entries:
        assert isinstance(actor_type, ActorType)
        assert isinstance(name, str) and name, "param name must be non-empty str"

        # Value must be a finite float.
        assert isinstance(value, float), f"{name}: value is not a float"
        assert math.isfinite(value), f"{name}: value is not finite ({value!r})"

        # Citation invariants.
        assert isinstance(citation, Citation)
        assert isinstance(citation.title, str) and citation.title.strip(), (
            f"{name}: citation.title must be a non-empty str"
        )
        assert isinstance(citation.authors, list) and len(citation.authors) > 0, (
            f"{name}: citation.authors must be a non-empty list"
        )
        for author in citation.authors:
            assert isinstance(author, str) and author.strip(), (
                f"{name}: each author must be a non-empty str"
            )
        assert isinstance(citation.year, int), (
            f"{name}: citation.year must be an int, got {type(citation.year).__name__}"
        )
        # Guard against bool (bool is a subclass of int in Python).
        assert not isinstance(citation.year, bool), (
            f"{name}: citation.year must be int, not bool"
        )
        assert 1900 <= citation.year <= max_year, (
            f"{name}: citation.year {citation.year} out of range [1900, {max_year}]"
        )
        assert isinstance(citation.source, str) and citation.source.strip(), (
            f"{name}: citation.source must be a non-empty str"
        )

        seen_keys.add((actor_type, name))

    # parameter_citations() must match iter_params() in size and key space.
    citations_map = lib.parameter_citations()
    assert isinstance(citations_map, dict)
    assert len(citations_map) == len(entries), (
        "parameter_citations() size must equal iter_params() count"
    )
    for key in citations_map:
        actor_type, name = key
        assert isinstance(actor_type, ActorType)
        assert isinstance(name, str) and name
    assert set(citations_map.keys()) == seen_keys

    # Library must cover every actor-construction-required field.
    for actor_type in ActorType:
        required = REQUIRED_PARAM_FIELDS.get(actor_type, ())
        for field_name in required:
            assert (actor_type, field_name) in citations_map, (
                f"PriorLibrary missing citation for ({actor_type.value}, {field_name})"
            )

    # lib.citation(name) must return the same Citation object as the dict lookup.
    for (actor_type, name), citation in citations_map.items():
        assert lib.citation(name, actor_type) is citation, (
            f"citation({name!r}, {actor_type}) mismatched the parameter_citations map"
        )
        # Unscoped lookup returns *a* Citation for that name; must exist.
        unscoped = lib.citation(name)
        assert isinstance(unscoped, Citation)

    # Unknown key must raise KeyError.
    with pytest.raises(KeyError):
        lib.citation("nonsense_key")
