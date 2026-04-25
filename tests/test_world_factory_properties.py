"""Property-based tests for the EconomicWorldFactory.

Covers:

- Property 2: Network Integrity
- Property 4: Actor Behavioral Completeness
- Property 6: Override Application
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from clyde.models.actors import Actor, REQUIRED_PARAM_FIELDS
from clyde.models.config import ShockConfig, VALID_SCOPES
from clyde.models.enums import RELATIONSHIP_TYPES, ActorType
from clyde.models.scenario import Scenario
from clyde.models.time import TimeHorizon
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary
from clyde.setup.world_factory import EconomicWorldFactory


# ---------- Hypothesis strategies -------------------------------------------


def shock_configs(
    min_households: int = 0,
    max_households: int = 8,
    min_firms: int = 0,
    max_firms: int = 4,
    min_banks: int = 0,
    max_banks: int = 4,
    min_central_banks: int = 0,
    max_central_banks: int = 1,
) -> st.SearchStrategy[ShockConfig]:
    """Generate ShockConfigs sized for fast Hypothesis runs (< 30 actors)."""

    return st.builds(
        lambda hh, fm, bk, cb, scope, severity, duration, sectors, geo, seed: ShockConfig(
            shock_type="synthetic",
            severity=severity,
            scope=scope,
            duration_steps=duration,
            geography=geo,
            sectors=sectors,
            agent_counts={
                ActorType.HOUSEHOLD.value: hh,
                ActorType.FIRM.value: fm,
                ActorType.BANK.value: bk,
                ActorType.CENTRAL_BANK.value: cb,
            },
            time_horizon=TimeHorizon(steps=duration, step_unit="day"),
            ensemble_seed=seed,
        ),
        st.integers(min_value=min_households, max_value=max_households),
        st.integers(min_value=min_firms, max_value=max_firms),
        st.integers(min_value=min_banks, max_value=max_banks),
        st.integers(min_value=min_central_banks, max_value=max_central_banks),
        st.sampled_from(sorted(VALID_SCOPES)),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=0, max_value=10),
        st.lists(st.sampled_from(["tech", "energy", "retail", "finance"]), max_size=2, unique=True),
        st.lists(st.sampled_from(["US", "EU", "UK", "JP"]), max_size=2, unique=True),
        st.integers(min_value=0, max_value=2**31 - 1),
    )


def shock_configs_dense() -> st.SearchStrategy[ShockConfig]:
    """ShockConfigs guaranteed to have >= 2 actors of each type.

    Used by Property 2 so the network builder actually produces edges of
    every flavour (labor market, supply chain, interbank).
    """
    return shock_configs(
        min_households=2, max_households=6,
        min_firms=2, max_firms=4,
        min_banks=2, max_banks=4,
        min_central_banks=1, max_central_banks=1,
    )


# ---------- helpers ---------------------------------------------------------


def _make_factory(seed: int = 0) -> EconomicWorldFactory:
    # Give both the factory and the NetworkBuilder deterministic seeded RNGs.
    nb = NetworkBuilder(rng=random.Random(seed))
    return EconomicWorldFactory(network_builder=nb, rng_seed=seed)


# ---------- Property 2: Network Integrity ----------------------------------
# Feature: clyde-economic-simulator, Property 2: Network Integrity


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(cfg=shock_configs_dense(), seed=st.integers(min_value=0, max_value=2**16))
def test_property_2_network_integrity(cfg: ShockConfig, seed: int) -> None:
    factory = _make_factory(seed)
    world = factory.build_world(cfg, PriorLibrary())

    actor_ids = {a.id for a in world.actors}

    # Every relationship attached to an actor must reference real ids and a
    # valid rel_type from the domain constant.
    for actor in world.actors:
        for rel in actor.relationships:
            assert rel.source_id in actor_ids, (
                f"relationship source_id {rel.source_id!r} on actor "
                f"{actor.id!r} not in world actor set"
            )
            assert rel.target_id in actor_ids, (
                f"relationship target_id {rel.target_id!r} on actor "
                f"{actor.id!r} not in world actor set"
            )
            assert rel.rel_type in RELATIONSHIP_TYPES, (
                f"relationship rel_type {rel.rel_type!r} on actor "
                f"{actor.id!r} not in RELATIONSHIP_TYPES"
            )

    # Every edge in every underlying graph must also reference real ids.
    for graph_name, graph in (
        ("labor_market", world.networks.labor_market),
        ("supply_chain", world.networks.supply_chain),
        ("interbank", world.networks.interbank),
    ):
        for src, tgt, _w in graph.edges:
            assert src in actor_ids, (
                f"{graph_name}: edge source {src!r} not in world actor set"
            )
            assert tgt in actor_ids, (
                f"{graph_name}: edge target {tgt!r} not in world actor set"
            )


# ---------- Property 4: Actor Behavioral Completeness ----------------------
# Feature: clyde-economic-simulator, Property 4: Actor Behavioral Completeness


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(cfg=shock_configs(), seed=st.integers(min_value=0, max_value=2**16))
def test_property_4_actor_behavioral_completeness(cfg: ShockConfig, seed: int) -> None:
    factory = _make_factory(seed)
    # If the factory raises, the test should fail; not re-wrap.
    world = factory.build_world(cfg, PriorLibrary())

    expected_total = sum(cfg.agent_counts.values())
    assert len(world.actors) == expected_total

    for actor in world.actors:
        assert isinstance(actor, Actor)
        required = REQUIRED_PARAM_FIELDS[actor.actor_type]
        assert required, f"no required fields registered for {actor.actor_type}"
        for name in required:
            value = getattr(actor.params, name, None)
            assert value is not None, (
                f"actor {actor.id!r} ({actor.actor_type.value}) "
                f"missing required param {name!r}"
            )


# ---------- Property 6: Override Application -------------------------------
# Feature: clyde-economic-simulator, Property 6: Override Application


# Candidate per-type overrides. Values are kept in plausible numeric ranges
# so the Actor post-init can't object to them; the point of the test is
# whether overrides were applied and recorded, not whether they are sensible.
_TYPE_LEVEL_CANDIDATES: dict[str, st.SearchStrategy[Any]] = {
    "household.mpc": st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    "firm.hurdle_rate": st.floats(min_value=0.01, max_value=0.40, allow_nan=False, allow_infinity=False),
    "bank.risk_appetite": st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    "central_bank.neutral_rate": st.floats(min_value=0.0, max_value=0.10, allow_nan=False, allow_infinity=False),
}


def _type_level_override_sets() -> st.SearchStrategy[dict[str, Any]]:
    keys = st.lists(
        st.sampled_from(sorted(_TYPE_LEVEL_CANDIDATES)),
        min_size=1,
        max_size=len(_TYPE_LEVEL_CANDIDATES),
        unique=True,
    )

    def _build(selected: list[str]) -> st.SearchStrategy[dict[str, Any]]:
        return st.fixed_dictionaries(
            {k: _TYPE_LEVEL_CANDIDATES[k] for k in selected}
        )

    return keys.flatmap(_build)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(
    # Guarantee at least 1 of each type so type-level overrides definitely
    # touch a real actor.
    cfg=shock_configs(
        min_households=1, max_households=4,
        min_firms=1, max_firms=3,
        min_banks=1, max_banks=3,
        min_central_banks=1, max_central_banks=1,
    ),
    overrides=_type_level_override_sets(),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_6_type_level_overrides_applied(
    cfg: ShockConfig,
    overrides: dict[str, Any],
    seed: int,
) -> None:
    factory = _make_factory(seed)
    scenario = factory.build_scenario(
        scenario_id="test-scen",
        description="override application property test",
        shock_config=cfg,
        prior_library=PriorLibrary(),
        param_overrides=overrides,
    )

    assert isinstance(scenario, Scenario)

    # The Scenario must record exactly the overrides that were applied.
    assert set(scenario.overrides.keys()) == set(overrides.keys())
    for key, expected_value in overrides.items():
        entry = scenario.overrides[key]
        assert entry["value"] == expected_value
        assert isinstance(entry["source"], str) and entry["source"], (
            f"override {key!r} missing non-empty source"
        )
        # Because overrides is param_overrides here, source should be that:
        assert entry["source"] == "param_overrides"

    # Every actor of the targeted type must show the overridden value.
    for key, expected_value in overrides.items():
        actor_type_str, _, param_name = key.partition(".")
        atype = ActorType(actor_type_str)
        matching = [a for a in scenario.actors if a.actor_type == atype]
        assert matching, f"no actors of type {atype.value} to check"
        for a in matching:
            assert getattr(a.params, param_name) == expected_value, (
                f"actor {a.id!r} did not pick up override {key}={expected_value}"
            )


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(
    cfg=shock_configs(
        min_households=2, max_households=4,
        min_firms=1, max_firms=2,
        min_banks=1, max_banks=2,
        min_central_banks=0, max_central_banks=1,
    ),
    type_mpc=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
    actor_mpc=st.floats(min_value=0.6, max_value=0.95, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_6_per_actor_override_beats_per_type(
    cfg: ShockConfig,
    type_mpc: float,
    actor_mpc: float,
    seed: int,
) -> None:
    factory = _make_factory(seed)
    # Target the first household deterministically — agent_counts is >= 2.
    target_actor_id = "household_0000"
    overrides = {
        "household.mpc": type_mpc,
        f"{target_actor_id}.mpc": actor_mpc,
    }
    scenario = factory.build_scenario(
        scenario_id="precedence-test",
        description="per-actor beats per-type",
        shock_config=cfg,
        prior_library=PriorLibrary(),
        param_overrides=overrides,
    )

    # Targeted actor should have the per-id value; other households should
    # have the per-type value.
    by_id = {a.id: a for a in scenario.actors}
    assert by_id[target_actor_id].params.mpc == actor_mpc
    for a in scenario.actors:
        if a.actor_type != ActorType.HOUSEHOLD or a.id == target_actor_id:
            continue
        assert a.params.mpc == type_mpc, (
            f"household {a.id!r} should have inherited type-level mpc override"
        )

    # Both keys recorded in Scenario.overrides with non-empty source.
    assert f"{target_actor_id}.mpc" in scenario.overrides
    assert "household.mpc" in scenario.overrides
    for entry in scenario.overrides.values():
        assert entry["source"]


def test_property_6_behavioral_overrides_vs_explicit_precedence() -> None:
    """Explicit ``param_overrides`` beats ``shock_config.behavioral_overrides``.

    Both keys are recorded, but the explicit value wins and the recorded
    source is ``"param_overrides"``.
    """
    cfg = ShockConfig(
        shock_type="test",
        severity=0.5,
        scope="macro",
        duration_steps=1,
        agent_counts={"household": 2, "firm": 1, "bank": 1, "central_bank": 1},
        behavioral_overrides={"household.mpc": 0.20},
        time_horizon=TimeHorizon(steps=1, step_unit="day"),
    )
    factory = _make_factory(seed=7)
    scenario = factory.build_scenario(
        scenario_id="precedence",
        description="explicit wins",
        shock_config=cfg,
        prior_library=PriorLibrary(),
        param_overrides={"household.mpc": 0.80},
    )
    for a in scenario.actors:
        if a.actor_type == ActorType.HOUSEHOLD:
            assert a.params.mpc == 0.80
    entry = scenario.overrides["household.mpc"]
    assert entry["value"] == 0.80
    assert entry["source"] == "param_overrides"


def test_property_6_behavioral_overrides_source_recorded() -> None:
    """An override supplied only via behavioral_overrides is recorded with
    ``source == 'behavioral_overrides'``."""
    cfg = ShockConfig(
        shock_type="test",
        severity=0.1,
        scope="macro",
        duration_steps=1,
        agent_counts={"household": 1, "firm": 1, "bank": 1, "central_bank": 1},
        behavioral_overrides={"firm.hurdle_rate": 0.33},
        time_horizon=TimeHorizon(steps=1, step_unit="day"),
    )
    factory = _make_factory(seed=3)
    scenario = factory.build_scenario(
        scenario_id="src-check",
        description="source tracking",
        shock_config=cfg,
        prior_library=PriorLibrary(),
    )
    for a in scenario.actors:
        if a.actor_type == ActorType.FIRM:
            assert a.params.hurdle_rate == 0.33
    entry = scenario.overrides["firm.hurdle_rate"]
    assert entry["value"] == 0.33
    assert entry["source"] == "behavioral_overrides"


# ---------- Error-path example tests ---------------------------------------


def test_unknown_param_key_raises_value_error() -> None:
    cfg = ShockConfig(
        shock_type="test",
        severity=0.0,
        scope="macro",
        duration_steps=1,
        agent_counts={"household": 1, "firm": 1, "bank": 1, "central_bank": 1},
        time_horizon=TimeHorizon(steps=1, step_unit="day"),
    )
    factory = _make_factory(seed=0)
    with pytest.raises(ValueError, match="unknown param"):
        factory.build_world(
            cfg,
            PriorLibrary(),
            param_overrides={"household.not_a_real_param": 0.5},
        )


def test_unknown_actor_type_target_raises_value_error() -> None:
    cfg = ShockConfig(
        shock_type="test",
        severity=0.0,
        scope="macro",
        duration_steps=1,
        agent_counts={"household": 1, "firm": 1, "bank": 1, "central_bank": 1},
        time_horizon=TimeHorizon(steps=1, step_unit="day"),
    )
    factory = _make_factory(seed=0)
    with pytest.raises(ValueError, match="unknown override target"):
        factory.build_world(
            cfg,
            PriorLibrary(),
            param_overrides={"alien.mpc": 0.5},
        )


def test_unknown_actor_id_target_raises_value_error() -> None:
    cfg = ShockConfig(
        shock_type="test",
        severity=0.0,
        scope="macro",
        duration_steps=1,
        agent_counts={"household": 1, "firm": 1, "bank": 1, "central_bank": 1},
        time_horizon=TimeHorizon(steps=1, step_unit="day"),
    )
    factory = _make_factory(seed=0)
    with pytest.raises(ValueError, match="unknown override target"):
        factory.build_world(
            cfg,
            PriorLibrary(),
            param_overrides={"household_9999.mpc": 0.5},
        )


def test_malformed_override_key_raises_value_error() -> None:
    cfg = ShockConfig(
        shock_type="test",
        severity=0.0,
        scope="macro",
        duration_steps=1,
        agent_counts={"household": 1, "firm": 1, "bank": 1, "central_bank": 1},
        time_horizon=TimeHorizon(steps=1, step_unit="day"),
    )
    factory = _make_factory(seed=0)
    with pytest.raises(ValueError, match="actor_type"):
        factory.build_world(
            cfg,
            PriorLibrary(),
            param_overrides={"no_dot_key": 0.5},
        )
