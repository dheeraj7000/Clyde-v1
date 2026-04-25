"""Unit tests for Scenario.pretty_print()."""

from __future__ import annotations

from clyde.models import (
    Actor,
    ActorType,
    BipartiteGraph,
    DirectedGraph,
    FirmParams,
    HistoricalAnalog,
    HouseholdParams,
    NetworkBundle,
    ScaleFreeGraph,
    Scenario,
    ShockConfig,
    TimeHorizon,
)


def _minimal_config(**overrides) -> ShockConfig:
    defaults = dict(
        shock_type="oil_price_spike",
        severity=0.42,
        scope="macro",
        duration_steps=12,
        time_horizon=TimeHorizon(steps=24, step_unit="week"),
    )
    defaults.update(overrides)
    return ShockConfig(**defaults)


def _household(actor_id: str = "h1") -> Actor:
    return Actor(
        id=actor_id,
        actor_type=ActorType.HOUSEHOLD,
        params=HouseholdParams(
            mpc=0.6,
            precautionary_savings_rate=0.1,
            unemployment_fear_threshold=0.2,
            wage_demand_elasticity=0.3,
            inflation_expectation_prior=0.02,
            inflation_expectation_lr=0.1,
            credit_seek_threshold=0.4,
        ),
    )


def _firm(actor_id: str = "f1") -> Actor:
    return Actor(
        id=actor_id,
        actor_type=ActorType.FIRM,
        params=FirmParams(
            hurdle_rate=0.08,
            hiring_elasticity=0.5,
            firing_threshold=0.2,
            cost_push_weight=0.4,
            demand_pull_weight=0.4,
            supplier_switch_stress=0.3,
            bankruptcy_threshold=0.1,
            investment_sensitivity=0.25,
        ),
    )


def test_pretty_print_minimal_scenario_contains_key_fragments() -> None:
    scenario = Scenario(
        scenario_id="scn-001",
        description="a minimal test scenario",
        config=_minimal_config(),
    )
    output = scenario.pretty_print()

    assert isinstance(output, str)
    assert output  # non-empty
    # Multiple lines
    assert output.count("\n") >= 5

    # Expected key fragments
    assert "scn-001" in output
    assert "Shock" in output
    assert "oil_price_spike" in output  # shock_type
    assert "Time horizon" in output
    assert "macro" in output  # scope
    assert "week" in output  # step unit


def test_pretty_print_returns_string_and_multiline() -> None:
    scenario = Scenario(
        scenario_id="s",
        description="d",
        config=_minimal_config(),
    )
    output = scenario.pretty_print()
    assert isinstance(output, str)
    lines = output.splitlines()
    assert len(lines) > 1
    # First line should mention the scenario header.
    assert lines[0].startswith("Scenario:")


def test_pretty_print_rich_scenario_includes_all_sections() -> None:
    analog = HistoricalAnalog(
        event_name="1973_oil_crisis",
        year=1973,
        similarity_score=0.87,
        source="BIS",
    )
    config = _minimal_config(
        geography=["US", "EU"],
        sectors=["energy", "transport"],
        initial_contact_actors=["f1"],
        agent_counts={"household": 100, "firm": 10},
        historical_analogs=[analog],
    )
    scenario = Scenario(
        scenario_id="scn-rich",
        description="rich scenario",
        config=config,
        actors=[_household("h1"), _firm("f1")],
        networks=NetworkBundle(
            labor_market=BipartiteGraph(edges=[("h1", "f1", 1.0)]),
            supply_chain=DirectedGraph(edges=[("f1", "h1", 0.5)]),
            interbank=ScaleFreeGraph(edges=[]),
        ),
        prior_library_version="v0.1.0",
        overrides={"liquidity_buffer": 0.15},
        metadata={"author": "dheeraj"},
    )

    output = scenario.pretty_print()

    # Geography and sectors
    assert "US" in output and "EU" in output
    assert "energy" in output and "transport" in output

    # Initial contact
    assert "initial_contact_actors" in output
    assert "f1" in output

    # Agent counts
    assert "Agent counts" in output
    assert "household" in output
    assert "firm" in output

    # Actors count line
    assert "Actors: 2" in output

    # Networks line with edge counts
    assert "labor_market=1" in output
    assert "supply_chain=1" in output
    assert "interbank=0" in output

    # Prior library version shown (not "(unset)")
    assert "v0.1.0" in output
    assert "(unset)" not in output

    # Historical analogs
    assert "Historical analogs" in output
    assert "1973_oil_crisis" in output
    assert "1973" in output

    # Overrides
    assert "Overrides" in output
    assert "liquidity_buffer" in output

    # Metadata
    assert "Metadata" in output
    assert "author" in output
    assert "dheeraj" in output
