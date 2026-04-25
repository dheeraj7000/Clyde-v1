"""Unit tests for the SQLite persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from clyde.models.causal import CausalEvent
from clyde.models.config import ShockConfig, ShockDelta
from clyde.models.metrics import StepMetrics
from clyde.models.reporting import HistoricalAnalog
from clyde.models.time import TimeHorizon
from clyde.persistence.db import SimulationDB


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> SimulationDB:
    with SimulationDB(db_path) as instance:
        yield instance


def _sample_config() -> ShockConfig:
    return ShockConfig(
        shock_type="credit_crunch",
        severity=0.6,
        scope="macro",
        duration_steps=10,
        geography=["US", "EU"],
        sectors=["banking", "manufacturing"],
        initial_contact_actors=["bank_01"],
        agent_counts={"households": 100, "firms": 20, "banks": 5},
        behavioral_overrides={"risk_appetite": 0.3},
        time_horizon=TimeHorizon(steps=10, step_unit="week"),
        ensemble_seed=42,
        historical_analogs=[
            HistoricalAnalog(
                event_name="GFC 2008",
                year=2008,
                similarity_score=0.75,
                param_adjustments={"credit_tightening": 0.9},
                source="Reinhart & Rogoff",
            )
        ],
    )


def _sample_metrics(step: int, *, bankruptcies: int = 0, freeze: bool = False) -> StepMetrics:
    return StepMetrics(
        step=step,
        gdp_index=100.0 - step,
        inflation_rate=0.02 + step * 0.001,
        unemployment_rate=0.05 + step * 0.002,
        gini_coefficient=0.4,
        credit_tightening_index=0.1 * step,
        firm_bankruptcy_count=bankruptcies,
        bank_stress_index=0.2,
        consumer_confidence=0.8 - step * 0.01,
        interbank_freeze=freeze,
        custom_metrics={"shadow_rate": 0.5 + step * 0.01},
    )


def test_insert_and_get_run_roundtrip(db: SimulationDB) -> None:
    config = _sample_config()
    db.insert_run(
        run_id="run_1",
        scenario_id="scen_1",
        seed=123,
        config=config,
        branch_id="branch_a",
        status="running",
    )

    row = db.get_run("run_1")

    assert row is not None
    assert row["run_id"] == "run_1"
    assert row["scenario_id"] == "scen_1"
    assert row["branch_id"] == "branch_a"
    assert row["seed"] == 123
    assert row["status"] == "running"
    restored_config = row["config"]
    assert isinstance(restored_config, ShockConfig)
    assert restored_config == config


def test_get_run_missing_returns_none(db: SimulationDB) -> None:
    assert db.get_run("does_not_exist") is None


def test_update_run_status_transitions(db: SimulationDB) -> None:
    db.insert_run("run_1", "scen_1", 1, _sample_config())
    assert db.get_run("run_1")["status"] == "running"

    db.update_run_status("run_1", "completed")
    assert db.get_run("run_1")["status"] == "completed"

    db.update_run_status("run_1", "failed")
    assert db.get_run("run_1")["status"] == "failed"


def test_insert_step_metrics_and_get_trajectory_ordered(db: SimulationDB) -> None:
    db.insert_run("run_1", "scen_1", 1, _sample_config())

    m0 = _sample_metrics(0)
    m1 = _sample_metrics(1, bankruptcies=2)
    m2 = _sample_metrics(2, freeze=True)
    m3 = _sample_metrics(3)

    for m in (m2, m0, m3, m1):
        db.insert_step_metrics("run_1", m)

    single = db.get_step_metrics("run_1", 2)
    assert single == m2
    assert single.interbank_freeze is True

    trajectory = db.get_trajectory("run_1")
    assert [m.step for m in trajectory] == [0, 1, 2, 3]
    assert trajectory[1] == m1
    assert trajectory[3].custom_metrics == m3.custom_metrics


def test_get_step_metrics_missing_returns_none(db: SimulationDB) -> None:
    db.insert_run("run_1", "scen_1", 1, _sample_config())
    assert db.get_step_metrics("run_1", 99) is None


def test_insert_step_metrics_batch_writes_all(db: SimulationDB) -> None:
    db.insert_run("run_1", "scen_1", 1, _sample_config())

    batch = [_sample_metrics(step=i) for i in range(5)]
    db.insert_step_metrics_batch("run_1", batch)

    trajectory = db.get_trajectory("run_1")
    assert len(trajectory) == 5
    assert trajectory == batch


def test_insert_causal_event_returns_id_and_orders_correctly(db: SimulationDB) -> None:
    db.insert_run("run_1", "scen_1", 1, _sample_config())

    events = [
        CausalEvent(step=2, source_actor_id="a", target_actor_id="b", channel="credit",
                    variable_affected="loans", magnitude=0.5, description="later step"),
        CausalEvent(step=1, source_actor_id="c", target_actor_id="d", channel="trade",
                    variable_affected="output", magnitude=0.3, description="first"),
        CausalEvent(step=1, source_actor_id="e", target_actor_id="f", channel="trade",
                    variable_affected="output", magnitude=0.2, description="second"),
    ]

    ids = [db.insert_causal_event("run_1", ev) for ev in events]
    assert all(isinstance(i, int) for i in ids)
    assert ids == sorted(ids)
    assert len(set(ids)) == 3

    retrieved = db.get_causal_events("run_1")
    assert [(e.step, e.description) for e in retrieved] == [
        (1, "first"),
        (1, "second"),
        (2, "later step"),
    ]


def test_insert_and_get_branch_roundtrip(db: SimulationDB) -> None:
    shock_delta = ShockDelta(
        intervention_step=5,
        param_overrides={"policy_rate": 0.01, "qe_size": 500},
        new_events=["rate_cut"],
        description="Emergency rate cut",
    )
    merged_config = _sample_config()

    db.insert_branch("branch_a", "scen_1", shock_delta, merged_config)

    branch = db.get_branch("branch_a")
    assert branch is not None
    assert branch["branch_id"] == "branch_a"
    assert branch["parent_scenario_id"] == "scen_1"
    assert isinstance(branch["shock_delta"], ShockDelta)
    assert branch["shock_delta"] == shock_delta
    assert isinstance(branch["merged_config"], ShockConfig)
    assert branch["merged_config"] == merged_config


def test_get_branch_missing_returns_none(db: SimulationDB) -> None:
    assert db.get_branch("missing") is None


def test_insert_and_get_backtest_results_roundtrip(db: SimulationDB) -> None:
    actual = {"gdp_drop_pct": 4.2, "unemployment_peak": 0.1}
    simulated = {"gdp_drop_pct": {"mean": 4.0, "std": 0.5}, "unemployment_peak": {"mean": 0.09}}

    db.insert_backtest_result(
        backtest_id="bt_1",
        scenario_id="scen_1",
        historical_event="GFC 2008",
        actual_outcome=actual,
        simulated_distribution=simulated,
        accuracy_score=0.87,
    )
    db.insert_backtest_result(
        backtest_id="bt_2",
        scenario_id="scen_1",
        historical_event="Covid 2020",
        actual_outcome={"gdp_drop_pct": 9.0},
        simulated_distribution={"gdp_drop_pct": {"mean": 8.5}},
        accuracy_score=None,
    )
    db.insert_backtest_result(
        backtest_id="bt_3",
        scenario_id="other_scenario",
        historical_event="Other",
        actual_outcome={},
        simulated_distribution={},
    )

    results = db.get_backtest_results("scen_1")
    assert len(results) == 2
    ids = {r["backtest_id"] for r in results}
    assert ids == {"bt_1", "bt_2"}

    bt1 = next(r for r in results if r["backtest_id"] == "bt_1")
    assert bt1["historical_event"] == "GFC 2008"
    assert bt1["actual_outcome"] == actual
    assert bt1["simulated_distribution"] == simulated
    assert bt1["accuracy_score"] == 0.87

    bt2 = next(r for r in results if r["backtest_id"] == "bt_2")
    assert bt2["accuracy_score"] is None


def test_schema_idempotency_open_twice(db_path: Path) -> None:
    db1 = SimulationDB(db_path)
    db1.insert_run("run_1", "scen_1", 7, _sample_config())
    db1.close()

    db2 = SimulationDB(db_path)
    row = db2.get_run("run_1")
    assert row is not None
    assert row["seed"] == 7
    db2.close()


def test_memory_db_does_not_enable_wal() -> None:
    with SimulationDB(":memory:") as db:
        db.insert_run("run_1", "scen_1", 1, _sample_config())
        assert db.get_run("run_1") is not None
