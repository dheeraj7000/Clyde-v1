"""SQLite persistence layer for Clyde."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from clyde.models.causal import CausalEvent
from clyde.models.config import ShockConfig, ShockDelta
from clyde.models.metrics import StepMetrics


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        branch_id TEXT,
        seed INTEGER NOT NULL,
        config_json TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS step_metrics (
        run_id TEXT NOT NULL,
        step INTEGER NOT NULL,
        gdp_index REAL,
        inflation_rate REAL,
        unemployment_rate REAL,
        gini_coefficient REAL,
        credit_tightening_index REAL,
        firm_bankruptcy_count INTEGER,
        bank_stress_index REAL,
        consumer_confidence REAL,
        interbank_freeze INTEGER,
        custom_metrics_json TEXT,
        PRIMARY KEY (run_id, step),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS causal_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        step INTEGER NOT NULL,
        source_actor_id TEXT NOT NULL,
        target_actor_id TEXT NOT NULL,
        channel TEXT NOT NULL,
        variable_affected TEXT NOT NULL,
        magnitude REAL NOT NULL,
        description TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS branches (
        branch_id TEXT PRIMARY KEY,
        parent_scenario_id TEXT NOT NULL,
        shock_delta_json TEXT NOT NULL,
        merged_config_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        historical_event TEXT NOT NULL,
        actual_outcome_json TEXT NOT NULL,
        simulated_distribution_json TEXT NOT NULL,
        accuracy_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
)


class SimulationDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        for stmt in _SCHEMA_STATEMENTS:
            self._conn.execute(stmt)
        self._conn.commit()

    def __enter__(self) -> "SimulationDB":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def insert_run(
        self,
        run_id: str,
        scenario_id: str,
        seed: int,
        config: ShockConfig,
        branch_id: str | None = None,
        status: str = "running",
    ) -> None:
        self._conn.execute(
            "INSERT INTO runs (run_id, scenario_id, branch_id, seed, config_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, scenario_id, branch_id, seed, json.dumps(config.to_dict()), status),
        )
        self._conn.commit()

    def update_run_status(self, run_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ? WHERE run_id = ?",
            (status, run_id),
        )
        self._conn.commit()

    def insert_step_metrics(self, run_id: str, metrics: StepMetrics) -> None:
        self._conn.execute(
            "INSERT INTO step_metrics (run_id, step, gdp_index, inflation_rate, "
            "unemployment_rate, gini_coefficient, credit_tightening_index, "
            "firm_bankruptcy_count, bank_stress_index, consumer_confidence, "
            "interbank_freeze, custom_metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._step_metrics_row(run_id, metrics),
        )
        self._conn.commit()

    def insert_step_metrics_batch(
        self, run_id: str, metrics_list: list[StepMetrics]
    ) -> None:
        rows = [self._step_metrics_row(run_id, m) for m in metrics_list]
        self._conn.executemany(
            "INSERT INTO step_metrics (run_id, step, gdp_index, inflation_rate, "
            "unemployment_rate, gini_coefficient, credit_tightening_index, "
            "firm_bankruptcy_count, bank_stress_index, consumer_confidence, "
            "interbank_freeze, custom_metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    @staticmethod
    def _step_metrics_row(run_id: str, m: StepMetrics) -> tuple:
        return (
            run_id,
            m.step,
            m.gdp_index,
            m.inflation_rate,
            m.unemployment_rate,
            m.gini_coefficient,
            m.credit_tightening_index,
            m.firm_bankruptcy_count,
            m.bank_stress_index,
            m.consumer_confidence,
            1 if m.interbank_freeze else 0,
            json.dumps(m.custom_metrics),
        )

    def insert_causal_event(self, run_id: str, event: CausalEvent) -> int:
        cur = self._conn.execute(
            "INSERT INTO causal_events (run_id, step, source_actor_id, target_actor_id, "
            "channel, variable_affected, magnitude, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                event.step,
                event.source_actor_id,
                event.target_actor_id,
                event.channel,
                event.variable_affected,
                event.magnitude,
                event.description,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def insert_branch(
        self,
        branch_id: str,
        parent_scenario_id: str,
        shock_delta: ShockDelta,
        merged_config: ShockConfig,
    ) -> None:
        self._conn.execute(
            "INSERT INTO branches (branch_id, parent_scenario_id, shock_delta_json, "
            "merged_config_json) VALUES (?, ?, ?, ?)",
            (
                branch_id,
                parent_scenario_id,
                json.dumps(shock_delta.to_dict()),
                json.dumps(merged_config.to_dict()),
            ),
        )
        self._conn.commit()

    def insert_backtest_result(
        self,
        backtest_id: str,
        scenario_id: str,
        historical_event: str,
        actual_outcome: dict,
        simulated_distribution: dict,
        accuracy_score: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO backtest_results (backtest_id, scenario_id, historical_event, "
            "actual_outcome_json, simulated_distribution_json, accuracy_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                backtest_id,
                scenario_id,
                historical_event,
                json.dumps(actual_outcome),
                json.dumps(simulated_distribution),
                accuracy_score,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["config"] = ShockConfig.from_dict(json.loads(data["config_json"]))
        return data

    def get_step_metrics(self, run_id: str, step: int) -> StepMetrics | None:
        row = self._conn.execute(
            "SELECT * FROM step_metrics WHERE run_id = ? AND step = ?",
            (run_id, step),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_step_metrics(row)

    def get_trajectory(self, run_id: str) -> list[StepMetrics]:
        rows = self._conn.execute(
            "SELECT * FROM step_metrics WHERE run_id = ? ORDER BY step ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_step_metrics(r) for r in rows]

    @staticmethod
    def _row_to_step_metrics(row: sqlite3.Row) -> StepMetrics:
        custom = json.loads(row["custom_metrics_json"]) if row["custom_metrics_json"] else {}
        return StepMetrics(
            step=int(row["step"]),
            gdp_index=float(row["gdp_index"]),
            inflation_rate=float(row["inflation_rate"]),
            unemployment_rate=float(row["unemployment_rate"]),
            gini_coefficient=float(row["gini_coefficient"]),
            credit_tightening_index=float(row["credit_tightening_index"]),
            firm_bankruptcy_count=int(row["firm_bankruptcy_count"]),
            bank_stress_index=float(row["bank_stress_index"]),
            consumer_confidence=float(row["consumer_confidence"]),
            interbank_freeze=bool(row["interbank_freeze"]),
            custom_metrics={k: float(v) for k, v in custom.items()},
        )

    def get_causal_events(self, run_id: str) -> list[CausalEvent]:
        rows = self._conn.execute(
            "SELECT * FROM causal_events WHERE run_id = ? ORDER BY step ASC, event_id ASC",
            (run_id,),
        ).fetchall()
        return [
            CausalEvent(
                step=int(r["step"]),
                source_actor_id=r["source_actor_id"],
                target_actor_id=r["target_actor_id"],
                channel=r["channel"],
                variable_affected=r["variable_affected"],
                magnitude=float(r["magnitude"]),
                description=r["description"] or "",
            )
            for r in rows
        ]

    def get_branch(self, branch_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM branches WHERE branch_id = ?", (branch_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["shock_delta"] = ShockDelta.from_dict(json.loads(data["shock_delta_json"]))
        data["merged_config"] = ShockConfig.from_dict(json.loads(data["merged_config_json"]))
        return data

    def get_backtest_results(self, scenario_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM backtest_results WHERE scenario_id = ? ORDER BY created_at ASC, backtest_id ASC",
            (scenario_id,),
        ).fetchall()
        results: list[dict] = []
        for r in rows:
            d = dict(r)
            d["actual_outcome"] = json.loads(d["actual_outcome_json"])
            d["simulated_distribution"] = json.loads(d["simulated_distribution_json"])
            results.append(d)
        return results
