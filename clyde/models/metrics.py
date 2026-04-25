"""Metrics and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

CORE_METRIC_NAMES: tuple[str, ...] = (
    "gdp_index",
    "inflation_rate",
    "unemployment_rate",
    "gini_coefficient",
    "credit_tightening_index",
    "firm_bankruptcy_count",
    "bank_stress_index",
    "consumer_confidence",
    "interbank_freeze",
)


@dataclass
class StepMetrics:
    step: int
    gdp_index: float
    inflation_rate: float
    unemployment_rate: float
    gini_coefficient: float
    credit_tightening_index: float
    firm_bankruptcy_count: int
    bank_stress_index: float
    consumer_confidence: float
    interbank_freeze: bool
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "gdp_index": self.gdp_index,
            "inflation_rate": self.inflation_rate,
            "unemployment_rate": self.unemployment_rate,
            "gini_coefficient": self.gini_coefficient,
            "credit_tightening_index": self.credit_tightening_index,
            "firm_bankruptcy_count": self.firm_bankruptcy_count,
            "bank_stress_index": self.bank_stress_index,
            "consumer_confidence": self.consumer_confidence,
            "interbank_freeze": self.interbank_freeze,
            "custom_metrics": {k: float(v) for k, v in self.custom_metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepMetrics":
        return cls(
            step=int(data["step"]),
            gdp_index=float(data["gdp_index"]),
            inflation_rate=float(data["inflation_rate"]),
            unemployment_rate=float(data["unemployment_rate"]),
            gini_coefficient=float(data["gini_coefficient"]),
            credit_tightening_index=float(data["credit_tightening_index"]),
            firm_bankruptcy_count=int(data["firm_bankruptcy_count"]),
            bank_stress_index=float(data["bank_stress_index"]),
            consumer_confidence=float(data["consumer_confidence"]),
            interbank_freeze=bool(data["interbank_freeze"]),
            custom_metrics={k: float(v) for k, v in data.get("custom_metrics", {}).items()},
        )


@dataclass
class TrajectoryResult:
    run_id: str
    seed: int
    steps: list[StepMetrics] = field(default_factory=list)
    causal_events: list = field(default_factory=list)  # list[CausalEvent] — forward-ref via duck-typing
    final_state_ref: str | None = None  # optional pointer; we don't serialize full world here


@dataclass
class EnsembleResult:
    scenario_id: str
    config: object  # ShockConfig — kept generic to avoid a circular import
    trajectories: list[TrajectoryResult] = field(default_factory=list)
    run_count: int = 0
    ensemble_seed: int = 0


@dataclass
class PathBundle:
    central: list[StepMetrics] = field(default_factory=list)
    optimistic: list[StepMetrics] = field(default_factory=list)
    pessimistic: list[StepMetrics] = field(default_factory=list)
    tail_upper: list[StepMetrics] = field(default_factory=list)
    tail_lower: list[StepMetrics] = field(default_factory=list)
