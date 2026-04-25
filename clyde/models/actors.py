"""Actor, Relationship, and per-type state/params dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

from clyde.models.enums import RELATIONSHIP_TYPES, ActorType


@dataclass
class Relationship:
    source_id: str
    target_id: str
    rel_type: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.rel_type not in RELATIONSHIP_TYPES:
            raise ValueError(
                f"Relationship.rel_type must be one of {sorted(RELATIONSHIP_TYPES)}, got {self.rel_type!r}"
            )

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            rel_type=data["rel_type"],
            weight=float(data.get("weight", 1.0)),
        )


# -- Household ----------------------------------------------------------------


@dataclass
class HouseholdState:
    income: float = 0.0
    savings: float = 0.0
    consumption: float = 0.0
    employed: bool = False
    employer_id: str | None = None
    debt: float = 0.0
    inflation_expectation: float = 0.0
    confidence: float = 0.5


@dataclass
class HouseholdParams:
    mpc: float
    precautionary_savings_rate: float
    unemployment_fear_threshold: float
    wage_demand_elasticity: float
    inflation_expectation_prior: float
    inflation_expectation_lr: float
    credit_seek_threshold: float


# -- Firm ---------------------------------------------------------------------


@dataclass
class FirmState:
    revenue: float = 0.0
    costs: float = 0.0
    inventory: float = 0.0
    employees: list[str] = field(default_factory=list)
    price_level: float = 1.0
    investment: float = 0.0
    debt: float = 0.0
    demand_pressure: float = 0.0
    is_bankrupt: bool = False
    suppliers: list[str] = field(default_factory=list)


@dataclass
class FirmParams:
    hurdle_rate: float
    hiring_elasticity: float
    firing_threshold: float
    cost_push_weight: float
    demand_pull_weight: float
    supplier_switch_stress: float
    bankruptcy_threshold: float
    investment_sensitivity: float


# -- Bank ---------------------------------------------------------------------


@dataclass
class BankState:
    reserves: float = 0.0
    loans_outstanding: float = 0.0
    npl_ratio: float = 0.0
    credit_tightness: float = 0.0
    interbank_borrowed: float = 0.0
    is_stressed: bool = False


@dataclass
class BankParams:
    npl_tightening_elasticity: float
    herding_weight: float
    reserve_threshold: float
    credit_approval_floor: float
    risk_appetite: float


# -- Central Bank -------------------------------------------------------------


@dataclass
class CentralBankState:
    policy_rate: float = 0.0
    inflation_target: float = 0.02
    output_gap_estimate: float = 0.0


@dataclass
class CentralBankParams:
    taylor_inflation_weight: float
    taylor_output_weight: float
    rate_increment: float
    discretionary_band: float
    neutral_rate: float


# Required param field names per actor type — used by property tests and the
# EconomicWorldFactory to confirm that every actor is fully populated at
# construction time (Property 4: Actor Behavioral Completeness).
REQUIRED_PARAM_FIELDS: dict[ActorType, tuple[str, ...]] = {
    ActorType.HOUSEHOLD: tuple(f.name for f in fields(HouseholdParams)),
    ActorType.FIRM: tuple(f.name for f in fields(FirmParams)),
    ActorType.BANK: tuple(f.name for f in fields(BankParams)),
    ActorType.CENTRAL_BANK: tuple(f.name for f in fields(CentralBankParams)),
}

PARAMS_CLASS_BY_TYPE: dict[ActorType, type] = {
    ActorType.HOUSEHOLD: HouseholdParams,
    ActorType.FIRM: FirmParams,
    ActorType.BANK: BankParams,
    ActorType.CENTRAL_BANK: CentralBankParams,
}


# -- Actor (generic) ----------------------------------------------------------


@dataclass
class Actor:
    """Generic actor with type-specific state dict and typed params object."""

    id: str
    actor_type: ActorType
    params: Any  # One of HouseholdParams | FirmParams | BankParams | CentralBankParams
    state: dict[str, float] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.actor_type, str):
            self.actor_type = ActorType(self.actor_type)
        required = REQUIRED_PARAM_FIELDS[self.actor_type]
        params_cls = PARAMS_CLASS_BY_TYPE[self.actor_type]
        if not isinstance(self.params, params_cls):
            raise TypeError(
                f"Actor params must be an instance of {params_cls.__name__} for actor_type "
                f"{self.actor_type.value}, got {type(self.params).__name__}"
            )
        for name in required:
            value = getattr(self.params, name, None)
            if value is None:
                raise ValueError(
                    f"Actor {self.id!r} ({self.actor_type.value}) missing required param {name!r}"
                )

    def to_dict(self) -> dict:
        assert is_dataclass(self.params)
        return {
            "id": self.id,
            "actor_type": self.actor_type.value,
            "params": asdict(self.params),
            "state": dict(self.state),
            "relationships": [r.to_dict() for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Actor":
        actor_type = ActorType(data["actor_type"])
        params_cls = PARAMS_CLASS_BY_TYPE[actor_type]
        params = params_cls(**data["params"])
        return cls(
            id=data["id"],
            actor_type=actor_type,
            params=params,
            state={k: float(v) if isinstance(v, (int, float)) else v for k, v in data.get("state", {}).items()},
            relationships=[Relationship.from_dict(r) for r in data.get("relationships", [])],
        )
