"""Core domain dataclasses for Clyde."""

from clyde.models.actors import (
    PARAMS_CLASS_BY_TYPE,
    REQUIRED_PARAM_FIELDS,
    Actor,
    BankParams,
    BankState,
    CentralBankParams,
    CentralBankState,
    FirmParams,
    FirmState,
    HouseholdParams,
    HouseholdState,
    Relationship,
)
from clyde.models.causal import CausalChain, CausalEvent
from clyde.models.config import VALID_SCOPES, ShockConfig, ShockDelta, SimulationWorld
from clyde.models.enums import RELATIONSHIP_TYPES, ActorType
from clyde.models.input import ActorHint, Ambiguity, Document, ParseResult, ShockParams
from clyde.models.metrics import (
    CORE_METRIC_NAMES,
    EnsembleResult,
    PathBundle,
    StepMetrics,
    TrajectoryResult,
)
from clyde.models.networks import BipartiteGraph, DirectedGraph, NetworkBundle, ScaleFreeGraph
from clyde.models.reporting import Citation, DivergenceMap, DivergenceVariable, HistoricalAnalog
from clyde.models.scenario import Scenario
from clyde.models.time import VALID_STEP_UNITS, TimeHorizon

__all__ = [
    # Enums & constants
    "ActorType",
    "RELATIONSHIP_TYPES",
    "VALID_SCOPES",
    "VALID_STEP_UNITS",
    "CORE_METRIC_NAMES",
    "REQUIRED_PARAM_FIELDS",
    "PARAMS_CLASS_BY_TYPE",
    # Actors
    "Actor",
    "Relationship",
    "HouseholdState",
    "HouseholdParams",
    "FirmState",
    "FirmParams",
    "BankState",
    "BankParams",
    "CentralBankState",
    "CentralBankParams",
    # Networks
    "BipartiteGraph",
    "DirectedGraph",
    "ScaleFreeGraph",
    "NetworkBundle",
    # Config & world
    "ShockConfig",
    "ShockDelta",
    "SimulationWorld",
    "TimeHorizon",
    # Metrics & results
    "StepMetrics",
    "TrajectoryResult",
    "EnsembleResult",
    "PathBundle",
    # Causal
    "CausalEvent",
    "CausalChain",
    # Scenario
    "Scenario",
    # Reporting
    "Citation",
    "HistoricalAnalog",
    "DivergenceMap",
    "DivergenceVariable",
    # Input
    "ParseResult",
    "Ambiguity",
    "Document",
    "ActorHint",
    "ShockParams",
]
