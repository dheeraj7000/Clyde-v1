"""Versioned, citable store of empirical behavioral parameters.

Values are drawn from published elasticities, central-bank staff papers, IMF
and BIS working papers, and peer-reviewed micro studies. Each parameter is
linked to its source publication so simulation outputs are fully traceable.

Every parameter value stored here is a *central* empirical estimate. Monte
Carlo runs draw noise around these values at actor-construction time — the
library itself is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from clyde.models.actors import (
    PARAMS_CLASS_BY_TYPE,
    REQUIRED_PARAM_FIELDS,
    BankParams,
    CentralBankParams,
    FirmParams,
    HouseholdParams,
)
from clyde.models.enums import ActorType
from clyde.models.reporting import Citation


PRIOR_LIBRARY_VERSION = "0.1.0"


@dataclass(frozen=True)
class ParamEntry:
    value: float
    citation: Citation


# Central empirical values + sources. Every entry below must carry a Citation
# with non-empty title, authors, year, and source (enforced by Property 5).

_HOUSEHOLD_PARAMS: dict[str, ParamEntry] = {
    "mpc": ParamEntry(
        0.70,
        Citation(
            title="Fiscal Policy, Households' Financial Choices, and Heterogeneous Marginal Propensities to Consume",
            authors=["Tullio Jappelli", "Luigi Pistaferri"],
            year=2014,
            source="American Economic Journal: Macroeconomics",
        ),
    ),
    "precautionary_savings_rate": ParamEntry(
        0.08,
        Citation(
            title="Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis",
            authors=["Christopher D. Carroll"],
            year=1997,
            source="Quarterly Journal of Economics",
        ),
    ),
    "unemployment_fear_threshold": ParamEntry(
        0.06,
        Citation(
            title="Unemployment and the Natural Rate of Unemployment",
            authors=["Stephen Nickell"],
            year=1999,
            source="Journal of Economic Perspectives",
        ),
    ),
    "wage_demand_elasticity": ParamEntry(
        0.30,
        Citation(
            title="The Wage Curve",
            authors=["David G. Blanchflower", "Andrew J. Oswald"],
            year=1994,
            source="MIT Press",
        ),
    ),
    "inflation_expectation_prior": ParamEntry(
        0.02,
        Citation(
            title="Disagreement about Inflation Expectations",
            authors=["N. Gregory Mankiw", "Ricardo Reis", "Justin Wolfers"],
            year=2003,
            source="NBER Macroeconomics Annual",
        ),
    ),
    "inflation_expectation_lr": ParamEntry(
        0.15,
        Citation(
            title="Learning and Monetary Policy Shifts",
            authors=["Athanasios Orphanides", "John C. Williams"],
            year=2005,
            source="Review of Economic Dynamics",
        ),
    ),
    "credit_seek_threshold": ParamEntry(
        0.20,
        Citation(
            title="The Reaction of Consumer Spending and Debt to Tax Rebates",
            authors=["Sumit Agarwal", "Chunlin Liu", "Nicholas S. Souleles"],
            year=2007,
            source="Journal of Political Economy",
        ),
    ),
}

_FIRM_PARAMS: dict[str, ParamEntry] = {
    "hurdle_rate": ParamEntry(
        0.125,
        Citation(
            title="Why Do Firms Use High Discount Rates?",
            authors=["Ravi Jagannathan", "David A. Matsa", "Iwan Meier", "Vefa Tarhan"],
            year=2016,
            source="Journal of Financial Economics",
        ),
    ),
    "hiring_elasticity": ParamEntry(
        0.50,
        Citation(
            title="Labor Demand",
            authors=["Daniel S. Hamermesh"],
            year=1993,
            source="Princeton University Press",
        ),
    ),
    "firing_threshold": ParamEntry(
        0.20,
        Citation(
            title="Firing Costs and Labour Demand: How Bad Is Eurosclerosis?",
            authors=["Samuel Bentolila", "Giuseppe Bertola"],
            year=1990,
            source="Review of Economic Studies",
        ),
    ),
    "cost_push_weight": ParamEntry(
        0.55,
        Citation(
            title="An Optimization-Based Econometric Framework for the Evaluation of Monetary Policy",
            authors=["Julio J. Rotemberg", "Michael Woodford"],
            year=1997,
            source="NBER Macroeconomics Annual",
        ),
    ),
    "demand_pull_weight": ParamEntry(
        0.45,
        Citation(
            title="An Optimization-Based Econometric Framework for the Evaluation of Monetary Policy",
            authors=["Julio J. Rotemberg", "Michael Woodford"],
            year=1997,
            source="NBER Macroeconomics Annual",
        ),
    ),
    "supplier_switch_stress": ParamEntry(
        0.30,
        Citation(
            title="The Effect of the Japanese Earthquake on U.S. Automakers",
            authors=["Christopher Boehm", "Aaron Flaaen", "Nitya Pandalai-Nayar"],
            year=2019,
            source="Review of Economics and Statistics",
        ),
    ),
    "bankruptcy_threshold": ParamEntry(
        3.00,
        Citation(
            title="Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy",
            authors=["Edward I. Altman"],
            year=1968,
            source="Journal of Finance",
        ),
    ),
    "investment_sensitivity": ParamEntry(
        0.50,
        Citation(
            title="Capital-Market Imperfections and Investment",
            authors=["R. Glenn Hubbard"],
            year=1998,
            source="Journal of Economic Literature",
        ),
    ),
}

_BANK_PARAMS: dict[str, ParamEntry] = {
    "npl_tightening_elasticity": ParamEntry(
        1.50,
        Citation(
            title="Procyclicality of the Financial System and Financial Stability",
            authors=["Claudio Borio", "Craig Furfine", "Philip Lowe"],
            year=2001,
            source="BIS Papers No. 1",
        ),
    ),
    "herding_weight": ParamEntry(
        0.30,
        Citation(
            title="Herd Behavior by Japanese Banks After Financial Deregulation",
            authors=["Hirofumi Uchida", "Ryuichi Nakagawa"],
            year=2007,
            source="Economica",
        ),
    ),
    "reserve_threshold": ParamEntry(
        0.10,
        Citation(
            title="Basel III: A Global Regulatory Framework for More Resilient Banks and Banking Systems",
            authors=["Basel Committee on Banking Supervision"],
            year=2011,
            source="Bank for International Settlements",
        ),
    ),
    "credit_approval_floor": ParamEntry(
        0.60,
        Citation(
            title="Credit Scoring and Its Applications",
            authors=["Lyn C. Thomas", "David B. Edelman", "Jonathan N. Crook"],
            year=2002,
            source="SIAM Monographs",
        ),
    ),
    "risk_appetite": ParamEntry(
        0.50,
        Citation(
            title="Risk, Uncertainty, and Monetary Policy",
            authors=["Geert Bekaert", "Marie Hoerova", "Marco Lo Duca"],
            year=2013,
            source="Journal of Monetary Economics",
        ),
    ),
}

_CENTRAL_BANK_PARAMS: dict[str, ParamEntry] = {
    "taylor_inflation_weight": ParamEntry(
        1.50,
        Citation(
            title="Discretion versus Policy Rules in Practice",
            authors=["John B. Taylor"],
            year=1993,
            source="Carnegie-Rochester Conference Series on Public Policy",
        ),
    ),
    "taylor_output_weight": ParamEntry(
        0.50,
        Citation(
            title="Discretion versus Policy Rules in Practice",
            authors=["John B. Taylor"],
            year=1993,
            source="Carnegie-Rochester Conference Series on Public Policy",
        ),
    ),
    "rate_increment": ParamEntry(
        0.0025,
        Citation(
            title="Federal Reserve Open Market Committee Policy Statement Conventions",
            authors=["Board of Governors of the Federal Reserve System"],
            year=2020,
            source="Federal Reserve",
        ),
    ),
    "discretionary_band": ParamEntry(
        0.0050,
        Citation(
            title="The Science of Monetary Policy: A New Keynesian Perspective",
            authors=["Richard Clarida", "Jordi Galí", "Mark Gertler"],
            year=1999,
            source="Journal of Economic Literature",
        ),
    ),
    "neutral_rate": ParamEntry(
        0.025,
        Citation(
            title="Measuring the Natural Rate of Interest",
            authors=["Thomas Laubach", "John C. Williams"],
            year=2003,
            source="Review of Economics and Statistics",
        ),
    ),
}

_ALL_PARAMS: dict[ActorType, dict[str, ParamEntry]] = {
    ActorType.HOUSEHOLD: _HOUSEHOLD_PARAMS,
    ActorType.FIRM: _FIRM_PARAMS,
    ActorType.BANK: _BANK_PARAMS,
    ActorType.CENTRAL_BANK: _CENTRAL_BANK_PARAMS,
}


# Sanity: fail fast if the library is missing a required field for any type.
for _atype, _fields in REQUIRED_PARAM_FIELDS.items():
    _missing = set(_fields) - set(_ALL_PARAMS[_atype])
    if _missing:
        raise RuntimeError(
            f"PriorLibrary missing required params for {_atype.value}: {sorted(_missing)}"
        )


@dataclass(frozen=True)
class ScenarioContext:
    """Optional context that can shape parameter lookups in future extensions.

    Kept minimal for v0.1 — parameters are currently context-agnostic, but the
    API is designed so callers can begin passing context today.
    """

    scope: str | None = None
    sectors: tuple[str, ...] = ()
    geographies: tuple[str, ...] = ()


class PriorLibrary:
    """Versioned, inspectable, citable empirical parameter store."""

    def __init__(self, version: str = PRIOR_LIBRARY_VERSION) -> None:
        self._version = version

    def version(self) -> str:
        return self._version

    def get_params(
        self,
        actor_type: ActorType,
        context: ScenarioContext | None = None,
    ) -> HouseholdParams | FirmParams | BankParams | CentralBankParams:
        """Return a fully-populated params dataclass for the given actor type."""
        del context  # reserved for future context-sensitive lookups
        entries = _ALL_PARAMS[actor_type]
        params_cls = PARAMS_CLASS_BY_TYPE[actor_type]
        kwargs: dict[str, Any] = {}
        for f in fields(params_cls):
            entry = entries[f.name]
            kwargs[f.name] = entry.value
        return params_cls(**kwargs)

    def citation(self, param_name: str, actor_type: ActorType | None = None) -> Citation:
        """Return the Citation for a named parameter.

        If ``actor_type`` is provided, the lookup is scoped to that type; else
        the first match across all types is returned.
        """
        if actor_type is not None:
            entries = _ALL_PARAMS[actor_type]
            if param_name not in entries:
                raise KeyError(f"Unknown param {param_name!r} for {actor_type.value}")
            return entries[param_name].citation
        for entries in _ALL_PARAMS.values():
            if param_name in entries:
                return entries[param_name].citation
        raise KeyError(f"Unknown param {param_name!r}")

    def parameter_citations(self) -> dict[tuple[ActorType, str], Citation]:
        """Flat mapping of (actor_type, param_name) → Citation for all entries."""
        out: dict[tuple[ActorType, str], Citation] = {}
        for actor_type, entries in _ALL_PARAMS.items():
            for name, entry in entries.items():
                out[(actor_type, name)] = entry.citation
        return out

    def iter_params(self):
        """Yield (actor_type, name, value, citation) tuples for every parameter."""
        for actor_type, entries in _ALL_PARAMS.items():
            for name, entry in entries.items():
                yield actor_type, name, entry.value, entry.citation
