"""Historical analog matching (Task 15.1, Requirements 10.1-10.3).

Given a ShockConfig (or a ParseResult) describing a novel scenario, find
structurally similar past events from a curated corpus and surface them as
HistoricalAnalog instances. The matcher informs parameter ranges (it does NOT
dictate outcomes — the analog disclosure makes the influence auditable).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Iterable

from clyde.models.config import ShockConfig
from clyde.models.input import ParseResult
from clyde.models.reporting import HistoricalAnalog


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalEvent:
    """Curated entry in the historical event corpus."""

    name: str
    year: int
    shock_type: str
    scope: str
    severity: float
    duration_months: int
    geographies: tuple[str, ...]
    sectors: tuple[str, ...]
    keywords: tuple[str, ...]
    canonical_outcomes: dict[str, float]
    param_adjustments: dict[str, float]
    source: str


@dataclass
class AnalogDisclosure:
    """How an analog influenced parameter ranges — auditable evidence trail."""

    analog: HistoricalAnalog
    rationale: str
    affected_params: list[str]
    shifted_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default corpus
# ---------------------------------------------------------------------------


_DEFAULT_CORPUS: list[HistoricalEvent] = [
    HistoricalEvent(
        name="Volcker Disinflation",
        year=1981,
        shock_type="rate_hike",
        scope="macro",
        severity=0.85,
        duration_months=24,
        geographies=("US",),
        sectors=("banking", "housing", "manufacturing"),
        keywords=(
            "interest", "rate", "monetary", "tightening", "inflation", "fed",
            "disinflation", "recession", "policy",
        ),
        canonical_outcomes={"gdp_drop_pct": 2.7, "unemployment_peak": 0.108},
        param_adjustments={
            "household.consumption_elasticity": -0.3,
            "firm.investment_elasticity": -0.5,
            "bank.loan_growth": -0.4,
            "monetary_transmission_lag": 0.6,
        },
        source="Goodfriend & King (2005), 'The Incredible Volcker Disinflation', JME.",
    ),
    HistoricalEvent(
        name="2008 Global Financial Crisis",
        year=2008,
        shock_type="banking_crisis",
        scope="macro",
        severity=0.95,
        duration_months=18,
        geographies=("US", "EU", "UK", "global"),
        sectors=("banking", "real_estate", "finance", "insurance"),
        keywords=(
            "lehman", "subprime", "mortgage", "banking", "crisis", "credit",
            "liquidity", "contagion", "deleveraging", "financial",
        ),
        canonical_outcomes={"gdp_drop_pct": 4.3, "unemployment_peak": 0.10},
        param_adjustments={
            "npl_tightening_elasticity": 0.7,
            "bank.loan_growth": -0.6,
            "household.consumption_elasticity": -0.4,
            "firm.investment_elasticity": -0.55,
            "interbank_contagion_strength": 0.5,
        },
        source="Bernanke (2018), 'The Real Effects of the Financial Crisis', BPEA.",
    ),
    HistoricalEvent(
        name="COVID-19 Pandemic Shock",
        year=2020,
        shock_type="supply_disruption",
        scope="macro",
        severity=0.90,
        duration_months=15,
        geographies=("global", "US", "EU", "China"),
        sectors=("services", "travel", "hospitality", "retail", "manufacturing", "healthcare"),
        keywords=(
            "pandemic", "covid", "lockdown", "supply", "shock", "shutdown",
            "service", "labor", "fiscal", "stimulus",
        ),
        canonical_outcomes={"gdp_drop_pct": 8.0, "unemployment_peak": 0.147},
        param_adjustments={
            "firm.cost_push_weight": 0.4,
            "household.consumption_elasticity": -0.6,
            "labor_market_flow_rate": -0.5,
            "supply_chain_friction": 0.7,
        },
        source="Brunnermeier (2021), 'The Resilient Society', JEP.",
    ),
    HistoricalEvent(
        name="1997 Asian Financial Crisis",
        year=1997,
        shock_type="currency_crisis",
        scope="cross_border",
        severity=0.85,
        duration_months=18,
        geographies=("Thailand", "Indonesia", "South Korea", "Asia"),
        sectors=("banking", "finance", "exports", "real_estate"),
        keywords=(
            "currency", "devaluation", "capital", "flight", "imf", "peg",
            "contagion", "asia", "baht", "rupiah",
        ),
        canonical_outcomes={"gdp_drop_pct": 7.0, "currency_depreciation_pct": 50.0},
        param_adjustments={
            "fx_pass_through": 0.6,
            "capital_flow_volatility": 0.7,
            "bank.loan_growth": -0.5,
            "sovereign_risk_premium": 0.5,
        },
        source="Radelet & Sachs (1998), 'The East Asian Financial Crisis', BPEA.",
    ),
    HistoricalEvent(
        name="1973 Oil Shock",
        year=1973,
        shock_type="supply_disruption",
        scope="macro",
        severity=0.80,
        duration_months=24,
        geographies=("US", "EU", "Japan", "global"),
        sectors=("energy", "oil", "transportation", "manufacturing", "automotive"),
        keywords=(
            "oil", "opec", "embargo", "energy", "price", "shock", "stagflation",
            "supply", "petroleum",
        ),
        canonical_outcomes={"gdp_drop_pct": 3.2, "inflation_peak_pct": 12.0},
        param_adjustments={
            "firm.cost_push_weight": 0.7,
            "energy_price_pass_through": 0.6,
            "household.consumption_elasticity": -0.3,
            "wage_price_spiral": 0.5,
        },
        source="Hamilton (2003), 'What Is an Oil Shock?', J. Econometrics.",
    ),
    HistoricalEvent(
        name="1992 ERM Crisis",
        year=1992,
        shock_type="currency_crisis",
        scope="sectoral",
        severity=0.65,
        duration_months=6,
        geographies=("UK", "Italy", "EU"),
        sectors=("banking", "finance", "exports"),
        keywords=(
            "erm", "currency", "peg", "soros", "sterling", "lira", "speculative",
            "attack", "europe",
        ),
        canonical_outcomes={"currency_depreciation_pct": 15.0, "gdp_drop_pct": 0.5},
        param_adjustments={
            "fx_pass_through": 0.4,
            "speculative_capital_volatility": 0.6,
            "central_bank_credibility": -0.5,
        },
        source="Eichengreen & Wyplosz (1993), 'The Unstable EMS', BPEA.",
    ),
    HistoricalEvent(
        name="Argentina 2001 Default",
        year=2001,
        shock_type="sovereign_default",
        scope="macro",
        severity=0.90,
        duration_months=24,
        geographies=("Argentina", "LatAm"),
        sectors=("banking", "finance", "government"),
        keywords=(
            "sovereign", "default", "peso", "currency", "convertibility", "imf",
            "argentina", "corralito", "debt",
        ),
        canonical_outcomes={"gdp_drop_pct": 11.0, "unemployment_peak": 0.21},
        param_adjustments={
            "sovereign_risk_premium": 0.9,
            "fx_pass_through": 0.7,
            "bank.loan_growth": -0.7,
            "household.consumption_elasticity": -0.5,
        },
        source="Hausmann & Velasco (2002), 'Hard Money's Soft Underbelly', BPEA.",
    ),
    HistoricalEvent(
        name="Greece Sovereign Debt Crisis",
        year=2010,
        shock_type="sovereign_debt",
        scope="macro",
        severity=0.85,
        duration_months=60,
        geographies=("Greece", "EU", "eurozone"),
        sectors=("banking", "government", "finance"),
        keywords=(
            "sovereign", "debt", "austerity", "eurozone", "greece", "bailout",
            "troika", "yields", "imf",
        ),
        canonical_outcomes={"gdp_drop_pct": 25.0, "unemployment_peak": 0.275},
        param_adjustments={
            "sovereign_risk_premium": 0.8,
            "fiscal_multiplier": -0.4,
            "bank.loan_growth": -0.6,
            "household.consumption_elasticity": -0.6,
        },
        source="Reinhart & Rogoff (2014), 'Recovery from Financial Crises', AER.",
    ),
    HistoricalEvent(
        name="Tohoku Earthquake & Tsunami",
        year=2011,
        shock_type="supply_disruption",
        scope="sectoral",
        severity=0.70,
        duration_months=12,
        geographies=("Japan", "global"),
        sectors=("manufacturing", "automotive", "electronics", "energy", "nuclear"),
        keywords=(
            "earthquake", "tsunami", "fukushima", "supply", "chain", "japan",
            "disruption", "nuclear", "manufacturing",
        ),
        canonical_outcomes={"gdp_drop_pct": 0.7, "industrial_production_drop_pct": 15.0},
        param_adjustments={
            "supply_chain_friction": 0.8,
            "firm.cost_push_weight": 0.4,
            "energy_price_pass_through": 0.3,
        },
        source="Carvalho et al. (2021), 'Supply Chain Disruptions: Evidence from Tohoku', QJE.",
    ),
    HistoricalEvent(
        name="Brexit Referendum",
        year=2016,
        shock_type="policy_uncertainty",
        scope="cross_border",
        severity=0.55,
        duration_months=36,
        geographies=("UK", "EU"),
        sectors=("finance", "manufacturing", "trade", "services"),
        keywords=(
            "brexit", "referendum", "policy", "uncertainty", "trade", "eu",
            "uk", "sterling", "vote",
        ),
        canonical_outcomes={"gdp_drop_pct": 1.5, "currency_depreciation_pct": 12.0},
        param_adjustments={
            "policy_uncertainty_premium": 0.6,
            "firm.investment_elasticity": -0.4,
            "fx_pass_through": 0.3,
            "trade_friction": 0.4,
        },
        source="Bloom et al. (2019), 'The Impact of Brexit on UK Firms', NBER WP.",
    ),
    HistoricalEvent(
        name="US-China Trade War",
        year=2018,
        shock_type="tariff",
        scope="cross_border",
        severity=0.45,
        duration_months=24,
        geographies=("US", "China", "global"),
        sectors=("manufacturing", "agriculture", "technology", "trade"),
        keywords=(
            "tariff", "trade", "war", "china", "us", "duties", "import",
            "export", "protectionism", "decoupling",
        ),
        canonical_outcomes={"gdp_drop_pct": 0.3, "trade_volume_drop_pct": 4.0},
        param_adjustments={
            "trade_friction": 0.6,
            "firm.cost_push_weight": 0.3,
            "supply_chain_friction": 0.4,
            "policy_uncertainty_premium": 0.4,
        },
        source="Amiti, Redding, Weinstein (2019), 'The Impact of the 2018 Tariffs', JEP.",
    ),
    HistoricalEvent(
        name="Silicon Valley Bank Failure",
        year=2023,
        shock_type="bank_run",
        scope="sectoral",
        severity=0.55,
        duration_months=3,
        geographies=("US",),
        sectors=("banking", "technology", "venture", "finance"),
        keywords=(
            "svb", "bank", "run", "deposit", "regional", "tech", "startup",
            "duration", "treasury", "uninsured",
        ),
        canonical_outcomes={"deposit_outflow_pct": 25.0, "regional_bank_equity_drop_pct": 30.0},
        param_adjustments={
            "deposit_flight_elasticity": 0.7,
            "bank.loan_growth": -0.3,
            "interbank_contagion_strength": 0.4,
            "duration_risk_repricing": 0.6,
        },
        source="Jiang et al. (2023), 'Monetary Tightening and US Bank Fragility', SSRN.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _normset(items: Iterable[str]) -> set[str]:
    return {str(x).strip().lower() for x in items if str(x).strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _overlap_ratio(query: set[str], reference: set[str]) -> float:
    """Asymmetric overlap: fraction of `query` that appears in `reference`."""
    if not query or not reference:
        return 0.0
    return len(query & reference) / len(query)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class _ScoreBreakdown:
    shock_type_exact: float
    shock_type_jaccard: float
    scope: float
    geography: float
    sectors: float
    keywords: float

    def total(self) -> float:
        return (
            self.shock_type_exact
            + self.shock_type_jaccard
            + self.scope
            + self.geography
            + self.sectors
            + self.keywords
        )

    def top_factor(self) -> tuple[str, float]:
        items = [
            ("shock-type exact match", self.shock_type_exact),
            ("shock-type token overlap", self.shock_type_jaccard),
            ("scope match", self.scope),
            ("geography overlap", self.geography),
            ("sector overlap", self.sectors),
            ("keyword overlap", self.keywords),
        ]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[0]


def _score(
    event: HistoricalEvent,
    *,
    shock_type: str,
    scope: str,
    geographies: set[str],
    sectors: set[str],
    keywords: set[str],
) -> _ScoreBreakdown:
    q_type = (shock_type or "").strip().lower()
    e_type = event.shock_type.strip().lower()

    shock_exact = 0.40 if q_type and q_type == e_type else 0.0

    # Token overlap on shock_type strings (e.g. "banking_crisis" vs "bank_run").
    q_tokens = _tokenize(q_type.replace("_", " "))
    e_tokens = _tokenize(e_type.replace("_", " "))
    shock_jaccard = 0.20 * _jaccard(q_tokens, e_tokens)

    scope_match = 0.15 if (scope or "").strip().lower() == event.scope.strip().lower() else 0.0

    e_geo = _normset(event.geographies)
    geography = 0.15 * _overlap_ratio(geographies, e_geo) if geographies else 0.0

    e_sec = _normset(event.sectors)
    sectors_score = 0.10 * _overlap_ratio(sectors, e_sec) if sectors else 0.0

    e_kw = _normset(event.keywords)
    keyword_score = 0.20 * _overlap_ratio(keywords, e_kw) if keywords else 0.0

    return _ScoreBreakdown(
        shock_type_exact=shock_exact,
        shock_type_jaccard=shock_jaccard,
        scope=scope_match,
        geography=geography,
        sectors=sectors_score,
        keywords=keyword_score,
    )


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


class AnalogMatcher:
    """Match novel scenarios to a curated corpus of historical events."""

    def __init__(
        self,
        corpus: list[HistoricalEvent] | None = None,
        *,
        min_similarity: float = 0.30,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError(f"min_similarity must be in [0,1], got {min_similarity}")
        # Defensive copy so the caller can't mutate our internal state and
        # vice-versa.
        self._corpus: list[HistoricalEvent] = list(corpus) if corpus is not None else list(_DEFAULT_CORPUS)
        self._min_similarity = float(min_similarity)
        self._top_k = int(top_k)

    @property
    def corpus(self) -> list[HistoricalEvent]:
        return list(self._corpus)

    # ------------------------------------------------------------------
    # Input feature extraction
    # ------------------------------------------------------------------

    def _features_from_shock(
        self, shock: ShockConfig, extra_keywords: list[str] | None
    ) -> tuple[str, str, set[str], set[str], set[str]]:
        shock_type = shock.shock_type
        scope = shock.scope
        geographies = _normset(shock.geography)
        sectors = _normset(shock.sectors)
        kws: set[str] = set()
        kws |= _tokenize(shock.shock_type.replace("_", " "))
        for s in shock.sectors:
            kws |= _tokenize(s)
        for g in shock.geography:
            kws |= _tokenize(g)
        if extra_keywords:
            for k in extra_keywords:
                kws |= _tokenize(k)
        return shock_type, scope, geographies, sectors, kws

    def _features_from_parse(
        self, parse: ParseResult, extra_keywords: list[str] | None
    ) -> tuple[str, str, set[str], set[str], set[str]]:
        shock_type = parse.shock_params.shock_type
        scope = parse.shock_params.scope
        geographies = _normset(parse.geographies)
        sectors = _normset(parse.markets)
        kws: set[str] = set()
        kws |= _tokenize(parse.triggering_event)
        kws |= _tokenize(shock_type.replace("_", " "))
        for m in parse.markets:
            kws |= _tokenize(m)
        for g in parse.geographies:
            kws |= _tokenize(g)
        if extra_keywords:
            for k in extra_keywords:
                kws |= _tokenize(k)
        return shock_type, scope, geographies, sectors, kws

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        shock_config: ShockConfig | None = None,
        parse_result: ParseResult | None = None,
        keywords: list[str] | None = None,
    ) -> list[HistoricalAnalog]:
        if shock_config is None and parse_result is None and not keywords:
            return []

        if shock_config is not None:
            shock_type, scope, geos, secs, kws = self._features_from_shock(shock_config, keywords)
        elif parse_result is not None:
            shock_type, scope, geos, secs, kws = self._features_from_parse(parse_result, keywords)
        else:
            shock_type = ""
            scope = ""
            geos = set()
            secs = set()
            kws = set()
            for k in keywords or []:
                kws |= _tokenize(k)

        scored: list[tuple[float, HistoricalEvent]] = []
        for event in self._corpus:
            breakdown = _score(
                event,
                shock_type=shock_type,
                scope=scope,
                geographies=geos,
                sectors=secs,
                keywords=kws,
            )
            total = min(breakdown.total(), 1.0)
            if total >= self._min_similarity:
                scored.append((total, event))

        scored.sort(key=lambda kv: (-kv[0], kv[1].year, kv[1].name))
        top = scored[: self._top_k]

        return [
            HistoricalAnalog(
                event_name=event.name,
                year=event.year,
                similarity_score=float(score),
                param_adjustments=dict(event.param_adjustments),
                source=event.source,
            )
            for score, event in top
        ]

    # ------------------------------------------------------------------
    # Disclosure / parameter range derivation (Requirement 10.3)
    # ------------------------------------------------------------------

    def _event_for_analog(self, analog: HistoricalAnalog) -> HistoricalEvent | None:
        for ev in self._corpus:
            if ev.name == analog.event_name and ev.year == analog.year:
                return ev
        return None

    def disclose(
        self,
        analogs: list[HistoricalAnalog],
        shock_config: ShockConfig,
    ) -> list[AnalogDisclosure]:
        disclosures: list[AnalogDisclosure] = []
        shock_type, scope, geos, secs, kws = self._features_from_shock(shock_config, None)
        for analog in analogs:
            event = self._event_for_analog(analog)
            if event is None:
                # Synthesize a minimal disclosure so callers always get one entry per analog.
                affected = sorted(analog.param_adjustments.keys())
                ranges = _build_ranges(analog.param_adjustments, base_params={})
                disclosures.append(
                    AnalogDisclosure(
                        analog=analog,
                        rationale=f"Selected analog '{analog.event_name}' (custom corpus entry).",
                        affected_params=affected,
                        shifted_ranges={k: ranges[k] for k in affected if k in ranges},
                    )
                )
                continue

            breakdown = _score(
                event,
                shock_type=shock_type,
                scope=scope,
                geographies=geos,
                sectors=secs,
                keywords=kws,
            )
            factor_name, factor_score = breakdown.top_factor()
            rationale = (
                f"Selected because of strong {factor_name} "
                f"(similarity={analog.similarity_score:.2f}, "
                f"top factor contributed {factor_score:.2f})."
            )
            affected = sorted(event.param_adjustments.keys())
            shifted = _build_ranges(
                event.param_adjustments,
                base_params=_extract_base_params(shock_config),
            )
            disclosures.append(
                AnalogDisclosure(
                    analog=analog,
                    rationale=rationale,
                    affected_params=affected,
                    shifted_ranges={k: shifted[k] for k in affected},
                )
            )
        return disclosures

    def informed_param_ranges(
        self,
        analogs: list[HistoricalAnalog],
        base_params: dict[str, float],
    ) -> dict[str, tuple[float, float]]:
        combined: dict[str, tuple[float, float]] = {}
        for analog in analogs:
            ranges = _build_ranges(analog.param_adjustments, base_params=base_params)
            for name, (lo, hi) in ranges.items():
                if name in combined:
                    cur_lo, cur_hi = combined[name]
                    combined[name] = (min(cur_lo, lo), max(cur_hi, hi))
                else:
                    combined[name] = (lo, hi)
        return combined


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------


def _extract_base_params(shock: ShockConfig) -> dict[str, float]:
    """Pull numeric scalars out of behavioral_overrides for use as base_params."""
    out: dict[str, float] = {}
    for k, v in (shock.behavioral_overrides or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
    return out


def _build_ranges(
    param_adjustments: dict[str, float],
    *,
    base_params: dict[str, float],
) -> dict[str, tuple[float, float]]:
    """Build (low, high) ranges for each adjusted param.

    - If `base_params[name]` exists, the range is base shifted by adjustment magnitude.
      Positive adjustment -> (base, base + |adj|); negative -> (base - |adj|, base).
      Either way the original base remains in the range, so we never collapse to
      a point estimate.
    - If `base_params` lacks the key, the adjustment value itself is the midpoint
      with +/-20% spread (always non-degenerate).
    """
    out: dict[str, tuple[float, float]] = {}
    for name, adj in param_adjustments.items():
        if name in base_params:
            base = float(base_params[name])
            mag = abs(float(adj))
            if adj >= 0:
                lo, hi = base, base + mag
            else:
                lo, hi = base - mag, base
            if lo == hi:
                # Defensive: never collapse to a point.
                spread = max(abs(base) * 0.05, 1e-6)
                lo, hi = base - spread, base + spread
        else:
            mid = float(adj)
            spread = max(abs(mid) * 0.20, 1e-6)
            lo, hi = mid - spread, mid + spread
        out[name] = (lo, hi)
    return out


__all__ = [
    "HistoricalEvent",
    "AnalogDisclosure",
    "AnalogMatcher",
]
