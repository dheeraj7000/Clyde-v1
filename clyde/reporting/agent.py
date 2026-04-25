"""Evidence-only Report Agent.

The :class:`ReportAgent` produces narrative reports whose factual content is
sourced *exclusively* from simulation artifacts (the :class:`SimulationDB`
trajectories / causal events and the :class:`SynthesisEngine` outputs). The
LLM client is used only as a *prose stylist*: it never injects numbers,
names, or facts of its own, and a prose response that omits a number does
not silence that number — the agent owns the factual section bodies and
appends LLM prose as a separate paragraph.

Contracts:

* Every numerical / structural claim is accompanied by a
  :class:`ProvenanceAnnotation` pointing at the artifact that supports it.
* If evidence cannot be retrieved, the corresponding claim is *dropped* and
  an :class:`EvidenceGapWarning` is emitted (Req 9.6, design "Synthesis /
  Report Errors").
* Uncertainty flags (Req 12.1–12.5) are detected programmatically and
  influence both the section bodies (heavy-tail softening, reflexivity
  pre/post split) and the trailing "Uncertainty Flags" section.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from clyde.llm.client import LLMClient, LLMMessage
from clyde.models.causal import CausalChain
from clyde.models.config import ShockConfig
from clyde.models.metrics import CORE_METRIC_NAMES, PathBundle, StepMetrics
from clyde.models.reporting import DivergenceMap

if TYPE_CHECKING:
    from clyde.persistence.db import SimulationDB
    from clyde.synthesis.engine import MetricSelection


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceAnnotation:
    """Trace from a single factual claim back to the artifact supporting it."""

    claim: str
    source_type: str  # "simulation_db" | "knowledge_graph"
    source_ref: str  # specific artifact id, run_id, or KG entity id
    query_used: str  # the query string used to retrieve the evidence


@dataclass
class ReportSection:
    heading: str
    body: str
    flags: list[str] = field(default_factory=list)
    provenance: list[ProvenanceAnnotation] = field(default_factory=list)


@dataclass
class NarrativeReport:
    scenario_id: str
    sections: list[ReportSection]
    provenance: list[ProvenanceAnnotation]
    uncertainty_flags: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Bundle the SynthesisEngine produces, that the agent consumes."""

    scenario_id: str
    config: ShockConfig
    paths: PathBundle
    divergence_map: DivergenceMap
    causal_chains: list[CausalChain]
    metric_selections: list  # list[MetricSelection] from synthesis.engine


class EvidenceGapWarning(UserWarning):
    """Emitted when the agent had to drop a claim due to missing evidence."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_SECTION_OUTCOME_RANGE = "Outcome Range"
_SECTION_CAUSAL_PATHWAYS = "Causal Pathways"
_SECTION_DIVERGENCE = "Divergence and Watchlist"
_SECTION_UNCERTAINTY = "Uncertainty Flags"

_INSUFFICIENT_DATA = "insufficient data"


def _format_value(metric: str, value: object) -> str:
    """Render a metric value compactly for narrative bodies."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sm(step: StepMetrics, metric: str) -> object:
    return getattr(step, metric)


def _band_summary(label: str, step: StepMetrics) -> str:
    """One-line per-band metric digest for the Outcome Range section."""
    parts = [f"{m}={_format_value(m, _sm(step, m))}" for m in CORE_METRIC_NAMES]
    return f"  - {label} (step={step.step}): " + ", ".join(parts)


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------


class ReportAgent:
    """Evidence-only narrative report generator (Requirement 9, 12)."""

    def __init__(
        self,
        llm_client: LLMClient,
        db: "SimulationDB",
        *,
        model: str | None = None,
        novel_regime_keywords: tuple[str, ...] = (
            "negative interest rate",
            "currency board collapse",
            "war economy",
            "post-pandemic",
            "AI productivity shock",
            "climate tail event",
        ),
        reflexivity_keywords: tuple[str, ...] = (
            "rate hike",
            "rate_hike",
            "policy announcement",
            "forward guidance",
            "tariff",
            "stimulus package",
            "sanctions",
        ),
        heavy_tail_keywords: tuple[str, ...] = (
            "panic",
            "fire sale",
            "bank run",
            "viral",
            "flash crash",
        ),
        geopolitical_keywords: tuple[str, ...] = (
            "sanctions",
            "war",
            "election",
            "coup",
            "diplomatic",
        ),
    ) -> None:
        self._llm = llm_client
        self._db = db
        self._model = model
        self._novel_regime_keywords = tuple(novel_regime_keywords)
        self._reflexivity_keywords = tuple(reflexivity_keywords)
        self._heavy_tail_keywords = tuple(heavy_tail_keywords)
        self._geopolitical_keywords = tuple(geopolitical_keywords)

    # ------------------------------------------------------------------ flags

    def detect_uncertainty_flags(
        self,
        config: ShockConfig,
        synthesis: SynthesisResult,
    ) -> list[str]:
        """Detect uncertainty flags (Req 12.1–12.5)."""
        flags: list[str] = []
        haystacks: list[str] = []
        if config.shock_type:
            haystacks.append(config.shock_type)
        haystacks.extend(config.geography or [])
        # Behavioral overrides may carry a free-text "description" we can scan.
        description = ""
        if isinstance(config.behavioral_overrides, dict):
            description = str(config.behavioral_overrides.get("description", ""))
        haystacks.append(description)

        joined = " | ".join(s for s in haystacks if s).lower()

        if any(kw.lower() in joined for kw in self._novel_regime_keywords):
            flags.append("unknown_regime")
        if any(kw.lower() in joined for kw in self._reflexivity_keywords):
            flags.append("reflexivity_risk")
        if any(kw.lower() in joined for kw in self._heavy_tail_keywords):
            flags.append("heavy_tail_dynamics")
        if any(kw.lower() in joined for kw in self._geopolitical_keywords):
            flags.append("exogenous_geopolitical")

        if self._has_high_outcome_dispersion(synthesis.paths):
            flags.append("high_outcome_dispersion")

        # Stable, dedup-preserving order.
        seen: set[str] = set()
        uniq: list[str] = []
        for f in flags:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq

    @staticmethod
    def _has_high_outcome_dispersion(paths: PathBundle) -> bool:
        """True iff tail bands diverge from optimistic/pessimistic by >=25% on any
        metric at end-of-horizon."""
        if (
            not paths.central
            or not paths.optimistic
            or not paths.pessimistic
            or not paths.tail_upper
            or not paths.tail_lower
        ):
            return False

        upper = paths.tail_upper[-1]
        lower = paths.tail_lower[-1]
        opt = paths.optimistic[-1]
        pes = paths.pessimistic[-1]

        for metric in CORE_METRIC_NAMES:
            try:
                tu = float(_sm(upper, metric))
                tl = float(_sm(lower, metric))
                o = float(_sm(opt, metric))
                p = float(_sm(pes, metric))
            except (TypeError, ValueError):
                continue
            # Reference is the larger absolute among optimistic/pessimistic.
            ref = max(abs(o), abs(p), 1e-9)
            if abs(tu - o) / ref >= 0.25 or abs(tl - p) / ref >= 0.25:
                return True
        return False

    # --------------------------------------------------------------- generate

    async def generate_report(
        self,
        synthesis: SynthesisResult,
        ensemble_run_ids: list[str] | None = None,
    ) -> NarrativeReport:
        """Compose the evidence-only narrative report.

        The factual content of every section is computed from ``synthesis``
        and the database; the LLM only produces decorative prose appended to
        each section body. Missing evidence drops the claim and emits an
        :class:`EvidenceGapWarning`.
        """
        flags = self.detect_uncertainty_flags(synthesis.config, synthesis)
        ensemble_run_ids = ensemble_run_ids or []

        # Probe DB for representative trajectories / events. Missing rows ->
        # warnings, but we still emit the structural sections (with empty
        # provenance for the affected claims).
        retrieved_trajectories: dict[str, list[StepMetrics]] = {}
        retrieved_events: dict[str, list] = {}
        for run_id in ensemble_run_ids:
            traj = self._db.get_trajectory(run_id)
            if not traj:
                warnings.warn(
                    EvidenceGapWarning(
                        f"No trajectory found for run_id={run_id!r}; "
                        f"dropping run-specific claims."
                    )
                )
            else:
                retrieved_trajectories[run_id] = traj
            evs = self._db.get_causal_events(run_id)
            if evs:
                retrieved_events[run_id] = evs

        outcome_section = await self._build_outcome_range_section(synthesis, flags)
        causal_section = await self._build_causal_pathways_section(synthesis)
        divergence_section = await self._build_divergence_section(synthesis)
        uncertainty_section = await self._build_uncertainty_section(synthesis, flags)

        sections = [outcome_section, causal_section, divergence_section, uncertainty_section]

        flat_provenance: list[ProvenanceAnnotation] = []
        for s in sections:
            flat_provenance.extend(s.provenance)

        return NarrativeReport(
            scenario_id=synthesis.scenario_id,
            sections=sections,
            provenance=flat_provenance,
            uncertainty_flags=list(flags),
        )

    # ---------------------------------------------------------- section: range

    async def _build_outcome_range_section(
        self,
        synthesis: SynthesisResult,
        flags: list[str],
    ) -> ReportSection:
        paths = synthesis.paths
        provenance: list[ProvenanceAnnotation] = []
        section_flags: list[str] = []
        body_lines: list[str] = []

        if not paths.central:
            warnings.warn(
                EvidenceGapWarning(
                    "Outcome Range: PathBundle is empty; reporting insufficient data."
                )
            )
            body = _INSUFFICIENT_DATA
            return ReportSection(
                heading=_SECTION_OUTCOME_RANGE,
                body=body,
                flags=section_flags,
                provenance=provenance,
            )

        last_idx = -1
        body_lines.append("End-of-horizon outcome bands:")
        for label, series in (
            ("central", paths.central),
            ("optimistic", paths.optimistic),
            ("pessimistic", paths.pessimistic),
            ("tail_upper", paths.tail_upper),
            ("tail_lower", paths.tail_lower),
        ):
            if not series:
                continue
            body_lines.append(_band_summary(label, series[last_idx]))
            provenance.append(
                ProvenanceAnnotation(
                    claim=f"{label} band at horizon",
                    source_type="simulation_db",
                    source_ref=synthesis.scenario_id,
                    query_used="paths_at_horizon",
                )
            )

        # Reflexivity-driven dual-path treatment (Req 12.3).
        if "reflexivity_risk" in flags:
            section_flags.append("reflexivity_risk")
            mid = max(1, len(paths.central) // 2)
            pre = paths.central[mid - 1]
            post = paths.central[-1]
            body_lines.append(
                "Reflexivity risk detected: reporting pre-announcement vs "
                "post-announcement outcomes from the central path."
            )
            body_lines.append(_band_summary("pre-announcement", pre))
            body_lines.append(_band_summary("post-announcement", post))
            provenance.append(
                ProvenanceAnnotation(
                    claim="pre-announcement central outcome (first half of horizon)",
                    source_type="simulation_db",
                    source_ref=synthesis.scenario_id,
                    query_used="paths_central_first_half",
                )
            )
            provenance.append(
                ProvenanceAnnotation(
                    claim="post-announcement central outcome (second half of horizon)",
                    source_type="simulation_db",
                    source_ref=synthesis.scenario_id,
                    query_used="paths_central_second_half",
                )
            )

        # Heavy-tail dynamics softening (Req 12.4).
        if "heavy_tail_dynamics" in flags:
            section_flags.append("heavy_tail_dynamics")
            body_lines.append(
                "_subject to heavy-tail dynamics; dispersion may be underestimated_"
            )

        # Unknown regime widens claims (Req 12.1).
        if "unknown_regime" in flags:
            section_flags.append("unknown_regime")
            body_lines.append(
                "Novel-regime conditions detected: outcome distributions are "
                "wider than the bands above might suggest, and causal claims "
                "are correspondingly weakened."
            )

        body_lines.append(
            await self._llm_polish(
                heading=_SECTION_OUTCOME_RANGE,
                facts=body_lines,
            )
        )

        return ReportSection(
            heading=_SECTION_OUTCOME_RANGE,
            body="\n".join(body_lines),
            flags=section_flags,
            provenance=provenance,
        )

    # -------------------------------------------------------- section: causal

    async def _build_causal_pathways_section(
        self,
        synthesis: SynthesisResult,
    ) -> ReportSection:
        provenance: list[ProvenanceAnnotation] = []
        body_lines: list[str] = []

        if not synthesis.causal_chains:
            return ReportSection(
                heading=_SECTION_CAUSAL_PATHWAYS,
                body=_INSUFFICIENT_DATA,
                flags=[],
                provenance=provenance,
            )

        for chain in synthesis.causal_chains:
            body_lines.append(
                f"Chain {chain.chain_id} (origin={chain.origin_shock!r}, "
                f"total_magnitude={chain.total_magnitude:.4f}):"
            )
            if not chain.events:
                body_lines.append(f"  - {_INSUFFICIENT_DATA}")
                continue
            for ev in chain.events:
                body_lines.append(
                    f"  - step={ev.step} {ev.source_actor_id} -> "
                    f"{ev.target_actor_id} via {ev.channel} "
                    f"(var={ev.variable_affected}, mag={ev.magnitude:.4f})"
                )
            provenance.append(
                ProvenanceAnnotation(
                    claim=f"causal chain {chain.chain_id}",
                    source_type="simulation_db",
                    source_ref=chain.chain_id,
                    query_used="causal_chain",
                )
            )

        body_lines.append(
            await self._llm_polish(
                heading=_SECTION_CAUSAL_PATHWAYS,
                facts=body_lines,
            )
        )

        return ReportSection(
            heading=_SECTION_CAUSAL_PATHWAYS,
            body="\n".join(body_lines),
            flags=[],
            provenance=provenance,
        )

    # ----------------------------------------------------- section: divergence

    async def _build_divergence_section(
        self,
        synthesis: SynthesisResult,
    ) -> ReportSection:
        provenance: list[ProvenanceAnnotation] = []
        body_lines: list[str] = []

        dm = synthesis.divergence_map
        if not dm.variables:
            return ReportSection(
                heading=_SECTION_DIVERGENCE,
                body=_INSUFFICIENT_DATA,
                flags=[],
                provenance=provenance,
            )

        body_lines.append("Top divergence drivers and recommended monitoring:")
        for var in dm.variables:
            body_lines.append(
                f"  - {var.name}: sensitivity={var.sensitivity:.6f}, "
                f"uncertainty={var.current_uncertainty:.4f}, "
                f"watch={var.monitoring_indicator}"
            )
            provenance.append(
                ProvenanceAnnotation(
                    claim=f"divergence driver {var.name}",
                    source_type="simulation_db",
                    source_ref="divergence_map",
                    query_used="divergence_map_top_k",
                )
            )

        body_lines.append(
            await self._llm_polish(
                heading=_SECTION_DIVERGENCE,
                facts=body_lines,
            )
        )

        return ReportSection(
            heading=_SECTION_DIVERGENCE,
            body="\n".join(body_lines),
            flags=[],
            provenance=provenance,
        )

    # ---------------------------------------------------- section: uncertainty

    async def _build_uncertainty_section(
        self,
        synthesis: SynthesisResult,
        flags: list[str],
    ) -> ReportSection:
        provenance: list[ProvenanceAnnotation] = []
        body_lines: list[str] = []

        if not flags:
            body_lines.append("No uncertainty flags fired for this scenario.")
        else:
            body_lines.append("Uncertainty flags fired:")
            for f in flags:
                body_lines.append(f"  - {f}")
                provenance.append(
                    ProvenanceAnnotation(
                        claim=f"uncertainty flag {f}",
                        source_type="simulation_db",
                        source_ref=synthesis.scenario_id,
                        query_used=f"uncertainty_flag:{f}",
                    )
                )

        body_lines.append(
            await self._llm_polish(
                heading=_SECTION_UNCERTAINTY,
                facts=body_lines,
            )
        )

        return ReportSection(
            heading=_SECTION_UNCERTAINTY,
            body="\n".join(body_lines),
            flags=list(flags),
            provenance=provenance,
        )

    # ----------------------------------------------------------- LLM polishing

    async def _llm_polish(self, *, heading: str, facts: list[str]) -> str:
        """Ask the LLM to write 2-3 paragraphs given pre-assembled facts.

        The prose returned is appended to the section body verbatim. The
        agent has *already* written every factual claim above; this call is
        purely decorative.
        """
        prompt_facts = "\n".join(facts)
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a writing assistant. Given the following "
                    "pre-computed facts and section heading, write 2-3 short "
                    "paragraphs of explanatory prose. DO NOT introduce any "
                    "numbers, names, dates, or facts that are not in the "
                    "provided facts list."
                ),
            ),
            LLMMessage(
                role="user",
                content=f"Section: {heading}\n\nFacts:\n{prompt_facts}",
            ),
        ]
        try:
            resp = await self._llm.complete(messages, model=self._model)
            return resp.content
        except Exception:
            # LLM failure must never fabricate; return an empty narrative
            # rider — the structural body is already complete.
            return ""


__all__ = [
    "EvidenceGapWarning",
    "NarrativeReport",
    "ProvenanceAnnotation",
    "ReportAgent",
    "ReportSection",
    "SynthesisResult",
]
