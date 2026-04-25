"""End-to-end pipeline orchestrator (Task 18, Requirements 15.1, 15.2, 15.4).

The :class:`ClydePipeline` glues together every existing subsystem into the
canonical flow described in the design document:

    Document Ingestion → Scenario Parser → Knowledge Graph
        → Analog Matcher (optional) → Economic World Factory
        → Monte Carlo Controller → Synthesis Engine → Report Agent.

The pipeline injects the :class:`LLMClient` only into the LLM-aware setup
and reporting subsystems (parser, knowledge graph, God's Eye, report
agent). The simulation and synthesis subsystems are LLM-free by contract
(Requirement 15.3/15.5) and never receive the client. The pipeline module
itself lives outside ``clyde/simulation/`` so the static boundary check
remains valid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from clyde.llm import LLMClient
from clyde.models.config import ShockConfig, SimulationWorld
from clyde.models.input import ParseResult
from clyde.persistence.db import SimulationDB
from clyde.reporting import NarrativeReport, ReportAgent, SynthesisResult
from clyde.setup import (
    EconomicWorldFactory,
    NetworkBuilder,
    PriorLibrary,
)
from clyde.setup.analog_matcher import AnalogMatcher
from clyde.setup.gods_eye import GodsEyeConsole
from clyde.setup.ingestion import DocumentIngester
from clyde.setup.knowledge_graph import KnowledgeGraph
from clyde.setup.parser import ScenarioParser
from clyde.simulation import BranchResult, MonteCarloController, PropagationEngine
from clyde.synthesis import SynthesisEngine


# ---------------------------------------------------------------------------
# Configuration / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """User-tunable knobs for :meth:`ClydePipeline.run`."""

    run_count: int = 50
    ensemble_seed: int = 0
    max_workers: int | None = None
    db_path: str | Path | None = None  # None ⇒ ":memory:"
    use_analogs: bool = True
    rng_seed: int = 0


@dataclass
class PipelineResult:
    """Captured artifacts produced by :meth:`ClydePipeline.run`.

    ``branches`` is mutated by :meth:`ClydePipeline.fork_branch` so callers
    can keep accumulating sibling branches against a single base run.
    """

    scenario_id: str
    parse_result: ParseResult
    knowledge_graph: KnowledgeGraph
    shock_config: ShockConfig
    world: SimulationWorld
    ensemble: object  # EnsembleResult
    synthesis: SynthesisResult
    report: NarrativeReport
    db_path: str
    branches: list[BranchResult] = field(default_factory=list)
    personas: list = field(default_factory=list)  # list[ActorPersona]
    influence_config: object = None  # InfluenceConfig
    influence_reasoning: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ClydePipeline:
    """Compose the setup → simulation → synthesis → report flow.

    The constructor is deliberately cheap: components are constructed lazily
    inside :meth:`run` so that callers can introspect / mutate ``config``
    after instantiation. The DB lives on the pipeline instance for the
    lifetime of one or more runs against the same store.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        config: PipelineConfig | None = None,
        analog_matcher: AnalogMatcher | None = None,
    ) -> None:
        self._llm = llm_client
        self.config = config or PipelineConfig()
        self._analog_matcher = analog_matcher

        # Lazily-initialised: see _ensure_components.
        self._db: SimulationDB | None = None
        self._owns_db: bool = False
        self._ingester: DocumentIngester | None = None
        self._parser: ScenarioParser | None = None
        self._kg_builder_factory = lambda: KnowledgeGraph(self._llm)
        self._gods_eye: GodsEyeConsole | None = None
        self._prior_library: PriorLibrary | None = None
        self._world_factory: EconomicWorldFactory | None = None
        self._controller: MonteCarloController | None = None
        self._synth: SynthesisEngine | None = None
        self._report_factory = None

    # ------------------------------------------------------------------ run

    async def run(
        self,
        description: str,
        document_paths: list[str | Path] | None = None,
        scenario_id: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline end-to-end."""
        self._ensure_components()
        assert self._db is not None
        assert self._ingester is not None
        assert self._parser is not None
        assert self._world_factory is not None
        assert self._controller is not None
        assert self._synth is not None

        # 1. Ingest documents (if any).
        documents = (
            self._ingester.ingest_many(document_paths) if document_paths else []
        )

        # 2. Parse the natural-language scenario.
        parse_result = await self._parser.parse(description, documents)

        # 3. Build the knowledge graph (one per run so artifacts don't leak).
        #    With no documents the call is a no-op on the LLM side and only
        #    merges NL-derived entities — KG.build_from_documents handles this
        #    natively, so we don't need to special-case it here.
        kg = self._kg_builder_factory()
        await kg.build_from_documents(documents, parse_result)

        # 4. Extract a ShockConfig from the graph + parse result and stamp
        #    in the configured ensemble seed (Property 4 / Req 6.7).
        shock_config = kg.extract_shock_config(parse_result=parse_result)
        from dataclasses import replace as _replace

        # Enforce a minimum time horizon: if the LLM returned 0 steps (or
        # the KG couldn't infer one), default to a sensible 12-step horizon
        # using the parsed step_unit (or "quarter" as fallback).
        if shock_config.time_horizon.steps <= 0:
            fallback_unit = parse_result.time_horizon.step_unit if parse_result.time_horizon.step_unit else "quarter"
            from clyde.models.time import TimeHorizon as _TH
            shock_config = _replace(
                shock_config,
                ensemble_seed=int(self.config.ensemble_seed),
                time_horizon=_TH(steps=12, step_unit=fallback_unit),
            )
        else:
            shock_config = _replace(shock_config, ensemble_seed=int(self.config.ensemble_seed))

        # 5. Optional historical-analog disclosure (Req 10.1–10.3).
        if self.config.use_analogs and self._analog_matcher is not None:
            analogs = self._analog_matcher.match(shock_config=shock_config)
            shock_config = _replace(shock_config, historical_analogs=list(analogs))

        # 6. Build the world (and a Scenario for downstream selection).
        sid = scenario_id or f"scenario_{self.config.ensemble_seed}"
        scenario = self._world_factory.build_scenario(
            scenario_id=sid,
            description=description,
            shock_config=shock_config,
            prior_library=self._prior_library,  # type: ignore[arg-type]
            param_overrides=None,
        )
        # Reconstruct a SimulationWorld view from the scenario (factory only
        # exposes one or the other; we want both).
        world = SimulationWorld(
            config=scenario.config,
            actors=list(scenario.actors),
            networks=scenario.networks,
            prior_library_version=scenario.prior_library_version,
        )

        # 6b. Generate rich actor personas and influence config via LLM.
        from clyde.setup.persona_generator import PersonaGenerator
        persona_gen = PersonaGenerator(self._llm)
        personas = await persona_gen.generate_personas(world.actors, shock_config)
        influence_config, influence_reasoning = await persona_gen.generate_influence_config(shock_config)

        # 7. Run the Monte Carlo ensemble (LLM-free).
        ensemble = self._controller.run_ensemble(
            world=world,
            run_count=int(self.config.run_count),
            scenario_id=scenario.scenario_id,
            db=self._db,
        )

        # 8. Synthesise distributional outputs (LLM-free).
        paths = self._synth.compute_paths(ensemble)
        divergence = self._synth.compute_divergence_map(ensemble)
        chains = self._synth.detect_causal_chains(ensemble)
        selections = self._synth.select_metrics(scenario, ensemble)
        synthesis = SynthesisResult(
            scenario_id=scenario.scenario_id,
            config=scenario.config,
            paths=paths,
            divergence_map=divergence,
            causal_chains=chains,
            metric_selections=selections,
        )

        # 9. Generate the evidence-only narrative report.
        report_agent = self._make_report_agent()
        report = await report_agent.generate_report(
            synthesis,
            ensemble_run_ids=[t.run_id for t in ensemble.trajectories],
        )

        return PipelineResult(
            scenario_id=scenario.scenario_id,
            parse_result=parse_result,
            knowledge_graph=kg,
            shock_config=shock_config,
            world=world,
            ensemble=ensemble,
            synthesis=synthesis,
            report=report,
            db_path=self._db.db_path,
            branches=[],
            personas=personas,
            influence_config=influence_config,
            influence_reasoning=influence_reasoning,
        )

    # ----------------------------------------------------------- fork_branch

    async def fork_branch(
        self,
        result: PipelineResult,
        injection_text: str,
    ) -> BranchResult:
        """Parse a NL injection against ``result`` and re-simulate from step 0."""
        self._ensure_components()
        assert self._db is not None
        assert self._controller is not None
        assert self._gods_eye is not None

        # Synthesize a minimal Scenario stub from the result so the
        # console can serialise the base config.
        from clyde.models.scenario import Scenario

        base_scenario = Scenario(
            scenario_id=result.scenario_id,
            description="",
            config=result.shock_config,
            actors=list(result.world.actors),
            networks=result.world.networks,
            prior_library_version=result.world.prior_library_version,
            overrides={},
            metadata={},
        )

        delta = await self._gods_eye.parse_injection(
            injection_text=injection_text,
            base_scenario=base_scenario,
        )

        branch = self._controller.fork_branch(
            base_world=result.world,
            delta=delta,
            run_count=int(self.config.run_count),
            parent_scenario_id=result.scenario_id,
            db=self._db,
        )
        result.branches.append(branch)
        return branch

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        """Close the :class:`SimulationDB` if the pipeline owns it."""
        if self._db is not None and self._owns_db:
            self._db.close()
        self._db = None
        self._owns_db = False

    def __enter__(self) -> "ClydePipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----------------------------------------------------------- internals

    def _ensure_components(self) -> None:
        """Construct lazily so the ctor stays cheap and re-use is cheap."""
        if self._db is None:
            db_path = self.config.db_path
            if db_path is None:
                self._db = SimulationDB(":memory:")
            else:
                self._db = SimulationDB(db_path)
            self._owns_db = True

        if self._ingester is None:
            self._ingester = DocumentIngester()

        if self._parser is None:
            self._parser = ScenarioParser(self._llm)

        if self._gods_eye is None:
            self._gods_eye = GodsEyeConsole(self._llm)

        if self._prior_library is None:
            self._prior_library = PriorLibrary()

        if self._analog_matcher is None and self.config.use_analogs:
            self._analog_matcher = AnalogMatcher()

        if self._world_factory is None:
            nb = NetworkBuilder(rng=random.Random(self.config.rng_seed))
            self._world_factory = EconomicWorldFactory(
                network_builder=nb,
                rng_seed=self.config.rng_seed,
            )

        if self._controller is None:
            self._controller = MonteCarloController(
                engine=PropagationEngine(),
                max_workers=self.config.max_workers,
            )

        if self._synth is None:
            self._synth = SynthesisEngine()

    def _make_report_agent(self) -> ReportAgent:
        """Construct a fresh ReportAgent bound to the current DB."""
        assert self._db is not None
        return ReportAgent(self._llm, self._db)


# Module-level alias for callers who prefer the shorter spelling.
Pipeline = ClydePipeline


__all__ = [
    "ClydePipeline",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
]
