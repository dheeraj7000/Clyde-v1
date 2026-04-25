# Implementation Plan: Clyde Economic Simulator

## Overview

Build Clyde from the ground up in Python, starting with core data models and persistence, then layering on the rule-based simulation engine, Monte Carlo orchestration, synthesis/reporting, and finally the LLM-powered setup phase (parsing, knowledge graph, God's Eye). Each task is concrete and self-contained — pick it up and go.

## Tasks

- [ ] 1. Project scaffolding, data models, and persistence
  - [ ] 1.1 Create project structure and core data models
    - Create `clyde/` package with subpackages: `models/`, `simulation/`, `setup/`, `synthesis/`, `reporting/`, `persistence/`, `llm/`
    - Implement all core dataclasses in `clyde/models/`: `ShockConfig`, `TimeHorizon`, `ShockDelta`, `SimulationWorld`, `Actor`, `Relationship`, `ActorType` enum, all actor state/param dataclasses (`HouseholdState`, `HouseholdParams`, `FirmState`, `FirmParams`, `BankState`, `BankParams`, `CentralBankState`, `CentralBankParams`), network structures (`NetworkBundle`, `BipartiteGraph`, `DirectedGraph`, `ScaleFreeGraph`), metrics (`StepMetrics`, `TrajectoryResult`, `EnsembleResult`, `PathBundle`), causal models (`CausalEvent`, `CausalChain`), scenario (`Scenario`), divergence/reporting models (`DivergenceMap`, `DivergenceVariable`, `HistoricalAnalog`, `Citation`), and input models (`ParseResult`, `Ambiguity`, `Document`, `ActorHint`)
    - Implement `CausalChain.serialize()` and `CausalChain.deserialize()` for JSON round-trip
    - Implement `Scenario.serialize()`, `Scenario.deserialize()`, and `Scenario.pretty_print()`
    - _Requirements: 13.1, 13.2, 13.3, 14.1, 14.2, 14.3, 14.4_

  - [ ]* 1.2 Write property tests for CausalChain serialization round-trip
    - **Property 16: CausalChain Serialization Round-Trip**
    - Use Hypothesis to generate random `CausalChain` objects, verify `serialize(deserialize(serialize(chain))) == serialize(chain)`
    - **Validates: Requirements 13.1, 13.2, 13.3**

  - [ ]* 1.3 Write property tests for Scenario serialization round-trip
    - **Property 17: Scenario Serialization Round-Trip**
    - Use Hypothesis to generate random `Scenario` objects, verify `serialize(deserialize(serialize(scenario))) == serialize(scenario)`
    - **Validates: Requirements 14.1, 14.2, 14.3**

  - [ ]* 1.4 Write unit test for Scenario pretty_print
    - Verify `Scenario.pretty_print()` produces non-empty, human-readable output
    - _Requirements: 14.4_

  - [ ] 1.5 Implement SQLite persistence layer
    - Create `clyde/persistence/db.py` with `SimulationDB` class
    - Implement schema creation (runs, step_metrics, causal_events, branches, backtest_results tables)
    - Implement write methods: `insert_run`, `insert_step_metrics`, `insert_causal_event`, `insert_branch`, `insert_backtest_result`
    - Implement read methods: `get_run`, `get_step_metrics(run_id, step)`, `get_trajectory(run_id)`, `get_branch(branch_id)`, `get_backtest_results(scenario_id)`
    - _Requirements: 17.3_

  - [ ]* 1.6 Write unit tests for persistence layer
    - Test insert + query round-trip for each table
    - Test querying metrics by (run_id, step)
    - _Requirements: 17.3_

- [ ] 2. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Prior Library and Economic World Factory
  - [ ] 3.1 Implement the Prior Library
    - Create `clyde/setup/prior_library.py` with `PriorLibrary` class
    - Populate with empirical behavioral parameters for each actor type (household, firm, bank, central bank) with source citations
    - Implement `get_params(actor_type, context)`, `version()`, and `citation(param_name)` methods
    - Each parameter must have a `Citation` with non-empty title, authors, year, source
    - Library must be versioned
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 3.2 Write property test for Prior Library citability
    - **Property 5: Prior Library Citability**
    - Iterate all parameters in the library, verify each has a non-empty version string and a Citation with non-empty title, authors, year, and source
    - **Validates: Requirements 3.3**

  - [ ] 3.3 Implement the Network Builder
    - Create `clyde/setup/network_builder.py` with `NetworkBuilder` class
    - Implement `build_labor_market(households, firms)` → `BipartiteGraph` (household ↔ firm edges only)
    - Implement `build_supply_chain(firms, households)` → `DirectedGraph` (firm → firm and firm → household edges only)
    - Implement `build_interbank(banks)` → `ScaleFreeGraph` (bank ↔ bank edges only)
    - _Requirements: 2.5, 2.6, 2.7_

  - [ ]* 3.4 Write property test for network topology constraints
    - **Property 3: Network Topology Constraints**
    - Generate random actor sets, build each network, verify: labor market is bipartite (household↔firm only), supply chain is directed (firm→firm or firm→household only), interbank is bank-to-bank only
    - **Validates: Requirements 2.5, 2.6, 2.7**

  - [ ] 3.5 Implement the Economic World Factory
    - Create `clyde/setup/world_factory.py` with `EconomicWorldFactory` class
    - Implement `build_world(shock_config, prior_library, param_overrides)` → `SimulationWorld`
    - Spawn actors based on `shock_config.agent_counts`, assign behavioral params from `PriorLibrary`
    - Apply user-provided `param_overrides`, record overrides in the Scenario spec
    - Wire network topology using `NetworkBuilder`
    - Validate all relationships reference valid actor IDs and use valid rel_types
    - All behavioral params must be fully populated at construction time — no lazy LLM calls
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 4.1, 15.4_

  - [ ]* 3.6 Write property test for network integrity
    - **Property 2: Network Integrity**
    - Generate random actor sets, build world, verify every Relationship references valid actor IDs and uses a rel_type from {employment, lending, trade, supply, ownership, regulation, trust}
    - **Validates: Requirements 2.3, 2.4**

  - [ ]* 3.7 Write property test for actor behavioral completeness
    - **Property 4: Actor Behavioral Completeness**
    - Generate random ShockConfigs, build world, verify every actor has all required params for its type fully populated (non-null)
    - **Validates: Requirements 3.1, 3.6, 3.7, 3.8, 3.9, 3.10, 4.1, 15.4**

  - [ ]* 3.8 Write property test for override application
    - **Property 6: Override Application**
    - Generate random actor types with random behavioral overrides, build world, verify overrides are applied (not defaults) and recorded in the Scenario
    - **Validates: Requirements 3.4, 3.5, 4.3**

- [ ] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Propagation Engine (rule-based simulation core)
  - [ ] 5.1 Implement the Propagation Engine
    - Create `clyde/simulation/propagation.py` with `PropagationEngine` class
    - Implement `run(world, seed, run_id, db)` → `TrajectoryResult`
    - Implement `_step(world, step, rng)` → `StepMetrics` with the fixed actor update order: Central Bank → Banks → Firms → Labor Market → Households → Interbank Network → Bankruptcies → Learning → Metrics
    - Implement each actor type's behavioral rules as pure functions: household consumption/savings/wage/credit, firm investment/hiring/pricing/bankruptcy, bank credit-tightening/herding/interbank, central bank Taylor rule
    - Accumulate stress at chokepoints where multiple causal chains converge
    - Allow feedback loops (actor responses affect upstream actors)
    - Produce second-order effects from actor interactions, not hard-coded sequences
    - Compute all 9 core metrics at each step: gdp_index, inflation_rate, unemployment_rate, gini_coefficient, credit_tightening_index, firm_bankruptcy_count, bank_stress_index, consumer_confidence, interbank_freeze
    - Support custom metrics alongside core set
    - Persist step metrics and causal events to SQLite via `SimulationDB`
    - Zero LLM imports in this module — enforce the architecture boundary
    - _Requirements: 3.6, 3.7, 3.8, 3.9, 5.1, 5.2, 5.3, 5.4, 15.3, 15.5, 17.1, 17.2_

  - [ ]* 5.2 Write property test for simulation step count
    - **Property 7: Simulation Step Count**
    - Generate random SimulationWorlds with random time horizons, run PropagationEngine, verify TrajectoryResult contains exactly N StepMetrics entries numbered 0 through N-1
    - **Validates: Requirements 5.1**

  - [ ]* 5.3 Write property test for deterministic reproducibility
    - **Property 10: Deterministic Reproducibility**
    - Generate random SimulationWorlds + seeds, run PropagationEngine twice with the same seed, verify identical TrajectoryResult outputs
    - **Validates: Requirements 6.5**

  - [ ]* 5.4 Write property test for core metrics completeness
    - **Property 18: Core Metrics Completeness**
    - Generate random worlds, run simulation, verify every StepMetrics has non-null values for all 9 core metrics, and custom metrics are present when configured
    - **Validates: Requirements 17.1, 17.2, 17.3**

  - [ ]* 5.5 Write unit tests for propagation engine
    - Test feedback loops: circular topology → upstream actors affected
    - Test stress accumulation: known chokepoint topology → stress accumulates
    - Test bankruptcy cascade: verify cascade removes actors and metrics reflect collapse
    - Test shock propagation only follows network edges
    - _Requirements: 2.8, 5.2, 5.3, 5.4_

- [ ] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Monte Carlo Controller
  - [ ] 7.1 Implement the Monte Carlo Controller
    - Create `clyde/simulation/monte_carlo.py` with `MonteCarloController` class
    - Implement `run_ensemble(world, run_count, max_workers)` → `EnsembleResult` using `ProcessPoolExecutor` for parallel execution
    - Implement `_generate_seeds(ensemble_seed, run_count)` for deterministic seed derivation — each run gets a unique seed
    - Implement `_run_single(world, seed, run_id)` to execute one simulation run
    - Vary behavioral response strength, timing, shock severity, and contagion thresholds across runs
    - Default run_count in [100, 500] range, configurable
    - Handle worker crashes gracefully: log failed seed, continue with remaining runs, note incomplete runs in result
    - Implement seed uniqueness check before run; re-derive on collision
    - Zero LLM imports — stochasticity through parameter variation and random seeds only
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 15.3, 15.5_

  - [ ] 7.2 Implement branch forking
    - Implement `fork_branch(base_config, delta, run_count)` → `BranchResult`
    - Merge `ShockDelta` into `ShockConfig` to produce a new merged config
    - Re-simulate from step 0 with the merged config (NOT mutation of running state)
    - Validate `ShockDelta.intervention_step` is within [0, time_horizon-1]
    - Validate `ShockDelta.param_overrides` reference existing parameters
    - Store branch in SQLite via `branches` table
    - Make branch available for comparison against base scenario and other branches
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.7_

  - [ ]* 7.3 Write property test for ensemble run distinctness
    - **Property 8: Ensemble Run Distinctness**
    - Generate random worlds, run small ensembles, verify each run uses a distinct seed and pairwise distinct parameter configurations
    - **Validates: Requirements 6.1, 6.3**

  - [ ]* 7.4 Write property test for ensemble trajectory count
    - **Property 9: Ensemble Trajectory Count**
    - Generate random run counts, run ensemble, verify EnsembleResult contains exactly K TrajectoryResult entries
    - **Validates: Requirements 6.2**

  - [ ]* 7.5 Write property test for branch re-simulation integrity
    - **Property 13: Branch Re-simulation Integrity**
    - Generate random ShockConfigs + ShockDeltas, fork branch, verify merged config contains all original values plus overrides, and trajectory has full time horizon
    - **Validates: Requirements 8.1, 8.4, 8.7**

  - [ ]* 7.6 Write unit tests for Monte Carlo Controller
    - Test configurable run count: default in [100, 500], custom value respected
    - Test parallel execution: verify multiple worker processes spawned
    - Test branch comparison availability: create branch → verify queryable alongside base scenario
    - _Requirements: 6.4, 6.6, 8.3_

- [ ] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Synthesis Engine
  - [ ] 9.1 Implement the Synthesis Engine
    - Create `clyde/synthesis/engine.py` with `SynthesisEngine` class
    - Implement `compute_paths(ensemble)` → `PathBundle` with percentile bands: central (p50), optimistic (p75–p90), pessimistic (p10–p25), tail_upper (>p90), tail_lower (<p10)
    - Implement `compute_divergence_map(ensemble)` → `DivergenceMap` identifying variables that most determine which outcome branch occurs; each variable gets a positive sensitivity score and a non-empty monitoring_indicator
    - Implement `detect_causal_chains(ensemble)` → `list[CausalChain]` extracting ordered actor-to-actor shock transmission sequences from causal events
    - Implement `select_metrics(scenario, ensemble)` → `list[MetricSelection]` picking situation-relevant metrics from the fixed output palette (outcome ranges, causal chain maps, actor impact scorecards, stress/contagion pathways, branch comparisons, indicator watchlists, narrative reports)
    - Derive indicator watchlist from DivergenceMap — must contain exactly the monitoring_indicator values from the map's variables
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 9.3, 9.4_

  - [ ]* 9.2 Write property test for percentile band ordering
    - **Property 11: Percentile Band Ordering**
    - Generate random EnsembleResults, compute paths, verify at every step and for every core metric: tail_lower ≤ pessimistic ≤ central ≤ optimistic ≤ tail_upper
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

  - [ ]* 9.3 Write property test for divergence map completeness and watchlist derivation
    - **Property 12: Divergence Map Completeness and Watchlist Derivation**
    - Generate random ensembles with non-zero variance, verify DivergenceMap is non-empty, each variable has positive sensitivity and non-empty monitoring_indicator, and watchlist matches map indicators exactly
    - **Validates: Requirements 7.5, 9.4**

  - [ ]* 9.4 Write property test for causal chain ordering
    - **Property 15: Causal Chain Ordering**
    - Generate random ensembles with causal events, detect chains, verify events within each chain are ordered by step (non-decreasing) and reference valid actor IDs
    - **Validates: Requirements 9.3**

- [ ] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Report Layer (evidence-only narrative reports)
  - [ ] 11.1 Implement the Report Agent
    - Create `clyde/reporting/agent.py` with `ReportAgent` class
    - Implement `generate_report(synthesis, db)` → `NarrativeReport` using a ReACT loop
    - Every factual claim must trace to a simulation artifact — LLM generates language, NOT factual content
    - Each claim gets a `ProvenanceAnnotation` with non-null `source_type` ("simulation_db" or "knowledge_graph"), `source_ref`, and `query_used`
    - If evidence can't be found for a claim, drop the claim (never fabricate)
    - Implement `NarrativeReport` and `ProvenanceAnnotation` dataclasses
    - _Requirements: 9.2, 9.5, 9.6_

  - [ ]* 11.2 Write property test for report provenance completeness
    - **Property 14: Report Provenance Completeness**
    - Generate random synthesis results, generate reports, verify every factual claim has a ProvenanceAnnotation with non-null source_type, source_ref, and query_used
    - **Validates: Requirements 9.2, 9.6**

  - [ ] 11.3 Implement uncertainty and limitation flagging
    - Add logic to flag novel regimes: wider outcome distributions + explicit "unknown" flags on affected outputs
    - Weaken causal claims in narrative for novel regime portions
    - Flag reflexivity risk when announcement changes behavior; distinguish pre/post-announcement outcomes
    - Flag heavy-tailed dynamics (panics, fire sales, viral shifts) as harder to simulate
    - Ensure geopolitical/political shocks are treated as exogenous inputs only
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 11.4 Write unit tests for uncertainty flagging
    - Test novel regime → unknown flags present and wider distributions
    - Test reflexivity → dual pre/post-announcement paths
    - Test heavy-tail dynamics → appropriate flag
    - Test geopolitical shocks are exogenous only (never generated endogenously by PropagationEngine)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 12. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Input Layer (LLM-powered setup phase)
  - [ ] 13.1 Implement the Document Ingester
    - Create `clyde/setup/ingestion.py` with `DocumentIngester` class
    - Implement `ingest(file_path)` → `Document` for PDF, Markdown, and plain text formats
    - Implement `supported_formats()` → `{'pdf', 'md', 'txt'}`
    - Raise `UnsupportedFormatError` for other formats; raise `DocumentIngestionError` for corrupted files
    - _Requirements: 16.1_

  - [ ] 13.2 Implement the Scenario Parser
    - Create `clyde/setup/parser.py` with `ScenarioParser` class
    - Implement `parse(description, documents)` → `ParseResult` extracting triggering event, geographies, markets, shock params, time horizon, and actor hints
    - Flag ambiguities for user confirmation instead of making silent assumptions
    - Implement `resolve_ambiguities(parse_result, resolutions)` → `ParseResult` incorporating user responses
    - Accept situations at any economic scale: micro, sectoral, macro, cross-border
    - Handle LLM timeout with 3 retries + exponential backoff
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 13.3 Write property test for ambiguity resolution merge
    - **Property 1: Ambiguity Resolution Merge**
    - Generate random ParseResults with ambiguities and valid resolutions, resolve them, verify every resolved field contains the user-supplied value and the ambiguity is marked resolved
    - **Validates: Requirements 1.3**

  - [ ]* 13.4 Write unit tests for input layer
    - Test DocumentIngester with one PDF, one MD, one TXT file
    - Test ScenarioParser with 3-5 curated scenarios covering micro, sectoral, macro, cross-border scales
    - _Requirements: 1.1, 1.4, 16.1_

- [ ] 14. Knowledge Graph
  - [ ] 14.1 Implement the Knowledge Graph
    - Create `clyde/setup/knowledge_graph.py` with `KnowledgeGraph` class
    - Implement `build_from_documents(documents, parse_result)` using an economic ontology for entity/relationship extraction
    - Implement `extract_shock_config()` → `ShockConfig` from the graph
    - Implement `query(query)` → `list[GraphNode]`
    - Implement `merge_sources(nl_entities, doc_entities)` → `list[Conflict]` flagging conflicts for user resolution
    - Implement `store_simulation_artifact(artifact)` for persisting simulation results back into the graph
    - Knowledge Graph serves as persistent truth store for seed data and simulation artifacts
    - _Requirements: 16.2, 16.3, 16.4, 16.5_

  - [ ]* 14.2 Write unit tests for Knowledge Graph
    - Test build from documents, extract ShockConfig, merge sources with conflict detection
    - _Requirements: 16.2, 16.4, 16.5_

- [ ] 15. Historical Analog Matching and Backtesting
  - [ ] 15.1 Implement historical analog matching
    - Add analog matching logic to the setup phase: match novel situations against structurally similar historical events
    - Use analogs to inform parameter ranges (not dictate outcomes)
    - Disclose which analogs were selected and how they influenced parameter ranges
    - Store analogs as `HistoricalAnalog` objects in `ShockConfig`
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ] 15.2 Implement backtesting support
    - Create `clyde/simulation/backtest.py` with backtesting logic
    - Run simulations against known historical shocks with actual outcome masked
    - Compare simulated distributions against actual historical outcomes
    - Record results in `backtest_results` SQLite table for accuracy tracking over time
    - _Requirements: 11.1, 11.2, 11.3_

  - [ ]* 15.3 Write unit tests for historical analogs and backtesting
    - Test known scenario → expected analog matches with disclosure
    - Test end-to-end backtest against a known historical shock
    - _Requirements: 10.1, 10.2, 10.3, 11.1, 11.2, 11.3_

- [ ] 16. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. God's Eye Console and LLM boundary enforcement
  - [ ] 17.1 Implement the God's Eye Console
    - Create `clyde/setup/gods_eye.py` with `GodsEyeConsole` class
    - Implement `parse_injection(injection_text, base_scenario)` → `ShockDelta` parsing NL interventions into structured deltas
    - Implement `apply_delta(base_config, delta)` → `ShockConfig` merging delta into base config
    - Handle unparseable NL: return partial ShockDelta with ambiguities flagged
    - Validate intervention_step and parameter references
    - _Requirements: 8.5, 8.6, 8.7_

  - [ ]* 17.2 Write unit tests for God's Eye Console
    - Test example interventions parsed into ShockDeltas
    - Test apply_delta merges correctly
    - _Requirements: 8.5, 8.6_

  - [ ] 17.3 Enforce LLM boundary at module level
    - Verify `clyde/simulation/` has zero imports from `clyde/llm/` or any LLM client library
    - Add a CI-compatible lint check (static import analysis) that fails if the boundary is violated
    - Verify the handoff between setup and simulation is a fully-resolved `ShockConfig` + `Actor` list with all params set
    - _Requirements: 15.1, 15.2, 15.3, 15.5_

  - [ ]* 17.4 Write integration test for LLM boundary enforcement
    - Static import analysis confirming `clyde/simulation/` has zero LLM imports
    - _Requirements: 15.3, 15.5_

- [ ] 18. End-to-end integration and wiring
  - [ ] 18.1 Wire the full pipeline together
    - Create `clyde/pipeline.py` (or similar entry point) connecting: Input Layer → Knowledge Graph → Economic World Factory → Monte Carlo Controller → Propagation Engine → Synthesis Engine → Report Layer
    - Wire God's Eye Console into the branch forking flow
    - Ensure the setup phase produces a fully-resolved `ShockConfig` + actors, then hands off to the rule-based simulation phase with no further LLM interaction
    - _Requirements: 15.1, 15.2, 15.4_

  - [ ]* 18.2 Write end-to-end integration tests
    - Test full pipeline: NL input → setup → simulation → synthesis → report with valid provenance
    - Test DB persistence: verify all metrics, causal events, and branches are queryable from SQLite after a run
    - _Requirements: 9.2, 9.6, 17.3_

- [ ] 19. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation — don't skip them
- Property tests validate universal correctness properties from the design document (18 total)
- The simulation phase (`clyde/simulation/`) must never import LLM libraries — this is a hard architectural boundary
- All behavioral parameters are set at actor construction time and varied across Monte Carlo runs, never determined by LLM during simulation
