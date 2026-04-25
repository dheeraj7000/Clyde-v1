# Clyde Economic Simulator — Architecture Diagrams

> **How to render:** These diagrams use [Mermaid](https://mermaid.js.org/) syntax and render automatically in:
> - **GitHub** — natively in any `.md` file preview
> - **VS Code** — with the [Mermaid Preview](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension
> - **Browser** — paste any block into [mermaid.live](https://mermaid.live) for instant rendering

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [End-to-End Pipeline Data Flow](#end-to-end-pipeline-data-flow)
3. [LLM Boundary](#llm-boundary)
4. [Actor Model](#actor-model)
5. [Network Topology](#network-topology)
6. [Monte Carlo Simulation Flow](#monte-carlo-simulation-flow)
7. [Propagation Engine Step Sequence](#propagation-engine-step-sequence)
8. [Data Model Hierarchy](#data-model-hierarchy)
9. [SQLite Persistence Schema](#sqlite-persistence-schema)
10. [Knowledge Graph and Setup Phase](#knowledge-graph-and-setup-phase)
11. [God's Eye Branching](#gods-eye-branching)
12. [Synthesis and Reporting Pipeline](#synthesis-and-reporting-pipeline)
13. [Web API and Job Lifecycle](#web-api-and-job-lifecycle)
14. [Causal Event Propagation Channels](#causal-event-propagation-channels)

---

## System Architecture Overview

Clyde is organized into seven top-level subsystems that form a linear pipeline from raw input to narrative report, with two cross-cutting concerns layered on top: a Web API entry point that wraps the entire pipeline, and a SQLite persistence side-channel that the Monte Carlo Controller writes to and the Report Agent reads from. The diagram below uses fill color to distinguish LLM-powered subsystems (pink) — which may call `clyde/llm/` — from rule-based subsystems (blue) that are strictly LLM-free. The God's Eye Console sits outside the main pipeline as a branching path: it accepts a natural-language injection, produces a `ShockDelta`, and feeds it back into the Monte Carlo Controller to re-simulate from step 0.

```mermaid
graph TD
    classDef llm fill:#f9d,stroke:#c7a,color:#000
    classDef rule fill:#bbf,stroke:#88c,color:#000
    classDef infra fill:#dfd,stroke:#8a8,color:#000

    Web_API["Web API\n(FastAPI + JobStore)"]
    Input_Layer["Input Layer\n(DocumentIngester + ScenarioParser)"]
    Knowledge_Graph["Knowledge Graph\n(KnowledgeGraph + AnalogMatcher)"]
    World_Factory["World Factory\n(EconomicWorldFactory + PriorLibrary)"]
    Monte_Carlo["Monte Carlo Controller\n(MonteCarloController)"]
    Synthesis_Engine["Synthesis Engine\n(SynthesisEngine)"]
    Report_Agent["Report Agent\n(ReportAgent)"]
    Persistence["Persistence\n(SimulationDB / SQLite)"]
    Gods_Eye["God's Eye Console\n(GodsEyeConsole)"]
    SQLite["SQLite\n(clyde/persistence/db.py)"]

    Web_API -->|"RunRequest"| Input_Layer
    Input_Layer -->|"ParseResult"| Knowledge_Graph
    Knowledge_Graph -->|"ShockConfig"| World_Factory
    World_Factory -->|"SimulationWorld"| Monte_Carlo
    Monte_Carlo -->|"EnsembleResult"| Synthesis_Engine
    Synthesis_Engine -->|"SynthesisResult"| Report_Agent
    Report_Agent -->|"NarrativeReport"| Web_API

    Gods_Eye -->|"ShockDelta"| Monte_Carlo
    Monte_Carlo -->|"run / step_metrics / causal_events"| Persistence
    Persistence --> SQLite
    Report_Agent -->|"reads run data"| Persistence

    class Input_Layer llm
    class Knowledge_Graph llm
    class World_Factory llm
    class Report_Agent llm
    class Web_API llm
    class Gods_Eye llm
    class Monte_Carlo rule
    class Synthesis_Engine rule
    class Persistence rule
    class SQLite infra
```

---

## End-to-End Pipeline Data Flow

`ClydePipeline.run()` sequences ten discrete steps from raw input to narrative report. Each step is owned by a single subsystem and passes a typed artifact to the next. Steps 8 and 9 — the Monte Carlo ensemble and the synthesis aggregation — are strictly LLM-free: they operate on fully-resolved `ShockConfig` and `SimulationWorld` objects and never call `clyde/llm/`. Step 5 (analog matching) is conditional: it runs only when `PipelineConfig.use_analogs=True` and an `AnalogMatcher` is present; when skipped, `ShockConfig` flows directly from step 4 to step 6 with an empty `historical_analogs` list.

```mermaid
flowchart LR
    classDef llmfree fill:#bbf,stroke:#88c,color:#000
    classDef llmpowered fill:#f9d,stroke:#c7a,color:#000

    S1["1 · Document Ingestion\ningest_many"]
    S2["2 · Scenario Parsing\nparse"]
    S3["3 · Knowledge Graph Build\nbuild_from_documents"]
    S4["4 · ShockConfig Extraction\nextract_shock_config"]
    USE_ANALOGS{"use_analogs?"}
    S5["5 · Analog Matching\nmatch"]
    S6["6 · World Building\nbuild_scenario"]
    S7["7 · Persona Generation\ngenerate_personas"]
    S8["8 · Monte Carlo Ensemble\nrun_ensemble\n⬡ LLM-free"]
    S9["9 · Synthesis\ncompute_paths\n⬡ LLM-free"]
    S10["10 · Report Generation\ngenerate_report"]

    S1 -->|"list[Document]"| S2
    S2 -->|"ParseResult"| S3
    S3 -->|"KnowledgeGraph"| S4
    S4 -->|"ShockConfig"| USE_ANALOGS
    USE_ANALOGS -->|"yes"| S5
    S5 -->|"ShockConfig + list[HistoricalAnalog]"| S6
    USE_ANALOGS -->|"no"| S6
    S6 -->|"SimulationWorld"| S7
    S7 -->|"list[ActorPersona]"| S8
    S8 -->|"EnsembleResult"| S9
    S9 -->|"SynthesisResult"| S10
    S10 -->|"NarrativeReport"| END(["PipelineResult"])

    class S8 llmfree
    class S9 llmfree
    class S1 llmpowered
    class S2 llmpowered
    class S3 llmpowered
    class S4 llmpowered
    class S5 llmpowered
    class S6 llmpowered
    class S7 llmpowered
    class S10 llmpowered
```

---

## LLM Boundary

The LLM boundary is the most critical architectural invariant in Clyde: modules inside `clyde/simulation/`, `clyde/synthesis/`, `clyde/models/`, and `clyde/persistence/` are permanently forbidden from importing anything in `clyde/llm/` or any third-party LLM provider SDK. This keeps the simulation deterministic, auditable, and free of network calls during the compute-intensive ensemble phase. The boundary is enforced statically by `tests/test_llm_boundary.py`, which walks the AST of every file under `clyde/simulation/` and fails the build if a forbidden import is found. The handoff across the boundary is a fully-resolved triple — `ShockConfig`, `list[Actor]`, and `NetworkBundle` — that carries all behavioral parameters the simulation needs without any further LLM involvement.

```mermaid
graph LR
    subgraph PERMITTED["LLM-Permitted Zone"]
        LLMClient["LLMClient\n(clyde/llm/client.py)"]
        Pipeline["pipeline.py\n(ClydePipeline)"]
        Parser["setup/parser.py\n(ScenarioParser)"]
        KG["setup/knowledge_graph.py\n(KnowledgeGraph)"]
        PersonaGen["setup/persona_generator.py\n(PersonaGenerator)"]
        GodsEye["setup/gods_eye.py\n(GodsEyeConsole)"]
        AgentSim["setup/agent_sim.py\n(AgentSimEngine)"]
        ReportAgent["reporting/agent.py\n(ReportAgent)"]
        WebServer["web/server.py\n(FastAPI app)"]
        WebJobs["web/jobs.py\n(JobStore)"]
        WebAgentRoutes["web/agent_routes.py"]
    end

    subgraph FORBIDDEN["LLM-Forbidden Zone"]
        Propagation["simulation/propagation.py\n(PropagationEngine)"]
        MonteCarlo["simulation/monte_carlo.py\n(MonteCarloController)"]
        Backtest["simulation/backtest.py"]
        SynthEngine["synthesis/engine.py\n(SynthesisEngine)"]
        Models["models/\n(all dataclasses)"]
        Persistence["persistence/db.py\n(SimulationDB)"]
    end

    Parser -->|"imports"| LLMClient
    KG -->|"imports"| LLMClient
    PersonaGen -->|"imports"| LLMClient
    GodsEye -->|"imports"| LLMClient
    AgentSim -->|"imports"| LLMClient
    ReportAgent -->|"imports"| LLMClient
    Pipeline -->|"imports"| LLMClient
    WebServer -->|"imports factory"| LLMClient
    WebJobs -->|"imports factory"| LLMClient
    WebAgentRoutes -->|"imports factory"| LLMClient

    PERMITTED -->|"BOUNDARY — enforced by\ntests/test_llm_boundary.py"| FORBIDDEN

    Pipeline -->|"ShockConfig\n+ list[Actor]\n+ NetworkBundle"| MonteCarlo
```

---

## Actor Model

Every simulation agent is an `Actor` instance carrying a type discriminator (`actor_type`), a typed params object frozen at construction time, a mutable state dict updated each step, and a list of `Relationship` edges to other actors. The four concrete params classes (`HouseholdParams`, `FirmParams`, `BankParams`, `CentralBankParams`) encode the behavioral elasticities drawn from the `PriorLibrary`; the four state classes hold the runtime variables written by the `PropagationEngine` each tick. The `Relationship` dataclass represents a directed, weighted edge between two actors with a validated `rel_type` drawn from the `RELATIONSHIP_TYPES` frozenset. The diagrams below are split into params classes and state classes for readability.

### Actor Base, Params Classes, Relationship, and ActorType

```mermaid
classDiagram
    class Actor {
        +id: str
        +actor_type: ActorType
        +params: Any
        +state: dict
        +relationships: "list~Relationship~"
    }

    class HouseholdParams {
        +mpc: float
        +precautionary_savings_rate: float
        +unemployment_fear_threshold: float
        +wage_demand_elasticity: float
        +inflation_expectation_prior: float
        +inflation_expectation_lr: float
        +credit_seek_threshold: float
    }

    class FirmParams {
        +hurdle_rate: float
        +hiring_elasticity: float
        +firing_threshold: float
        +cost_push_weight: float
        +demand_pull_weight: float
        +supplier_switch_stress: float
        +bankruptcy_threshold: float
        +investment_sensitivity: float
    }

    class BankParams {
        +npl_tightening_elasticity: float
        +herding_weight: float
        +reserve_threshold: float
        +credit_approval_floor: float
        +risk_appetite: float
    }

    class CentralBankParams {
        +taylor_inflation_weight: float
        +taylor_output_weight: float
        +rate_increment: float
        +discretionary_band: float
        +neutral_rate: float
    }

    class Relationship {
        +source_id: str
        +target_id: str
        +rel_type: str
        +weight: float
    }

    class ActorType {
        <<enumeration>>
        HOUSEHOLD
        FIRM
        BANK
        CENTRAL_BANK
    }

    Actor *-- HouseholdParams : params
    Actor *-- FirmParams : params
    Actor *-- BankParams : params
    Actor *-- CentralBankParams : params
    Actor *-- Relationship : relationships
    Actor --> ActorType : actor_type
```

### State Classes

```mermaid
classDiagram
    class HouseholdState {
        +income: float
        +savings: float
        +consumption: float
        +employed: bool
        +employer_id: str
        +debt: float
        +inflation_expectation: float
        +confidence: float
    }

    class FirmState {
        +revenue: float
        +costs: float
        +inventory: float
        +employees: "list~str~"
        +price_level: float
        +investment: float
        +debt: float
        +demand_pressure: float
        +is_bankrupt: bool
        +suppliers: "list~str~"
    }

    class BankState {
        +reserves: float
        +loans_outstanding: float
        +npl_ratio: float
        +credit_tightness: float
        +interbank_borrowed: float
        +is_stressed: bool
    }

    class CentralBankState {
        +policy_rate: float
        +inflation_target: float
        +output_gap_estimate: float
    }
```

---

## Network Topology

The three networks inside a `NetworkBundle` are the wiring layer that turns a flat list of `Actor` objects into an interacting economy. Each network enforces a strict topology constraint: the labor market (`BipartiteGraph`) only connects households to firms, the supply chain (`DirectedGraph`) carries directed firm-to-firm and firm-to-household edges, and the interbank network (`ScaleFreeGraph`) is bank-to-bank only, grown via Barabási–Albert preferential attachment. Every edge in each network maps to a named relationship type (`employment`, `supply`, `trade`, or `lending`) that the `PropagationEngine` reads when propagating shocks. The `NetworkBundle` aggregator holds all three graphs and is the single object handed across the LLM boundary into the simulation phase.

```mermaid
graph LR
    subgraph BUNDLE["NetworkBundle"]
        subgraph LABOR["BipartiteGraph — Labor Market"]
            household_0001(["household_0001"])
            household_0002(["household_0002"])
            firm_0001(["firm_0001"])
            firm_0002(["firm_0002"])
            household_0001 -->|"employment\n(household ↔ firm)"| firm_0001
            household_0002 -->|"employment\n(household ↔ firm)"| firm_0002
        end

        subgraph SUPPLY["DirectedGraph — Supply Chain"]
            firm_A(["firm_0001"])
            firm_B(["firm_0002"])
            firm_C(["firm_0003"])
            hh_A(["household_0001"])
            firm_A -->|"supply\n(firm → firm)"| firm_B
            firm_B -->|"supply\n(firm → firm)"| firm_C
            firm_C -->|"trade\n(firm → household)"| hh_A
        end

        subgraph INTERBANK["ScaleFreeGraph — Interbank"]
            bank_0001(["bank_0001"])
            bank_0002(["bank_0002"])
            bank_0003(["bank_0003"])
            bank_0001 -->|"lending\n(bank → bank)"| bank_0002
            bank_0002 -->|"lending\n(bank → bank)"| bank_0003
            bank_0001 -->|"lending\n(bank → bank)"| bank_0003
        end
    end

    BipartiteGraph["BipartiteGraph\nconstraint: household ↔ firm only\nrel_type: employment"]
    DirectedGraph["DirectedGraph\nconstraint: firm → firm or firm → household\nrel_type: supply / trade"]
    ScaleFreeGraph["ScaleFreeGraph\nconstraint: bank → bank only\nrel_type: lending"]
    NetworkBundle["NetworkBundle\nlabor_market: BipartiteGraph\nsupply_chain: DirectedGraph\ninterbank: ScaleFreeGraph"]

    NetworkBundle --> BipartiteGraph
    NetworkBundle --> DirectedGraph
    NetworkBundle --> ScaleFreeGraph
```

---

## Monte Carlo Simulation Flow

`MonteCarloController.run_ensemble()` turns a single `SimulationWorld` into a statistically robust `EnsembleResult` by running hundreds of deterministically seeded, independently jittered trajectories. The flow begins with Knuth-multiplicative seed derivation — each per-run seed is produced by multiplying the ensemble seed by the golden-ratio constant `2654435761`, XOR-ing with the run index, and masking to 32 bits, with collision re-derivation via a large-prime bump. Each seed then drives `_jitter_world_impl`, which perturbs actor params across four axes: behavioral response strength (multiplicative noise on every numeric param), timing (additive offset on firm hiring/firing thresholds), shock severity (clamped multiplicative noise on `ShockConfig.severity`), and contagion thresholds (bank `reserve_threshold` and `credit_approval_floor` are part of the multiplicative pass). The jittered world is dispatched to a `ProcessPoolExecutor` worker pool; if the pool breaks or `parallel=False` is set, execution falls back to a serial loop in the orchestrator process. Each worker calls `PropagationEngine.run()` and returns a `TrajectoryResult`; failed workers are logged and warned about but do not abort the ensemble. Successful trajectories are aggregated into an `EnsembleResult`. When a `SimulationDB` handle is supplied, the orchestrator process (never the worker) writes each run record, its step metrics batch, and its causal events to SQLite — keeping the non-picklable DB connection safely out of the process pool.

```mermaid
flowchart TD
    START(["SimulationWorld\n+ ensemble_seed"])
    GEN_SEEDS["_generate_seeds\nKnuth: seed = (ensemble_seed × 2654435761) XOR i\nmask to 32-bit · collision → bump by prime"]
    JITTER["_jitter_world_impl\nper-run jittered SimulationWorld"]
    JITTER_AXES["Jitter axes:\n① behavioral response strength\n② timing — hiring/firing thresholds\n③ shock severity — clamped\n④ contagion thresholds — reserve/credit floor"]
    PARALLEL{"parallel=True\nand pool healthy?"}
    POOL["ProcessPoolExecutor\n_run_one worker per seed"]
    SERIAL["Serial fallback loop\n_run_serial"]
    ENGINE["PropagationEngine.run()\nper worker / per iteration"]
    TRAJ["TrajectoryResult\nsteps + causal_events"]
    AGGREGATE["Aggregate results\ncollect successes · log failures · warn"]
    ENSEMBLE(["EnsembleResult\nscenario_id · config · trajectories\nrun_count · ensemble_seed"])
    DB_CHECK{"db supplied?"}
    DB_WRITE["SimulationDB\ninsert_run\ninsert_step_metrics_batch\ninsert_causal_event"]

    START --> GEN_SEEDS
    GEN_SEEDS --> JITTER
    JITTER -. "four axes" .-> JITTER_AXES
    JITTER --> PARALLEL
    PARALLEL -->|"yes"| POOL
    PARALLEL -->|"no / BrokenProcessPool"| SERIAL
    POOL --> ENGINE
    SERIAL --> ENGINE
    ENGINE --> TRAJ
    TRAJ --> AGGREGATE
    AGGREGATE --> ENSEMBLE
    AGGREGATE --> DB_CHECK
    DB_CHECK -->|"yes"| DB_WRITE
    DB_CHECK -->|"no"| DONE(["done"])
    DB_WRITE --> DONE
```

---

## Propagation Engine Step Sequence

`PropagationEngine._step()` executes nine actor-update phases in strict order every simulation tick. Before phase 1 runs at step 0, a pre-phase applies the raw shock severity directly to each `initial_contact_actors` entry — this is the only hardcoded injection point; all downstream effects must be network-mediated. Phases 2 and 3 are the primary `CausalEvent` emitters: phase 2 fires a `monetary_policy` event when the central-bank policy-rate signal exceeds 0.2 and a `lending` event when a bank's `credit_tightness` rises by ≥ 5%; phase 3 fires a `trade` event when a firm's price level jumps ≥ 2%; phases 4 and 7 fire `employment` events when a firm fires a worker or goes bankrupt. Phase 9 computes the nine core `StepMetrics` — including `observed_inflation` and `output_gap_estimate` — which the central bank reads at the start of the next tick to update its Taylor-rule rate, closing the feedback loop from phase 9 back to phase 1.

```mermaid
flowchart TD
    PRE["Pre-phase — step 0 only\n_apply_initial_shock\nReads: severity, initial_contact_actors\nWrites: revenue / income / npl_ratio / policy_rate"]

    P1["Phase 1 — Central Bank\n_update_central_banks\nReads: observed_inflation, output_gap_estimate\nWrites: policy_rate"]

    P2["Phase 2 — Banks\n_update_banks\nReads: policy_rate, npl_ratio, peer credit_tightness\nWrites: credit_tightness, npl_ratio, reserves, is_stressed"]

    P2_MP["CausalEvent: monetary_policy\nCentralBank → Bank\npolicy_rate_signal > 0.2 and delta > 0\nvariable_affected: credit_tightness"]

    P2_LN["CausalEvent: lending\nBank → Firm\ncredit_tightness delta ≥ 5%\nvariable_affected: credit_tightness"]

    P3["Phase 3 — Firms\n_update_firms + _fire_one_employee\nReads: credit_tightness, policy_rate, demand_pressure\nWrites: investment, price_level, revenue, costs, is_bankrupt, employee_count"]

    P3_TR["CausalEvent: trade\nFirm → Household\nprice spike ≥ 2%\nvariable_affected: price_level"]

    P3_EM["CausalEvent: employment\nFirm → Household\nfirm fires employee\nvariable_affected: employed"]

    P4["Phase 4 — Labor Market\n_clear_labor_market\nReads: is_bankrupt, household_employer\nWrites: employed, income\n(no new CausalEvents — firings already emitted in phase 3)"]

    P5["Phase 5 — Households\n_update_households\nReads: income, inflation_expectation, unemployment_fear\nWrites: consumption, savings, wage_demand, confidence, demand_pressure on firms"]

    P6["Phase 6 — Interbank Settlement\n_interbank_settlement\nReads: is_stressed, interbank edges\nWrites: npl_ratio (peer contagion)\n(no CausalEvents emitted)"]

    P7["Phase 7 — Bankruptcy Cascade\n_bankruptcy_cascade\nReads: is_bankrupt, firm_suppliers, firm_employees\nWrites: demand_pressure, stress, employed, income, revenue"]

    P7_EM["CausalEvent: employment\nFirm → Household\nemployer bankruptcy\nvariable_affected: employed"]

    P7_SP["CausalEvent: supply\nBankrupt Firm → Supplier\nbuyer went bankrupt\nvariable_affected: revenue"]

    P8["Phase 8 — Bayesian Learning\n_learning_updates\nReads: prev_price_level, price_level\nWrites: inflation_expectation on households"]

    P9["Phase 9 — Metrics\n_compute_metrics\nReads: all actor state\nWrites: StepMetrics\n(gdp_index, inflation_rate, unemployment_rate,\ngini, credit_tightening_index, firm_bankruptcy_count,\nbank_stress_index, consumer_confidence, interbank_freeze)"]

    NEXT_STEP(["Next step →\nPhase 1 reads StepMetrics\nto update Taylor-rule rate"])

    PRE --> P1
    P1 --> P2
    P2 -. "emits" .-> P2_MP
    P2 -. "emits" .-> P2_LN
    P2 --> P3
    P3 -. "emits" .-> P3_TR
    P3 -. "emits" .-> P3_EM
    P3 --> P4
    P4 --> P5
    P5 --> P6
    P6 --> P7
    P7 -. "emits" .-> P7_EM
    P7 -. "emits" .-> P7_SP
    P7 --> P8
    P8 --> P9
    P9 -->|"observed_inflation\noutput_gap_estimate\nfeed into next tick"| NEXT_STEP
    NEXT_STEP -->|"feedback loop"| P1
```

---

## Data Model Hierarchy

The sixteen core dataclasses form three loosely coupled clusters. The **input/config cluster** (`TimeHorizon`, `HistoricalAnalog`, `ShockConfig`, `ShockDelta`, `SimulationWorld`, `Scenario`) carries everything needed to initialise and fork a simulation run. The **simulation output cluster** (`StepMetrics`, `CausalEvent`, `TrajectoryResult`, `EnsembleResult`) holds the raw per-step and per-run results produced by the `PropagationEngine` and `MonteCarloController`. The **synthesis/reporting cluster** (`PathBundle`, `CausalChain`, `DivergenceVariable`, `DivergenceMap`, `SynthesisResult`, `NarrativeReport`) holds the aggregated, human-readable outputs produced by `SynthesisEngine` and `ReportAgent`. Composition arrows (`*--`) indicate that the parent owns and serialises the child; aggregation arrows (`o--`) indicate that the parent holds a reference list but the child objects are also meaningful on their own.

```mermaid
classDiagram
    class TimeHorizon {
        +steps: int
        +step_unit: str
    }

    class HistoricalAnalog {
        +event_name: str
        +year: int
        +similarity_score: float
        +param_adjustments: dict
        +source: str
    }

    class ShockConfig {
        +shock_type: str
        +severity: float
        +scope: str
        +duration_steps: int
        +geography: "list~str~"
        +sectors: "list~str~"
        +initial_contact_actors: "list~str~"
        +agent_counts: dict
        +behavioral_overrides: dict
        +ensemble_seed: int
    }

    class ShockDelta {
        +intervention_step: int
        +param_overrides: dict
        +new_events: "list~str~"
        +description: str
    }

    class SimulationWorld {
        +prior_library_version: str
    }

    class Scenario {
        +scenario_id: str
        +description: str
        +prior_library_version: str
        +overrides: dict
        +metadata: dict
    }

    class StepMetrics {
        +step: int
        +gdp_index: float
        +inflation_rate: float
        +unemployment_rate: float
        +gini_coefficient: float
        +credit_tightening_index: float
        +firm_bankruptcy_count: int
        +bank_stress_index: float
        +consumer_confidence: float
        +interbank_freeze: bool
        +custom_metrics: dict
    }

    class CausalEvent {
        +step: int
        +source_actor_id: str
        +target_actor_id: str
        +channel: str
        +variable_affected: str
        +magnitude: float
        +description: str
    }

    class TrajectoryResult {
        +run_id: str
        +seed: int
        +final_state_ref: str
    }

    class EnsembleResult {
        +scenario_id: str
        +run_count: int
        +ensemble_seed: int
    }

    class PathBundle {
        +central: "list~StepMetrics~"
        +optimistic: "list~StepMetrics~"
        +pessimistic: "list~StepMetrics~"
        +tail_upper: "list~StepMetrics~"
        +tail_lower: "list~StepMetrics~"
    }

    class CausalChain {
        +chain_id: str
        +origin_shock: str
        +total_magnitude: float
    }

    class DivergenceVariable {
        +name: str
        +sensitivity: float
        +current_uncertainty: float
        +monitoring_indicator: str
    }

    class DivergenceMap {
    }

    class SynthesisResult {
        +scenario_id: str
    }

    class NarrativeReport {
        +scenario_id: str
        +uncertainty_flags: "list~str~"
    }

    %% ShockConfig composition
    ShockConfig *-- TimeHorizon : time_horizon
    ShockConfig o-- HistoricalAnalog : historical_analogs

    %% SimulationWorld composition
    SimulationWorld *-- ShockConfig : config

    %% Scenario composition
    Scenario *-- ShockConfig : config

    %% EnsembleResult composition
    EnsembleResult *-- ShockConfig : config
    EnsembleResult o-- TrajectoryResult : trajectories

    %% TrajectoryResult composition
    TrajectoryResult o-- StepMetrics : steps
    TrajectoryResult o-- CausalEvent : causal_events

    %% PathBundle (aggregated from StepMetrics bands)
    PathBundle o-- StepMetrics : bands

    %% CausalChain composition
    CausalChain o-- CausalEvent : events

    %% DivergenceMap composition
    DivergenceMap o-- DivergenceVariable : variables

    %% SynthesisResult composition
    SynthesisResult *-- ShockConfig : config
    SynthesisResult *-- PathBundle : paths
    SynthesisResult *-- DivergenceMap : divergence_map
    SynthesisResult o-- CausalChain : causal_chains
```

---

## SQLite Persistence Schema

The five SQLite tables form two distinct relationship clusters. The `runs` table is the central hub: both `step_metrics` and `causal_events` hang off it via `run_id` foreign keys, giving each run a full time-series of economic metrics and a complete log of actor-to-actor shock transmissions. The `branches` table is intentionally decoupled — it references a `parent_scenario_id` (a scenario identifier, not a run identifier) because a branch is a re-simulation of an entire scenario under a modified `ShockConfig`, not a continuation of a specific run. The `backtest_results` table is similarly standalone, keyed by `scenario_id` to allow multiple historical comparisons per scenario. All JSON columns (`config_json`, `custom_metrics_json`, `shock_delta_json`, `merged_config_json`, `actual_outcome_json`, `simulated_distribution_json`) store serialized dataclass payloads that are deserialized by `SimulationDB` read methods before being returned to callers.

```mermaid
erDiagram
    runs {
        TEXT run_id PK "UUID for this simulation run"
        TEXT scenario_id "Scenario this run belongs to"
        TEXT branch_id "NULL for baseline runs"
        INTEGER seed "Deterministic RNG seed"
        TEXT config_json "Serialized ShockConfig"
        TEXT status "running | completed | failed"
        TIMESTAMP created_at "Row creation time"
    }

    step_metrics {
        TEXT run_id PK FK "References runs.run_id"
        INTEGER step PK "Simulation tick (0-based)"
        REAL gdp_index "GDP index value"
        REAL inflation_rate "Observed inflation rate"
        REAL unemployment_rate "Unemployment rate"
        REAL gini_coefficient "Income inequality measure"
        REAL credit_tightening_index "Credit tightness 0-1"
        INTEGER firm_bankruptcy_count "Bankrupt firms this step"
        REAL bank_stress_index "Bank stress 0-1"
        REAL consumer_confidence "Consumer confidence 0-1"
        INTEGER interbank_freeze "1 if interbank frozen else 0"
        TEXT custom_metrics_json "Extra metrics as JSON dict"
    }

    causal_events {
        INTEGER event_id PK "Auto-increment surrogate key"
        TEXT run_id FK "References runs.run_id"
        INTEGER step "Simulation tick when event fired"
        TEXT source_actor_id "Originating actor ID"
        TEXT target_actor_id "Receiving actor ID"
        TEXT channel "monetary_policy|lending|trade|employment|supply"
        TEXT variable_affected "State variable changed"
        REAL magnitude "Signed shock magnitude"
        TEXT description "Human-readable event summary"
    }

    branches {
        TEXT branch_id PK "UUID for this branch"
        TEXT parent_scenario_id "Scenario that was forked"
        TEXT shock_delta_json "Serialized ShockDelta override"
        TEXT merged_config_json "Serialized merged ShockConfig"
        TIMESTAMP created_at "Row creation time"
    }

    backtest_results {
        TEXT backtest_id PK "UUID for this backtest"
        TEXT scenario_id "Scenario being validated"
        TEXT historical_event "Name of historical reference event"
        TEXT actual_outcome_json "Observed historical outcome"
        TEXT simulated_distribution_json "Simulated outcome distribution"
        REAL accuracy_score "Comparison score (nullable)"
        TIMESTAMP created_at "Row creation time"
    }

    runs ||--o{ step_metrics : "has many steps"
    runs ||--o{ causal_events : "has many events"
```

---

## Knowledge Graph and Setup Phase

The setup phase transforms two heterogeneous input streams — raw documents (PDF, Markdown, plain text) and a natural-language scenario description — into a fully resolved `ShockConfig` and `SimulationWorld` ready for the simulation engine. `DocumentIngester.ingest_many` loads files into `Document` objects while `ScenarioParser.parse` converts the NL description into a structured `ParseResult`; both streams converge at `KnowledgeGraph.build_from_documents`, which calls the LLM once per document to extract entities and relations. The `merge_sources()` step reconciles NL-derived and document-derived entities by id: any field-level disagreement is recorded as a `Conflict` and the entity is withheld until the user resolves it. `extract_shock_config()` then assembles a `ShockConfig` from the merged graph and the `ParseResult`, extending geography and sector lists from every matching graph entity. `EconomicWorldFactory._build()` takes that config through five deterministic steps: spawn actors from `PriorLibrary` empirical priors, validate and apply behavioral overrides, wire the three networks via `NetworkBuilder`, attach per-actor `Relationship` lists from network edges, and cross-validate every relationship. Finally, `AnalogMatcher.match()` is an optional enrichment step — when `use_analogs=True`, it scores the `ShockConfig` against a curated corpus of historical events and appends the top-k `HistoricalAnalog` records to the config before it crosses the LLM boundary into the simulation phase.

```mermaid
flowchart LR
    RAW_DOCS(["Raw Documents\nPDF / MD / TXT"])
    NL_DESC(["NL Scenario Description"])

    INGESTER["DocumentIngester\ningest_many\n→ list[Document]"]
    PARSER["ScenarioParser\nparse\n→ ParseResult"]

    KG["KnowledgeGraph\nbuild_from_documents\n(LLM: one call per document)"]

    MERGE["merge_sources()\nNL entities vs doc entities\nConflict → withheld until resolved"]

    EXTRACT["extract_shock_config()\nParseResult + graph entities\n→ ShockConfig"]

    ANALOG_CHECK{"use_analogs?"}

    ANALOG["AnalogMatcher\nmatch(shock_config)\nscores corpus · top-k\n→ list[HistoricalAnalog]\nappended to ShockConfig"]

    PRIOR["PriorLibrary\nget_params(actor_type)\nversioned empirical priors"]

    FACTORY["EconomicWorldFactory\n_build()"]

    SPAWN["① Spawn actors\nfrom PriorLibrary priors\nper agent_counts in ShockConfig"]

    OVERRIDES["② Apply overrides\nbehavioral_overrides + param_overrides\nper-id beats per-type"]

    NETWORKS["③ Build networks\nNetworkBuilder\nlabor_market · supply_chain · interbank"]

    ATTACH["④ Attach relationships\nemployment · supply · trade · lending\nfrom network edges"]

    VALIDATE["⑤ Validate\ncross-check all rel source/target ids\nand rel_type values"]

    WORLD(["SimulationWorld\nconfig · actors · networks\nprior_library_version"])

    RAW_DOCS --> INGESTER
    NL_DESC --> PARSER
    INGESTER -->|"list[Document]"| KG
    PARSER -->|"ParseResult"| KG
    KG --> MERGE
    MERGE -->|"merged entities + relations"| EXTRACT
    PARSER -->|"ParseResult"| EXTRACT
    EXTRACT -->|"ShockConfig"| ANALOG_CHECK
    ANALOG_CHECK -->|"yes"| ANALOG
    ANALOG -->|"ShockConfig + HistoricalAnalogs"| FACTORY
    ANALOG_CHECK -->|"no"| FACTORY
    PRIOR -->|"typed params"| SPAWN
    FACTORY --> SPAWN
    SPAWN --> OVERRIDES
    OVERRIDES --> NETWORKS
    NETWORKS --> ATTACH
    ATTACH --> VALIDATE
    VALIDATE --> WORLD
```

---

## God's Eye Branching

The God's Eye Console is the only mechanism that lets an analyst fork a live scenario mid-timeline without touching the running simulation state. The lifecycle begins when `ClydePipeline.fork_branch()` receives a natural-language injection string and passes it — together with a serialized snapshot of the base `Scenario` — to `GodsEyeConsole.parse_injection()`. The console sends a structured prompt to the LLM and retries up to `max_retries` times with exponential backoff; on success it builds a `ShockDelta` from the JSON response. Any ambiguities the LLM could not resolve (or that fail range validation, such as an out-of-range `intervention_step`) are stashed under the magic key `AMBIGUITY_KEY` (`"_ambiguities"`) inside `ShockDelta.param_overrides` so callers can surface a UI prompt — the key is stripped before the delta is merged so it never pollutes a runnable `ShockConfig`. `MonteCarloController.merge_delta()` then applies the delta to the base config using two promotion rules: top-level fields (`severity`, `duration_steps`, `shock_type`, `scope`) found in `param_overrides` replace the corresponding `ShockConfig` fields directly, while all remaining keys are merged into `behavioral_overrides` with delta winning on conflict. The merged config drives a fresh `run_ensemble()` call from step 0 — never mutating any running simulation state — and the resulting `BranchResult` (carrying `branch_id`, `parent_scenario_id`, `delta`, `merged_config`, and `ensemble`) is appended to `PipelineResult.branches` and persisted via `SimulationDB.insert_branch()`.

```mermaid
sequenceDiagram
    participant Analyst as "Analyst"
    participant Pipeline as "ClydePipeline"
    participant GodsEye as "GodsEyeConsole"
    participant LLM as "LLMClient"
    participant MC as "MonteCarloController"
    participant Engine as "PropagationEngine"
    participant DB as "SimulationDB"

    Analyst->>Pipeline: fork_branch(result, injection_text)
    activate Pipeline

    Pipeline->>Pipeline: build base_scenario stub\n(scenario_id, shock_config, actors, networks)
    Pipeline->>GodsEye: parse_injection(injection_text, base_scenario)
    activate GodsEye

    Note over GodsEye: Build system + user messages\n(base scenario snapshot + injection text)

    loop up to max_retries (default 3)
        GodsEye->>LLM: complete_json(messages)
        LLM-->>GodsEye: JSON dict or exception
        Note over GodsEye: Retry with exponential backoff\non failure
    end

    GodsEye->>GodsEye: _build_delta(response, injection_text, base_scenario)
    Note over GodsEye: Validate intervention_step against\ntime_horizon.steps — clip if out of range

    alt ambiguities exist (LLM-flagged or validation-clipped)
        GodsEye->>GodsEye: stash under param_overrides[AMBIGUITY_KEY]\n("_ambiguities": [list of Ambiguity dicts])
    end

    GodsEye-->>Pipeline: ShockDelta\n(intervention_step, param_overrides,\nnew_events, description)
    deactivate GodsEye

    Pipeline->>MC: fork_branch(base_world, delta, run_count,\nparent_scenario_id, db)
    activate MC

    MC->>MC: merge_delta(base_config, delta)
    Note over MC: Strip AMBIGUITY_KEY from param_overrides\nPromote top-level keys (severity, duration_steps,\nshock_type, scope) directly onto ShockConfig\nMerge remaining keys into behavioral_overrides\n(delta wins on conflict)\nAppend new_events into behavioral_overrides["new_events"]

    MC->>MC: Build forked SimulationWorld\n(merged_config + base actors + base networks)

    Note over MC: Re-simulate from step 0\nNEVER mutates running state

    MC->>Engine: run_ensemble(forked_world, run_count, db)
    activate Engine
    Engine-->>MC: EnsembleResult
    deactivate Engine

    MC->>DB: insert_branch(branch_id, parent_scenario_id,\nshock_delta, merged_config)
    activate DB
    DB-->>MC: ok
    deactivate DB

    MC-->>Pipeline: BranchResult\n(branch_id, parent_scenario_id,\ndelta, merged_config, ensemble)
    deactivate MC

    Pipeline->>Pipeline: result.branches.append(branch_result)
    Pipeline-->>Analyst: BranchResult
    deactivate Pipeline
```

---

## Synthesis and Reporting Pipeline

The synthesis and reporting pipeline is the final, fully deterministic stage of Clyde's processing chain — with one narrow, clearly annotated exception. `SynthesisEngine` (LLM-free) runs four operations in sequence on the raw `EnsembleResult`: `compute_paths()` collapses all trajectories into five percentile bands (p03/p10/p50/p90/p97) per metric per step, inverting the band assignment for "lower is better" metrics so that `optimistic` always means the best outcome regardless of direction; `compute_divergence_map()` ranks metrics by cross-trajectory variance to surface the top drivers of outcome uncertainty; `detect_causal_chains()` groups per-trajectory `CausalEvent` sequences by their (source, target, channel) signature and deduplicates them into canonical `CausalChain` records; and `select_metrics()` combines divergence-map variables with the metrics showing the largest end-to-end delta in the central path. `ReportAgent.generate_report()` then consumes the resulting `SynthesisResult` in five steps: it first detects uncertainty flags programmatically from the `ShockConfig` and `PathBundle` (scanning for keyword matches and tail-band dispersion), then builds four evidence-only section bodies — Outcome Range, Causal Pathways, Divergence and Watchlist, and Uncertainty Flags — each backed by `ProvenanceAnnotation` records. Only after all factual content is assembled does the agent make a single call to `_llm_polish()`, which asks the LLM to write decorative prose paragraphs without introducing any new numbers, names, or facts. This is the only LLM call in the entire synthesis and reporting pipeline.

```mermaid
flowchart LR
    ENSEMBLE(["EnsembleResult\ntrajectories · config · run_count"])

    subgraph SYNTH["SynthesisEngine — LLM-free"]
        CP["compute_paths()\nper step · per metric\nsort values across trajectories"]
        PCTILE["Percentile logic\np03 · p10 · p50 · p90 · p97\nhigher_is_better: opt=p90 · pes=p10 · tu=p97 · tl=p03\nlower_is_better: opt=p10 · pes=p90 · tu=p03 · tl=p97"]
        PB(["PathBundle\ncentral · optimistic · pessimistic\ntail_upper · tail_lower"])

        CDM["compute_divergence_map()\ncross-traj variance per metric\nCV at final step · rank top-k"]
        DM(["DivergenceMap\nvariables: name · sensitivity\nuncertainty · monitoring_indicator"])

        DCC["detect_causal_chains()\ngroup by (source, target, channel) signature\ndeduplicate · sort by frequency"]
        CC(["list[CausalChain]\nchain_id · origin_shock\nevents · total_magnitude"])

        SM["select_metrics()\ndivergence-map vars\n+ top-3 end-to-end delta metrics"]
        MS(["list[MetricSelection]\nmetric · reason"])
    end

    subgraph REPORT["ReportAgent.generate_report()"]
        FLAGS["detect_uncertainty_flags()\nkeyword scan on shock_type · geography · overrides\n+ tail-band dispersion check"]
        FLAG_LIST(["Uncertainty flags detected\nunknown_regime\nreflexivity_risk\nheavy_tail_dynamics\nexogenous_geopolitical\nhigh_outcome_dispersion"])

        S1["_build_outcome_range_section()\nband summaries at horizon\nreflexivity: pre/post split\nheavy-tail softening · unknown-regime widening"]
        S2["_build_causal_pathways_section()\nchain-by-chain event listing\nstep · source → target · channel · magnitude"]
        S3["_build_divergence_section()\ntop divergence drivers\nsensitivity · uncertainty · monitoring indicator"]
        S4["_build_uncertainty_section()\nlist fired flags with provenance"]

        POLISH["_llm_polish()\n⚠ ONLY LLM CALL in this pipeline\nwrites decorative prose paragraphs\nDO NOT introduce new facts · numbers · names"]

        NR(["NarrativeReport\nscenario_id · sections\nprovenance · uncertainty_flags"])
    end

    ENSEMBLE --> CP
    CP -. "percentile logic" .-> PCTILE
    CP --> PB
    PB --> CDM
    CDM --> DM
    DM --> DCC
    DCC --> CC
    CC --> SM
    SM --> MS

    MS --> FLAGS
    PB --> FLAGS
    FLAGS --> FLAG_LIST
    FLAG_LIST --> S1
    FLAG_LIST --> S2
    FLAG_LIST --> S3
    FLAG_LIST --> S4
    PB --> S1
    CC --> S2
    DM --> S3

    S1 --> POLISH
    S2 --> POLISH
    S3 --> POLISH
    S4 --> POLISH
    POLISH --> NR
```

---

## Web API and Job Lifecycle

The web layer is a thin FastAPI application that wraps the entire pipeline behind an async job queue. Every `POST /api/runs` request creates a `Job` in the `JobStore`, spawns an `asyncio.Task` that drives `run_pipeline_job()`, and immediately returns a `job_id` for polling. Progress advances through five coarse stages — `parsing`, `building world`, `running ensemble`, `synthesizing`, and `generating report` — before landing in `completed` or `failed`. Branch runs follow the same pattern: `POST /api/runs/{job_id}/branches` creates a `BranchJob` (a child of the parent `Job`) and spawns a second task that calls `run_branch_job()`. The `PipelineFactory` callable is the single injection point that lets tests substitute a stub pipeline without touching any route logic — the server checks `app.state.pipeline_factory_override` first and falls back to the real LLM-backed factory. The four agent-sim endpoints (`/agent-sim/start`, `/agent-sim/{sim_id}/round`, `/agent-sim/{sim_id}/inject`, `/agent-sim/{sim_id}/state`) operate on a per-job in-memory `agent_sim` dict and require the parent job to be in `completed` status before they can be called.

```mermaid
flowchart TD
    subgraph ENDPOINTS["REST Endpoints"]
        EP_ROOT["GET /\nServe index.html"]
        EP_HEALTH["GET /api/health\nProvider availability check"]
        EP_SAMPLE["GET /api/scenarios/sample\nReturn sample scenario seeds"]
        EP_POST_RUN["POST /api/runs\nCreate job + spawn asyncio.Task"]
        EP_GET_RUN["GET /api/runs/{job_id}\nPoll job status + result"]
        EP_POST_BRANCH["POST /api/runs/{job_id}/branches\nFork branch off completed job"]
        EP_GET_BRANCH["GET /api/runs/{job_id}/branches/{branch_id}\nPoll branch status + result"]
        EP_SIM_START["POST /api/runs/{job_id}/agent-sim/start\nInit AgentSimEngine from completed job"]
        EP_SIM_ROUND["POST /api/runs/{job_id}/agent-sim/{sim_id}/round\nAdvance one simulation round"]
        EP_SIM_INJECT["POST /api/runs/{job_id}/agent-sim/{sim_id}/inject\nInject NL event into running sim"]
        EP_SIM_STATE["GET /api/runs/{job_id}/agent-sim/{sim_id}/state\nRead current sim state + memories"]
    end

    subgraph STORE["JobStore"]
        JS["JobStore\nOrderedDict of Job objects\nmax_jobs=32 soft eviction\nasync lock for create/get"]
    end

    subgraph JOB_LIFECYCLE["Job Lifecycle"]
        JOB_CREATE["Job created\nstatus: pending\nprogress: queued 0%"]

        FACTORY_CHECK{"app.state.pipeline_factory_override\nset?"}
        FACTORY_REAL["PipelineFactory\n(real LLM-backed pipeline)"]
        FACTORY_STUB["PipelineFactory\n(test stub override)"]
        PIPELINE_FACTORY["PipelineFactory\nCallable[[PipelineConfig], ClydePipeline]\ninjection point for tests"]

        TASK["asyncio.Task\nrun_pipeline_job()"]

        STAGE1["Stage: parsing\n5%"]
        STAGE2["Stage: building world\n25%"]
        STAGE3["Stage: simulating ensemble\n35%"]
        STAGE4["Stage: synthesizing\n80%"]
        STAGE5["Stage: generating report\n92%"]

        JOB_DONE["status: completed\nprogress: 100%\nresult: serialized PipelineResult"]
        JOB_FAIL["status: failed\nerror: ExcType: message\ndetails: traceback (if CLYDE_DEBUG)"]

        subgraph BRANCH_LIFECYCLE["BranchJob Lifecycle (child of Job)"]
            BJ_CREATE["BranchJob created\nbranch_id: UUID\nstatus: pending"]
            BJ_TASK["asyncio.Task\nrun_branch_job()"]
            BJ_STAGE1["Stage: parsing injection\n10%"]
            BJ_STAGE2["Stage: running branch ensemble\n50%"]
            BJ_DONE["status: completed\nresult: serialized BranchResult"]
            BJ_FAIL["status: failed\nerror: ExcType: message"]
        end
    end

    EP_POST_RUN -->|"create_job_async()"| JS
    JS --> JOB_CREATE
    JOB_CREATE --> FACTORY_CHECK
    FACTORY_CHECK -->|"yes"| FACTORY_STUB
    FACTORY_CHECK -->|"no"| FACTORY_REAL
    FACTORY_STUB --> PIPELINE_FACTORY
    FACTORY_REAL --> PIPELINE_FACTORY
    PIPELINE_FACTORY --> TASK
    TASK --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> STAGE3
    STAGE3 --> STAGE4
    STAGE4 --> STAGE5
    STAGE5 --> JOB_DONE
    STAGE5 --> JOB_FAIL

    EP_GET_RUN -->|"store.get(job_id)"| JS
    EP_POST_BRANCH -->|"requires status=completed"| JOB_DONE
    JOB_DONE --> BJ_CREATE
    BJ_CREATE --> BJ_TASK
    BJ_TASK --> BJ_STAGE1
    BJ_STAGE1 --> BJ_STAGE2
    BJ_STAGE2 --> BJ_DONE
    BJ_STAGE2 --> BJ_FAIL
    EP_GET_BRANCH -->|"job.branches[branch_id]"| BJ_DONE

    EP_SIM_START -->|"requires status=completed"| JOB_DONE
    EP_SIM_ROUND --> EP_SIM_STATE
    EP_SIM_INJECT --> EP_SIM_STATE
```

---

## Causal Event Propagation Channels

Every shock that propagates through the Clyde network is recorded as a `CausalEvent` — a typed, actor-to-actor transmission record carrying a channel name, the state variable it affects, and a signed magnitude. There are four channels, each emitted by a distinct phase of `PropagationEngine._step()`: `monetary_policy` fires in phase 2 when the central-bank policy-rate signal exceeds 0.2 and the bank's credit tightness is rising (CentralBank → Bank, `variable_affected: credit_tightness`); `lending` fires in phase 2 when a bank's credit tightness jumps by ≥ 5% in a single step (Bank → Firm, `variable_affected: credit_tightness`); `trade` fires in phase 3 when a firm's price level spikes by ≥ 2% (Firm → Household, `variable_affected: price_level`); and `employment` fires in phases 3 and 7 when a firm fires a worker or goes bankrupt (Firm → Household, `variable_affected: employed`). After the ensemble completes, `SynthesisEngine.detect_causal_chains()` collapses the per-trajectory `CausalEvent` streams into canonical `CausalChain` records by grouping on the (source, target, channel) signature tuple, deduplicating within each group, and sorting chains by cross-trajectory frequency — turning thousands of raw events into a compact, human-readable causal narrative.

```mermaid
graph LR
    CB(["CentralBank"])
    Bank(["Bank"])
    Firm(["Firm"])
    HH(["Household"])

    CB -->|"monetary_policy\ncondition: policy_rate_signal > 0.2 and delta > 0\nvariable_affected: credit_tightness"| Bank
    Bank -->|"lending\ncondition: credit_tightness delta >= 5%\nvariable_affected: credit_tightness"| Firm
    Firm -->|"trade\ncondition: price spike >= 2%\nvariable_affected: price_level"| HH
    Firm -->|"employment\ncondition: firm fires employee\nvariable_affected: employed"| HH

    subgraph EVENTS["Per-trajectory CausalEvent records"]
        EV["CausalEvent\nstep · source_actor_id · target_actor_id\nchannel · variable_affected · magnitude"]
    end

    subgraph COLLAPSE["SynthesisEngine.detect_causal_chains()"]
        SIG["Group by signature\n(source, target, channel) tuple\nper trajectory"]
        DEDUP["Deduplicate within group\nkeep first representative\nsort events by step"]
        HASH["chain_id = SHA-256[:12]\nof signature string"]
        FREQ["Sort chains by\ncross-trajectory frequency\n(most common first)"]
        CHAIN(["CausalChain\nchain_id · origin_shock\nevents · total_magnitude"])
    end

    EV --> SIG
    SIG --> DEDUP
    DEDUP --> HASH
    HASH --> FREQ
    FREQ --> CHAIN
```
