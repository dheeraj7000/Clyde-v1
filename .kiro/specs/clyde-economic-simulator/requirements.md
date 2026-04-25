# Requirements Document

## Introduction

Clyde is a situation-agnostic economic simulator that accepts any economic event, policy, business decision, or external shock described in natural language, constructs a causal model of affected actors and their incentives, runs an ensemble of plausible futures, and explains how effects propagate. It produces outcome ranges, risk pathways, and intervention points rather than a single forecast. The system is grounded in empirical behavioral priors, historical analogs, and backtesting — and is honest about its limits.

## Glossary

- **Simulator**: The core Clyde system that orchestrates scenario construction, actor instantiation, behavioral rule assignment, propagation, ensemble simulation, and synthesis.
- **Actor**: An economic entity relevant to a given scenario — households, firms, banks, governments, suppliers, unions, or other participants selected based on the situation.
- **Relationship**: A directed edge between two Actors along which shocks propagate, such as employment, lending, trade, supply, ownership, regulation, or trust links.
- **Shock**: The triggering event that initiates a simulation, characterized by severity, scope, duration, and the Actors it first touches.
- **Behavior**: A response function assigned to an Actor that defines how the Actor reacts to changes in its constraints and incentives, drawn from empirical priors.
- **Scenario**: The complete specification of a simulation run, including the parsed input situation, instantiated Actors, assigned Behaviors, Relationships, Shock parameters, and time horizon.
- **Ensemble**: A collection of simulation runs executed across a distribution of parameter values to produce a cloud of trajectories.
- **Trajectory**: A single simulated path of outcomes over the time horizon, produced by one run within an Ensemble.
- **Central_Path**: The Trajectory representing outcomes under median parameter assumptions.
- **Optimistic_Path**: The Trajectory bounding the favorable end of the plausible outcome range.
- **Pessimistic_Path**: The Trajectory bounding the unfavorable end of the plausible outcome range.
- **Tail_Path**: A Trajectory representing low-probability, high-impact configurations surfaced explicitly.
- **Divergence_Map**: An analysis identifying the variables whose real-world values most determine which branch of outcomes reality follows.
- **Branch**: A full re-simulation from a user-specified intervention point in an existing Trajectory, preserving nonlinear and threshold effects.
- **Prior_Library**: A versioned, inspectable, citable collection of empirically-grounded behavioral parameters drawn from published research.
- **Causal_Chain**: An ordered sequence of Actor-to-Actor transmissions showing how a Shock propagates through the system.
- **Scenario_Parser**: The subsystem that interprets natural language input and extracts triggering events, geographies, markets, Shock parameters, and time horizon.
- **Propagation_Engine**: The subsystem that steps the simulation forward in time, letting each Actor respond to changing local conditions.
- **Synthesis_Engine**: The subsystem that extracts summary outputs from Ensemble results, including paths, divergence analysis, and narrative reports.
- **ShockConfig**: The structured parameter set extracted from the Knowledge_Graph that defines initial conditions for a simulation — shock type, affected variables, geography, sectors, severity, and agent counts.
- **ShockDelta**: A parameter override introduced via the God's_Eye_Console that modifies the ShockConfig at a specific simulation step to create a Branch.
- **Knowledge_Graph**: The persistent graph store built from seed documents using an economic ontology, serving as the single truth store for both seed data and simulation-generated artifacts.
- **God's_Eye_Console**: The natural language injection interface that lets users introduce new events or policy changes mid-simulation, triggering Branch creation.
- **Monte_Carlo_Controller**: The orchestration subsystem that manages parallel simulation runs across varied parameter seeds.

## Requirements

### Requirement 1: Natural Language Scenario Input

**User Story:** As an analyst, I want to describe any economic situation in natural language, so that I can initiate a simulation without manual model construction.

#### Acceptance Criteria

1. WHEN a natural language description of an economic situation is provided, THE Scenario_Parser SHALL extract the triggering event, affected geographies, relevant markets, initial Shock parameters, and time horizon from the input.
2. WHEN the Scenario_Parser encounters ambiguities in the input description, THE Scenario_Parser SHALL flag each ambiguity for user confirmation rather than making silent assumptions.
3. WHEN the user confirms or resolves flagged ambiguities, THE Scenario_Parser SHALL incorporate the user's responses into the Scenario specification.
4. THE Scenario_Parser SHALL accept situations spanning any economic scale, including microeconomic decisions, sectoral shocks, macroeconomic policy changes, and cross-border events.

### Requirement 2: Situation-Relevant Actor Instantiation and Network Topology

**User Story:** As an analyst, I want only the actors relevant to my scenario instantiated with realistic network connections, so that the simulation is focused and shocks propagate through the right channels.

#### Acceptance Criteria

1. WHEN a Scenario is constructed, THE Simulator SHALL instantiate only the Actors relevant to the described situation.
2. THE Simulator SHALL select Actor types from the full range of economic entities, including households, firms, banks, governments, suppliers, unions, and other participants, based on the situation context.
3. WHEN an Actor is instantiated, THE Simulator SHALL assign the Actor a set of Relationships to other instantiated Actors that reflect the economic linkages relevant to the Scenario.
4. THE Simulator SHALL represent each Relationship with a type label drawn from a defined set including employment, lending, trade, supply, ownership, regulation, and trust.
5. WHEN households and firms are instantiated, THE Simulator SHALL construct a bipartite labor market matching network connecting households to firms via employment Relationships.
6. WHEN firms are instantiated, THE Simulator SHALL construct a directed supply chain network connecting firms to other firms and firms to households via trade and supply Relationships.
7. WHEN banks are instantiated, THE Simulator SHALL construct an interbank lending network connecting banks to other banks via lending Relationships.
8. THE Propagation_Engine SHALL use the labor market, supply chain, and interbank network structures as the channels through which shocks propagate between Actors.

### Requirement 3: Empirically-Grounded Behavioral Rule Assignment

**User Story:** As an analyst, I want actor behaviors grounded in published empirical data with concrete decision rules, so that simulation results are credible, traceable, and deterministic during runs.

#### Acceptance Criteria

1. WHEN an Actor is instantiated, THE Simulator SHALL assign a Behavior to the Actor drawn from the Prior_Library.
2. THE Prior_Library SHALL store behavioral parameters sourced from published elasticities, central-bank staff papers, IMF and BIS working papers, and peer-reviewed micro-level studies.
3. THE Prior_Library SHALL be versioned, inspectable, and citable, with each parameter linked to its source publication.
4. WHEN a user provides override values for behavioral parameters, THE Simulator SHALL replace the Prior_Library defaults with the user-supplied values for that Scenario.
5. WHEN a user overrides a behavioral parameter, THE Simulator SHALL record the override and its source alongside the Scenario specification.
6. WHEN a household Actor is instantiated, THE Simulator SHALL assign rules for consumption vs. savings decisions including precautionary savings triggered by unemployment fear, wage demand adjustment based on inflation expectations, and credit-seeking behavior.
7. WHEN a firm Actor is instantiated, THE Simulator SHALL assign rules for investment decisions based on demand pressure vs. hurdle rate, hiring and firing based on demand, pricing based on cost-push and demand-pull factors, supply chain sourcing with stress-based supplier switching, and bankruptcy detection.
8. WHEN a bank Actor is instantiated, THE Simulator SHALL assign rules for credit tightening based on non-performing loan ratios, herding behavior following peer bank risk appetite, credit approval based on reserves and borrower score, and interbank borrowing when reserves fall below threshold.
9. WHEN a central bank Actor is instantiated, THE Simulator SHALL assign a Taylor-rule-inspired policy rate decision rule with configurable discretionary bands.
10. THE Simulator SHALL set all behavioral parameters at Actor initialization time and vary those parameters across Monte Carlo runs rather than determining them via LLM calls during simulation.

### Requirement 4: Shock Characterization

**User Story:** As an analyst, I want shocks fully characterized by severity, scope, duration, and initial contact points, so that the simulation accurately reflects the triggering event.

#### Acceptance Criteria

1. WHEN a Scenario is constructed, THE Simulator SHALL characterize the Shock with severity, scope, duration, and the set of Actors the Shock first touches.
2. THE Simulator SHALL derive Shock parameters from the natural language input and flag any parameters that cannot be confidently extracted for user confirmation.
3. WHEN the user provides explicit Shock parameter values, THE Simulator SHALL use those values instead of derived estimates.

### Requirement 5: Time-Stepped Propagation

**User Story:** As an analyst, I want shocks to propagate through the actor network over time, so that I can observe second-order effects and feedback loops.

#### Acceptance Criteria

1. WHEN a simulation run begins, THE Propagation_Engine SHALL step forward in time, letting each Actor respond to its changing local conditions at each time step.
2. WHILE the Propagation_Engine is stepping forward, THE Propagation_Engine SHALL accumulate stress at chokepoints where multiple Causal_Chains converge.
3. WHILE the Propagation_Engine is stepping forward, THE Propagation_Engine SHALL allow feedback loops to form when Actor responses create conditions that affect upstream Actors.
4. THE Propagation_Engine SHALL produce second-order effects from Actor interactions rather than from hard-coded sequences.

### Requirement 6: Ensemble Simulation with Monte Carlo Orchestration

**User Story:** As an analyst, I want the simulation run across many parameter variations in parallel, so that I get a distribution of outcomes rather than a single forecast — efficiently and reproducibly.

#### Acceptance Criteria

1. WHEN a Scenario is ready for simulation, THE Monte_Carlo_Controller SHALL execute the simulation across a distribution of parameter values, varying behavioral response strength, timing, Shock severity, and contagion thresholds.
2. THE Monte_Carlo_Controller SHALL produce a collection of Trajectories forming an Ensemble for each Scenario.
3. THE Monte_Carlo_Controller SHALL ensure each Trajectory in the Ensemble uses a distinct, sampled combination of parameter values.
4. THE Monte_Carlo_Controller SHALL distribute simulation runs across multiple worker processes for parallel execution.
5. THE Monte_Carlo_Controller SHALL assign each run a deterministic seed so that any individual run is fully reproducible given its seed.
6. THE Monte_Carlo_Controller SHALL support a configurable run count with a default range of 100 to 500 runs per Ensemble.
7. THE Monte_Carlo_Controller SHALL achieve stochasticity through behavioral parameter variation and random seeds, not through LLM output variation.

### Requirement 7: Four-View Uncertainty Output

**User Story:** As an analyst, I want distributional outputs with central, optimistic, pessimistic, and tail views, so that I understand the full range of plausible outcomes.

#### Acceptance Criteria

1. THE Synthesis_Engine SHALL extract a Central_Path from the Ensemble showing outcomes under median parameter assumptions.
2. THE Synthesis_Engine SHALL extract an Optimistic_Path bounding the favorable end of the plausible outcome range.
3. THE Synthesis_Engine SHALL extract a Pessimistic_Path bounding the unfavorable end of the plausible outcome range.
4. THE Synthesis_Engine SHALL extract at least one Tail_Path surfacing low-probability, high-impact configurations explicitly.
5. THE Synthesis_Engine SHALL produce a Divergence_Map identifying the variables whose real-world values most determine which outcome branch is realized.

### Requirement 8: Branching, Intervention, and God's-Eye Live Injection

**User Story:** As an analyst, I want to introduce new events or policies at any point in a simulated timeline using natural language, and compare branches, so that I can evaluate intervention strategies without writing code.

#### Acceptance Criteria

1. WHEN a user introduces a new event or policy at a point in an existing Trajectory, THE Simulator SHALL create a Branch by performing a full re-simulation from step 0 with the merged ShockConfig and ShockDelta, rather than mutating a running simulation.
2. THE Simulator SHALL preserve nonlinear and threshold effects in each Branch by re-simulating rather than perturbing the original Trajectory.
3. WHEN a Branch is created, THE Simulator SHALL make the Branch available for comparison against the base Scenario run and other Branches.
4. THE Simulator SHALL allow Branches to be created at any time step within a completed Trajectory.
5. THE God's_Eye_Console SHALL accept natural language descriptions of interventions — new events, policy changes, or parameter adjustments.
6. WHEN a natural language injection is submitted, THE God's_Eye_Console SHALL parse the injection into a ShockDelta specifying the affected parameters and the simulation step at which the intervention applies.
7. WHEN a ShockDelta is produced, THE Simulator SHALL fork the simulation by merging the ShockDelta into the base ShockConfig and performing a full re-simulation from step 0 with the merged configuration.

### Requirement 9: Scenario-Relevant Output Generation with Evidence-Only Reporting

**User Story:** As an analyst, I want outputs tailored to my specific scenario drawn from a consistent palette, so that I receive relevant metrics and explanations — with every claim backed by simulation data, not LLM guesswork.

#### Acceptance Criteria

1. THE Synthesis_Engine SHALL select situation-relevant metrics from a fixed output palette that includes outcome ranges, Causal_Chain maps, Actor impact scorecards, stress and contagion pathways, Branch comparisons, indicator watchlists, and narrative reports.
2. WHEN a narrative report is generated, THE Synthesis_Engine SHALL link every claim in the report back to a specific simulation artifact.
3. THE Synthesis_Engine SHALL produce Causal_Chain maps showing which Actors are affected and in what order.
4. THE Synthesis_Engine SHALL produce indicator watchlists derived from the Divergence_Map, identifying real-world indicators worth monitoring.
5. THE Synthesis_Engine SHALL NOT use LLM parametric or training knowledge for any factual claim in a narrative report; all factual claims must be traceable to retrieved simulation data from the Knowledge_Graph or simulation database.
6. WHEN a narrative report is generated, THE Synthesis_Engine SHALL include a provenance annotation for each factual claim identifying the simulation artifact or Knowledge_Graph entry that supports the claim.

### Requirement 10: Historical Analog Matching

**User Story:** As an analyst, I want novel situations matched against structurally similar past events, so that parameter ranges are informed by historical precedent.

#### Acceptance Criteria

1. WHEN a Scenario involves a novel situation, THE Simulator SHALL match the situation against structurally similar historical events.
2. THE Simulator SHALL use historical analogs to inform parameter ranges without dictating outcomes.
3. WHEN historical analogs are used, THE Simulator SHALL disclose which analogs were selected and how they influenced parameter ranges.

### Requirement 11: Backtesting

**User Story:** As a system operator, I want the simulator regularly backtested against known historical shocks, so that I can assess and improve simulation accuracy.

#### Acceptance Criteria

1. THE Simulator SHALL support backtesting by running simulations against known historical shocks with the actual outcome masked during the run.
2. WHEN a backtest is completed, THE Simulator SHALL compare simulated outcome distributions against the actual historical outcome.
3. THE Simulator SHALL record backtest results for tracking simulation accuracy over time.

### Requirement 12: Uncertainty and Limitation Flagging

**User Story:** As an analyst, I want the simulator to be transparent about its confidence and limitations, so that I can calibrate my trust in the results.

#### Acceptance Criteria

1. WHEN a Scenario involves a novel regime with limited empirical precedent, THE Simulator SHALL produce wider outcome distributions and attach explicit "unknown" flags to affected outputs.
2. WHEN a Scenario involves a novel regime, THE Simulator SHALL weaken causal claims in the narrative report for the affected portions.
3. WHEN a Scenario involves an event where the announcement itself changes behavior (reflexivity risk), THE Simulator SHALL flag the reflexivity risk and distinguish pre-announcement and post-announcement outcomes.
4. WHEN a Scenario involves heavy-tailed individual actions such as panics, fire sales, or viral consumer shifts, THE Simulator SHALL flag those dynamics as harder to simulate reliably.
5. THE Simulator SHALL model geopolitical and political shocks as exogenous inputs rather than endogenously generated events.

### Requirement 13: Causal Chain Serialization and Parsing

**User Story:** As a developer, I want Causal_Chains serialized to and parsed from a structured format, so that simulation artifacts are portable and inspectable.

#### Acceptance Criteria

1. THE Simulator SHALL serialize each Causal_Chain into a structured data format.
2. WHEN a serialized Causal_Chain is provided, THE Simulator SHALL parse the serialized representation back into an equivalent Causal_Chain object.
3. FOR ALL valid Causal_Chain objects, serializing then parsing then serializing SHALL produce an identical serialized representation (round-trip property).

### Requirement 14: Scenario Serialization and Parsing

**User Story:** As a developer, I want Scenarios serialized to and parsed from a structured format, so that simulation configurations are reproducible and shareable.

#### Acceptance Criteria

1. THE Simulator SHALL serialize each Scenario, including Actors, Relationships, Behaviors, Shock parameters, and time horizon, into a structured data format.
2. WHEN a serialized Scenario is provided, THE Simulator SHALL parse the serialized representation back into an equivalent Scenario object.
3. FOR ALL valid Scenario objects, serializing then parsing then serializing SHALL produce an identical serialized representation (round-trip property).
4. THE Simulator SHALL include a human-readable pretty-print format for serialized Scenarios.

### Requirement 15: LLM and Rule-Based Architecture Split

**User Story:** As a system architect, I want a clean separation between LLM-powered setup and rule-based simulation, so that ensemble and Monte Carlo runs are feasible without LLM bottlenecks.

#### Acceptance Criteria

1. THE Simulator SHALL operate in two distinct phases: a setup phase and a simulation phase.
2. WHILE in the setup phase, THE Simulator SHALL use LLM capabilities for scenario parsing, Actor construction, persona generation, and Knowledge_Graph building.
3. WHILE in the simulation phase, THE Propagation_Engine SHALL execute using pure rule-based logic with zero LLM calls.
4. WHEN the setup phase completes, THE Simulator SHALL produce a fully resolved ShockConfig and set of initialized Actors with fixed behavioral parameters, requiring no further LLM interaction for simulation execution.
5. THE Simulator SHALL enforce the architecture boundary such that no component in the simulation step loop can invoke an LLM call.

### Requirement 16: Document Ingestion and Knowledge Graph

**User Story:** As an analyst, I want to feed seed documents into the system alongside natural language descriptions, so that the simulator can ground its scenario construction in real-world data.

#### Acceptance Criteria

1. WHEN seed documents are provided, THE Simulator SHALL accept PDF, Markdown, and plain text formats for ingestion.
2. WHEN seed documents are ingested, THE Simulator SHALL build a Knowledge_Graph from the document content using an economic ontology that captures entities, relationships, and quantitative parameters.
3. THE Knowledge_Graph SHALL serve as the persistent truth store for both seed data and simulation-generated artifacts.
4. WHEN a Scenario is constructed, THE Simulator SHALL extract ShockConfig parameters from the Knowledge_Graph.
5. WHEN both natural language input and seed documents are provided, THE Simulator SHALL merge information from both sources into the Knowledge_Graph, flagging conflicts for user resolution.

### Requirement 17: Core Metric Set

**User Story:** As an analyst, I want every simulation to produce a standard set of economic health metrics, so that I can compare runs and scenarios on a common basis.

#### Acceptance Criteria

1. THE Propagation_Engine SHALL compute the following core metrics at each simulation step for each run: GDP index, inflation rate, unemployment rate, Gini coefficient, credit tightening index, firm bankruptcy count, bank stress index, consumer confidence, and interbank freeze detection.
2. WHEN a Scenario requires additional situation-specific metrics beyond the core set, THE Simulator SHALL compute those additional metrics alongside the core set.
3. THE Simulator SHALL make all computed metrics available per step and per run for downstream analysis by the Synthesis_Engine.
