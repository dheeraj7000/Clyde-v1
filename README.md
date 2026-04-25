# Clyde

> Situation-agnostic economic simulator. Describe an event in plain English; get distributional outcomes with traceable causal chains.

```
   ____ _           _
  / ___| |_   _  __| | ___
 | |   | | | | |/ _` |/ _ \
 | |___| | |_| | (_| |  __/
  \____|_|\__, |\__,_|\___|
          |___/
```

[![tests](https://img.shields.io/badge/tests-196%20passing-brightgreen)](#development) [![python](https://img.shields.io/badge/python-3.11%2B-blue)](#) [![status](https://img.shields.io/badge/status-hackathon%20demo-orange)](#)

---

## What is Clyde?

Clyde is a **situation-agnostic economic simulator**. You describe any economic event, policy, or shock in natural language — a Fed rate hike, a regional drought, a supply-chain disruption, a payroll-tax change — and Clyde builds a causal model of the affected actors and incentives, runs an ensemble of Monte Carlo simulations, and explains how the effects propagate.

The output is **distributional, not point-forecast**: central, optimistic, pessimistic, and tail paths, plus a divergence map showing which variables most drive uncertainty and which leading indicators to watch. Every claim in the report traces back to simulation data — no LLM hallucination in outputs. This is enforced architecturally: the simulation phase is strictly rule-based and has no LLM imports. The LLM is used only at setup (parsing the scenario, building the knowledge graph, spawning actors) and at reporting (narrative polish over evidence-only facts).

Behavioral parameters are drawn from a citable **prior library** (published elasticities, central-bank papers, peer-reviewed studies). Users can also fork **branches** at any step via natural-language injections in the God's Eye Console ("cut rates by 75bp at step 2") and compare alternate futures side-by-side.

## Architecture

```
                     ┌──────────────────────────────┐
                     │  God's Eye Console (LLM)     │
                     │  NL injection → ShockDelta   │
                     └──────────────┬───────────────┘
                                    │ branches
                                    ▼
  Input        Knowledge       World          Monte Carlo       Synthesis        Report
  Layer   ──►  Graph      ──►  Factory   ──►  Controller   ──►  Engine      ──►  Agent
  (LLM)        (LLM)           (LLM)          (rules only)      (rules only)     (LLM polish,
   PDF/MD/                     ShockConfig    100–500 runs      bands, divergence  evidence-only)
   TXT, NL                     + Actors       parallel,          map, causal
                               + Networks     deterministic      chains
                                              seeds
                                              │
                                              ▼
                                          ┌────────┐
                                          │ SQLite │  step metrics + causal events
                                          └────────┘
```

Setup is LLM-powered (parse the scenario, build the world). Simulation is 100% rule-based and runs hundreds of trajectories in parallel via `ProcessPoolExecutor`. Synthesis aggregates trajectories into percentile bands and divergence maps. The Report Agent then writes a narrative whose every fact is sourced from the SQLite store of simulation artifacts. An optional **Agent Mode** (`clyde/setup/agent_sim.py`) runs each actor as an autonomous LLM agent for richer narrative simulations — it lives in `setup/` to preserve the LLM boundary invariant.

## Quickstart

```bash
git clone <repo-url> clyde && cd clyde
pip install -e .[web]
```

Get an API key from one of:

- **OpenRouter** (recommended, single key fronts dozens of models): https://openrouter.ai/keys
- **Cerebras** (optional, fast inference): https://cloud.cerebras.ai

…or skip both and run in **Demo Mode** (`MockLLMClient`, no key needed).

```bash
cp .env.example .env
# edit .env, paste your key
export $(cat .env | xargs)

python -m clyde.web
# → open http://localhost:8000
```

Type a scenario into the input box ("A 50bp Fed rate hike in 2026 affecting US markets."), hit **Run**, and watch the ensemble play out.

## LLM providers

Clyde auto-detects which provider to use based on env vars (resolution order: explicit `CLYDE_LLM_PROVIDER` → `OPENROUTER_API_KEY` → `CEREBRAS_API_KEY` → mock).

### OpenRouter (recommended)

A single key fronts dozens of upstream models. Set `OPENROUTER_API_KEY` and Clyde will use it automatically.

| Setting       | Default                          | Notes                          |
|---------------|----------------------------------|--------------------------------|
| Model         | `anthropic/claude-3.5-sonnet`    | Override via `CLYDE_MODEL`     |
| Cheap option  | `anthropic/claude-3.5-haiku`     | ~10x cheaper, still solid JSON |
| Free option   | `meta-llama/llama-3.3-70b-instruct` | Free tier on OpenRouter     |

### Cerebras (optional, fast inference)

Cerebras is fast — useful when iterating on scenarios live during a demo. Models are limited to a curated llama set.

| Setting | Default          | Notes                               |
|---------|------------------|-------------------------------------|
| Model   | `llama3.1-8b`    | Override via `CLYDE_CEREBRAS_MODEL` |

Force this provider (even if `OPENROUTER_API_KEY` is also set) with `CLYDE_LLM_PROVIDER=cerebras`.

### Demo mode

If neither key is set, Clyde falls back to `MockLLMClient`. The full pipeline runs end-to-end (parse, build, simulate, synthesize, report) with deterministic stub responses from the LLM-aware components. The web UI shows a banner indicating Demo Mode. Useful for showing the architecture without paying for tokens.

## Programmatic use

```python
import asyncio
from clyde.llm import make_llm_client
from clyde.pipeline import Pipeline, PipelineConfig  # Pipeline is a short alias for ClydePipeline

async def main():
    pipeline = Pipeline(
        llm_client=make_llm_client(),
        config=PipelineConfig(run_count=50, ensemble_seed=0),
    )
    result = await pipeline.run("A 50bp Fed rate hike in 2026 affecting US markets.")
    print(result.scenario_id, len(result.report.sections), "sections")

    branch = await pipeline.fork_branch(result, "Cut rates by 75bp at step 2.")
    print(branch.branch_id, "vs", result.scenario_id)

asyncio.run(main())
```

A runnable version lives at [`examples/run_pipeline.py`](examples/run_pipeline.py).

## CLI

| Command         | Entry point                  | What it does                                        |
|-----------------|------------------------------|-----------------------------------------------------|
| `clyde-web`     | `clyde.web.__main__:main`    | Start the FastAPI server (defaults: 0.0.0.0:8000)   |
| `python -m clyde.web` | same                   | Equivalent module form                              |
| `clyde` (CLI)   | TODO                         | Headless runner / batch mode — not implemented yet  |

## REST API

The web layer (`pip install -e .[web]`) exposes:

| Method | Path                                            | Description                                               |
|--------|-------------------------------------------------|-----------------------------------------------------------|
| GET    | `/`                                             | Frontend HTML bundle                                      |
| GET    | `/api/health`                                   | `{ status, provider, providers_available, model }`        |
| POST   | `/api/runs`                                     | Kick off a run; returns `{ job_id, status }`              |
| GET    | `/api/runs/{job_id}`                            | Poll: `{ status, progress, result?, error? }`             |
| POST   | `/api/runs/{job_id}/branches`                   | Fork a branch with a natural-language injection           |
| GET    | `/api/runs/{job_id}/branches/{branch_id}`       | Poll a branch                                             |
| GET    | `/api/scenarios/sample`                         | Demo scenarios for the UI                                 |
| POST   | `/api/runs/{job_id}/agent-sim/start`            | Start an agent-based simulation from a completed run      |
| POST   | `/api/runs/{job_id}/agent-sim/{sim_id}/round`   | Execute one round of the agent simulation                 |
| POST   | `/api/runs/{job_id}/agent-sim/{sim_id}/inject`  | Inject a mid-simulation event (God's Eye live)            |
| GET    | `/api/runs/{job_id}/agent-sim/{sim_id}/state`   | Get current state of an agent simulation                  |

## Architecture invariants

The headline invariant is the **LLM-vs-rule-based split** (Requirement 15):

- `clyde/llm/`, `clyde/setup/`, `clyde/reporting/` — may import an LLM client.
- `clyde/simulation/`, `clyde/synthesis/` — must not. Ever.

This is enforced two ways:

1. **Static import analysis** at test time (`tests/test_llm_boundary.py`) walks the AST of every module under `clyde/simulation/` and fails if any import path resolves into `clyde.llm`.
2. **Runtime contract tests** assert that `MonteCarloController` and `PropagationEngine` produce identical outputs given identical seeds, with no LLM client constructed.

The whole project ships with **196+ passing tests** covering ingestion, parsing, knowledge-graph construction, world-factory determinism, Monte Carlo reproducibility, synthesis percentile bands, divergence maps, causal-chain detection, evidence-only reporting, and branch forking. Run them with `pytest`.

## Development

```bash
pip install -e .[dev]
pytest
```

Where things live:

```
clyde/
  llm/           OpenRouter, Cerebras, Mock clients + factory
  models/        Dataclasses (ShockConfig, Actor, StepMetrics, …)
  setup/         Document ingestion, scenario parser, knowledge graph,
                 economic world factory, prior library, God's Eye console,
                 persona generator, agent-based simulation engine
  simulation/    Monte Carlo controller, propagation engine (rule-based)
  synthesis/     Percentile bands, divergence map, causal chains
  reporting/     ReACT report agent (evidence-only)
  persistence/   SQLite store for trajectories + causal events
  web/           FastAPI server, job store + background runners, static frontend
  pipeline.py    End-to-end orchestrator (ClydePipeline)
tests/           pytest suite (~196 tests)
examples/        Runnable demos
```

Design doc: `.kiro/specs/clyde-economic-simulator/design.md`. Requirements:
`.kiro/specs/clyde-economic-simulator/requirements.md`.

## Changelog

- **PropagationEngine fully implemented (`clyde/simulation/propagation.py`)** — The core rule-based, deterministic time-stepped simulation engine is now complete. Given a `SimulationWorld` and a seed, it produces a `TrajectoryResult` with per-step metrics and a full causal-event trace. Each simulation step executes a fixed 9-phase actor update order: Central Bank (Taylor-rule rate), Banks (NPL/credit tightening/herding + monetary policy transmission), Firms (investment/hiring/firing/pricing/bankruptcy), Labor Market clearing, Households (consumption/savings/wage demands/confidence), Interbank settlement with peer contagion, Bankruptcy cascade propagation, Bayesian household learning (inflation expectations), and Metrics computation (GDP index, inflation, unemployment, Gini, credit tightening, bank stress, consumer confidence, interbank freeze). Initial shocks are applied at step 0 to designated contact actors and all downstream effects are purely network-mediated. The engine enforces zero LLM imports, full determinism (same seed + world = identical output including causal event sequence), and bounded state via stress caps. Chokepoint detection flags firms with multiple bankrupt suppliers and banks with many distressed borrowers. Optional `SimulationDB` integration persists run records, causal events inline, and step metrics in batch.

- **Job store and background runner extracted to `clyde/web/jobs.py`** — The web layer now has a dedicated module for in-memory job management, background pipeline execution, progress tracking, and result serialization. `JobStore` holds up to 32 concurrent jobs (soft-evicting the oldest), each driven by an `asyncio.Task` that wraps `ClydePipeline.run`. Branch forking follows the same pattern. The pipeline factory is injectable, so tests can swap in stubs without touching the server. Progress is reported in coarse stages (parsing → building world → running ensemble → synthesizing → generating report) and exposed via the existing polling endpoints. All serialization helpers (`serialize_pipeline_result`, `serialize_branch_result`) live here too, keeping `server.py` thin.
- **End-to-end web test added** — `tests/_e2e_web_test.py` is a standalone script (not collected by pytest) that exercises the full Clyde pipeline through the REST API using a real LLM provider (Cerebras). It POSTs a scenario, polls until completion, validates the entire result structure (shock config, parse result, paths, divergence, causal chains, evidence-only report, watchlist, historical analogs), forks a branch via the God's Eye injection endpoint, and confirms the branch completes with a merged config. Run it manually against a live server: `python tests/_e2e_web_test.py`.
- **SynthesisEngine fully implemented** — `clyde/synthesis/engine.py` now contains the complete synthesis pipeline: `compute_paths` (percentile bands across ensemble trajectories — central, optimistic, pessimistic, tail), `compute_divergence_map` (ranks metrics by cross-run variance and maps each to a real-world monitoring indicator), `detect_causal_chains` (collapses per-trajectory causal events into canonical chains with stable IDs), `select_metrics` (picks situation-relevant metrics for reporting based on divergence + end-to-end movement), and `indicator_watchlist` (extracts leading indicators from the divergence map). Metric directionality is handled correctly — "lower is better" metrics like unemployment flip the optimistic/pessimistic bands. Boolean metrics (interbank freeze) use fraction-True thresholds. Integer metrics (firm bankruptcy count) are rounded.
- **Synthesis engine metric mapping completed** — The `_build_step_metrics` helper in `SynthesisEngine` was truncated mid-field (`inflation_r…`), leaving all metrics after `gdp_index` unmapped. The function now correctly deserializes every `StepMetrics` field (inflation rate, unemployment, Gini, credit tightening, firm bankruptcies, bank stress, consumer confidence, interbank freeze). Without this fix, percentile-band and divergence-map outputs would contain default/zero values for most indicators. Also added an `__all__` export list to `clyde/synthesis/engine.py`.
- **Cerebras default model changed to `llama3.1-8b`** — The default Cerebras model was switched from `llama-3.3-70b` to `llama3.1-8b`. The smaller model is faster and better suited for live demo iteration where latency matters more than raw capability. Override with `CLYDE_CEREBRAS_MODEL` if you need the larger model.
- **Interbank network builder fix** — Fixed an operator-precedence bug in `NetworkBuilder._build_interbank()` where `and`/`or` grouping in the preferential-attachment candidate filter could exclude seed nodes with zero degree, potentially producing disconnected interbank graphs. Added parentheses to ensure `degrees[b.id] > 0 or b in seed_nodes` is evaluated as a unit.
- **Frontend app logic added (`clyde/web/static/app.js`)** — The web UI now has a full Alpine.js component (`clydeApp`) backed by Chart.js and D3.js for visualization. It drives the entire client-side experience: scenario submission, async job polling with a multi-stage progress bar (parse → synthesize → simulate → diverge → report), fan-chart rendering (central/optimistic/pessimistic/tail bands), a horizontal-bar divergence-sensitivity chart, tabbed result views (report, causal chains, watchlist, network graph, simulation, branches, raw JSON), and counterfactual branching via the God's Eye injection endpoint. Branch trajectories overlay the fan chart with distinct dashed lines. Sample scenarios can be loaded from the backend. Includes a D3 force-directed network graph that visualizes actor relationships (households, firms, banks, central bank) with drag, zoom, and shocked-node glow rings, plus a step-by-step simulation replay mode with play/pause controls that animates stress propagation across the network — nodes turn red on active causal events and orange for cumulative stress, with edges highlighting active transmission paths. Graphs render lazily on tab switch. The component is self-contained and exposed globally as `window.clydeApp`.

- **PersonaGenerator added (`clyde/setup/persona_generator.py`)** — New LLM-powered module that gives simulation actors rich narrative identities ("Maria Chen, 34, risk-averse saver") instead of opaque IDs like `household_0003`. Personas are grounded in each actor's actual behavioral parameters (MPC, risk appetite, bankruptcy threshold, etc.) and the shock context, so the display names and descriptions reflect real economic behavior. Actors are batched by type for efficient LLM calls, with automatic fallback to minimal personas when the LLM is unavailable. The module also includes `InfluenceConfig` generation — scenario-specific propagation weights (monetary transmission lag, herding strength, credit channel weight, etc.) that tune how economic signals travel through the actor network. Personas are purely decorative metadata; they never influence the rule-based simulation engine.

- **Agent-Based Simulation Engine fully implemented (`clyde/setup/agent_sim.py`)** — New LLM-powered "Agent Mode" where each economic actor runs as an autonomous LLM agent that observes its environment, recalls its history, and makes economic decisions each round — producing emergent behavior instead of rule-based deterministic outcomes. The engine is now complete end-to-end: actors maintain rolling `AgentMemory` (observations, decisions, received events with configurable window), each round queries actors in parallel batches ordered by type (Central Bank → Banks → Firms → Households) via `asyncio.gather`, the LLM returns structured JSON actions, and `_apply_action` updates state and emits `CausalEvent`s across network links. Supported actions include rate-setting, credit tightening/easing, price adjustments, hiring/firing (with household employment state updates), consumption/savings shifts, investment, and cost-cutting. Mid-simulation event injection is supported via `inject_event()` for God's Eye live interventions, and `run_all` streams `RoundResult`s as an `AsyncIterator` for live UI updates. Aggregate `StepMetrics` (GDP, inflation, unemployment, credit tightening, bank stress, consumer confidence, interbank freeze) are computed from actor states each round. The module lives in `clyde/setup/` (not `clyde/simulation/`) because it imports `LLMClient`, preserving the LLM boundary invariant. The existing rule-based `PropagationEngine` remains the "Fast Mode" for Monte Carlo ensembles; this is the complementary "Agent Mode" for rich narrative simulations.

- **Pipeline module alias and `__all__` exports** — `clyde/pipeline.py` now exports a `Pipeline` alias for `ClydePipeline` so callers can use the shorter name. An explicit `__all__` list (`ClydePipeline`, `Pipeline`, `PipelineConfig`, `PipelineResult`) makes star-imports predictable and documents the module's public API.

- **End-to-end persona test added** — `tests/_e2e_persona_test.py` is a standalone script (not collected by pytest) that exercises the persona generation and influence config pipeline through the REST API using a real LLM provider (Cerebras). It POSTs a Fed rate-hike scenario, polls until completion, then validates that the result contains rich persona metadata (display names, roles, descriptions, economic behavior, vulnerability, tags) and scenario-specific influence configuration parameters with LLM reasoning. Run it manually against a live server: `python tests/_e2e_persona_test.py`.

- **`__all__` export added to `clyde/simulation/propagation.py`** — The propagation module now declares `__all__ = ["PropagationEngine"]`, making its public API explicit and ensuring star-imports only surface the engine class. Replaces a stray `_` that was left at the end of the file.

- **Agent Mode prompt templates added (`clyde/setup/agent_sim.py`)** — The agent-based simulation engine now includes structured prompt templates for all four actor types (Central Bank, Bank, Firm, Household). Each template injects the actor's current state variables, memory, and environment context, then constrains the LLM to a type-specific action menu — central banks can set rates or signal forward guidance, banks can tighten/ease credit or call loans, firms can adjust prices/hiring/investment, and households can shift consumption/savings or seek credit. All responses are requested as JSON with a reasoning field, keeping agent decisions parseable and traceable.

- **Agent simulation REST routes added (`clyde/web/agent_routes.py`)** — New API endpoints expose the agent-based simulation engine over HTTP. After a standard pipeline run completes, clients can POST to `/agent-sim/start` to spin up an LLM-driven agent simulation using the run's world and actors. From there, `/agent-sim/{sim_id}/round` advances one round at a time (useful for step-through UI playback), `/agent-sim/{sim_id}/inject` accepts natural-language God's Eye interventions mid-simulation, and `/agent-sim/{sim_id}/state` returns the current round, actor count, recent memories, and last few round results. Simulation state is held in-memory on the job object, keyed by a short UUID. The routes live in `clyde/web/` and import from `clyde/setup/agent_sim` (not `clyde/simulation/`), preserving the LLM boundary invariant.

- **Agent routes refactored from router to plain functions (`clyde/web/agent_routes.py`)** — The agent-sim module no longer defines its own `APIRouter`. Route decorators and the `router` object were removed; the four endpoints (`start_agent_sim`, `run_agent_round`, `inject_agent_event`, `get_agent_sim_state`) are now plain async functions that accept an explicit `JobStore` parameter instead of relying on FastAPI dependency injection. `inject_agent_event` takes a raw `description: str` instead of an `InjectRequest` body — the server-side route handler in `server.py` unwraps the Pydantic model before calling in. Return-type annotations and `Job` type hints were added throughout. This keeps `agent_routes.py` decoupled from FastAPI's routing machinery, making the functions easier to test in isolation and giving `server.py` full control over URL layout and middleware.

- **FastAPI server module added (`clyde/web/server.py`)** — The web server is now fully wired up as a thin `create_app()` factory. It registers all REST routes (health, sample scenarios, run creation/polling, branch forking, and agent-sim endpoints), mounts the static frontend, configures CORS, and attaches a `JobStore` to `app.state`. Five hand-crafted sample scenarios (Fed rate hike, banking crisis, oil shock, housing correction, trade tariff) ship as demo seeds for the UI. The pipeline factory is injectable via `app.state.pipeline_factory_override`, so tests can swap in stubs without touching provider resolution. A module-level `app` instance is exported for `uvicorn clyde.web.server:app` usage.

- **End-to-end agent simulation test added** — `tests/_e2e_agent_sim.py` is a standalone script (not collected by pytest) that exercises the full agent-based simulation lifecycle through the REST API using a real LLM provider (Cerebras). It runs a standard pipeline simulation, starts an agent-mode session from the completed run, executes 3 rounds (validating actions, causal events, and aggregate metrics each round), injects a mid-simulation God's Eye event ("major regional bank failure"), runs a post-injection round to confirm actors react to the shock, and finally checks simulation state. Run it manually against a live server: `python tests/_e2e_agent_sim.py`.

- **Split-screen view added (`clyde/web/static/split.js`)** — New Alpine.js component (`window.splitScreen`) that provides a dual-pane UI: a live D3 force-directed knowledge graph on the left and a real-time simulation event feed on the right. In pipeline mode, the KG populates from the run's network and the feed shows parsed phases (scenario parse, personas, influence config, causal chains, report sections). Switching to Agent Mode starts an LLM-driven agent simulation with play/pause/step controls, speed adjustment, and mid-simulation God's Eye event injection. Agent actions, causal propagation events, and per-round aggregate metrics (GDP, inflation, unemployment, bankruptcies) stream into the feed with filterable categories (all / decisions / events / metrics / system). Causal events animate in the KG — edges flash red on transmission and target nodes pulse — giving a live visual of stress propagation through the actor network. The feed is capped at 200 entries and supports category filtering. The component is self-contained and works alongside `app.js` and `_mixin.js`.

- **Interactive network graph mixin (`clyde/web/static/_mixin.js`)** — New JS mixin that extends the frontend app with click-to-inspect node details and a full simulation replay mode. Clicking any node in the D3 network graph now opens a detail panel showing the actor's type, behavioral parameters (with human-readable labels), connections, and causal events. The simulation replay tab adds play/pause/step controls that animate stress propagation across the network over time — nodes pulse red on active causal events and orange for cumulative stress, edges highlight active transmission paths, and a per-step metrics dashboard shows real-time values with color-coded deltas. Also includes persona display helpers and influence-config labels for the upcoming persona UI. The mixin wraps `window.clydeApp` transparently so it layers on top of `app.js` without modifying it.

## License / Credits

TBD.
