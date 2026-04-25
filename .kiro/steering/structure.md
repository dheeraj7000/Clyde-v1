# Project Structure

```
clyde/
├── models/          # All core dataclasses (ShockConfig, Actor, Metrics, etc.)
├── simulation/      # Rule-based engine — NO LLM imports allowed
│   ├── propagation.py    # PropagationEngine: time-stepped actor updates
│   ├── monte_carlo.py    # MonteCarloController: parallel ensemble runs
│   └── backtest.py       # Backtesting against historical shocks
├── setup/           # LLM-powered setup phase
│   ├── parser.py         # ScenarioParser: NL → ParseResult
│   ├── ingestion.py      # DocumentIngester: PDF/MD/TXT → Document
│   ├── knowledge_graph.py # KnowledgeGraph: GraphRAG economic ontology
│   ├── world_factory.py  # EconomicWorldFactory: actors + networks
│   ├── prior_library.py  # PriorLibrary: versioned empirical params
│   ├── network_builder.py # NetworkBuilder: labor/supply/interbank graphs
│   └── gods_eye.py       # GodsEyeConsole: NL intervention → ShockDelta
├── synthesis/       # Output aggregation
│   └── engine.py         # SynthesisEngine: percentile bands, divergence maps, causal chains
├── reporting/       # Evidence-only narrative reports
│   └── agent.py          # ReportAgent: ReACT loop with provenance
├── persistence/     # Data storage
│   └── db.py             # SimulationDB: SQLite schema + read/write
├── llm/             # LLM client wrappers (setup phase only)
└── pipeline.py      # End-to-end wiring: Input → KG → Factory → MC → Synth → Report
```

## Key Boundaries

- `clyde/simulation/` must never import from `clyde/llm/` or any LLM client library
- The handoff between setup and simulation is a fully-resolved `ShockConfig` + list of `Actor` objects
- Tests live in a top-level `tests/` directory
- Property tests use Hypothesis and should be marked with `@pytest.mark.property`
