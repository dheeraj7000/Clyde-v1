# Tech Stack & Build

## Language

- Python (modern — uses `dataclasses`, `type` unions with `|`, `async/await`)

## Key Libraries

- **Hypothesis** — property-based testing
- **SQLite** — persistence (via stdlib `sqlite3`)
- **ProcessPoolExecutor** — parallel Monte Carlo runs (stdlib `concurrent.futures`)
- **LLM client** — used only in setup phase (`clyde/llm/`), never in simulation

## Architecture: LLM Boundary

This is the most important architectural constraint:

- **Setup phase** (`clyde/setup/`, `clyde/llm/`): LLM-powered. Parses scenarios, builds actors, constructs knowledge graph.
- **Simulation phase** (`clyde/simulation/`): Pure rule-based. Zero LLM calls. Zero LLM imports. Enforced at module level.
- A CI lint check must fail if any import from `clyde/llm/` appears in `clyde/simulation/`.

## Common Commands

```bash
# Run all tests
pytest

# Run property-based tests only
pytest -m property

# Run a specific test file
pytest tests/test_propagation.py

# Run with verbose output
pytest -v
```

## Conventions

- Use `dataclasses` for all domain models (not Pydantic, not dicts)
- All behavioral parameters are set at actor construction time, varied across Monte Carlo runs — never determined by LLM during simulation
- Every simulation run gets a deterministic seed. Same seed = same trajectory.
- Branches re-simulate from step 0 with merged config — never mutate running state.

## MCP Tools

- **Context7**: Always use the Context7 MCP server when looking up library/API documentation, generating code that depends on external libraries, or following setup/configuration steps. Do not wait for the user to ask — proactively resolve docs via Context7 whenever a library or API is involved.
