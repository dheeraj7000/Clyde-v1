"""Programmatic Clyde pipeline demo.

Runs the full setup → simulation → synthesis → report flow against a
single scenario, prints a few key outputs, then forks a branch with a
natural-language intervention.

Usage:

    python examples/run_pipeline.py            # auto-detects provider
    python examples/run_pipeline.py --help

Set ``OPENROUTER_API_KEY`` or ``CEREBRAS_API_KEY`` in your environment to
use a real model. With neither set, Clyde falls back to ``MockLLMClient``
so the architecture still runs end-to-end (useful as a smoke test).
"""

from __future__ import annotations

import argparse
import asyncio

from clyde.llm import make_llm_client, resolve_provider
from clyde.pipeline import ClydePipeline, PipelineConfig


SCENARIO = "A 50bp Fed rate hike in 2026 affecting US markets."
INJECTION = "Cut rates by 75bp at step 2."


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo run count")
    parser.add_argument("--seed", type=int, default=0, help="Ensemble seed")
    args = parser.parse_args()

    provider = resolve_provider()
    print(f"[clyde] provider = {provider}")

    llm = make_llm_client()
    config = PipelineConfig(run_count=args.runs, ensemble_seed=args.seed)

    with ClydePipeline(llm_client=llm, config=config) as pipeline:
        result = await pipeline.run(SCENARIO)

        shock = result.shock_config
        print(f"\nscenario_id      : {result.scenario_id}")
        print(
            f"shock            : {shock.shock_type} "
            f"severity={shock.severity} duration={shock.duration_steps} steps"
        )

        central = result.synthesis.paths.central
        if central:
            head, tail = central[0], central[-1]
            print(
                f"GDP central path : step {head.step} -> {head.gdp_index:.3f} "
                f"... step {tail.step} -> {tail.gdp_index:.3f}"
            )

        top_div = result.synthesis.divergence_map.variables[:3]
        print("top divergence   : " + (", ".join(v.name for v in top_div) or "(none)"))
        print(f"report sections  : {len(result.report.sections)}")

        branch = await pipeline.fork_branch(result, INJECTION)
        print(
            f"\nbranch           : {branch.branch_id} "
            f"({len(branch.ensemble.trajectories)} trajectories) "
            f"forked from {branch.parent_scenario_id}"
        )


if __name__ == "__main__":
    asyncio.run(main())
