"""Auto-detect and construct the right LLM provider at runtime.

Resolution order:
1. ``CLYDE_LLM_PROVIDER`` env var if set (one of: ``openrouter``, ``cerebras``, ``mock``).
2. ``OPENROUTER_API_KEY`` is set → OpenRouter.
3. ``CEREBRAS_API_KEY`` is set → Cerebras.
4. Fallback → ``MockLLMClient`` (demo mode, always available).
"""

from __future__ import annotations

import os
from typing import Literal

from clyde.llm.client import LLMClient
from clyde.llm.mock import MockLLMClient


ProviderName = Literal["openrouter", "cerebras", "mock", "auto"]


def available_providers() -> dict[str, bool]:
    """Map provider name → whether the necessary env var is present."""
    return {
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        "cerebras": bool(os.environ.get("CEREBRAS_API_KEY")),
        "mock": True,
    }


def resolve_provider(provider: ProviderName | None = None) -> str:
    """Pick the actual provider name based on env + explicit override."""
    explicit = provider if (provider and provider != "auto") else None
    requested = (explicit or os.environ.get("CLYDE_LLM_PROVIDER") or "auto").lower()
    avail = available_providers()
    if requested == "auto":
        if avail["openrouter"]:
            return "openrouter"
        if avail["cerebras"]:
            return "cerebras"
        return "mock"
    if requested not in {"openrouter", "cerebras", "mock"}:
        raise ValueError(
            f"Unknown LLM provider {requested!r}. "
            f"Use one of: openrouter, cerebras, mock, auto."
        )
    if requested != "mock" and not avail.get(requested, False):
        raise RuntimeError(
            f"Provider {requested!r} requested but its API key is not set. "
            f"Set {'OPENROUTER_API_KEY' if requested == 'openrouter' else 'CEREBRAS_API_KEY'}."
        )
    return requested


def make_llm_client(
    provider: ProviderName | None = None,
    *,
    model: str | None = None,
) -> LLMClient:
    """Construct an LLMClient. Defaults to env-driven auto-detection."""
    name = resolve_provider(provider)
    if name == "openrouter":
        from clyde.llm.openrouter import OpenRouterLLMClient

        return OpenRouterLLMClient(model=model)
    if name == "cerebras":
        from clyde.llm.cerebras import CerebrasLLMClient

        return CerebrasLLMClient(model=model)
    from clyde.llm.demo_router import demo_router

    return MockLLMClient(router=demo_router, default_model="demo-mock")
