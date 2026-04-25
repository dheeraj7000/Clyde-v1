"""OpenRouter LLMClient implementation.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint at
``https://openrouter.ai/api/v1`` with a single API key that fronts dozens of
upstream models. Default model is biased toward strong JSON adherence; users
can override via ``CLYDE_MODEL`` env var or ``model`` argument.
"""

from __future__ import annotations

import os

from clyde.llm.http_client import HTTPLLMClient, HTTPLLMConfig


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class OpenRouterLLMClient(HTTPLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        timeout_s: float = 60.0,
        site_url: str | None = None,
        app_name: str | None = "Clyde Economic Simulator",
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Either pass api_key=... or export "
                "OPENROUTER_API_KEY in your shell."
            )
        extra: dict[str, str] = {}
        # OpenRouter encourages identifying the calling app; both headers are optional.
        if site_url:
            extra["HTTP-Referer"] = site_url
        if app_name:
            extra["X-Title"] = app_name
        super().__init__(
            HTTPLLMConfig(
                api_key=key,
                base_url=OPENROUTER_BASE_URL,
                default_model=model or os.environ.get("CLYDE_MODEL") or OPENROUTER_DEFAULT_MODEL,
                extra_headers=extra,
                timeout_s=timeout_s,
            )
        )
