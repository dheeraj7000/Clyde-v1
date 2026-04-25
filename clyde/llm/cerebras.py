"""Cerebras LLMClient implementation (optional fast-inference provider).

Cerebras exposes an OpenAI-compatible API at ``https://api.cerebras.ai/v1``.
The selling point is speed — useful when iterating on scenarios live during
a demo. Models are limited to a curated llama set as of 2026.
"""

from __future__ import annotations

import os

from clyde.llm.http_client import HTTPLLMClient, HTTPLLMConfig


CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_DEFAULT_MODEL = "llama3.1-8b"


class CerebrasLLMClient(HTTPLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not key:
            raise RuntimeError(
                "CEREBRAS_API_KEY not set. Either pass api_key=... or export "
                "CEREBRAS_API_KEY in your shell."
            )
        super().__init__(
            HTTPLLMConfig(
                api_key=key,
                base_url=CEREBRAS_BASE_URL,
                default_model=model or os.environ.get("CLYDE_CEREBRAS_MODEL") or CEREBRAS_DEFAULT_MODEL,
                timeout_s=timeout_s,
            )
        )
