"""Provider-agnostic LLM client protocol.

This module defines the abstract contract that every LLM-dependent component
in the *setup* and *reporting* phases programs against. No concrete provider
SDK (anthropic, openai, etc.) is imported here -- the protocol is pure.

The simulation phase (``clyde.simulation``) must *never* import from this
module. Enforcement lives in ``tests/test_llm_boundary.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    """A single chat message passed to an LLM client.

    ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
    """

    role: str
    content: str


@dataclass
class LLMResponse:
    """A completion returned from an LLM client.

    ``usage`` is an optional free-form token accounting dict; ``raw`` holds
    the provider-native response object when callers need it for debugging.
    """

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Abstract LLM client used only in the setup / reporting phases.

    Implementations MUST be safe to call from ``async`` code paths. The
    simulation loop never touches this protocol.
    """

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,  # "json" | None
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        ...

    async def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        ...


__all__ = ["LLMClient", "LLMMessage", "LLMResponse"]
