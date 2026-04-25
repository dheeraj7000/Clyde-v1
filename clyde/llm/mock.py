"""In-memory :class:`LLMClient` implementation for deterministic tests.

Use :class:`MockLLMClient` wherever a real provider would otherwise be
called. Configure canned responses via either a FIFO ``responses`` queue or
a ``router`` callable that inspects the incoming messages and returns a
response on demand.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from clyde.llm.client import LLMMessage, LLMResponse


# A single configured response may be provided in three forms:
#   * a ready-made :class:`LLMResponse`
#   * a ``str`` (wrapped into ``LLMResponse(content=..., model=default_model)``)
#   * a ``dict`` (for ``complete`` -> JSON-serialized into ``content``; for
#     ``complete_json`` -> returned directly)
ResponseLike = LLMResponse | str | dict
RouterFn = Callable[[list[LLMMessage]], ResponseLike]


class MockLLMClient:
    """In-memory mock for deterministic testing.

    Configure responses either via a *queue* (FIFO, one response dequeued per
    call) or via a *router* (a callable that maps the incoming messages ->
    :class:`LLMResponse` / ``dict`` / ``str``). When both are provided, the
    router takes precedence.
    """

    def __init__(
        self,
        *,
        responses: list[ResponseLike] | None = None,
        router: RouterFn | None = None,
        default_model: str = "mock-1",
    ) -> None:
        self._queue: list[ResponseLike] = list(responses) if responses else []
        self._router: RouterFn | None = router
        self._default_model = default_model
        self._call_log: list[tuple[str, list[LLMMessage], dict[str, Any]]] = []

    # ------------------------------------------------------------------ helpers

    def enqueue(self, response: ResponseLike) -> None:
        """Append a response to the FIFO queue."""
        self._queue.append(response)

    @property
    def call_log(self) -> list[tuple[str, list[LLMMessage], dict[str, Any]]]:
        """Return the recorded call log (method_name, messages, kwargs)."""
        return list(self._call_log)

    def _next_raw(self, messages: list[LLMMessage]) -> ResponseLike:
        """Obtain the next raw response from router or queue."""
        if self._router is not None:
            return self._router(messages)
        if not self._queue:
            raise IndexError(
                "MockLLMClient response queue is empty: no queued response "
                "available for this call. Enqueue more responses or provide "
                "a router."
            )
        return self._queue.pop(0)

    @staticmethod
    def _coerce_to_response(raw: ResponseLike, default_model: str) -> LLMResponse:
        """Normalise a queue/router payload into an :class:`LLMResponse`."""
        if isinstance(raw, LLMResponse):
            return raw
        if isinstance(raw, str):
            return LLMResponse(content=raw, model=default_model)
        if isinstance(raw, dict):
            return LLMResponse(content=json.dumps(raw), model=default_model)
        raise TypeError(
            f"Unsupported mock response type: {type(raw).__name__}. "
            "Expected LLMResponse, str, or dict."
        )

    @staticmethod
    def _coerce_to_dict(raw: ResponseLike) -> dict:
        """Normalise a queue/router payload into a ``dict`` (for ``complete_json``)."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, LLMResponse):
            try:
                parsed = json.loads(raw.content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "MockLLMClient.complete_json received an LLMResponse whose "
                    f"content was not valid JSON: {raw.content!r}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    "MockLLMClient.complete_json expected the response JSON to "
                    f"decode to a dict, got {type(parsed).__name__}."
                )
            return parsed
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "MockLLMClient.complete_json received a string response "
                    f"that was not valid JSON: {raw!r}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    "MockLLMClient.complete_json expected the response JSON to "
                    f"decode to a dict, got {type(parsed).__name__}."
                )
            return parsed
        raise TypeError(
            f"Unsupported mock response type: {type(raw).__name__}. "
            "Expected LLMResponse, str, or dict."
        )

    # -------------------------------------------------------------- LLMClient API

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "tools": tools,
        }
        self._call_log.append(("complete", list(messages), kwargs))
        raw = self._next_raw(messages)
        response = self._coerce_to_response(raw, self._default_model)
        if model is not None and response.model == self._default_model:
            # Reflect the caller-requested model if we synthesised the response
            # from a bare str / dict; preserves useful metadata without
            # overriding an explicitly-constructed LLMResponse.
            response = LLMResponse(
                content=response.content,
                model=model,
                usage=dict(response.usage),
                raw=response.raw,
            )
        return response

    async def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "schema": schema,
            "model": model,
            "temperature": temperature,
        }
        self._call_log.append(("complete_json", list(messages), kwargs))
        raw = self._next_raw(messages)
        return self._coerce_to_dict(raw)


__all__ = ["MockLLMClient"]
