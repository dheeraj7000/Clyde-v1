"""Shared HTTP-based LLM client.

OpenRouter and Cerebras both expose OpenAI-compatible chat-completions APIs.
This module factors the common request/response handling so each provider is
just a thin subclass that picks the base URL, headers, and default model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from clyde.llm.client import LLMMessage, LLMResponse


logger = logging.getLogger(__name__)


class LLMRequestError(RuntimeError):
    """Raised when the LLM provider returns a non-success response."""


@dataclass
class HTTPLLMConfig:
    api_key: str
    base_url: str
    default_model: str
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 60.0
    max_retries: int = 2
    backoff_base_s: float = 0.5
    # When the provider returns a 429 (or quota / queue error), suppress
    # further requests for this many seconds so callers fall through to their
    # deterministic fallback paths instead of making the rate-limit worse.
    rate_limit_cooldown_s: float = 30.0


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        " 429 " in f" {msg} "
        or "429 from" in msg
        or "too_many_requests" in msg
        or "request_quota_exceeded" in msg
        or "queue_exceeded" in msg
        or "rate limit" in msg
    )


class HTTPLLMClient:
    """Generic OpenAI-compatible chat-completions client over httpx."""

    def __init__(self, config: HTTPLLMConfig) -> None:
        self._config = config
        # Monotonic timestamp until which we should fast-fail without hitting
        # the network. Set whenever retries exhaust on a 429-class error.
        self._cooldown_until: float = 0.0

    @property
    def default_model(self) -> str:
        return self._config.default_model

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        h.update(self._config.extra_headers)
        return h

    async def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        # Global cool-down: callers' fallbacks are far cheaper than another
        # round-trip that we already know will be 429'd.
        now = time.monotonic()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            raise LLMRequestError(
                f"LLM client cooling down after rate limit (429); "
                f"{remaining:.1f}s remaining"
            )
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._config.timeout_s) as client:
                    resp = await client.post(url, headers=self._headers(), json=payload)
                if resp.status_code >= 400:
                    raise LLMRequestError(
                        f"{resp.status_code} from {url}: {resp.text[:300]}"
                    )
                return resp.json()
            except (httpx.HTTPError, LLMRequestError) as exc:
                last_exc = exc
                if attempt >= self._config.max_retries:
                    break
                delay = self._config.backoff_base_s * (2 ** attempt)
                logger.warning(
                    "LLM request to %s failed (attempt %d/%d): %s; retrying in %.2fs",
                    url, attempt + 1, self._config.max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
        # If the final failure was a rate-limit, latch a cool-down so peer
        # callers in the same run short-circuit.
        if last_exc is not None and _is_rate_limit_error(last_exc):
            self._cooldown_until = time.monotonic() + self._config.rate_limit_cooldown_s
            logger.warning(
                "LLM provider rate-limited; cooling down for %.0fs (clients will fall back).",
                self._config.rate_limit_cooldown_s,
            )
        raise LLMRequestError(f"LLM request failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _to_openai_messages(messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

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
        payload: dict[str, Any] = {
            "model": model or self._config.default_model,
            "messages": self._to_openai_messages(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools

        raw = await self._post_chat_completions(payload)
        choice = (raw.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        return LLMResponse(
            content=content,
            model=raw.get("model") or payload["model"],
            usage=raw.get("usage") or {},
            raw=raw,
        )

    async def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        del schema  # OpenAI-compatible servers vary in JSON-schema support; ignore
        # Append a strict instruction so models that don't honor response_format
        # still produce JSON.
        nudged = list(messages)
        if nudged and nudged[0].role == "system":
            nudged[0] = LLMMessage(
                role="system",
                content=nudged[0].content
                + "\n\nReturn ONLY a single JSON object. No prose, no code fences.",
            )
        else:
            nudged.insert(
                0,
                LLMMessage(
                    role="system",
                    content="Return ONLY a single JSON object. No prose, no code fences.",
                ),
            )
        resp = await self.complete(
            nudged,
            model=model,
            temperature=temperature,
            response_format="json",
        )
        text = resp.content.strip()
        # Strip ```json fences if a model added them anyway.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMRequestError(
                f"LLM response was not valid JSON: {text[:300]}"
            ) from exc
