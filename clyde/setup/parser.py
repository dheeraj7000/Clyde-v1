"""Natural-language scenario parser.

The :class:`ScenarioParser` translates a free-form English description (plus
optional supporting documents) into a structured :class:`ParseResult`. The
parser hits a single ``LLMClient.complete_json`` endpoint, validates the
returned schema against the system's enum constants, and surfaces every
uncertain field as an :class:`Ambiguity` for downstream user resolution.

This module is the *only* component of the setup phase that converts raw
natural language into structured shock parameters. It does not build actors,
networks, or knowledge graphs -- those concerns live in
``world_factory.py`` / ``knowledge_graph.py``.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from typing import Any

from clyde.llm import LLMClient, LLMMessage
from clyde.models.config import VALID_SCOPES
from clyde.models.input import (
    ActorHint,
    Ambiguity,
    Document,
    ParseResult,
    ShockParams,
)
from clyde.models.time import VALID_STEP_UNITS, TimeHorizon


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParsingTimeoutError(RuntimeError):
    """Raised when the parser exhausts all retries without a usable LLM reply.

    The original failure (``asyncio.TimeoutError``, JSON decode error, or any
    other ``Exception``) is preserved on the ``cause`` attribute and chained
    via ``raise ... from cause``.
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Per-document excerpt size injected into the LLM user message.
_DOC_EXCERPT_CHARS = 2000

# Default scope / step unit clamps when the LLM produces an out-of-range value.
_FALLBACK_SCOPE = "sectoral"
_FALLBACK_STEP_UNIT = "week"

_SYSTEM_PROMPT = (
    "You are the Clyde economic-simulator scenario parser. Given a natural "
    "language description (and optional supporting documents), extract a "
    "structured scenario specification as JSON. Flag every uncertain field "
    "as an ambiguity rather than guessing.\n\n"
    "Return JSON with exactly this schema:\n"
    "{\n"
    '  "triggering_event": "string",\n'
    '  "geographies": ["string"],\n'
    '  "markets": ["string"],\n'
    '  "shock_params": {\n'
    '    "shock_type": "string",\n'
    '    "severity": 0.0,\n'
    '    "scope": "micro|sectoral|macro|cross_border",\n'
    '    "duration_steps": 0,\n'
    '    "initial_contact_actors": ["string"]\n'
    "  },\n"
    '  "time_horizon": {"steps": 0, "step_unit": "day|week|month|quarter"},\n'
    '  "ambiguities": [\n'
    '    {"field": "string", "description": "string", "options": ["string"]}\n'
    "  ],\n"
    '  "actor_hints": [\n'
    '    {"actor_type": "string", "count_estimate": 0, "description": "string"}\n'
    "  ]\n"
    "}\n\n"
    "Use dotted field paths in ambiguities (e.g. 'shock_params.severity', "
    "'time_horizon.step_unit'). Severity is a fraction in [0, 1]. "
    "Scope must be one of: micro, sectoral, macro, cross_border. "
    "Step unit must be one of: day, week, month, quarter."
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class ScenarioParser:
    """Parse natural-language scenarios into structured :class:`ParseResult`.

    Parameters
    ----------
    llm_client:
        Any object satisfying the :class:`LLMClient` protocol. The parser
        calls ``complete_json`` exactly once per attempt.
    model:
        Model identifier forwarded to the LLM client.
    max_retries:
        Maximum number of attempts (so a value of 3 means up to 3 calls in
        total). Each retry uses an exponential backoff of
        ``backoff_base_s * 2 ** attempt`` seconds.
    backoff_base_s:
        Base delay between retries. Tiny by default to keep tests fast.
    request_timeout_s:
        Per-attempt timeout enforced via :func:`asyncio.wait_for`.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str | None = None,
        max_retries: int = 3,
        backoff_base_s: float = 0.05,
        request_timeout_s: float = 30.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if backoff_base_s < 0:
            raise ValueError(f"backoff_base_s must be >= 0, got {backoff_base_s}")
        if request_timeout_s <= 0:
            raise ValueError(
                f"request_timeout_s must be > 0, got {request_timeout_s}"
            )
        self._llm = llm_client
        self._model = model
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._request_timeout_s = request_timeout_s

    # ------------------------------------------------------------------ parse

    async def parse(
        self,
        description: str,
        documents: list[Document] | None = None,
    ) -> ParseResult:
        """Parse a natural-language description into a :class:`ParseResult`.

        Empty or nonsensical input (after stripping whitespace, fewer than
        five characters) short-circuits without calling the LLM and returns a
        :class:`ParseResult` whose every structured field is flagged as an
        :class:`Ambiguity`.
        """
        if len(description.strip()) < 5:
            return self._all_ambiguous_result(description)

        messages = self._build_messages(description, documents)
        raw = await self._call_llm_with_retries(messages)
        return self._build_parse_result(raw)

    # ------------------------------------------------------------ resolve

    async def resolve_ambiguities(
        self,
        parse_result: ParseResult,
        resolutions: dict[str, str],
    ) -> ParseResult:
        """Apply user resolutions to a :class:`ParseResult`.

        Returns a *new* :class:`ParseResult` -- the input is never mutated.
        Resolutions whose ``field`` does not match an existing ambiguity are
        silently ignored (Property 1: only flagged fields are updated).
        """
        # Deep-copy so we never mutate caller state.
        new_result = copy.deepcopy(parse_result)

        # Index ambiguities by field for quick lookup. Only the FIRST
        # ambiguity for a given field is considered -- duplicate fields in
        # the input are unusual but we tolerate them by leaving later
        # entries untouched.
        flagged_fields = {amb.field for amb in new_result.ambiguities}

        # Apply field updates for resolutions that match a flagged field.
        for field_path, raw_value in resolutions.items():
            if field_path not in flagged_fields:
                # Property 1: silently ignore resolutions referencing fields
                # that were never flagged.
                continue
            new_result = _apply_resolution(new_result, field_path, raw_value)

        # Mark the matching ambiguities as resolved (replace dataclass
        # instances rather than mutating in place to keep things tidy).
        updated_ambiguities: list[Ambiguity] = []
        for amb in new_result.ambiguities:
            if amb.field in resolutions:
                updated_ambiguities.append(
                    replace(
                        amb,
                        resolved=True,
                        resolution=str(resolutions[amb.field]),
                    )
                )
            else:
                updated_ambiguities.append(amb)
        new_result.ambiguities = updated_ambiguities
        return new_result

    # ------------------------------------------------------ internal helpers

    def _build_messages(
        self,
        description: str,
        documents: list[Document] | None,
    ) -> list[LLMMessage]:
        user_parts: list[str] = [f"SCENARIO DESCRIPTION:\n{description.strip()}"]
        if documents:
            for idx, doc in enumerate(documents, start=1):
                excerpt = (doc.content or "")[:_DOC_EXCERPT_CHARS]
                header = f"--- DOCUMENT {idx}: {doc.path} ({doc.format}) ---"
                user_parts.append(f"{header}\n{excerpt}")
        user_parts.append(
            "Return the structured JSON described in the system prompt."
        )
        user_message = "\n\n".join(user_parts)
        return [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_message),
        ]

    async def _call_llm_with_retries(
        self,
        messages: list[LLMMessage],
    ) -> dict:
        """Invoke the LLM with exponential-backoff retries.

        Retries on ``asyncio.TimeoutError``, ``json.JSONDecodeError``, any
        ``ValueError`` from the mock validation layer, and any other
        ``Exception`` raised by the client. After ``max_retries`` failures
        the last exception is wrapped in :class:`ParsingTimeoutError`.
        """
        last_exc: BaseException | None = None
        for attempt in range(self._max_retries):
            try:
                response = await asyncio.wait_for(
                    self._llm.complete_json(messages, model=self._model),
                    timeout=self._request_timeout_s,
                )
                if not isinstance(response, dict):
                    raise ValueError(
                        "LLM complete_json returned a non-dict payload: "
                        f"{type(response).__name__}"
                    )
                return response
            except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as exc:
                last_exc = exc
                if attempt + 1 >= self._max_retries:
                    break
                delay = self._backoff_base_s * (2 ** attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        raise ParsingTimeoutError(
            f"ScenarioParser failed after {self._max_retries} attempts: {last_exc!r}",
            cause=last_exc,
        ) from last_exc

    # ---------------------------------------------------- result construction

    def _build_parse_result(self, raw: dict) -> ParseResult:
        """Validate and structure the LLM response into a :class:`ParseResult`."""
        ambiguities: list[Ambiguity] = []

        # 1) triggering_event
        triggering_event = _coerce_str(raw.get("triggering_event"))
        if not triggering_event:
            ambiguities.append(
                Ambiguity(
                    field="triggering_event",
                    description="Triggering event missing or empty in LLM response.",
                    options=None,
                )
            )

        # 2) geographies / markets
        geographies = _coerce_str_list(raw.get("geographies"))
        markets = _coerce_str_list(raw.get("markets"))

        # 3) shock_params block
        sp_raw = raw.get("shock_params") or {}
        if not isinstance(sp_raw, dict):
            sp_raw = {}
            ambiguities.append(
                Ambiguity(
                    field="shock_params",
                    description="LLM did not return a shock_params object.",
                    options=None,
                )
            )

        shock_type = _coerce_str(sp_raw.get("shock_type"))
        if not shock_type:
            ambiguities.append(
                Ambiguity(
                    field="shock_params.shock_type",
                    description="Shock type missing or empty.",
                    options=None,
                )
            )

        severity = _coerce_float(sp_raw.get("severity"), default=0.0)
        if not (0.0 <= severity <= 1.0):
            ambiguities.append(
                Ambiguity(
                    field="shock_params.severity",
                    description=(
                        f"Severity {severity!r} is outside [0, 1]; clamped to 0."
                    ),
                    options=None,
                )
            )
            severity = 0.0

        scope = _coerce_str(sp_raw.get("scope"))
        if scope not in VALID_SCOPES:
            ambiguities.append(
                Ambiguity(
                    field="shock_params.scope",
                    description=(
                        scope
                        if scope
                        else "Scope missing; clamped to placeholder."
                    ),
                    options=sorted(VALID_SCOPES),
                )
            )
            scope = _FALLBACK_SCOPE

        duration_steps = _coerce_int(sp_raw.get("duration_steps"), default=0)
        if duration_steps < 0:
            ambiguities.append(
                Ambiguity(
                    field="shock_params.duration_steps",
                    description=(
                        f"duration_steps {duration_steps} is negative; clamped to 0."
                    ),
                    options=None,
                )
            )
            duration_steps = 0

        initial_actors = _coerce_str_list(sp_raw.get("initial_contact_actors"))

        shock_params = ShockParams(
            shock_type=shock_type,
            severity=severity,
            scope=scope,
            duration_steps=duration_steps,
            initial_contact_actors=initial_actors,
        )

        # 4) time_horizon
        th_raw = raw.get("time_horizon") or {}
        if not isinstance(th_raw, dict):
            th_raw = {}
            ambiguities.append(
                Ambiguity(
                    field="time_horizon",
                    description="LLM did not return a time_horizon object.",
                    options=None,
                )
            )
        steps = _coerce_int(th_raw.get("steps"), default=0)
        if steps < 0:
            ambiguities.append(
                Ambiguity(
                    field="time_horizon.steps",
                    description=f"time_horizon.steps {steps} is negative; clamped to 0.",
                    options=None,
                )
            )
            steps = 0
        step_unit = _coerce_str(th_raw.get("step_unit"))
        if step_unit not in VALID_STEP_UNITS:
            ambiguities.append(
                Ambiguity(
                    field="time_horizon.step_unit",
                    description=(
                        step_unit
                        if step_unit
                        else "step_unit missing; clamped to 'week'."
                    ),
                    options=sorted(VALID_STEP_UNITS),
                )
            )
            step_unit = _FALLBACK_STEP_UNIT
        time_horizon = TimeHorizon(steps=steps, step_unit=step_unit)

        # 5) LLM-supplied ambiguities (passed through after coercion)
        for amb_raw in raw.get("ambiguities") or []:
            if not isinstance(amb_raw, dict):
                continue
            field_name = _coerce_str(amb_raw.get("field"))
            if not field_name:
                continue
            ambiguities.append(
                Ambiguity(
                    field=field_name,
                    description=_coerce_str(amb_raw.get("description")),
                    options=_coerce_optional_str_list(amb_raw.get("options")),
                )
            )

        # 6) actor_hints
        actor_hints: list[ActorHint] = []
        for hint_raw in raw.get("actor_hints") or []:
            if not isinstance(hint_raw, dict):
                continue
            actor_type = _coerce_str(hint_raw.get("actor_type"))
            if not actor_type:
                continue
            count_estimate_raw = hint_raw.get("count_estimate")
            count_estimate: int | None
            if count_estimate_raw is None:
                count_estimate = None
            else:
                try:
                    count_estimate = int(count_estimate_raw)
                except (TypeError, ValueError):
                    count_estimate = None
            actor_hints.append(
                ActorHint(
                    actor_type=actor_type,
                    count_estimate=count_estimate,
                    description=_coerce_str(hint_raw.get("description")),
                )
            )

        # De-duplicate ambiguities by (field, description) preserving order so
        # that LLM-supplied entries don't get clobbered by validator-added
        # ones for the same field, but exact duplicates collapse.
        ambiguities = _dedupe_ambiguities(ambiguities)

        return ParseResult(
            triggering_event=triggering_event,
            geographies=geographies,
            markets=markets,
            shock_params=shock_params,
            time_horizon=time_horizon,
            ambiguities=ambiguities,
            actor_hints=actor_hints,
        )

    @staticmethod
    def _all_ambiguous_result(description: str) -> ParseResult:
        """Construct a fully-flagged ParseResult for empty / nonsensical input.

        Every structured field is added to the ambiguity list per the design
        document's "Empty or nonsensical NL input" handling rule.
        """
        flagged_fields = [
            ("triggering_event", "Triggering event could not be inferred from input."),
            ("geographies", "Geographies could not be inferred from input."),
            ("markets", "Markets could not be inferred from input."),
            ("shock_params.shock_type", "Shock type could not be inferred."),
            ("shock_params.severity", "Severity could not be inferred."),
            ("shock_params.scope", "Scope could not be inferred."),
            ("shock_params.duration_steps", "Duration could not be inferred."),
            (
                "shock_params.initial_contact_actors",
                "Initial contact actors could not be inferred.",
            ),
            ("time_horizon.steps", "Time horizon steps could not be inferred."),
            ("time_horizon.step_unit", "Time horizon unit could not be inferred."),
        ]
        ambiguities = [
            Ambiguity(field=name, description=desc, options=None)
            for name, desc in flagged_fields
        ]
        return ParseResult(
            triggering_event="",
            geographies=[],
            markets=[],
            shock_params=ShockParams(scope=_FALLBACK_SCOPE),
            time_horizon=TimeHorizon(steps=0, step_unit=_FALLBACK_STEP_UNIT),
            ambiguities=ambiguities,
            actor_hints=[],
        )


# ---------------------------------------------------------------------------
# Field-path resolver for resolve_ambiguities()
# ---------------------------------------------------------------------------


# Type coercion strategy per dotted field path. Functions take the raw user
# string and return the value to assign on the dataclass.
_LIST_FROM_CSV = lambda v: [s.strip() for s in v.split(",") if s.strip()]  # noqa: E731


def _coerce_resolution(field_path: str, value: str) -> Any:
    """Coerce a resolution string into the right Python type for ``field_path``."""
    if field_path == "shock_params.severity":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if field_path == "shock_params.duration_steps":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if field_path == "time_horizon.steps":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if field_path in {"geographies", "markets", "shock_params.initial_contact_actors"}:
        return _LIST_FROM_CSV(value)
    # Default: keep as-is (string).
    return value


def _apply_resolution(
    parse_result: ParseResult,
    field_path: str,
    raw_value: str,
) -> ParseResult:
    """Return a new ParseResult with ``field_path`` updated to ``raw_value``.

    Field path syntax: dotted names such as ``"shock_params.severity"`` or
    ``"time_horizon.step_unit"``. ``TimeHorizon`` is a frozen dataclass so we
    rebuild it with :func:`dataclasses.replace`. ``ShockParams`` is mutable
    but we still copy it to avoid sharing references with the input.
    """
    coerced = _coerce_resolution(field_path, raw_value)
    parts = field_path.split(".")
    target = parse_result
    # Walk all but the last component, copying mutable containers along the
    # way so we don't accidentally mutate the input ParseResult.
    if len(parts) == 1:
        attr = parts[0]
        if not hasattr(parse_result, attr):
            return parse_result
        # Special-case: time_horizon is frozen, but if someone aimed at the
        # whole field we'd need a more complex coercion. For now only leaf
        # fields are supported on the top-level dataclass.
        try:
            new_result = copy.deepcopy(parse_result)
            setattr(new_result, attr, coerced)
        except AttributeError:
            return parse_result
        return new_result

    # Two-component path (e.g. "shock_params.scope" or "time_horizon.steps").
    head, leaf = parts[0], ".".join(parts[1:])
    if not hasattr(parse_result, head):
        return parse_result
    container = getattr(parse_result, head)

    if head == "time_horizon":
        # TimeHorizon is frozen; build a replacement that satisfies its
        # __post_init__ invariants (else fall back to the original).
        kwargs = {leaf: coerced}
        try:
            new_th = replace(container, **kwargs)
        except (TypeError, ValueError):
            return parse_result
        parse_result.time_horizon = new_th
        return parse_result

    # Mutable nested dataclass (ShockParams): copy + setattr.
    new_container = copy.deepcopy(container)
    if not hasattr(new_container, leaf):
        return parse_result
    try:
        setattr(new_container, leaf, coerced)
    except AttributeError:
        return parse_result
    setattr(parse_result, head, new_container)
    return parse_result


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _coerce_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return _coerce_str_list(value)


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_ambiguities(ambiguities: list[Ambiguity]) -> list[Ambiguity]:
    seen: set[tuple[str, str]] = set()
    out: list[Ambiguity] = []
    for amb in ambiguities:
        key = (amb.field, amb.description)
        if key in seen:
            continue
        seen.add(key)
        out.append(amb)
    return out


__all__ = ["ScenarioParser", "ParsingTimeoutError"]
