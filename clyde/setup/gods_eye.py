"""God's Eye Console: natural-language intervention parser.

The :class:`GodsEyeConsole` translates analyst-authored natural-language
"injections" (e.g. *"raise the policy rate by 50bp at step 5"*) into the
structured :class:`~clyde.models.config.ShockDelta` form used by the
branch-forking machinery, then merges that delta into a base
:class:`~clyde.models.config.ShockConfig` so the simulator can re-run from
step 0 against the patched configuration.

Design notes
------------

* The console is the **only** component allowed to call an LLM during
  intervention parsing.  Once a :class:`ShockDelta` is produced, the rest
  of the branch lifecycle is rule-based.
* Ambiguities that the LLM cannot resolve (or that we discover during
  validation, e.g. an out-of-range ``intervention_step``) are smuggled
  back to the caller via a magic key inside ``param_overrides``.  The
  constant :data:`AMBIGUITY_KEY` names that key so callers can probe for
  it; :meth:`GodsEyeConsole.apply_delta` strips the key before merging
  into ``behavioral_overrides`` so it never leaks into a runnable
  ``ShockConfig``.
* The retry loop mirrors the established setup-phase pattern: at most
  ``max_retries`` attempts with exponential backoff seeded by
  ``backoff_base_s``.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from clyde.llm.client import LLMClient, LLMMessage
from clyde.models.config import ShockConfig, ShockDelta, VALID_SCOPES
from clyde.models.input import Ambiguity
from clyde.models.scenario import Scenario


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Magic key smuggled inside ``ShockDelta.param_overrides`` to carry a list
#: of unresolved ambiguities back to the caller.  Callers can check for
#: ``AMBIGUITY_KEY in delta.param_overrides`` to surface a UI prompt.  The
#: key is stripped during :meth:`GodsEyeConsole.apply_delta` so it never
#: pollutes a real :class:`ShockConfig`.
AMBIGUITY_KEY: str = "_ambiguities"


# ---------------------------------------------------------------------------
# Keys that, when present in ``param_overrides``, override top-level
# ShockConfig fields rather than landing in ``behavioral_overrides``.
# ---------------------------------------------------------------------------
_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"severity", "duration_steps", "shock_type", "scope"}
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """\
You are the God's Eye Console for the Clyde economic simulator.  Translate
the analyst's natural-language intervention into a JSON object describing a
ShockDelta that will be merged into the base ShockConfig and re-simulated
from step 0.

Return STRICTLY a single JSON object with the following keys.  Do not
include any prose, code fences, or commentary outside the JSON.

{
  "intervention_step": <int>,                 // simulation step at which the intervention applies
  "param_overrides":   {<string>: <any>},     // shock parameter overrides (e.g. {"severity": 0.7})
  "new_events":        [<string>, ...],       // optional new event identifiers triggered by the injection
  "description":       "<string>",            // echo of the original injection text
  "ambiguities":       [                      // optional list; omit or empty when fully resolved
    {"field": "<string>", "description": "<string>", "options": ["<string>", ...]}
  ]
}

Recognised top-level shock fields that may appear in `param_overrides`:
`severity` (float in [0, 1]), `duration_steps` (int >= 0), `shock_type`
(string), `scope` (one of "micro", "sectoral", "macro", "cross_border").
Any other override is treated as a behavioural parameter and merged into
the resulting ShockConfig.behavioral_overrides dict.

If a required field cannot be inferred from the injection, set a sensible
placeholder and add an entry to `ambiguities` describing what was unclear
and a short list of plausible options.
"""


# ---------------------------------------------------------------------------
# GodsEyeConsole
# ---------------------------------------------------------------------------


class GodsEyeConsole:
    """Natural-language intervention interface.

    Parameters
    ----------
    llm_client:
        Any object satisfying :class:`~clyde.llm.LLMClient`.  Only used by
        :meth:`parse_injection`; :meth:`apply_delta` is pure.
    model:
        Model identifier passed through to ``llm_client.complete_json``.
    max_retries:
        Maximum number of attempts (including the first) to obtain a
        well-formed JSON response from the LLM.  Must be >= 1.
    backoff_base_s:
        Base seconds for the exponential backoff between retries.  The
        sleep before attempt ``n`` (0-indexed) is
        ``backoff_base_s * (2 ** n)``.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str | None = None,
        max_retries: int = 3,
        backoff_base_s: float = 0.05,
    ) -> None:
        if max_retries < 1:
            raise ValueError(
                f"GodsEyeConsole.max_retries must be >= 1, got {max_retries}"
            )
        if backoff_base_s < 0:
            raise ValueError(
                f"GodsEyeConsole.backoff_base_s must be >= 0, got {backoff_base_s}"
            )
        self._llm = llm_client
        self._model = model
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    # ------------------------------------------------------------------ #
    # parse_injection
    # ------------------------------------------------------------------ #
    async def parse_injection(
        self,
        injection_text: str,
        base_scenario: Scenario,
    ) -> ShockDelta:
        """Parse a natural-language intervention into a :class:`ShockDelta`.

        On success, returns a :class:`ShockDelta` whose
        ``intervention_step`` is clipped into the legal range
        ``[0, time_horizon.steps - 1]``.  Any unresolved ambiguities (raw
        from the LLM, plus any synthesised during validation) are stashed
        under ``param_overrides[AMBIGUITY_KEY]`` as a list of
        :class:`Ambiguity` ``to_dict`` payloads.

        Raises
        ------
        RuntimeError
            If every retry attempt fails to return a usable dict.  The
            final underlying exception is chained as ``__cause__``.
        """
        messages = self._build_messages(injection_text, base_scenario)

        last_exc: BaseException | None = None
        response: dict | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._llm.complete_json(
                    messages,
                    model=self._model,
                )
                if not isinstance(response, dict):
                    raise ValueError(
                        "LLM complete_json returned a non-dict payload: "
                        f"{type(response).__name__}"
                    )
                # Successful retrieval; break out of the retry loop.
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 -- we re-raise after retries
                last_exc = exc
                response = None
                if attempt + 1 < self._max_retries:
                    await asyncio.sleep(self._backoff_base_s * (2 ** attempt))

        if response is None:
            raise RuntimeError(
                f"god's eye parsing failed after {self._max_retries} retries"
            ) from last_exc

        return self._build_delta(response, injection_text, base_scenario)

    # ------------------------------------------------------------------ #
    # apply_delta -- pure, no LLM
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_delta(
        base_config: ShockConfig,
        delta: ShockDelta,
    ) -> ShockConfig:
        """Merge ``delta`` into ``base_config`` and return a NEW
        :class:`ShockConfig`.

        The merge rules are:

        * Top-level ShockConfig fields (``severity``, ``duration_steps``,
          ``shock_type``, ``scope``) present in ``delta.param_overrides``
          replace the corresponding base value.
        * All other entries in ``param_overrides`` are merged into
          ``behavioral_overrides`` (delta wins on conflict).
        * ``delta.new_events`` is appended into
          ``behavioral_overrides["new_events"]`` (concatenated with any
          existing list).
        * The :data:`AMBIGUITY_KEY` entry is metadata and is never copied
          into the returned config.

        The returned config is freshly constructed, so its
        ``__post_init__`` validates field ranges.  Inputs are not mutated.
        """
        # Deep-copy the snapshot of base behavioral_overrides + scalar fields
        # so we never mutate the caller's objects.
        merged_behavioral: dict[str, Any] = copy.deepcopy(
            dict(base_config.behavioral_overrides)
        )
        overrides: dict[str, Any] = copy.deepcopy(dict(delta.param_overrides))

        # Promote top-level keys.
        promoted: dict[str, Any] = {}
        for key in _TOP_LEVEL_KEYS:
            if key in overrides:
                promoted[key] = overrides.pop(key)

        # Drop the ambiguity smuggling key -- it is metadata only.
        overrides.pop(AMBIGUITY_KEY, None)

        # Strip top-level keys from behavioral_overrides as well so we
        # don't end up double-bookkeeping. (Defensive; usually absent.)
        for key in _TOP_LEVEL_KEYS:
            merged_behavioral.pop(key, None)

        # Merge the remaining overrides; delta wins on conflict.
        for key, value in overrides.items():
            merged_behavioral[key] = value

        # Concatenate new_events into behavioral_overrides["new_events"].
        if delta.new_events:
            existing_events_raw = merged_behavioral.get("new_events", [])
            if not isinstance(existing_events_raw, list):
                # Be liberal: coerce a scalar into a single-element list.
                existing_events: list[Any] = [existing_events_raw]
            else:
                existing_events = list(existing_events_raw)
            merged_behavioral["new_events"] = existing_events + list(delta.new_events)

        # Build the new config; let __post_init__ validate.
        return ShockConfig(
            shock_type=promoted.get("shock_type", base_config.shock_type),
            severity=promoted.get("severity", base_config.severity),
            scope=promoted.get("scope", base_config.scope),
            duration_steps=promoted.get("duration_steps", base_config.duration_steps),
            geography=list(base_config.geography),
            sectors=list(base_config.sectors),
            initial_contact_actors=list(base_config.initial_contact_actors),
            agent_counts=dict(base_config.agent_counts),
            behavioral_overrides=merged_behavioral,
            time_horizon=base_config.time_horizon,
            ensemble_seed=base_config.ensemble_seed,
            historical_analogs=list(base_config.historical_analogs),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_messages(
        self,
        injection_text: str,
        base_scenario: Scenario,
    ) -> list[LLMMessage]:
        """Construct the system + user messages for ``complete_json``."""
        snapshot = {
            "scenario_id": base_scenario.scenario_id,
            "config": base_scenario.config.to_dict(),
        }
        user_content = (
            "Base scenario snapshot:\n"
            f"{json.dumps(snapshot, sort_keys=True)}\n\n"
            "Injection text:\n"
            f"{injection_text}"
        )
        return [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

    @staticmethod
    def _build_delta(
        response: dict,
        injection_text: str,
        base_scenario: Scenario,
    ) -> ShockDelta:
        """Translate a validated LLM JSON response into a :class:`ShockDelta`."""
        param_overrides_raw = response.get("param_overrides", {})
        if not isinstance(param_overrides_raw, dict):
            param_overrides_raw = {}
        param_overrides: dict[str, Any] = dict(param_overrides_raw)

        new_events_raw = response.get("new_events", [])
        new_events: list[str] = (
            [str(e) for e in new_events_raw]
            if isinstance(new_events_raw, list)
            else []
        )

        description = response.get("description") or injection_text
        if not isinstance(description, str):
            description = injection_text

        # Collect ambiguities that the LLM surfaced.
        ambiguities: list[Ambiguity] = []
        raw_ambs = response.get("ambiguities", [])
        if isinstance(raw_ambs, list):
            for entry in raw_ambs:
                if isinstance(entry, dict) and "field" in entry and "description" in entry:
                    opts = entry.get("options")
                    ambiguities.append(
                        Ambiguity(
                            field=str(entry["field"]),
                            description=str(entry["description"]),
                            options=(
                                [str(o) for o in opts]
                                if isinstance(opts, list)
                                else None
                            ),
                        )
                    )

        # Validate / clip intervention_step against the base time horizon.
        max_step = max(0, base_scenario.config.time_horizon.steps - 1)
        raw_step = response.get("intervention_step")

        intervention_step: int
        if not isinstance(raw_step, int) or isinstance(raw_step, bool):
            # Missing or non-integer -> clip to 0 and flag.
            intervention_step = 0
            ambiguities.append(
                Ambiguity(
                    field="intervention_step",
                    description=(
                        "intervention_step missing or non-integer in LLM "
                        f"response (got {raw_step!r}); defaulted to 0"
                    ),
                    options=None,
                )
            )
        elif raw_step < 0 or raw_step > max_step:
            clipped = 0 if raw_step < 0 else max_step
            ambiguities.append(
                Ambiguity(
                    field="intervention_step",
                    description=(
                        f"intervention_step={raw_step} out of range "
                        f"[0, {max_step}]; clipped to {clipped}"
                    ),
                    options=None,
                )
            )
            intervention_step = clipped
        else:
            intervention_step = raw_step

        if ambiguities:
            param_overrides[AMBIGUITY_KEY] = [a.to_dict() for a in ambiguities]

        return ShockDelta(
            intervention_step=intervention_step,
            param_overrides=param_overrides,
            new_events=new_events,
            description=description,
        )


__all__ = ["AMBIGUITY_KEY", "GodsEyeConsole"]


# Re-exported for completeness so importers don't have to dig into models.
_ = VALID_SCOPES  # noqa: F841 -- kept as an intentional reference for tooling
