"""LLM-powered economic actor persona generator.

Given a ShockConfig and a list of Actor objects (with their typed params),
the :class:`PersonaGenerator` asks the LLM to produce rich narrative
identities grounded in the actor's behavioral parameters and role in the
economy. The personas are purely decorative metadata — they never influence
the rule-based simulation. They exist so the UI can show "Maria Chen, 34,
software engineer, risk-averse saver" instead of "household_0003".
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from clyde.llm.client import LLMClient, LLMMessage
from clyde.models.actors import Actor
from clyde.models.config import ShockConfig
from clyde.models.enums import ActorType


@dataclass
class ActorPersona:
    """Rich narrative identity for a single actor."""
    actor_id: str
    actor_type: str
    display_name: str
    role: str
    description: str
    economic_behavior: str
    vulnerability: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "display_name": self.display_name,
            "role": self.role,
            "description": self.description,
            "economic_behavior": self.economic_behavior,
            "vulnerability": self.vulnerability,
            "tags": list(self.tags),
        }


@dataclass
class InfluenceConfig:
    """Economic influence propagation weights — the economic equivalent
    of a social media recommendation algorithm."""
    monetary_transmission_lag: float = 0.6
    information_asymmetry: float = 0.4
    herding_strength: float = 0.3
    credit_channel_weight: float = 0.5
    expectation_channel_weight: float = 0.4
    supply_chain_friction: float = 0.3
    interbank_contagion: float = 0.5
    confidence_multiplier: float = 0.35

    def to_dict(self) -> dict[str, float]:
        return {
            "monetary_transmission_lag": self.monetary_transmission_lag,
            "information_asymmetry": self.information_asymmetry,
            "herding_strength": self.herding_strength,
            "credit_channel_weight": self.credit_channel_weight,
            "expectation_channel_weight": self.expectation_channel_weight,
            "supply_chain_friction": self.supply_chain_friction,
            "interbank_contagion": self.interbank_contagion,
            "confidence_multiplier": self.confidence_multiplier,
        }


_PERSONA_SYSTEM_PROMPT = """\
You are an economic persona generator. Given actors with behavioral parameters \
and a shock scenario, generate a short narrative identity for each actor.

Ground each persona in the actor's actual parameters. Examples:
- High MPC (0.85) + low savings → "young gig worker, paycheck to paycheck"
- High bankruptcy_threshold (5.0) → "well-capitalized tech company"
- High risk_appetite (0.8) → "aggressive regional lender"

Return JSON: {"personas": [{"actor_id":"string","display_name":"string",\
"role":"string (short)","description":"string (1 sentence)",\
"economic_behavior":"string (1 sentence)","vulnerability":"string (1 sentence)",\
"tags":["tag1","tag2"]}]}

Keep descriptions SHORT (under 20 words each). Return ONLY JSON.
"""

_INFLUENCE_SYSTEM_PROMPT = """\
You are an economic simulation configurator. Given a shock scenario, generate
the influence propagation weights that control how economic signals travel
through the actor network. These are the economic equivalent of a social
media recommendation algorithm — they determine how fast and strongly
central bank signals reach banks, how bank credit tightening propagates to
firms, and how firm distress affects household confidence.

Return a JSON object with these fields (all floats in [0, 1]):
{
  "monetary_transmission_lag": float,  // how slowly CB rate changes reach the real economy (higher = slower)
  "information_asymmetry": float,      // how unevenly information spreads (higher = more asymmetric)
  "herding_strength": float,           // how strongly actors copy peer behavior (bank herding, consumer panic)
  "credit_channel_weight": float,      // importance of the bank lending channel
  "expectation_channel_weight": float, // importance of inflation expectations channel
  "supply_chain_friction": float,      // how much supply chain disruptions amplify shocks
  "interbank_contagion": float,        // how fast stress spreads between banks
  "confidence_multiplier": float,      // how much consumer confidence amplifies/dampens effects
  "reasoning": "string explaining your choices for this specific scenario"
}

Return ONLY the JSON object. No prose, no code fences.
"""


def _salvage_truncated_json(text: str) -> dict | list | None:
    """Attempt to parse truncated JSON by closing open brackets/braces."""
    # Find the last complete object in a personas array
    # Strategy: find all complete {...} blocks inside the personas array
    import re
    # Try progressively closing the JSON
    for suffix in ['}]}', '"}]}', '"}],"tags":[]}]}', '"]}}]}']:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
    # Try to extract individual persona objects via regex
    pattern = r'\{[^{}]*"actor_id"\s*:\s*"[^"]*"[^{}]*\}'
    matches = re.findall(pattern, text)
    if matches:
        personas = []
        for m in matches:
            try:
                personas.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        if personas:
            return {"personas": personas}
    return None


def _actor_summary(actor: Actor, shock: ShockConfig) -> dict:
    """Compact summary of an actor for the LLM prompt."""
    from dataclasses import asdict
    params = asdict(actor.params)
    # Only send the 3 most distinctive params to keep prompt short
    return {
        "id": actor.id,
        "type": actor.actor_type.value,
        "params": params,
        "shocked": actor.id in (shock.initial_contact_actors or []),
    }


class PersonaGenerator:
    """Generate rich economic personas for simulation actors via LLM."""

    def __init__(self, llm_client: LLMClient, *, model: str | None = None) -> None:
        self._llm = llm_client
        self._model = model

    async def generate_personas(
        self,
        actors: list[Actor],
        shock_config: ShockConfig,
    ) -> list[ActorPersona]:
        """Generate personas for all actors, batched by type."""
        all_personas: list[ActorPersona] = []

        # Batch by actor type to keep prompts focused
        by_type: dict[ActorType, list[Actor]] = {}
        for a in actors:
            by_type.setdefault(a.actor_type, []).append(a)

        scenario_ctx = (
            f"Shock: {shock_config.shock_type}, severity={shock_config.severity}, "
            f"scope={shock_config.scope}, geography={shock_config.geography}, "
            f"sectors={shock_config.sectors}"
        )

        # Circuit breaker: once a batch fails because of provider rate-limiting
        # (HTTP 429 / quota / queue), stop calling the LLM for the remaining
        # batches in this run and fall through to the deterministic fallback.
        # Personas are decorative (they don't affect simulation outcomes), so
        # losing LLM enrichment for some actors is acceptable; what's NOT
        # acceptable is making the rate limit worse by issuing more calls.
        rate_limited = False

        for atype, type_actors in by_type.items():
            # For households (many), batch in groups of 8; others in 5
            batch_size = 8 if atype == ActorType.HOUSEHOLD else 5
            for i in range(0, len(type_actors), batch_size):
                batch = type_actors[i:i + batch_size]
                if rate_limited:
                    for a in batch:
                        all_personas.append(self._fallback_persona(a))
                    continue
                summaries = [_actor_summary(a, shock_config) for a in batch]
                user_msg = (
                    f"Scenario: {scenario_ctx}\n\n"
                    f"Generate personas for these {atype.value} actors:\n"
                    f"{json.dumps(summaries, indent=2)}"
                )
                messages = [
                    LLMMessage(role="system", content=_PERSONA_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_msg),
                ]
                try:
                    resp = await self._llm.complete(
                        messages, model=self._model, max_tokens=4096, response_format="json"
                    )
                    text = resp.content.strip()
                    if text.startswith("```"):
                        text = text.strip("`")
                        if text.lower().startswith("json"):
                            text = text[4:].lstrip()
                    # Try to parse; if truncated, attempt to salvage partial JSON
                    try:
                        raw = json.loads(text)
                    except json.JSONDecodeError:
                        # Try closing the JSON structure to salvage partial results
                        salvaged = _salvage_truncated_json(text)
                        if salvaged is not None:
                            raw = salvaged
                            logger.info("Salvaged %d personas from truncated JSON for %s batch", len(raw.get("personas", raw if isinstance(raw, list) else [])), atype.value)
                        else:
                            raise
                    # LLM may return {"personas": [...]} or {"actors": [...]} or just [...]
                    if isinstance(raw, list):
                        personas = raw
                    elif isinstance(raw, dict):
                        # Try common wrapper keys
                        personas = (
                            raw.get("personas")
                            or raw.get("actors")
                            or raw.get("profiles")
                            or raw.get("results")
                            or raw.get("data")
                            or []
                        )
                        # If none of those keys exist, check if the dict itself
                        # looks like a single persona (has actor_id)
                        if not personas and "actor_id" in raw:
                            personas = [raw]
                        # Last resort: grab the first list-valued key
                        if not personas:
                            for v in raw.values():
                                if isinstance(v, list):
                                    personas = v
                                    break
                    else:
                        personas = []
                    for p in personas:
                        if not isinstance(p, dict):
                            continue
                        all_personas.append(ActorPersona(
                            actor_id=p.get("actor_id", ""),
                            actor_type=atype.value,
                            display_name=p.get("display_name", ""),
                            role=p.get("role", ""),
                            description=p.get("description", ""),
                            economic_behavior=p.get("economic_behavior", ""),
                            vulnerability=p.get("vulnerability", ""),
                            tags=p.get("tags", []),
                        ))
                except Exception as exc:
                    msg = str(exc)
                    is_rate_limit = (
                        "429" in msg
                        or "too_many_requests" in msg.lower()
                        or "quota" in msg.lower()
                        or "queue_exceeded" in msg.lower()
                        or "rate limit" in msg.lower()
                    )
                    if is_rate_limit and not rate_limited:
                        logger.warning(
                            "Persona generator hit provider rate limit (%s) — "
                            "switching remaining batches to deterministic fallback for this run.",
                            atype.value,
                        )
                        rate_limited = True
                    else:
                        logger.warning("Persona generation failed for %s batch: %s", atype.value, exc)
                    # Fallback: generate minimal personas without LLM
                    for a in batch:
                        all_personas.append(self._fallback_persona(a))

        # Fill in any actors that didn't get a persona
        covered = {p.actor_id for p in all_personas}
        for a in actors:
            if a.id not in covered:
                all_personas.append(self._fallback_persona(a))

        return all_personas

    async def generate_influence_config(
        self,
        shock_config: ShockConfig,
    ) -> tuple[InfluenceConfig, str]:
        """Generate scenario-specific influence propagation weights.
        Returns (config, reasoning_text)."""
        user_msg = (
            f"Scenario: shock_type={shock_config.shock_type}, "
            f"severity={shock_config.severity}, scope={shock_config.scope}, "
            f"geography={shock_config.geography}, sectors={shock_config.sectors}, "
            f"duration_steps={shock_config.duration_steps}"
        )
        messages = [
            LLMMessage(role="system", content=_INFLUENCE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_msg),
        ]
        try:
            raw = await self._llm.complete_json(messages, model=self._model)
            reasoning = raw.pop("reasoning", "")
            cfg = InfluenceConfig(
                monetary_transmission_lag=float(raw.get("monetary_transmission_lag", 0.6)),
                information_asymmetry=float(raw.get("information_asymmetry", 0.4)),
                herding_strength=float(raw.get("herding_strength", 0.3)),
                credit_channel_weight=float(raw.get("credit_channel_weight", 0.5)),
                expectation_channel_weight=float(raw.get("expectation_channel_weight", 0.4)),
                supply_chain_friction=float(raw.get("supply_chain_friction", 0.3)),
                interbank_contagion=float(raw.get("interbank_contagion", 0.5)),
                confidence_multiplier=float(raw.get("confidence_multiplier", 0.35)),
            )
            return cfg, str(reasoning)
        except Exception:
            return InfluenceConfig(), "Default configuration (LLM unavailable)"

    @staticmethod
    def _fallback_persona(actor: Actor) -> ActorPersona:
        """Minimal persona when LLM is unavailable."""
        type_roles = {
            ActorType.HOUSEHOLD: ("Household Member", "Consumer & Worker"),
            ActorType.FIRM: ("Business Entity", "Producer & Employer"),
            ActorType.BANK: ("Financial Institution", "Lender & Intermediary"),
            ActorType.CENTRAL_BANK: ("Central Bank", "Monetary Authority"),
        }
        name, role = type_roles.get(actor.actor_type, ("Economic Actor", "Participant"))
        idx = actor.id.split("_")[-1]
        return ActorPersona(
            actor_id=actor.id,
            actor_type=actor.actor_type.value,
            display_name=f"{name} #{idx}",
            role=role,
            description=f"A {actor.actor_type.value} participant in the economic simulation.",
            economic_behavior="Behavior determined by empirical parameters from the Prior Library.",
            vulnerability="Vulnerability depends on network position and shock characteristics.",
            tags=[actor.actor_type.value],
        )
