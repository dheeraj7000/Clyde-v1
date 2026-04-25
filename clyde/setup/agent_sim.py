"""LLM-powered Agent-Based Simulation Engine (MiroFish-style).

Each economic actor is an autonomous LLM agent that observes its environment,
recalls its history, and makes economic decisions each round. This produces
emergent behavior rather than rule-based deterministic outcomes.

This module lives in clyde/setup/ (not clyde/simulation/) because it uses
LLM calls. The rule-based PropagationEngine remains the "Fast Mode" for
Monte Carlo ensembles; this is "Agent Mode" for rich narrative simulations.

Architecture:
- Each actor has a Memory (recent events, observations, decisions)
- Each round: actors observe neighbors → LLM decides actions → state updates
- Causal events are emitted from LLM-decided actions
- Mid-simulation event injection supported via inject_event()
- Rounds stream results back for live UI updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from clyde.llm.client import LLMClient, LLMMessage
from clyde.models.actors import Actor
from clyde.models.causal import CausalEvent
from clyde.models.config import ShockConfig, SimulationWorld
from clyde.models.enums import ActorType
from clyde.models.metrics import StepMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory & Action dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentMemory:
    """Rolling memory for one actor — what they've seen and done."""
    actor_id: str
    observations: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    received_events: list[str] = field(default_factory=list)
    max_items: int = 20

    def add_observation(self, text: str) -> None:
        self.observations.append(text)
        if len(self.observations) > self.max_items:
            self.observations = self.observations[-self.max_items:]

    def add_decision(self, text: str) -> None:
        self.decisions.append(text)
        if len(self.decisions) > self.max_items:
            self.decisions = self.decisions[-self.max_items:]

    def add_event(self, text: str) -> None:
        self.received_events.append(text)
        if len(self.received_events) > self.max_items:
            self.received_events = self.received_events[-self.max_items:]

    def summary(self) -> str:
        parts = []
        if self.observations:
            parts.append("Recent observations: " + "; ".join(self.observations[-5:]))
        if self.decisions:
            parts.append("My recent decisions: " + "; ".join(self.decisions[-5:]))
        if self.received_events:
            parts.append("Events affecting me: " + "; ".join(self.received_events[-5:]))
        return "\n".join(parts) if parts else "No history yet."

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "observations": list(self.observations[-10:]),
            "decisions": list(self.decisions[-10:]),
            "received_events": list(self.received_events[-10:]),
        }


@dataclass
class AgentAction:
    """A single action decided by an LLM agent."""
    actor_id: str
    action_type: str  # "set_rate", "tighten_credit", "invest", "hire", "fire", "consume", "save", etc.
    target_id: str | None = None
    magnitude: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "magnitude": self.magnitude,
            "reasoning": self.reasoning,
        }


@dataclass
class RoundResult:
    """Output of a single simulation round."""
    round_num: int
    actions: list[AgentAction] = field(default_factory=list)
    events: list[CausalEvent] = field(default_factory=list)
    metrics: StepMetrics | None = None
    actor_states: dict[str, dict[str, float]] = field(default_factory=dict)
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "round_num": self.round_num,
            "actions": [a.to_dict() for a in self.actions],
            "events": [e.to_dict() for e in self.events],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "actor_states": {k: dict(v) for k, v in self.actor_states.items()},
            "narrative": self.narrative,
        }


@dataclass
class AgentSimConfig:
    """Configuration for the agent-based simulation."""
    total_rounds: int = 12
    round_duration_label: str = "1 quarter"
    concurrent_agents: int = 8  # how many agents to query in parallel per round
    temperature: float = 0.3


@dataclass
class AgentSimState:
    """Full state of a running agent simulation."""
    config: AgentSimConfig
    world: SimulationWorld
    shock_config: ShockConfig
    personas: dict[str, dict] = field(default_factory=dict)  # actor_id -> persona dict
    memories: dict[str, AgentMemory] = field(default_factory=dict)
    actor_states: dict[str, dict[str, float]] = field(default_factory=dict)
    rounds: list[RoundResult] = field(default_factory=list)
    current_round: int = 0
    status: str = "ready"  # ready, running, paused, completed
    injected_events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ACTOR_PROMPTS = {
    ActorType.CENTRAL_BANK: """\
You are {name}, {role}. {description}

Current state: policy_rate={policy_rate:.4f}, inflation_target={inflation_target:.4f}, observed_inflation={observed_inflation:.4f}
{memory}
{environment}

As the central bank, decide your action this round. You can:
- "set_rate": adjust the policy rate (magnitude = new rate, e.g. 0.05 for 5%)
- "signal": send forward guidance (magnitude = hawkish 0.0-1.0 or dovish -1.0-0.0)
- "hold": maintain current stance (magnitude = 0)

Return JSON: {{"action_type":"string","magnitude":float,"reasoning":"string (1-2 sentences)"}}""",

    ActorType.BANK: """\
You are {name}, {role}. {description}

Current state: credit_tightness={credit_tightness:.4f}, npl_ratio={npl_ratio:.4f}, reserves={reserves:.4f}, stressed={is_stressed}
{memory}
{environment}

As a bank, decide your action this round. You can:
- "tighten_credit": increase lending standards (magnitude = how much, 0.0-0.3)
- "ease_credit": loosen lending standards (magnitude = how much, 0.0-0.3)
- "call_loans": demand early repayment from distressed borrowers (magnitude = fraction 0.0-1.0)
- "hold": maintain current stance

Return JSON: {{"action_type":"string","magnitude":float,"target_id":"firm_id or null","reasoning":"string (1-2 sentences)"}}""",

    ActorType.FIRM: """\
You are {name}, {role}. {description}

Current state: revenue={revenue:.4f}, costs={costs:.4f}, price_level={price_level:.4f}, employees={employee_count:.0f}, debt={debt:.4f}, bankrupt={is_bankrupt}
{memory}
{environment}

As a firm, decide your action this round. You can:
- "raise_prices": increase prices (magnitude = % increase, e.g. 0.05 for 5%)
- "cut_prices": decrease prices (magnitude = % decrease)
- "hire": hire workers (magnitude = number of workers)
- "fire": lay off workers (magnitude = number of workers)
- "invest": increase investment (magnitude = amount)
- "cut_costs": reduce operations (magnitude = % reduction)
- "hold": maintain current operations

Return JSON: {{"action_type":"string","magnitude":float,"reasoning":"string (1-2 sentences)"}}""",

    ActorType.HOUSEHOLD: """\
You are {name}, {role}. {description}

Current state: income={income:.4f}, savings={savings:.4f}, consumption={consumption:.4f}, employed={employed}, confidence={confidence:.4f}
{memory}
{environment}

As a household, decide your action this round. You can:
- "spend_more": increase consumption (magnitude = % increase)
- "save_more": increase savings, cut spending (magnitude = % of income to save)
- "seek_credit": borrow money (magnitude = amount)
- "demand_raise": ask for higher wages (magnitude = % increase demanded)
- "hold": maintain current behavior

Return JSON: {{"action_type":"string","magnitude":float,"reasoning":"string (1-2 sentences)"}}""",
}


# ---------------------------------------------------------------------------
# Agent Simulation Engine
# ---------------------------------------------------------------------------

class AgentSimEngine:
    """MiroFish-style LLM agent simulation for economics."""

    def __init__(self, llm_client: LLMClient, *, model: str | None = None) -> None:
        self._llm = llm_client
        self._model = model

    def init_state(
        self,
        world: SimulationWorld,
        shock_config: ShockConfig,
        personas: list[dict],
        config: AgentSimConfig | None = None,
    ) -> AgentSimState:
        """Initialize the simulation state from a built world."""
        cfg = config or AgentSimConfig(total_rounds=shock_config.time_horizon.steps or 12)
        persona_map = {p["actor_id"]: p for p in personas}

        # Initialize actor states (same as PropagationEngine._initialise_state but simplified)
        actor_states: dict[str, dict[str, float]] = {}
        for a in world.actors:
            if a.actor_type == ActorType.HOUSEHOLD:
                actor_states[a.id] = {"income": 1.0, "savings": 0.5, "consumption": 0.7,
                    "employed": 1.0, "confidence": 0.5, "debt": 0.0}
            elif a.actor_type == ActorType.FIRM:
                actor_states[a.id] = {"revenue": 2.0, "costs": 1.2, "price_level": 1.0,
                    "employee_count": 3.0, "investment": 0.1, "debt": 0.5, "is_bankrupt": 0.0,
                    "demand_pressure": 0.0}
            elif a.actor_type == ActorType.BANK:
                actor_states[a.id] = {"credit_tightness": 0.2, "npl_ratio": 0.02,
                    "reserves": 1.0, "loans_outstanding": 1.0, "is_stressed": 0.0}
            elif a.actor_type == ActorType.CENTRAL_BANK:
                actor_states[a.id] = {"policy_rate": float(a.params.neutral_rate),
                    "inflation_target": 0.02, "observed_inflation": 0.02}

        # Initialize memories
        memories = {a.id: AgentMemory(actor_id=a.id) for a in world.actors}

        # Apply initial shock to memories
        severity = shock_config.severity
        for aid in (shock_config.initial_contact_actors or []):
            if aid in memories:
                memories[aid].add_event(
                    f"SHOCK: {shock_config.shock_type} with severity {severity:.0%} "
                    f"has hit. This directly affects you."
                )

        return AgentSimState(
            config=cfg, world=world, shock_config=shock_config,
            personas=persona_map, memories=memories,
            actor_states=actor_states, status="ready",
        )

    async def run_round(self, state: AgentSimState) -> RoundResult:
        """Execute one round of the agent simulation."""
        round_num = state.current_round
        all_actions: list[AgentAction] = []
        all_events: list[CausalEvent] = []

        # Build environment summary visible to all actors
        env_summary = self._build_environment(state)

        # Process injected events for this round
        for inj in state.injected_events:
            if inj.get("round") == round_num:
                for a in state.world.actors:
                    state.memories[a.id].add_event(f"BREAKING: {inj['description']}")

        # Query actors in parallel batches, ordered by type (CB first, then banks, firms, households)
        type_order = [ActorType.CENTRAL_BANK, ActorType.BANK, ActorType.FIRM, ActorType.HOUSEHOLD]
        for atype in type_order:
            actors_of_type = [a for a in state.world.actors if a.actor_type == atype]
            if not actors_of_type:
                continue

            # Process in batches
            batch_size = state.config.concurrent_agents
            for i in range(0, len(actors_of_type), batch_size):
                batch = actors_of_type[i:i + batch_size]
                tasks = [self._query_agent(a, state, env_summary) for a in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for actor, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.warning("Agent %s failed: %s", actor.id, result)
                        action = AgentAction(actor_id=actor.id, action_type="hold",
                                             reasoning="(agent error, holding)")
                    else:
                        action = result
                    all_actions.append(action)

                    # Apply action to state and generate causal events
                    events = self._apply_action(action, actor, state)
                    all_events.extend(events)

            # After each type processes, broadcast results to other actors' memories
            self._broadcast_actions(all_actions, state, round_num)

        # Compute metrics from current state
        metrics = self._compute_metrics(state, round_num)

        # Capture actor states snapshot
        actor_snapshot = {aid: dict(s) for aid, s in state.actor_states.items()}

        result = RoundResult(
            round_num=round_num,
            actions=all_actions,
            events=all_events,
            metrics=metrics,
            actor_states=actor_snapshot,
        )

        state.rounds.append(result)
        state.current_round += 1

        if state.current_round >= state.config.total_rounds:
            state.status = "completed"

        return result

    async def run_all(self, state: AgentSimState) -> AsyncIterator[RoundResult]:
        """Run all rounds, yielding each result for streaming."""
        state.status = "running"
        while state.current_round < state.config.total_rounds and state.status == "running":
            result = await self.run_round(state)
            yield result

    def inject_event(self, state: AgentSimState, description: str, round_num: int | None = None) -> None:
        """Inject a mid-simulation event (God's Eye live injection)."""
        target_round = round_num if round_num is not None else state.current_round
        state.injected_events.append({
            "round": target_round,
            "description": description,
            "injected_at": state.current_round,
        })

    # ------------------------------------------------------------------ internals

    async def _query_agent(self, actor: Actor, state: AgentSimState, env: str) -> AgentAction:
        """Ask one actor's LLM agent what to do this round."""
        persona = state.personas.get(actor.id, {})
        memory = state.memories.get(actor.id, AgentMemory(actor_id=actor.id))
        actor_state = state.actor_states.get(actor.id, {})

        template = _ACTOR_PROMPTS.get(actor.actor_type, "")
        if not template:
            return AgentAction(actor_id=actor.id, action_type="hold", reasoning="no template")

        # Build the prompt
        prompt = template.format(
            name=persona.get("display_name", actor.id),
            role=persona.get("role", actor.actor_type.value),
            description=persona.get("description", ""),
            memory=memory.summary(),
            environment=env,
            **{k: actor_state.get(k, 0.0) for k in [
                "policy_rate", "inflation_target", "observed_inflation",
                "credit_tightness", "npl_ratio", "reserves", "is_stressed",
                "revenue", "costs", "price_level", "employee_count", "debt",
                "is_bankrupt", "demand_pressure",
                "income", "savings", "consumption", "employed", "confidence",
            ]},
        )

        messages = [LLMMessage(role="user", content=prompt)]
        try:
            resp = await self._llm.complete(
                messages, model=self._model,
                temperature=state.config.temperature,
                max_tokens=256, response_format="json",
            )
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            raw = json.loads(text)
            action = AgentAction(
                actor_id=actor.id,
                action_type=raw.get("action_type", "hold"),
                target_id=raw.get("target_id"),
                magnitude=float(raw.get("magnitude", 0.0)),
                reasoning=raw.get("reasoning", ""),
            )
            memory.add_decision(f"Round {state.current_round}: {action.action_type} (mag={action.magnitude:.3f}) — {action.reasoning}")
            return action
        except Exception as exc:
            logger.warning("Agent %s LLM error: %s", actor.id, exc)
            return AgentAction(actor_id=actor.id, action_type="hold", reasoning=f"LLM error: {exc}")

    def _build_environment(self, state: AgentSimState) -> str:
        """Build the environment summary visible to all actors."""
        lines = [f"Round {state.current_round} of {state.config.total_rounds} ({state.config.round_duration_label} per round)"]
        lines.append(f"Shock: {state.shock_config.shock_type}, severity={state.shock_config.severity:.0%}")

        # Aggregate metrics
        if state.rounds:
            last = state.rounds[-1]
            if last.metrics:
                m = last.metrics
                lines.append(f"Economy: GDP={m.gdp_index:.3f}, inflation={m.inflation_rate:.3f}, "
                             f"unemployment={m.unemployment_rate:.3f}, bankruptcies={m.firm_bankruptcy_count}")

        # Recent notable actions
        if state.rounds:
            recent = state.rounds[-1].actions
            notable = [a for a in recent if a.action_type != "hold"][:5]
            if notable:
                lines.append("Recent actions by other actors:")
                for a in notable:
                    lines.append(f"  - {a.actor_id}: {a.action_type} (mag={a.magnitude:.3f})")

        return "\n".join(lines)

    def _apply_action(self, action: AgentAction, actor: Actor, state: AgentSimState) -> list[CausalEvent]:
        """Apply an agent's action to the simulation state. Returns causal events."""
        events: list[CausalEvent] = []
        s = state.actor_states.get(actor.id, {})
        rnd = state.current_round

        if action.action_type == "set_rate":
            old_rate = s.get("policy_rate", 0.025)
            s["policy_rate"] = max(0.0, min(0.20, action.magnitude))
            if abs(s["policy_rate"] - old_rate) > 0.001:
                for b in state.world.actors:
                    if b.actor_type == ActorType.BANK:
                        events.append(CausalEvent(step=rnd, source_actor_id=actor.id,
                            target_actor_id=b.id, channel="monetary_policy",
                            variable_affected="policy_rate",
                            magnitude=s["policy_rate"] - old_rate,
                            description=action.reasoning))
                        state.memories[b.id].add_event(
                            f"Central bank changed rate to {s['policy_rate']:.2%}: {action.reasoning}")

        elif action.action_type == "tighten_credit":
            old = s.get("credit_tightness", 0.2)
            s["credit_tightness"] = min(1.0, old + action.magnitude)
            for fid in self._linked_firms(actor, state):
                events.append(CausalEvent(step=rnd, source_actor_id=actor.id,
                    target_actor_id=fid, channel="lending",
                    variable_affected="credit_tightness",
                    magnitude=action.magnitude, description=action.reasoning))
                state.memories[fid].add_event(
                    f"Bank {actor.id} tightened credit by {action.magnitude:.1%}: {action.reasoning}")

        elif action.action_type == "ease_credit":
            old = s.get("credit_tightness", 0.2)
            s["credit_tightness"] = max(0.0, old - action.magnitude)

        elif action.action_type == "raise_prices":
            old = s.get("price_level", 1.0)
            s["price_level"] = old * (1.0 + action.magnitude)
            for hid in self._linked_households(actor, state):
                events.append(CausalEvent(step=rnd, source_actor_id=actor.id,
                    target_actor_id=hid, channel="trade",
                    variable_affected="price_level",
                    magnitude=action.magnitude, description=action.reasoning))
                state.memories[hid].add_event(
                    f"Firm {actor.id} raised prices by {action.magnitude:.1%}")

        elif action.action_type == "fire":
            count = max(0, min(int(action.magnitude), int(s.get("employee_count", 0))))
            s["employee_count"] = max(0, s.get("employee_count", 0) - count)
            fired = 0
            for hid in self._linked_households(actor, state):
                if fired >= count:
                    break
                hs = state.actor_states.get(hid, {})
                if hs.get("employed", 0) >= 1.0:
                    hs["employed"] = 0.0
                    hs["income"] = max(0.0, hs.get("income", 1.0) - 0.5)
                    fired += 1
                    events.append(CausalEvent(step=rnd, source_actor_id=actor.id,
                        target_actor_id=hid, channel="employment",
                        variable_affected="employed", magnitude=-1.0,
                        description=action.reasoning))
                    state.memories[hid].add_event(f"FIRED by {actor.id}: {action.reasoning}")

        elif action.action_type == "hire":
            s["employee_count"] = s.get("employee_count", 0) + action.magnitude

        elif action.action_type == "spend_more":
            s["consumption"] = s.get("consumption", 0.7) * (1.0 + action.magnitude)

        elif action.action_type == "save_more":
            s["savings"] = s.get("savings", 0.5) + action.magnitude * s.get("income", 1.0)
            s["consumption"] = max(0.0, s.get("consumption", 0.7) * (1.0 - action.magnitude))

        elif action.action_type == "cut_costs":
            s["costs"] = max(0.0, s.get("costs", 1.0) * (1.0 - action.magnitude))

        elif action.action_type == "invest":
            s["investment"] = max(0.0, action.magnitude)

        state.actor_states[actor.id] = s
        return events

    def _broadcast_actions(self, actions: list[AgentAction], state: AgentSimState, rnd: int) -> None:
        """Add notable actions to all actors' observation memories."""
        notable = [a for a in actions if a.action_type != "hold"]
        if not notable:
            return
        summary_parts = []
        for a in notable[:8]:
            p = state.personas.get(a.actor_id, {})
            name = p.get("display_name", a.actor_id)
            summary_parts.append(f"{name} decided to {a.action_type} (mag={a.magnitude:.3f})")
        summary = "Round " + str(rnd) + " actions: " + "; ".join(summary_parts)
        for actor in state.world.actors:
            state.memories[actor.id].add_observation(summary)

    def _linked_firms(self, bank: Actor, state: AgentSimState) -> list[str]:
        """Get firm IDs linked to a bank."""
        return [a.id for a in state.world.actors if a.actor_type == ActorType.FIRM]

    def _linked_households(self, firm: Actor, state: AgentSimState) -> list[str]:
        """Get household IDs linked to a firm via supply chain."""
        hids = []
        for src, tgt, _w in state.world.networks.supply_chain.edges:
            if src == firm.id:
                a = next((x for x in state.world.actors if x.id == tgt and x.actor_type == ActorType.HOUSEHOLD), None)
                if a:
                    hids.append(a.id)
        return hids

    def _compute_metrics(self, state: AgentSimState, step: int) -> StepMetrics:
        """Compute aggregate metrics from current actor states."""
        firms = [(aid, s) for aid, s in state.actor_states.items() if aid.startswith("firm_")]
        banks = [(aid, s) for aid, s in state.actor_states.items() if aid.startswith("bank_")]
        households = [(aid, s) for aid, s in state.actor_states.items() if aid.startswith("household_")]

        total_rev = sum(s.get("revenue", 1.0) for _, s in firms) or 1.0
        gdp = total_rev / max(1.0, len(firms) * 2.0)

        inflation = 0.0
        if firms:
            avg_price = sum(s.get("price_level", 1.0) for _, s in firms) / len(firms)
            inflation = avg_price - 1.0

        unemployed = sum(1 for _, s in households if s.get("employed", 1.0) < 1.0)
        unemp_rate = unemployed / max(1, len(households))

        avg_tight = sum(s.get("credit_tightness", 0.2) for _, s in banks) / max(1, len(banks))
        bankruptcies = sum(1 for _, s in firms if s.get("is_bankrupt", 0) >= 1.0)
        bank_stress = sum(1 for _, s in banks if s.get("is_stressed", 0) >= 1.0) / max(1, len(banks))
        confidence = sum(s.get("confidence", 0.5) for _, s in households) / max(1, len(households))

        return StepMetrics(
            step=step, gdp_index=max(0.0, gdp), inflation_rate=inflation,
            unemployment_rate=unemp_rate, gini_coefficient=0.0,
            credit_tightening_index=avg_tight,
            firm_bankruptcy_count=bankruptcies,
            bank_stress_index=bank_stress,
            consumer_confidence=confidence,
            interbank_freeze=bank_stress >= 0.5,
        )
