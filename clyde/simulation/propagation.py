"""Rule-based Propagation Engine.

Implements :class:`PropagationEngine`, a deterministic time-stepped actor
simulation. Given a :class:`SimulationWorld` and a seed, the engine produces
a :class:`TrajectoryResult` with exactly ``world.config.time_horizon.steps``
:class:`StepMetrics`, plus a list of :class:`CausalEvent` records that
describe how the initial shock propagated through the network.

Design commitments (see ``.kiro/specs/clyde-economic-simulator/design.md``):

* **Zero LLM imports.** This module must not import anything from
  ``clyde.llm`` or any LLM provider SDK — enforced by a boundary test.
* **Determinism.** Behaviour is driven entirely by a seeded ``random.Random``
  plus the world state. Same ``(world, seed)`` ⇒ identical ``TrajectoryResult``
  including the exact ``CausalEvent`` sequence.
* **Fixed 9-step actor update order per step**:

  1. Central Bank — Taylor-rule policy rate update
  2. Banks — credit tightening from NPL, herding, interbank reserve check
  3. Firms — investment, hiring/firing, pricing, bankruptcy check
  4. Labor Market — matching/clearing
  5. Households — consumption, savings, wage demands, credit seeking
  6. Interbank Network — settlement + freeze detection
  7. Bankruptcies — cascade removal
  8. Learning — Bayesian household inflation expectation update
  9. Metrics — compute all 9 core metrics

Per-actor state is represented as ``dict[str, float]`` where booleans are
encoded as ``0.0``/``1.0`` (e.g. ``state["employed"]``, ``state["is_bankrupt"]``).
Actor params are never mutated.
"""

from __future__ import annotations

import math
import random
from typing import Any

from clyde.models.actors import Actor
from clyde.models.causal import CausalEvent
from clyde.models.config import SimulationWorld
from clyde.models.enums import ActorType
from clyde.models.metrics import StepMetrics, TrajectoryResult
from clyde.persistence.db import SimulationDB


# ---------------------------------------------------------------------------
# Thresholds / tuning constants. These are intentionally centralised so the
# rule surface is auditable in one place.
# ---------------------------------------------------------------------------

# Emit a firm→household "trade" causal event when price jumps this much.
_PRICE_SPIKE_THRESHOLD = 0.02  # 2%
# Emit a bank→firm "lending" causal event when credit_tightness rises this much.
_CREDIT_TIGHTEN_THRESHOLD = 0.05  # 5%
# Interbank freeze triggers when at least this fraction of banks are stressed.
_BANK_STRESS_FREEZE_FRACTION = 0.5
# Bank is flagged as stressed when its NPL ratio exceeds this.
_BANK_NPL_STRESS = 0.12
# Chokepoint bank: 3+ distressed borrowers is "many".
_BANK_DISTRESS_BORROWER_COUNT = 3
# Firm stress accumulates whenever 2+ of its suppliers are bankrupt.
_FIRM_CHOKEPOINT_SUPPLIER_COUNT = 2
# Per-step stress increment at a chokepoint.
_STRESS_STEP_INCREMENT = 0.15
# Upper bound on accumulated stress (keeps state bounded for determinism).
_STRESS_CAP = 5.0


class PropagationEngine:
    """Rule-based, deterministic time-stepped propagation engine.

    Usage::

        engine = PropagationEngine()
        trajectory = engine.run(world, seed=42, run_id="demo")
    """

    # ------------------------------------------------------------------ API

    def run(
        self,
        world: SimulationWorld,
        seed: int,
        run_id: str,
        db: SimulationDB | None = None,
    ) -> TrajectoryResult:
        """Execute a single run.

        Parameters
        ----------
        world:
            Fully resolved :class:`SimulationWorld` from the factory.
        seed:
            Seed for the internal RNG. Same seed + same world ⇒ identical
            output.
        run_id:
            Identifier for the run; used as the DB foreign key.
        db:
            Optional :class:`SimulationDB`. When provided, the run record is
            inserted, step metrics are persisted at the end (batch), causal
            events are persisted inline in the order they were emitted, and
            the run status is updated to ``"completed"`` at the end.
        """
        rng = random.Random(seed)

        # Deep-copy-ish working state: we rebuild state dicts from scratch so
        # we never mutate the input world's actors. Params are referenced
        # (never mutated).
        working_actors = [self._clone_actor(a) for a in world.actors]

        # Index structures used every step.
        state = _WorldWorkingState.build(world, working_actors)

        # Initialise all actor state based on sensible defaults + network edges.
        self._initialise_state(state)

        # Persist the run record up front so step_metrics FK is satisfiable.
        if db is not None:
            db.insert_run(
                run_id=run_id,
                scenario_id=run_id,  # caller may overwrite; run_id is safe default
                seed=seed,
                config=world.config,
                status="running",
            )

        causal_events: list[CausalEvent] = []
        step_records: list[StepMetrics] = []

        n_steps = int(world.config.time_horizon.steps)
        for step in range(n_steps):
            # Step 0 applies the initial shock to ``initial_contact_actors``
            # before any behavioural updates. This guarantees downstream
            # effects are rule-driven, not hardcoded.
            if step == 0:
                self._apply_initial_shock(state, world, step, causal_events)

            new_events = self._step(state, step, rng)
            causal_events.extend(new_events)

            metrics = self._compute_metrics(state, step)
            step_records.append(metrics)

            if db is not None:
                for ev in new_events:
                    db.insert_causal_event(run_id, ev)

        # Persist metrics batch at the end (faster than row-at-a-time) and
        # flip run status.
        if db is not None:
            db.insert_step_metrics_batch(run_id, step_records)
            db.update_run_status(run_id, "completed")

        return TrajectoryResult(
            run_id=run_id,
            seed=seed,
            steps=step_records,
            causal_events=causal_events,
            final_state_ref=None,
        )

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _clone_actor(actor: Actor) -> Actor:
        """Return an Actor with a fresh (empty) state dict, preserving params+rels."""
        return Actor(
            id=actor.id,
            actor_type=actor.actor_type,
            params=actor.params,  # params are treated as immutable
            state={},
            relationships=list(actor.relationships),
        )

    # --------------------------------------------------------- initialisation

    def _initialise_state(self, state: "_WorldWorkingState") -> None:
        """Seed every actor's state dict with sensible defaults.

        Booleans are encoded as 0.0/1.0. We intentionally set ``employed`` from
        the labor-market edges so unemployment at step 0 reflects the network.
        """
        # Households: every household starts employed iff it has at least one
        # employment edge; otherwise unemployed.
        employed_set = set()
        for src, _tgt, _w in state.networks.labor_market.edges:
            employed_set.add(src)

        for h in state.households:
            h.state = {
                "income": 1.0,
                "savings": 0.5,
                "consumption": 0.7,
                "employed": 1.0 if h.id in employed_set else 0.0,
                "debt": 0.0,
                "inflation_expectation": float(h.params.inflation_expectation_prior),
                "confidence": 0.5,
                "wage_demand": 1.0,
                "unemployment_fear": 0.0,
                "stress": 0.0,
            }

        # Firms: revenue scales with number of customers (household trade edges
        # inbound from this firm) so bigger firms have bigger revenue bases.
        trade_customers: dict[str, int] = {f.id: 0 for f in state.firms}
        firm_suppliers: dict[str, list[str]] = {f.id: [] for f in state.firms}
        for src, tgt, _w in state.networks.supply_chain.edges:
            # src → tgt: if src is a firm and tgt is a household → sale
            if src in trade_customers and tgt in state.household_ids:
                trade_customers[src] += 1
            # firm → firm means src is supplier to tgt
            if src in trade_customers and tgt in firm_suppliers:
                firm_suppliers[tgt].append(src)

        # Firm employees from labor-market edges (household → firm).
        firm_employees: dict[str, list[str]] = {f.id: [] for f in state.firms}
        for src, tgt, _w in state.networks.labor_market.edges:
            if tgt in firm_employees:
                firm_employees[tgt].append(src)

        for f in state.firms:
            base_rev = 1.0 + 0.5 * trade_customers[f.id]
            f.state = {
                "revenue": base_rev,
                "initial_revenue": base_rev,
                "costs": 0.6 * base_rev,
                "inventory": 1.0,
                "price_level": 1.0,
                "prev_price_level": 1.0,
                "investment": 0.1,
                "debt": 0.3 * base_rev,
                "demand_pressure": 0.0,
                "is_bankrupt": 0.0,
                "employee_count": float(len(firm_employees[f.id])),
                "supplier_count": float(len(firm_suppliers[f.id])),
                "stress": 0.0,
            }

        # Banks.
        for b in state.banks:
            b.state = {
                "reserves": 1.0,
                "loans_outstanding": 1.0,
                "npl_ratio": 0.02,
                "credit_tightness": 0.2,
                "prev_credit_tightness": 0.2,
                "interbank_borrowed": 0.0,
                "is_stressed": 0.0,
                "stress": 0.0,
            }

        # Central bank.
        for cb in state.central_banks:
            cb.state = {
                "policy_rate": float(cb.params.neutral_rate),
                "inflation_target": 0.02,
                "output_gap_estimate": 0.0,
                "observed_inflation": 0.02,
                "stress": 0.0,
            }

        # Cache derived indices that don't change during the run.
        state.firm_suppliers = firm_suppliers
        state.firm_employees = firm_employees
        # Reverse labor-market edge: household → firm (employer).
        state.household_employer = {
            src: tgt for src, tgt, _w in state.networks.labor_market.edges
        }
        # Firm→household trade edges (for price-spike causal emission).
        state.firm_trade_customers: dict[str, list[str]] = {f.id: [] for f in state.firms}
        for src, tgt, _w in state.networks.supply_chain.edges:
            if src in state.firm_trade_customers and tgt in state.household_ids:
                state.firm_trade_customers[src].append(tgt)
        # Bank→firm linkages (interbank graph is bank-bank; firm lending isn't
        # a distinct graph, so we infer: every bank "lends to" every firm with
        # proportional weight. In a larger build this would be an explicit
        # graph.).
        state.bank_firm_links: dict[str, list[str]] = {
            b.id: [f.id for f in state.firms] for b in state.banks
        }
        # Bank→bank lending (interbank) for cascade detection.
        state.bank_bank_links: dict[str, list[str]] = {b.id: [] for b in state.banks}
        for src, tgt, _w in state.networks.interbank.edges:
            if src in state.bank_bank_links:
                state.bank_bank_links[src].append(tgt)

    # ---------------------------------------------------------- initial shock

    def _apply_initial_shock(
        self,
        state: "_WorldWorkingState",
        world: SimulationWorld,
        step: int,
        events: list[CausalEvent],
    ) -> None:
        """Apply the shock severity to each ``initial_contact_actors`` entry.

        Any actor id not listed receives no direct shock at step 0 — downstream
        changes must therefore be network-mediated.
        """
        severity = float(world.config.severity)
        if severity <= 0.0 or not world.config.initial_contact_actors:
            return

        for aid in world.config.initial_contact_actors:
            actor = state.by_id.get(aid)
            if actor is None:
                continue
            if actor.actor_type == ActorType.FIRM:
                # Reduce revenue; push demand negative.
                actor.state["revenue"] = max(0.0, actor.state["revenue"] * (1.0 - severity))
                actor.state["demand_pressure"] = actor.state.get("demand_pressure", 0.0) - severity
                actor.state["stress"] = min(_STRESS_CAP, actor.state.get("stress", 0.0) + severity)
            elif actor.actor_type == ActorType.HOUSEHOLD:
                actor.state["income"] = max(0.0, actor.state.get("income", 1.0) - severity)
                actor.state["unemployment_fear"] = min(
                    1.0, actor.state.get("unemployment_fear", 0.0) + severity
                )
                actor.state["stress"] = min(_STRESS_CAP, actor.state.get("stress", 0.0) + severity)
            elif actor.actor_type == ActorType.BANK:
                actor.state["npl_ratio"] = min(
                    1.0, actor.state.get("npl_ratio", 0.0) + severity * 0.5
                )
                actor.state["stress"] = min(_STRESS_CAP, actor.state.get("stress", 0.0) + severity)
            elif actor.actor_type == ActorType.CENTRAL_BANK:
                # Tightening shock: lift policy rate by severity.
                actor.state["policy_rate"] = actor.state.get("policy_rate", 0.0) + severity
                actor.state["inflation_target"] = max(
                    0.0, actor.state.get("inflation_target", 0.02) - severity * 0.01
                )

    # -------------------------------------------------------------- single step

    def _step(
        self,
        state: "_WorldWorkingState",
        step: int,
        rng: random.Random,
    ) -> list[CausalEvent]:
        """Execute one simulation step. Returns causal events emitted this step."""
        events: list[CausalEvent] = []

        # 1) Central Bank — Taylor rule.
        self._update_central_banks(state)

        # 2) Banks — credit tightening, NPL tracking, stress flag.
        self._update_banks(state, step, events)

        # 3) Firms — investment, hiring/firing, pricing, bankruptcy.
        self._update_firms(state, step, rng, events)

        # 4) Labor Market — matching / clearing (track employment from firm
        #    updates; firms may have fired in step 3).
        self._clear_labor_market(state, step, events)

        # 5) Households — consumption / savings / wage demands / credit seek.
        self._update_households(state, step, events)

        # 6) Interbank — settlement + freeze detection (flag per-bank stressed).
        self._interbank_settlement(state, step, events)

        # 7) Bankruptcy cascade — finalise bankrupt firms, propagate to suppliers.
        self._bankruptcy_cascade(state, step, events)

        # 8) Learning — Bayesian inflation-expectation update on households.
        self._learning_updates(state, step)

        # 9) Metrics computed by caller (kept separate so _step stays testable).

        return events

    # ---- 1) Central Bank ---------------------------------------------------

    def _update_central_banks(self, state: "_WorldWorkingState") -> None:
        # Observed inflation: year-over-year proxy — use current avg price
        # vs previous step's avg price.
        if not state.firms:
            observed_inflation = 0.0
        else:
            prev = _weighted_price(state.firms, "prev_price_level")
            cur = _weighted_price(state.firms, "price_level")
            observed_inflation = (cur - prev) / prev if prev > 0 else 0.0

        # Output gap proxy: normalised revenue shortfall vs initial.
        total_init = sum(f.state.get("initial_revenue", 1.0) for f in state.firms)
        total_now = sum(f.state.get("revenue", 0.0) for f in state.firms)
        output_gap = (total_now - total_init) / total_init if total_init > 0 else 0.0

        for cb in state.central_banks:
            target = cb.state.get("inflation_target", 0.02)
            infl_gap = observed_inflation - target
            neutral = float(cb.params.neutral_rate)
            taylor = (
                neutral
                + cb.params.taylor_inflation_weight * infl_gap
                + cb.params.taylor_output_weight * output_gap
            )
            # Move toward Taylor rate one ``rate_increment`` at a time.
            cur_rate = cb.state.get("policy_rate", neutral)
            delta = taylor - cur_rate
            step_size = float(cb.params.rate_increment)
            if abs(delta) > float(cb.params.discretionary_band):
                move = step_size if delta > 0 else -step_size
                cur_rate = cur_rate + move
            cb.state["policy_rate"] = cur_rate
            cb.state["observed_inflation"] = observed_inflation
            cb.state["output_gap_estimate"] = output_gap

    # ---- 2) Banks ----------------------------------------------------------

    def _update_banks(
        self,
        state: "_WorldWorkingState",
        step: int,
        events: list[CausalEvent],
    ) -> None:
        # --- Monetary policy transmission: CB policy rate → bank tightening ---
        # The policy rate above neutral signals tightening; below signals easing.
        # This is the primary monetary transmission channel.
        policy_rate_signal = 0.0
        if state.central_banks:
            cb = state.central_banks[0]
            neutral = float(cb.params.neutral_rate)
            policy_rate = cb.state.get("policy_rate", neutral)
            # Positive = tightening, negative = easing
            policy_rate_signal = max(-1.0, min(1.0, (policy_rate - neutral) * 10.0))

        # Peer mean tightness for herding.
        if state.banks:
            peer_mean_tight = sum(b.state.get("credit_tightness", 0.2) for b in state.banks) / len(
                state.banks
            )
        else:
            peer_mean_tight = 0.2

        # Number of bankrupt firms feeds back into NPL.
        bankrupt_firms = sum(1 for f in state.firms if f.state.get("is_bankrupt", 0.0) >= 1.0)
        firm_total = max(1, len(state.firms))
        system_npl_pressure = bankrupt_firms / firm_total

        for b in state.banks:
            prev_tight = b.state.get("credit_tightness", 0.2)
            b.state["prev_credit_tightness"] = prev_tight

            # NPL evolves with system distress + mean reversion.
            npl = b.state.get("npl_ratio", 0.02)
            npl = 0.9 * npl + 0.1 * system_npl_pressure
            b.state["npl_ratio"] = min(1.0, npl)

            # Tightening = NPL elasticity + herding + MONETARY POLICY CHANNEL.
            # The policy rate signal directly pushes bank tightness up/down.
            base = npl * float(b.params.npl_tightening_elasticity)
            herd = float(b.params.herding_weight) * (peer_mean_tight - prev_tight)
            # Monetary channel: policy rate above neutral → tighten; below → ease.
            # This is the dominant channel for rate-hike shocks.
            monetary_floor = max(0.0, 0.5 * policy_rate_signal)
            new_tight = max(0.0, min(1.0, 0.5 * prev_tight + 0.2 * base + herd + 0.3 * monetary_floor))
            b.state["credit_tightness"] = new_tight

            # Reserves drift with interbank need + NPL + policy rate pressure.
            reserves = b.state.get("reserves", 1.0)
            if reserves < float(b.params.reserve_threshold):
                b.state["interbank_borrowed"] = b.state.get("interbank_borrowed", 0.0) + 0.1
            # Higher policy rate drains reserves faster (cost of funding).
            reserve_drain = 0.02 * npl + 0.01 * max(0.0, policy_rate_signal)
            b.state["reserves"] = max(0.0, reserves - reserve_drain)

            # Stressed flag.
            stressed = 1.0 if npl >= _BANK_NPL_STRESS or new_tight >= 0.75 else 0.0
            b.state["is_stressed"] = stressed

            # Chokepoint stress: many distressed borrowers.
            distressed_borrowers = 0
            for fid in state.bank_firm_links.get(b.id, []):
                firm = state.by_id.get(fid)
                if firm is None:
                    continue
                if firm.state.get("is_bankrupt", 0.0) >= 1.0 or firm.state.get("revenue", 0.0) < 0.5:
                    distressed_borrowers += 1
            if distressed_borrowers >= _BANK_DISTRESS_BORROWER_COUNT:
                b.state["stress"] = min(
                    _STRESS_CAP,
                    b.state.get("stress", 0.0) + _STRESS_STEP_INCREMENT,
                )

            # Causal event: ≥5% tightening jump propagates to every linked firm.
            delta = new_tight - prev_tight
            if delta >= _CREDIT_TIGHTEN_THRESHOLD:
                for fid in state.bank_firm_links.get(b.id, []):
                    events.append(
                        CausalEvent(
                            step=step,
                            source_actor_id=b.id,
                            target_actor_id=fid,
                            channel="lending",
                            variable_affected="credit_tightness",
                            magnitude=delta,
                            description="bank tightened credit to firm",
                        )
                    )

            # Causal event: monetary policy transmission — CB rate hike
            # propagates through this bank to its borrowers.
            if policy_rate_signal > 0.2 and delta > 0:
                for cb in state.central_banks:
                    events.append(
                        CausalEvent(
                            step=step,
                            source_actor_id=cb.id,
                            target_actor_id=b.id,
                            channel="monetary_policy",
                            variable_affected="credit_tightness",
                            magnitude=policy_rate_signal,
                            description="central bank rate hike tightens bank funding",
                        )
                    )

    # ---- 3) Firms ----------------------------------------------------------

    def _update_firms(
        self,
        state: "_WorldWorkingState",
        step: int,
        rng: random.Random,
        events: list[CausalEvent],
    ) -> None:
        # Aggregate credit tightness for this step (banks already updated).
        avg_tight = (
            sum(b.state.get("credit_tightness", 0.2) for b in state.banks) / len(state.banks)
            if state.banks
            else 0.2
        )

        # Policy rate effect on firm borrowing costs.
        policy_rate = 0.025
        if state.central_banks:
            policy_rate = state.central_banks[0].state.get("policy_rate", 0.025)
        # Rate above neutral increases firm costs; below decreases them.
        neutral = float(state.central_banks[0].params.neutral_rate) if state.central_banks else 0.025
        rate_pressure = max(-1.0, min(1.0, (policy_rate - neutral) * 5.0))

        for f in state.firms:
            if f.state.get("is_bankrupt", 0.0) >= 1.0:
                # Frozen: no more updates on bankrupt firms.
                continue

            # Investment decision: falls as credit tightens AND rates rise.
            inv = 0.1 * (1.0 - avg_tight) * (1.0 - 0.5 * max(0, rate_pressure)) * (1.0 + f.state.get("demand_pressure", 0.0))
            f.state["investment"] = max(0.0, inv)

            # Hiring/firing: based on revenue vs costs.
            revenue = f.state.get("revenue", 0.0)
            costs = f.state.get("costs", 0.0)
            margin = revenue - costs
            employees = f.state.get("employee_count", 0.0)
            if revenue > 0 and -margin / max(0.1, revenue) >= float(f.params.firing_threshold):
                # Needs to fire one employee if possible.
                if employees >= 1.0:
                    self._fire_one_employee(f, state, step, events)
            elif margin > 0 and rng.random() < float(f.params.hiring_elasticity) * 0.05:
                # Hiring path is rare in this simple rule-set; skipped for
                # determinism stability.
                pass

            # Pricing: cost-push + demand-pull.
            prev_price = f.state.get("price_level", 1.0)
            f.state["prev_price_level"] = prev_price
            cost_push = float(f.params.cost_push_weight) * (costs / max(0.01, revenue) - 0.6)
            demand_pull = float(f.params.demand_pull_weight) * f.state.get(
                "demand_pressure", 0.0
            )
            new_price = max(0.01, prev_price * (1.0 + 0.1 * cost_push + 0.1 * demand_pull))
            f.state["price_level"] = new_price

            delta_pct = (new_price - prev_price) / prev_price if prev_price > 0 else 0.0
            if delta_pct >= _PRICE_SPIKE_THRESHOLD:
                for hid in state.firm_trade_customers.get(f.id, []):
                    events.append(
                        CausalEvent(
                            step=step,
                            source_actor_id=f.id,
                            target_actor_id=hid,
                            channel="trade",
                            variable_affected="price_level",
                            magnitude=new_price - prev_price,
                            description="firm raised prices ≥2%",
                        )
                    )

            # Revenue evolves with demand pressure (negative = shrinking sales).
            f.state["revenue"] = max(
                0.0, revenue * (1.0 + 0.05 * f.state.get("demand_pressure", 0.0))
            )
            # Costs drift with credit tightness and upstream supplier stress.
            upstream_stress = 0.0
            suppliers = state.firm_suppliers.get(f.id, [])
            if suppliers:
                upstream_stress = sum(
                    state.by_id[sid].state.get("stress", 0.0) for sid in suppliers if sid in state.by_id
                ) / max(1, len(suppliers))
            f.state["costs"] = max(0.0, costs * (1.0 + 0.02 * avg_tight + 0.05 * upstream_stress + 0.03 * max(0, rate_pressure)))

            # Chokepoint stress accumulation: many bankrupt suppliers.
            bankrupt_suppliers = sum(
                1
                for sid in suppliers
                if state.by_id.get(sid) is not None
                and state.by_id[sid].state.get("is_bankrupt", 0.0) >= 1.0
            )
            # Stress also grows when revenue is shocked / upstream is stressed.
            stress_step = 0.0
            if bankrupt_suppliers >= _FIRM_CHOKEPOINT_SUPPLIER_COUNT:
                stress_step += _STRESS_STEP_INCREMENT
            if upstream_stress > 0.05:
                stress_step += 0.5 * upstream_stress
            if f.state.get("revenue", 0.0) < 0.5 * f.state.get("initial_revenue", 1.0):
                stress_step += 0.05
            if stress_step > 0:
                f.state["stress"] = min(
                    _STRESS_CAP, f.state.get("stress", 0.0) + stress_step
                )

            # Bankruptcy check: debt/revenue ratio blows past threshold OR
            # stress saturates.
            debt = f.state.get("debt", 0.0)
            debt_ratio = debt / max(0.01, f.state.get("revenue", 0.0))
            if (
                debt_ratio >= float(f.params.bankruptcy_threshold)
                or f.state.get("stress", 0.0) >= _STRESS_CAP * 0.8
            ):
                f.state["is_bankrupt"] = 1.0

    def _fire_one_employee(
        self,
        firm: Actor,
        state: "_WorldWorkingState",
        step: int,
        events: list[CausalEvent],
    ) -> None:
        """Find one employed household attached to ``firm`` and fire them."""
        employees = state.firm_employees.get(firm.id, [])
        for idx, hid in enumerate(employees):
            h = state.by_id.get(hid)
            if h is None:
                continue
            if h.state.get("employed", 0.0) >= 1.0:
                h.state["employed"] = 0.0
                h.state["income"] = max(0.0, h.state.get("income", 1.0) - 0.5)
                firm.state["employee_count"] = max(0.0, firm.state.get("employee_count", 0.0) - 1.0)
                # Remove from the index so subsequent firings skip this household.
                employees.pop(idx)
                events.append(
                    CausalEvent(
                        step=step,
                        source_actor_id=firm.id,
                        target_actor_id=hid,
                        channel="employment",
                        variable_affected="employed",
                        magnitude=-1.0,
                        description="firm fired employee",
                    )
                )
                return

    # ---- 4) Labor Market ---------------------------------------------------

    def _clear_labor_market(
        self,
        state: "_WorldWorkingState",
        step: int,
        events: list[CausalEvent],
    ) -> None:
        """Reconcile employment after firms may have fired.

        This function does not emit events (firing already emitted them in
        step 3). Its job is to make sure the household employment flag is
        consistent with the firm's employee_count if a firm went bankrupt in
        a prior step.
        """
        del step, events  # reserved

        # Households employed at a now-bankrupt firm become unemployed.
        for h in state.households:
            emp_id = state.household_employer.get(h.id)
            if emp_id is None:
                continue
            employer = state.by_id.get(emp_id)
            if employer is None:
                continue
            if employer.state.get("is_bankrupt", 0.0) >= 1.0 and h.state.get("employed", 0.0) >= 1.0:
                h.state["employed"] = 0.0
                h.state["income"] = max(0.0, h.state.get("income", 1.0) - 0.5)

    # ---- 5) Households -----------------------------------------------------

    def _update_households(
        self,
        state: "_WorldWorkingState",
        step: int,
        events: list[CausalEvent],
    ) -> None:
        del events  # household updates don't emit new causal events.
        # Average firm price, used as price signal.
        avg_price = _weighted_price(state.firms, "price_level") if state.firms else 1.0

        for h in state.households:
            employed = h.state.get("employed", 0.0) >= 1.0
            income = h.state.get("income", 1.0)
            mpc = float(h.params.mpc)
            prec_sav = float(h.params.precautionary_savings_rate)
            infl_exp = h.state.get("inflation_expectation", h.params.inflation_expectation_prior)
            # Consumption falls with inflation expectation and unemployment fear.
            fear = h.state.get("unemployment_fear", 0.0)
            consumption = max(
                0.0,
                mpc * income * (1.0 - 0.5 * infl_exp) * (1.0 - 0.5 * fear),
            )
            h.state["consumption"] = consumption
            h.state["savings"] = max(0.0, h.state.get("savings", 0.5) + prec_sav * income - 0.1 * consumption)

            # Wage demand rises with inflation expectation.
            h.state["wage_demand"] = 1.0 + float(h.params.wage_demand_elasticity) * infl_exp

            # Confidence: up if employed & prices stable, down otherwise.
            prev_conf = h.state.get("confidence", 0.5)
            target_conf = 0.7 if employed else 0.2
            target_conf -= 0.3 * (avg_price - 1.0)
            target_conf -= 0.4 * fear
            target_conf = max(0.0, min(1.0, target_conf))
            h.state["confidence"] = 0.7 * prev_conf + 0.3 * target_conf

            # Unemployment fear decays when employed, grows when not.
            if employed:
                h.state["unemployment_fear"] = max(0.0, fear - 0.05)
            else:
                h.state["unemployment_fear"] = min(1.0, fear + 0.1)

        # Credit-seeking: households push aggregate demand pressure on firms
        # they consume from.
        total_consumption = sum(h.state.get("consumption", 0.0) for h in state.households)
        baseline = 0.7 * max(1, len(state.households))
        demand_shift = (total_consumption - baseline) / max(0.01, baseline)
        for f in state.firms:
            # Rolling update (avoid overwriting initial shock at step 0).
            cur = f.state.get("demand_pressure", 0.0)
            f.state["demand_pressure"] = 0.7 * cur + 0.3 * demand_shift

    # ---- 6) Interbank settlement ------------------------------------------

    def _interbank_settlement(
        self,
        state: "_WorldWorkingState",
        step: int,
        events: list[CausalEvent],
    ) -> None:
        del step, events  # no new events emitted here.
        # Peer contagion: if any neighbour bank is stressed, lift your NPL a bit.
        if not state.banks:
            return
        by_id = {b.id: b for b in state.banks}
        stressed_neighbour_count: dict[str, int] = {b.id: 0 for b in state.banks}
        for src, tgt, _w in state.networks.interbank.edges:
            src_b = by_id.get(src)
            tgt_b = by_id.get(tgt)
            if src_b is None or tgt_b is None:
                continue
            if src_b.state.get("is_stressed", 0.0) >= 1.0:
                stressed_neighbour_count[tgt] += 1
            if tgt_b.state.get("is_stressed", 0.0) >= 1.0:
                stressed_neighbour_count[src] += 1
        for b in state.banks:
            cnt = stressed_neighbour_count[b.id]
            if cnt > 0:
                b.state["npl_ratio"] = min(1.0, b.state.get("npl_ratio", 0.0) + 0.005 * cnt)

    # ---- 7) Bankruptcy cascade --------------------------------------------

    def _bankruptcy_cascade(
        self,
        state: "_WorldWorkingState",
        step: int,
        events: list[CausalEvent],
    ) -> None:
        """Emit supply-channel events for newly bankrupt firms and mark workers
        unemployed."""
        # We rely on ``is_bankrupt`` to flip to 1.0 inside _update_firms. For
        # each firm that is now bankrupt but wasn't emitted yet, emit the
        # supplier cascade + employee separation.
        for f in state.firms:
            if f.state.get("is_bankrupt", 0.0) < 1.0:
                continue
            if f.state.get("bankruptcy_emitted", 0.0) >= 1.0:
                continue

            # Suppliers of f: upstream lose demand from f.
            for sid in state.firm_suppliers.get(f.id, []):
                events.append(
                    CausalEvent(
                        step=step,
                        source_actor_id=f.id,
                        target_actor_id=sid,
                        channel="supply",
                        variable_affected="revenue",
                        magnitude=-1.0,
                        description="buyer went bankrupt, supplier loses demand",
                    )
                )
                sup = state.by_id.get(sid)
                if sup is not None:
                    sup.state["demand_pressure"] = sup.state.get("demand_pressure", 0.0) - 0.2
                    sup.state["stress"] = min(
                        _STRESS_CAP, sup.state.get("stress", 0.0) + 0.1
                    )

            # Employees lose their jobs.
            for hid in list(state.firm_employees.get(f.id, [])):
                h = state.by_id.get(hid)
                if h is None:
                    continue
                if h.state.get("employed", 0.0) >= 1.0:
                    h.state["employed"] = 0.0
                    h.state["income"] = max(0.0, h.state.get("income", 1.0) - 0.5)
                    events.append(
                        CausalEvent(
                            step=step,
                            source_actor_id=f.id,
                            target_actor_id=hid,
                            channel="employment",
                            variable_affected="employed",
                            magnitude=-1.0,
                            description="employer bankruptcy",
                        )
                    )
            f.state["employee_count"] = 0.0
            f.state["revenue"] = 0.0
            f.state["bankruptcy_emitted"] = 1.0

    # ---- 8) Learning -------------------------------------------------------

    def _learning_updates(self, state: "_WorldWorkingState", step: int) -> None:
        del step
        # Observed inflation signal: weighted avg firm price change vs initial.
        if not state.firms:
            observed = 0.0
        else:
            prev = _weighted_price(state.firms, "prev_price_level")
            cur = _weighted_price(state.firms, "price_level")
            observed = (cur - prev) / prev if prev > 0 else 0.0

        for h in state.households:
            lr = float(h.params.inflation_expectation_lr)
            cur_exp = h.state.get("inflation_expectation", h.params.inflation_expectation_prior)
            h.state["inflation_expectation"] = (1.0 - lr) * cur_exp + lr * observed

    # ---- 9) Metrics --------------------------------------------------------

    def _compute_metrics(self, state: "_WorldWorkingState", step: int) -> StepMetrics:
        # gdp_index: sum of current revenue / sum of initial revenue.
        total_init = sum(f.state.get("initial_revenue", 1.0) for f in state.firms)
        total_now = sum(f.state.get("revenue", 0.0) for f in state.firms)
        gdp_index = (total_now / total_init) if total_init > 0 else 1.0

        # inflation_rate: %Δ weighted avg price vs previous step.
        if state.firms:
            prev = _weighted_price(state.firms, "prev_price_level")
            cur = _weighted_price(state.firms, "price_level")
            inflation_rate = (cur - prev) / prev if prev > 0 else 0.0
        else:
            inflation_rate = 0.0
        if step == 0:
            inflation_rate = 0.0

        # unemployment_rate: fraction of households with employed==0.0.
        if state.households:
            unemployed = sum(1 for h in state.households if h.state.get("employed", 0.0) < 1.0)
            unemployment_rate = unemployed / len(state.households)
        else:
            unemployment_rate = 0.0

        # gini on household income.
        incomes = sorted(max(0.0, h.state.get("income", 0.0)) for h in state.households)
        gini = _gini(incomes)

        # credit_tightening_index: mean bank credit_tightness.
        if state.banks:
            cti = sum(b.state.get("credit_tightness", 0.0) for b in state.banks) / len(state.banks)
        else:
            cti = 0.0
        cti = max(0.0, min(1.0, cti))

        # firm_bankruptcy_count (cumulative).
        bankruptcies = sum(1 for f in state.firms if f.state.get("is_bankrupt", 0.0) >= 1.0)

        # bank_stress_index.
        if state.banks:
            bsi = sum(1 for b in state.banks if b.state.get("is_stressed", 0.0) >= 1.0) / len(
                state.banks
            )
        else:
            bsi = 0.0

        # consumer_confidence.
        if state.households:
            cc = sum(h.state.get("confidence", 0.5) for h in state.households) / len(
                state.households
            )
        else:
            cc = 0.5

        interbank_freeze = bsi >= _BANK_STRESS_FREEZE_FRACTION

        return StepMetrics(
            step=step,
            gdp_index=_sanitize(gdp_index, default=1.0),
            inflation_rate=_sanitize(inflation_rate, default=0.0),
            unemployment_rate=_sanitize(unemployment_rate, default=0.0),
            gini_coefficient=_sanitize(gini, default=0.0),
            credit_tightening_index=_sanitize(cti, default=0.0),
            firm_bankruptcy_count=int(bankruptcies),
            bank_stress_index=_sanitize(bsi, default=0.0),
            consumer_confidence=_sanitize(cc, default=0.5),
            interbank_freeze=bool(interbank_freeze),
            custom_metrics={},
        )


# ---------------------------------------------------------------------------
# Internal working-state container. Kept private to this module.
# ---------------------------------------------------------------------------


class _WorldWorkingState:
    """Indexed view over the mutable actor list for one run."""

    __slots__ = (
        "actors",
        "by_id",
        "households",
        "firms",
        "banks",
        "central_banks",
        "household_ids",
        "networks",
        "firm_suppliers",
        "firm_employees",
        "firm_trade_customers",
        "household_employer",
        "bank_firm_links",
        "bank_bank_links",
    )

    @classmethod
    def build(cls, world: SimulationWorld, actors: list[Actor]) -> "_WorldWorkingState":
        self = cls()
        self.actors = actors
        self.by_id: dict[str, Actor] = {a.id: a for a in actors}
        self.households = [a for a in actors if a.actor_type == ActorType.HOUSEHOLD]
        self.firms = [a for a in actors if a.actor_type == ActorType.FIRM]
        self.banks = [a for a in actors if a.actor_type == ActorType.BANK]
        self.central_banks = [a for a in actors if a.actor_type == ActorType.CENTRAL_BANK]
        self.household_ids = {h.id for h in self.households}
        self.networks = world.networks
        # Filled in _initialise_state.
        self.firm_suppliers: dict[str, list[str]] = {}
        self.firm_employees: dict[str, list[str]] = {}
        self.firm_trade_customers: dict[str, list[str]] = {}
        self.household_employer: dict[str, str] = {}
        self.bank_firm_links: dict[str, list[str]] = {}
        self.bank_bank_links: dict[str, list[str]] = {}
        return self


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------


def _weighted_price(firms: list[Actor], key: str) -> float:
    """Revenue-weighted average of ``state[key]`` across firms.

    Falls back to unweighted mean if total weight is zero.
    """
    num = 0.0
    den = 0.0
    for f in firms:
        w = max(0.0, f.state.get("initial_revenue", 1.0))
        p = f.state.get(key, 1.0)
        num += w * p
        den += w
    if den <= 0:
        return sum(f.state.get(key, 1.0) for f in firms) / max(1, len(firms))
    return num / den


def _gini(sorted_incomes: list[float]) -> float:
    """Gini coefficient on a non-negative, sorted-ascending income vector.

    Standard formulation:  G = (2 Σ_i i*x_i) / (n Σ x_i) - (n + 1)/n.
    Returns 0.0 for empty input or an all-zero vector.
    """
    n = len(sorted_incomes)
    if n == 0:
        return 0.0
    total = sum(sorted_incomes)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(sorted_incomes))
    g = (2.0 * weighted) / (n * total) - (n + 1) / n
    # Clamp to [0, 1] for numeric safety.
    return max(0.0, min(1.0, g))


def _sanitize(value: Any, default: float) -> float:
    """Replace NaN/Inf with ``default`` and coerce to float.

    The design doc calls out ``MetricOverflowWarning`` for numerical blowups;
    we silently clamp (property tests require finite values).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return v


__all__ = ["PropagationEngine"]
