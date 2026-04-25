"""EconomicWorldFactory: assembles a fully resolved SimulationWorld.

This module is deliberately LLM-free. All behavioral parameters are drawn
from the :class:`~clyde.setup.prior_library.PriorLibrary` and then optionally
mutated by ``param_overrides`` and ``shock_config.behavioral_overrides``.

The factory is responsible for:

* spawning actors with deterministic, zero-padded IDs
* looking up default params per actor from the Prior Library
* applying two scopes of overrides (per actor-type, per actor-id), with
  per-id overrides winning and explicit ``param_overrides`` beating
  ``shock_config.behavioral_overrides`` on conflict
* wiring networks via an injected :class:`NetworkBuilder`
* cross-validating that every produced ``Relationship`` references a real
  actor and a known ``rel_type``
* recording every applied override onto :class:`Scenario.overrides` so the
  evidence trail required by Property 6 / Requirement 3.5 survives
  serialization
"""

from __future__ import annotations

import random
from dataclasses import fields, replace
from typing import Any, Iterable

from clyde.models.actors import (
    PARAMS_CLASS_BY_TYPE,
    REQUIRED_PARAM_FIELDS,
    Actor,
    Relationship,
)
from clyde.models.config import ShockConfig, SimulationWorld
from clyde.models.enums import RELATIONSHIP_TYPES, ActorType, canonicalize_actor_type
from clyde.models.networks import NetworkBundle
from clyde.models.scenario import Scenario
from clyde.setup.network_builder import NetworkBuilder
from clyde.setup.prior_library import PriorLibrary, ScenarioContext


# Zero-pad ids to this width so sorting is lexicographic-stable for realistic
# ensemble sizes (< 10k actors of a given type).
_ID_PAD_WIDTH = 4


def _make_actor_id(actor_type: ActorType, idx: int) -> str:
    return f"{actor_type.value}_{idx:0{_ID_PAD_WIDTH}d}"


def _parse_override_key(key: str) -> tuple[str, str]:
    """Split an override key into (target, param_name).

    ``target`` is either an :class:`ActorType` value (e.g. ``"household"``)
    or a fully qualified actor id (e.g. ``"household_0003"``).
    """
    if "." not in key:
        raise ValueError(
            f"Override key {key!r} must be of the form "
            f"'<actor_type>.<param_name>' or '<actor_id>.<param_name>'"
        )
    target, _, param = key.partition(".")
    if not target or not param:
        raise ValueError(
            f"Override key {key!r} has empty target or param component"
        )
    return target, param


class EconomicWorldFactory:
    """Assembles a :class:`SimulationWorld` from a :class:`ShockConfig`."""

    def __init__(
        self,
        network_builder: NetworkBuilder | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self._rng = random.Random(rng_seed)
        # Inject the NetworkBuilder (so tests can supply a deterministic one);
        # otherwise construct one seeded from the factory's RNG for determinism.
        if network_builder is None:
            seeded_rng = random.Random(self._rng.random())
            network_builder = NetworkBuilder(rng=seeded_rng)
        self._network_builder = network_builder

    # ------------------------------------------------------------------ API

    def build_world(
        self,
        shock_config: ShockConfig,
        prior_library: PriorLibrary,
        param_overrides: dict[str, Any] | None = None,
    ) -> SimulationWorld:
        world, _applied = self._build(shock_config, prior_library, param_overrides)
        return world

    def build_scenario(
        self,
        scenario_id: str,
        description: str,
        shock_config: ShockConfig,
        prior_library: PriorLibrary,
        param_overrides: dict[str, Any] | None = None,
    ) -> Scenario:
        world, applied = self._build(shock_config, prior_library, param_overrides)
        return Scenario(
            scenario_id=scenario_id,
            description=description,
            config=world.config,
            actors=list(world.actors),
            networks=world.networks,
            prior_library_version=world.prior_library_version,
            overrides=applied,
            metadata={},
        )

    # --------------------------------------------------------------- Internals

    def _build(
        self,
        shock_config: ShockConfig,
        prior_library: PriorLibrary,
        param_overrides: dict[str, Any] | None,
    ) -> tuple[SimulationWorld, dict[str, dict[str, Any]]]:
        # 1. Merge the two override sources. Explicit param_overrides wins
        #    on key collision; we still track the source for the scenario.
        merged_sources = self._merge_overrides(
            behavioral=dict(shock_config.behavioral_overrides or {}),
            explicit=dict(param_overrides or {}),
        )

        # 2. Spawn actors with default params from the Prior Library.
        context = ScenarioContext(
            scope=shock_config.scope,
            sectors=tuple(shock_config.sectors),
            geographies=tuple(shock_config.geography),
        )
        actors = self._spawn_actors(shock_config, prior_library, context)
        actors_by_id = {a.id: a for a in actors}
        actors_by_type: dict[ActorType, list[Actor]] = {
            atype: [a for a in actors if a.actor_type == atype]
            for atype in ActorType
        }

        # 3. Validate override keys against the actor universe BEFORE mutating
        #    anything. Unknown type / id / param names must raise clearly.
        self._validate_override_keys(merged_sources, actors_by_id, actors_by_type)

        # 4. Apply overrides. Per-id overrides beat per-type overrides.
        applied: dict[str, dict[str, Any]] = {}
        actors = self._apply_overrides(
            actors=actors,
            actors_by_type=actors_by_type,
            merged_sources=merged_sources,
            applied=applied,
        )

        # 5. Wire networks. We inject the same RNG instance so downstream
        #    calls share entropy but remain deterministic across runs.
        networks = self._build_networks(actors)

        # 6. Attach per-actor Relationship lists derived from network edges.
        actors = self._attach_relationships(actors, networks)

        # 7. Cross-validate every relationship references real ids + known rel_type.
        self._validate_relationships(actors, networks)

        world = SimulationWorld(
            config=shock_config,
            actors=actors,
            networks=networks,
            prior_library_version=prior_library.version(),
        )
        return world, applied

    # ---- override plumbing -------------------------------------------------

    @staticmethod
    def _merge_overrides(
        behavioral: dict[str, Any],
        explicit: dict[str, Any],
    ) -> dict[str, tuple[Any, str]]:
        """Merge the two override dicts into a single ``key → (value, source)``.

        ``explicit`` (i.e. ``param_overrides`` passed to :meth:`build_world`)
        wins when both sides define the same key.
        """
        merged: dict[str, tuple[Any, str]] = {}
        for k, v in behavioral.items():
            merged[k] = (v, "behavioral_overrides")
        for k, v in explicit.items():
            merged[k] = (v, "param_overrides")
        return merged

    @staticmethod
    def _validate_override_keys(
        merged: dict[str, tuple[Any, str]],
        actors_by_id: dict[str, Actor],
        actors_by_type: dict[ActorType, list[Actor]],
    ) -> None:
        actor_type_values = {t.value for t in ActorType}
        errors: list[str] = []
        for key in merged:
            target, param = _parse_override_key(key)

            # Determine whether target refers to a type or to a single actor.
            if target in actor_type_values:
                atype = ActorType(target)
                allowed = set(REQUIRED_PARAM_FIELDS[atype])
                if param not in allowed:
                    errors.append(
                        f"unknown param {param!r} for actor_type {target!r}; "
                        f"valid params: {sorted(allowed)}"
                    )
            elif target in actors_by_id:
                atype = actors_by_id[target].actor_type
                allowed = set(REQUIRED_PARAM_FIELDS[atype])
                if param not in allowed:
                    errors.append(
                        f"unknown param {param!r} for actor {target!r} "
                        f"(type {atype.value}); valid params: {sorted(allowed)}"
                    )
            else:
                errors.append(
                    f"unknown override target {target!r} (not an ActorType "
                    f"value nor a known actor id)"
                )
        if errors:
            raise ValueError(
                "Invalid override keys: " + "; ".join(errors)
            )

    def _apply_overrides(
        self,
        actors: list[Actor],
        actors_by_type: dict[ActorType, list[Actor]],
        merged_sources: dict[str, tuple[Any, str]],
        applied: dict[str, dict[str, Any]],
    ) -> list[Actor]:
        """Return a new list of actors with overrides applied.

        Per-id overrides win over per-type overrides for the same param.
        The ``applied`` dict is mutated in place and records every change.
        """
        # Separate into type-scoped and id-scoped pools.
        type_scoped: dict[ActorType, dict[str, tuple[Any, str]]] = {
            t: {} for t in ActorType
        }
        id_scoped: dict[str, dict[str, tuple[Any, str]]] = {}
        actor_type_values = {t.value for t in ActorType}

        for key, (value, source) in merged_sources.items():
            target, param = _parse_override_key(key)
            if target in actor_type_values:
                type_scoped[ActorType(target)][param] = (value, source)
            else:
                id_scoped.setdefault(target, {})[param] = (value, source)

        new_actors: list[Actor] = []
        for actor in actors:
            type_overrides = type_scoped.get(actor.actor_type, {})
            per_actor = id_scoped.get(actor.id, {})

            # Per-id overrides win: start with type, layer per-id on top.
            effective: dict[str, tuple[Any, str]] = {}
            effective.update(type_overrides)
            effective.update(per_actor)

            if not effective:
                new_actors.append(actor)
                continue

            # Build new params dataclass via replace() so we stay immutable-ish
            # and still get field-level validation via Actor.__post_init__.
            new_param_kwargs = {name: value for name, (value, _src) in effective.items()}
            new_params = replace(actor.params, **new_param_kwargs)
            new_actor = replace(actor, params=new_params)
            new_actors.append(new_actor)

            # Record the applied overrides using the canonical key form the
            # user supplied (type-level vs id-level) so the evidence trail
            # matches what the caller sent in.
            for param, (value, source) in type_overrides.items():
                # A per-id override for the same param supersedes this record.
                if param in per_actor:
                    continue
                key = f"{actor.actor_type.value}.{param}"
                applied[key] = {"value": value, "source": source}
            for param, (value, source) in per_actor.items():
                key = f"{actor.id}.{param}"
                applied[key] = {"value": value, "source": source}

        return new_actors

    # ---- actor spawning ----------------------------------------------------

    def _spawn_actors(
        self,
        shock_config: ShockConfig,
        prior_library: PriorLibrary,
        context: ScenarioContext,
    ) -> list[Actor]:
        actors: list[Actor] = []
        # Iterate ActorType in declaration order for stable ids across runs.
        # Be tolerant of free-form actor labels coming from upstream callers
        # (LLM-driven shock-config builders, programmatic branches): bucket
        # them onto the canonical four via `canonicalize_actor_type`. We only
        # raise when a label can't be mapped at all — that's a programming
        # error worth surfacing.
        raw_counts = dict(shock_config.agent_counts or {})
        actor_type_values = {t.value for t in ActorType}
        counts: dict[str, int] = {}
        unknown: list[str] = []
        for k, v in raw_counts.items():
            if k in actor_type_values:
                counts[k] = counts.get(k, 0) + int(v)
                continue
            canon = canonicalize_actor_type(k)
            if canon is None:
                unknown.append(k)
                continue
            counts[canon.value] = counts.get(canon.value, 0) + int(v)
        if unknown:
            raise ValueError(
                f"ShockConfig.agent_counts contains unknown actor types: "
                f"{sorted(unknown)}; valid values: {sorted(actor_type_values)}"
            )

        for atype in ActorType:
            n = int(counts.get(atype.value, 0))
            if n <= 0:
                continue
            for i in range(n):
                params = prior_library.get_params(atype, context)
                actor = Actor(
                    id=_make_actor_id(atype, i),
                    actor_type=atype,
                    params=params,
                    state={},
                    relationships=[],
                )
                actors.append(actor)
        return actors

    # ---- networks + relationships -----------------------------------------

    def _build_networks(self, actors: list[Actor]) -> NetworkBundle:
        households = [a for a in actors if a.actor_type == ActorType.HOUSEHOLD]
        firms = [a for a in actors if a.actor_type == ActorType.FIRM]
        banks = [a for a in actors if a.actor_type == ActorType.BANK]
        nb = self._network_builder
        return NetworkBundle(
            labor_market=nb.build_labor_market(households, firms),
            supply_chain=nb.build_supply_chain(firms, households),
            interbank=nb.build_interbank(banks),
        )

    def _attach_relationships(
        self,
        actors: list[Actor],
        networks: NetworkBundle,
    ) -> list[Actor]:
        """Populate each actor's ``relationships`` from the network edges.

        Edges are interpreted as outbound relationships on the source actor.
        Mapping of network to ``rel_type``:

        - ``labor_market`` → ``employment``
        - ``supply_chain`` → ``supply`` for firm→firm, ``trade`` for firm→household
        - ``interbank`` → ``lending``
        """
        actors_by_id = {a.id: a for a in actors}
        rels_by_source: dict[str, list[Relationship]] = {a.id: [] for a in actors}

        # Labor market: bipartite edges (household, firm). The builder writes
        # (household_id, firm_id, weight) — we treat the household as the
        # source of an "employment" relationship to its firm.
        for src, tgt, w in networks.labor_market.edges:
            rels_by_source.setdefault(src, []).append(
                Relationship(source_id=src, target_id=tgt, rel_type="employment", weight=float(w))
            )

        # Supply chain: firm→firm is "supply"; firm→household is "trade".
        for src, tgt, w in networks.supply_chain.edges:
            src_actor = actors_by_id.get(src)
            tgt_actor = actors_by_id.get(tgt)
            if src_actor is None or tgt_actor is None:
                # Skip silently here; _validate_relationships will catch it.
                continue
            if tgt_actor.actor_type == ActorType.HOUSEHOLD:
                rel_type = "trade"
            else:
                rel_type = "supply"
            rels_by_source.setdefault(src, []).append(
                Relationship(source_id=src, target_id=tgt, rel_type=rel_type, weight=float(w))
            )

        # Interbank: bank→bank edges are "lending" relationships.
        for src, tgt, w in networks.interbank.edges:
            rels_by_source.setdefault(src, []).append(
                Relationship(source_id=src, target_id=tgt, rel_type="lending", weight=float(w))
            )

        # Rebuild the actor list with attached relationships.
        new_actors: list[Actor] = []
        for a in actors:
            new_actors.append(replace(a, relationships=list(rels_by_source.get(a.id, []))))
        return new_actors

    @staticmethod
    def _validate_relationships(
        actors: list[Actor],
        networks: NetworkBundle,
    ) -> None:
        ids = {a.id for a in actors}
        offenders: list[str] = []

        for actor in actors:
            for rel in actor.relationships:
                if rel.source_id not in ids:
                    offenders.append(
                        f"actor {actor.id}: unknown source_id {rel.source_id!r}"
                    )
                if rel.target_id not in ids:
                    offenders.append(
                        f"actor {actor.id}: unknown target_id {rel.target_id!r}"
                    )
                if rel.rel_type not in RELATIONSHIP_TYPES:
                    offenders.append(
                        f"actor {actor.id}: invalid rel_type {rel.rel_type!r}"
                    )

        for label, edges in (
            ("labor_market", networks.labor_market.edges),
            ("supply_chain", networks.supply_chain.edges),
            ("interbank", networks.interbank.edges),
        ):
            for src, tgt, _w in edges:
                if src not in ids:
                    offenders.append(f"{label}: unknown source_id {src!r}")
                if tgt not in ids:
                    offenders.append(f"{label}: unknown target_id {tgt!r}")

        if offenders:
            raise ValueError(
                "NetworkIntegrity violation(s): " + "; ".join(offenders)
            )


__all__ = ["EconomicWorldFactory"]
