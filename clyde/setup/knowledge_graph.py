"""GraphRAG-based economic ontology store.

The :class:`KnowledgeGraph` is the persistent truth store for both seed-data
entities (extracted from documents and natural-language scenario input) and
simulation-generated artifacts (trajectory summaries, divergence findings,
causal chains). It is built during the *setup* phase using an
:class:`~clyde.llm.LLMClient`; from then on, the graph is queried by
the :class:`~clyde.setup.world_factory.EconomicWorldFactory` (to derive
:class:`~clyde.models.config.ShockConfig`) and by the reporting layer (to
fetch artifacts produced by the simulation).

Design notes
------------

* Storage is purely in-memory (``dict``s and ``list``s). There is no
  external graph database -- the graph is small enough to live alongside
  the simulation world.
* The economic ontology is intentionally lightweight: every entity has a
  ``type`` drawn from a small open vocabulary and every relation has a
  ``rel_type`` similarly drawn from a small set. The vocabulary is
  documented on the dataclasses below and on
  :data:`ECONOMIC_ENTITY_TYPES` / :data:`ECONOMIC_RELATION_TYPES`.
* :meth:`KnowledgeGraph.merge_sources` is *non-destructive*: when an
  entity appears in both NL-derived and document-derived input with
  conflicting fields, the conflict is recorded and **neither side is
  added**. The user resolves conflicts later (Requirement 16.5).
* :meth:`KnowledgeGraph.query` is a pure substring search over entity
  names and attribute values -- no LLM call required.
* :meth:`KnowledgeGraph.extract_shock_config` is also pure: it builds a
  :class:`ShockConfig` from a supplied :class:`ParseResult` (preferred)
  or, failing that, from the first ``"shock"``-typed entity in the
  graph. Geographies / sectors / actor counts are sourced from the
  graph and the parse result respectively.

LLM boundary
~~~~~~~~~~~~

This module imports :class:`~clyde.llm.LLMClient`; only
:meth:`build_from_documents` actually calls the client. Every other
method is deterministic and pure-Python so that callers can build
graphs ad hoc (e.g. in tests) without an LLM at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from clyde.llm import LLMClient, LLMMessage
from clyde.models.config import VALID_SCOPES, ShockConfig
from clyde.models.enums import canonicalize_actor_type
from clyde.models.input import Document, ParseResult
from clyde.models.time import TimeHorizon


# ---------------------------------------------------------------- Vocabulary

#: Recognised entity types in the economic ontology. The vocabulary is
#: open in the sense that the graph itself does not enforce membership;
#: this constant exists for documentation and prompt construction.
ECONOMIC_ENTITY_TYPES: tuple[str, ...] = (
    "actor",
    "market",
    "geography",
    "policy",
    "indicator",
    "shock",
)

#: Recognised relation types. Same caveat as
#: :data:`ECONOMIC_ENTITY_TYPES` -- not enforced at the storage layer.
ECONOMIC_RELATION_TYPES: tuple[str, ...] = (
    "supplies",
    "lends_to",
    "located_in",
    "regulates",
    "affects",
    "employs",
    "owns",
)


# ---------------------------------------------------------------- Dataclasses


@dataclass
class Entity:
    """An entity vertex in the knowledge graph.

    ``id`` is the stable join key (used by :meth:`KnowledgeGraph.merge_sources`).
    ``sources`` lists the document paths or the literal string ``"nl_input"``
    indicating where the entity was learned about.
    """

    id: str
    type: str
    name: str
    attributes: dict[str, str | float | int | bool] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


@dataclass
class GraphRelation:
    """A directed, weighted edge between two entities."""

    source_id: str
    target_id: str
    rel_type: str
    weight: float = 1.0
    sources: list[str] = field(default_factory=list)


@dataclass
class GraphNode:
    """A view of an entity together with its inbound and outbound edges.

    Returned by :meth:`KnowledgeGraph.query`. The relations are *copies*,
    so mutating a returned :class:`GraphNode` does not mutate the graph.
    """

    entity: Entity
    inbound: list[GraphRelation] = field(default_factory=list)
    outbound: list[GraphRelation] = field(default_factory=list)


@dataclass
class Conflict:
    """A merge conflict between NL-derived and document-derived entities.

    ``field`` is the attribute or top-level key that disagrees (e.g.
    ``"name"`` or ``"attributes.severity"``). ``nl_value`` and
    ``doc_value`` are the conflicting values; ``note`` is a human-readable
    explanation suitable for surfacing in a UI.
    """

    field: str
    nl_value: Any
    doc_value: Any
    note: str = ""


@dataclass
class SimulationArtifact:
    """A persistent piece of simulation output written back to the graph.

    The graph stores artifacts so the reporting layer can cite them by
    ``artifact_id`` without having to re-derive them. ``refs`` lists the
    entity ids the artifact pertains to, enabling later joins with the
    rest of the ontology.
    """

    artifact_id: str
    kind: str
    payload: dict
    refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- Prompting


_EXTRACTION_SYSTEM_PROMPT = (
    "You are an economic-ontology extractor. Read the document and "
    "return a JSON object with two keys: 'entities' and 'relations'.\n\n"
    "Each entity must have: 'id' (stable string), 'type' (one of "
    f"{list(ECONOMIC_ENTITY_TYPES)}), 'name', and 'attributes' "
    "(an object of primitive values).\n\n"
    "Each relation must have: 'source_id', 'target_id', 'rel_type' "
    f"(one of {list(ECONOMIC_RELATION_TYPES)}), and 'weight' (float).\n\n"
    "Return ONLY the JSON object, no commentary."
)


def _extraction_messages(document: Document) -> list[LLMMessage]:
    """Build the chat-style prompt for a single-document extraction call."""
    user_payload = {
        "path": document.path,
        "format": document.format,
        "content": document.content,
    }
    return [
        LLMMessage(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                "Extract entities and relations from the following document:\n"
                + json.dumps(user_payload, ensure_ascii=False)
            ),
        ),
    ]


# ---------------------------------------------------------------- Helpers


def _coerce_attributes(raw: Any) -> dict[str, str | float | int | bool]:
    """Coerce a raw attribute mapping into the strict primitive-value shape."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str | float | int | bool] = {}
    for k, v in raw.items():
        if isinstance(v, (str, float, int, bool)):
            out[str(k)] = v
        elif v is None:
            continue
        else:
            # Stringify lists/dicts/etc. so we never silently drop information.
            out[str(k)] = json.dumps(v, ensure_ascii=False, default=str)
    return out


def _entity_from_dict(raw: dict, sources: list[str]) -> Entity | None:
    """Parse a single entity payload returned by the LLM."""
    eid = raw.get("id")
    etype = raw.get("type")
    name = raw.get("name")
    if not isinstance(eid, str) or not eid:
        return None
    if not isinstance(etype, str) or not etype:
        return None
    if not isinstance(name, str):
        name = ""
    attrs = _coerce_attributes(raw.get("attributes"))
    return Entity(id=eid, type=etype, name=name, attributes=attrs, sources=list(sources))


def _relation_from_dict(raw: dict, sources: list[str]) -> GraphRelation | None:
    """Parse a single relation payload returned by the LLM."""
    src = raw.get("source_id")
    tgt = raw.get("target_id")
    rtype = raw.get("rel_type")
    if not isinstance(src, str) or not src:
        return None
    if not isinstance(tgt, str) or not tgt:
        return None
    if not isinstance(rtype, str) or not rtype:
        return None
    weight_raw = raw.get("weight", 1.0)
    try:
        weight = float(weight_raw)
    except (TypeError, ValueError):
        weight = 1.0
    return GraphRelation(
        source_id=src,
        target_id=tgt,
        rel_type=rtype,
        weight=weight,
        sources=list(sources),
    )


def _nl_entities_from_parse_result(parse_result: ParseResult) -> list[Entity]:
    """Derive a list of :class:`Entity` objects from a :class:`ParseResult`.

    The parser exposes geographies, markets, actor hints, and shock
    parameters; we lift each into an :class:`Entity` tagged with
    source ``"nl_input"`` so it can flow through
    :meth:`KnowledgeGraph.merge_sources`.
    """
    entities: list[Entity] = []
    for geo in parse_result.geographies:
        entities.append(
            Entity(
                id=f"geography:{geo}",
                type="geography",
                name=geo,
                attributes={},
                sources=["nl_input"],
            )
        )
    for mkt in parse_result.markets:
        entities.append(
            Entity(
                id=f"market:{mkt}",
                type="market",
                name=mkt,
                attributes={},
                sources=["nl_input"],
            )
        )
    for hint in parse_result.actor_hints:
        attrs: dict[str, str | float | int | bool] = {"actor_type": hint.actor_type}
        if hint.count_estimate is not None:
            attrs["count_estimate"] = int(hint.count_estimate)
        if hint.description:
            attrs["description"] = hint.description
        entities.append(
            Entity(
                id=f"actor:{hint.actor_type}",
                type="actor",
                name=hint.actor_type,
                attributes=attrs,
                sources=["nl_input"],
            )
        )
    sp = parse_result.shock_params
    if sp.shock_type or sp.severity or sp.duration_steps or sp.scope != "micro":
        entities.append(
            Entity(
                id=f"shock:{sp.shock_type or 'unspecified'}",
                type="shock",
                name=sp.shock_type or "unspecified",
                attributes={
                    "shock_type": sp.shock_type,
                    "severity": float(sp.severity),
                    "scope": sp.scope,
                    "duration_steps": int(sp.duration_steps),
                },
                sources=["nl_input"],
            )
        )
    return entities


# ---------------------------------------------------------------- KnowledgeGraph


class KnowledgeGraph:
    """In-memory economic ontology graph.

    Construct with an optional :class:`LLMClient`. Only
    :meth:`build_from_documents` requires the client; every other method
    works without one.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client
        self._entities: dict[str, Entity] = {}
        self._relations: list[GraphRelation] = []
        self._artifacts: dict[str, SimulationArtifact] = {}
        self._conflicts: list[Conflict] = []

    # ----------------------------------------------------------------- Build

    async def build_from_documents(
        self,
        documents: list[Document],
        parse_result: ParseResult | None = None,
    ) -> None:
        """Extract entities/relations from documents and merge with NL input.

        Calls :meth:`LLMClient.complete_json` once per document. Failures
        on a single document do **not** abort the whole call -- the
        document is skipped (so the graph contains a partial result).
        After every document is processed, NL-derived entities (from
        ``parse_result``) are merged via :meth:`merge_sources`; conflicts
        are stored on the graph and exposed via :meth:`conflicts`.

        Calling this method twice with overlapping documents is safe but
        idempotency is **not** guaranteed -- the second call will re-add
        all entities (with sources merged via :meth:`add_entity`).
        """
        if documents and self._llm is None:
            raise RuntimeError(
                "KnowledgeGraph.build_from_documents requires an LLMClient when "
                "documents are provided; instantiate KnowledgeGraph(llm_client=...)."
            )

        doc_entities: list[Entity] = []
        doc_relations: list[GraphRelation] = []

        for doc in documents:
            try:
                payload = await self._llm.complete_json(_extraction_messages(doc))
            except Exception as exc:
                # A single bad document must not poison the rest of the build.
                # Log a warning so callers can diagnose silent failures.
                import logging
                logging.getLogger(__name__).warning(
                    "KnowledgeGraph: failed to extract entities from %s: %s",
                    doc.path,
                    exc,
                )
                continue
            if not isinstance(payload, dict):
                continue
            sources = [doc.path]
            for raw in payload.get("entities", []) or []:
                if not isinstance(raw, dict):
                    continue
                ent = _entity_from_dict(raw, sources)
                if ent is not None:
                    doc_entities.append(ent)
            for raw in payload.get("relations", []) or []:
                if not isinstance(raw, dict):
                    continue
                rel = _relation_from_dict(raw, sources)
                if rel is not None:
                    doc_relations.append(rel)

        # Derive NL-side entities from the parse result (if provided) and
        # merge against the doc-side entities. Conflicts are recorded and
        # the non-conflicting entities are added to the graph as a
        # side-effect of merge_sources.
        nl_entities: list[Entity] = (
            _nl_entities_from_parse_result(parse_result) if parse_result else []
        )
        conflicts = self.merge_sources(nl_entities, doc_entities)
        self._conflicts.extend(conflicts)

        # Relations are added unconditionally; conflicts are only defined
        # for entities (the document is the only source of relations in
        # the current pipeline).
        for rel in doc_relations:
            self.add_relation(rel)

    # --------------------------------------------------------- Mutation API

    def add_entity(self, entity: Entity) -> None:
        """Add or merge an entity by id.

        If an entity with the same id already exists, the *new* entity's
        ``sources`` list is folded into the existing entry (deduplicated,
        preserving order) and unknown attributes are added; existing
        attributes are **not** overwritten. Use :meth:`merge_sources`
        when conflicts must be detected explicitly.
        """
        existing = self._entities.get(entity.id)
        if existing is None:
            self._entities[entity.id] = Entity(
                id=entity.id,
                type=entity.type,
                name=entity.name,
                attributes=dict(entity.attributes),
                sources=list(entity.sources),
            )
            return
        # Fold sources (preserve order, dedupe).
        merged_sources = list(existing.sources)
        for s in entity.sources:
            if s not in merged_sources:
                merged_sources.append(s)
        existing.sources = merged_sources
        # Add unknown attributes only.
        for k, v in entity.attributes.items():
            existing.attributes.setdefault(k, v)

    def add_relation(self, relation: GraphRelation) -> None:
        """Append a relation to the graph (no deduplication)."""
        self._relations.append(
            GraphRelation(
                source_id=relation.source_id,
                target_id=relation.target_id,
                rel_type=relation.rel_type,
                weight=float(relation.weight),
                sources=list(relation.sources),
            )
        )

    # ----------------------------------------------------------- Read API

    def query(self, query: str) -> list[GraphNode]:
        """Substring/keyword search over entity names + attribute values.

        Returns a list of :class:`GraphNode` -- one per matching entity --
        each populated with its inbound and outbound edges. The search is
        case-insensitive; an empty query returns no results.

        This is a pure function: no LLM call.
        """
        if not query:
            return []
        needle = query.lower()
        results: list[GraphNode] = []
        for ent in self._entities.values():
            if self._matches(ent, needle):
                inbound = [
                    self._copy_relation(r)
                    for r in self._relations
                    if r.target_id == ent.id
                ]
                outbound = [
                    self._copy_relation(r)
                    for r in self._relations
                    if r.source_id == ent.id
                ]
                results.append(
                    GraphNode(
                        entity=Entity(
                            id=ent.id,
                            type=ent.type,
                            name=ent.name,
                            attributes=dict(ent.attributes),
                            sources=list(ent.sources),
                        ),
                        inbound=inbound,
                        outbound=outbound,
                    )
                )
        return results

    @staticmethod
    def _matches(entity: Entity, needle_lower: str) -> bool:
        if needle_lower in entity.name.lower():
            return True
        if needle_lower in entity.id.lower():
            return True
        if needle_lower in entity.type.lower():
            return True
        for k, v in entity.attributes.items():
            if needle_lower in str(k).lower():
                return True
            if needle_lower in str(v).lower():
                return True
        return False

    @staticmethod
    def _copy_relation(rel: GraphRelation) -> GraphRelation:
        return GraphRelation(
            source_id=rel.source_id,
            target_id=rel.target_id,
            rel_type=rel.rel_type,
            weight=rel.weight,
            sources=list(rel.sources),
        )

    def list_entities(self, type: str | None = None) -> list[Entity]:
        """List entities, optionally filtered by ``type``.

        The order is the insertion order of the underlying dict (Python
        guarantees this for ``dict`` since 3.7).
        """
        if type is None:
            return list(self._entities.values())
        return [e for e in self._entities.values() if e.type == type]

    def conflicts(self) -> list[Conflict]:
        """Return the list of unresolved merge conflicts."""
        return list(self._conflicts)

    # ----------------------------------------------------------- Merge API

    def merge_sources(
        self,
        nl_entities: list[Entity],
        doc_entities: list[Entity],
    ) -> list[Conflict]:
        """Merge two entity lists by ``id`` and emit conflicts.

        Semantics
        ---------

        For each entity present in both lists (matched by ``id``):

        * If ``type`` differs OR ``name`` differs OR any *common*
          attribute key has a different value, a :class:`Conflict` is
          emitted **and the entity is not added** -- the user must
          resolve the disagreement manually (Requirement 16.5).
        * Otherwise the union of attributes is added with the union of
          sources (NL first, then document).

        Entities present in only one list are added directly.

        The method has the side-effect of mutating the graph
        (calling :meth:`add_entity`). It returns the list of conflicts;
        the caller may keep or discard them at will.
        """
        nl_by_id: dict[str, Entity] = {e.id: e for e in nl_entities}
        doc_by_id: dict[str, Entity] = {e.id: e for e in doc_entities}

        conflicts: list[Conflict] = []
        seen: set[str] = set()

        for eid, nl_ent in nl_by_id.items():
            seen.add(eid)
            doc_ent = doc_by_id.get(eid)
            if doc_ent is None:
                self.add_entity(nl_ent)
                continue
            ent_conflicts = self._diff_entities(nl_ent, doc_ent)
            if ent_conflicts:
                conflicts.extend(ent_conflicts)
                continue
            # No conflicts -> add the union.
            merged = self._union_entities(nl_ent, doc_ent)
            self.add_entity(merged)

        for eid, doc_ent in doc_by_id.items():
            if eid in seen:
                continue
            self.add_entity(doc_ent)

        return conflicts

    @staticmethod
    def _diff_entities(nl: Entity, doc: Entity) -> list[Conflict]:
        conflicts: list[Conflict] = []
        if nl.type != doc.type:
            conflicts.append(
                Conflict(
                    field="type",
                    nl_value=nl.type,
                    doc_value=doc.type,
                    note=f"Entity {nl.id!r}: type mismatch between NL and document sources.",
                )
            )
        if nl.name != doc.name:
            conflicts.append(
                Conflict(
                    field="name",
                    nl_value=nl.name,
                    doc_value=doc.name,
                    note=f"Entity {nl.id!r}: name mismatch between NL and document sources.",
                )
            )
        common_keys = set(nl.attributes) & set(doc.attributes)
        for k in sorted(common_keys):
            if nl.attributes[k] != doc.attributes[k]:
                conflicts.append(
                    Conflict(
                        field=f"attributes.{k}",
                        nl_value=nl.attributes[k],
                        doc_value=doc.attributes[k],
                        note=(
                            f"Entity {nl.id!r}: attribute {k!r} disagrees "
                            "between NL and document sources."
                        ),
                    )
                )
        return conflicts

    @staticmethod
    def _union_entities(nl: Entity, doc: Entity) -> Entity:
        merged_attrs: dict[str, str | float | int | bool] = dict(nl.attributes)
        for k, v in doc.attributes.items():
            merged_attrs.setdefault(k, v)
        merged_sources: list[str] = list(nl.sources)
        for s in doc.sources:
            if s not in merged_sources:
                merged_sources.append(s)
        return Entity(
            id=nl.id,
            type=nl.type,
            name=nl.name,
            attributes=merged_attrs,
            sources=merged_sources,
        )

    # -------------------------------------------------------- ShockConfig API

    def extract_shock_config(
        self,
        parse_result: ParseResult | None = None,
    ) -> ShockConfig:
        """Build a :class:`ShockConfig` from the graph + optional parse result.

        Resolution order
        ----------------

        1. If ``parse_result`` is provided, its ``shock_params``,
           ``triggering_event``, ``time_horizon``, ``geographies``,
           ``markets``, and ``actor_hints`` are the primary source.
        2. Otherwise, the first ``"shock"``-typed entity in the graph
           supplies ``shock_type`` (its ``name`` or ``attributes['shock_type']``),
           ``severity``, ``scope``, and ``duration_steps`` (read from
           ``attributes``).
        3. ``geography`` and ``sectors`` are extended with every
           ``"geography"`` / ``"market"`` entity in the graph.
        4. ``agent_counts`` defaults to a small mixed economy
           (50 households, 10 firms, 3 banks, 1 central bank) when the
           parse result omits hints.

        The resulting :class:`ShockConfig` is validated by its own
        ``__post_init__`` (severity in [0, 1], scope in
        :data:`VALID_SCOPES`, ``duration_steps >= 0``).
        """
        # ---- defaults
        shock_type = ""
        severity = 0.0
        scope = "micro"
        duration_steps = 0
        time_horizon = TimeHorizon(steps=0, step_unit="day")
        initial_contact_actors: list[str] = []
        agent_counts: dict[str, int] = {}
        geographies: list[str] = []
        sectors: list[str] = []

        # ---- 1. parse_result wins for the scenario header
        if parse_result is not None:
            sp = parse_result.shock_params
            shock_type = sp.shock_type or parse_result.triggering_event or ""
            severity = float(sp.severity)
            scope = sp.scope or "micro"
            duration_steps = int(sp.duration_steps)
            time_horizon = parse_result.time_horizon
            initial_contact_actors = list(sp.initial_contact_actors)
            geographies = list(parse_result.geographies)
            sectors = list(parse_result.markets)
            # The LLM frequently returns free-form labels here ("European
            # central banks", "OPEC+", "energy-intensive manufacturers").
            # Bucket them onto the canonical ActorType vocabulary so the
            # downstream world-factory validation accepts the config.
            # Hints that don't match any keyword are folded into the FIRM
            # bucket, biased toward producer-like entities.
            for hint in parse_result.actor_hints:
                if hint.count_estimate is None or hint.count_estimate <= 0:
                    continue
                canon = canonicalize_actor_type(hint.actor_type)
                key = canon.value if canon is not None else "firm"
                agent_counts[key] = agent_counts.get(key, 0) + int(hint.count_estimate)
        else:
            # ---- 2. fall back to a "shock" entity if there is one.
            shock_entity = next(
                (e for e in self._entities.values() if e.type == "shock"), None
            )
            if shock_entity is not None:
                attrs = shock_entity.attributes
                raw_st = attrs.get("shock_type", shock_entity.name)
                shock_type = str(raw_st) if raw_st is not None else shock_entity.name
                try:
                    severity = float(attrs.get("severity", 0.0))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    severity = 0.0
                raw_scope = attrs.get("scope", "micro")
                scope = str(raw_scope) if raw_scope in VALID_SCOPES else "micro"
                try:
                    duration_steps = int(attrs.get("duration_steps", 0))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    duration_steps = 0
                if duration_steps > 0:
                    time_horizon = TimeHorizon(steps=duration_steps, step_unit="day")

        # ---- 3. extend geography/sectors from graph entities (dedupe).
        for ent in self._entities.values():
            if ent.type == "geography" and ent.name and ent.name not in geographies:
                geographies.append(ent.name)
            elif ent.type == "market" and ent.name and ent.name not in sectors:
                sectors.append(ent.name)

        # ---- 4. agent_counts default if still empty.
        if not agent_counts:
            agent_counts = {
                "household": 50,
                "firm": 10,
                "bank": 3,
                "central_bank": 1,
            }

        return ShockConfig(
            shock_type=shock_type,
            severity=severity,
            scope=scope,
            duration_steps=duration_steps,
            geography=geographies,
            sectors=sectors,
            initial_contact_actors=initial_contact_actors,
            agent_counts=agent_counts,
            behavioral_overrides={},
            time_horizon=time_horizon,
            ensemble_seed=0,
            historical_analogs=[],
        )

    # -------------------------------------------------------- Artifact API

    def store_simulation_artifact(self, artifact: SimulationArtifact) -> None:
        """Persist a simulation artifact in the graph.

        Storing an artifact with an existing id overwrites the previous
        entry (idempotent re-runs of the synthesis layer are common).
        """
        self._artifacts[artifact.artifact_id] = SimulationArtifact(
            artifact_id=artifact.artifact_id,
            kind=artifact.kind,
            payload=dict(artifact.payload),
            refs=list(artifact.refs),
        )

    def get_artifact(self, artifact_id: str) -> SimulationArtifact | None:
        """Return the stored artifact, or ``None`` if absent."""
        existing = self._artifacts.get(artifact_id)
        if existing is None:
            return None
        return SimulationArtifact(
            artifact_id=existing.artifact_id,
            kind=existing.kind,
            payload=dict(existing.payload),
            refs=list(existing.refs),
        )

    # ----------------------------------------------------------- Introspection

    @property
    def relations(self) -> list[GraphRelation]:
        """Read-only view of every relation in the graph."""
        return [self._copy_relation(r) for r in self._relations]


__all__ = [
    "Entity",
    "GraphRelation",
    "GraphNode",
    "Conflict",
    "SimulationArtifact",
    "KnowledgeGraph",
    "ECONOMIC_ENTITY_TYPES",
    "ECONOMIC_RELATION_TYPES",
]
