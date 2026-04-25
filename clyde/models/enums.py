"""Enums used across the domain model."""

from enum import Enum


class ActorType(str, Enum):
    HOUSEHOLD = "household"
    FIRM = "firm"
    BANK = "bank"
    CENTRAL_BANK = "central_bank"


RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {"employment", "lending", "trade", "supply", "ownership", "regulation", "trust"}
)


# Keyword → canonical ActorType. Order matters: more specific phrases (e.g.
# "central bank") must come before less specific ones ("bank") so the
# bucketing in `canonicalize_actor_type` doesn't classify "european central
# banks" as a regular Bank. Each tuple is checked as a substring against the
# lowercased label.
_ACTOR_TYPE_KEYWORDS: tuple[tuple[str, ActorType], ...] = (
    # Central banks & monetary authorities first.
    ("central bank", ActorType.CENTRAL_BANK),
    ("monetary authority", ActorType.CENTRAL_BANK),
    ("federal reserve", ActorType.CENTRAL_BANK),
    ("reserve bank", ActorType.CENTRAL_BANK),
    ("ecb", ActorType.CENTRAL_BANK),
    ("boe", ActorType.CENTRAL_BANK),
    ("boj", ActorType.CENTRAL_BANK),
    ("pboc", ActorType.CENTRAL_BANK),
    (" the fed", ActorType.CENTRAL_BANK),
    ("regulator", ActorType.CENTRAL_BANK),
    # Households / consumers.
    ("household", ActorType.HOUSEHOLD),
    ("consumer", ActorType.HOUSEHOLD),
    ("worker", ActorType.HOUSEHOLD),
    ("wage earner", ActorType.HOUSEHOLD),
    ("family", ActorType.HOUSEHOLD),
    ("individual", ActorType.HOUSEHOLD),
    ("citizen", ActorType.HOUSEHOLD),
    # Banks (commercial / lenders).
    ("bank", ActorType.BANK),
    ("lender", ActorType.BANK),
    ("creditor", ActorType.BANK),
    ("financial institution", ActorType.BANK),
    ("credit union", ActorType.BANK),
    # Firms — anything producing or trading goods/services.
    ("firm", ActorType.FIRM),
    ("company", ActorType.FIRM),
    ("corporation", ActorType.FIRM),
    ("manufacturer", ActorType.FIRM),
    ("producer", ActorType.FIRM),
    ("supplier", ActorType.FIRM),
    ("business", ActorType.FIRM),
    ("industry", ActorType.FIRM),
    ("exporter", ActorType.FIRM),
    ("importer", ActorType.FIRM),
    ("trader", ActorType.FIRM),
    ("retailer", ActorType.FIRM),
    ("operator", ActorType.FIRM),
    ("opec", ActorType.FIRM),       # cartel of producers
    ("cartel", ActorType.FIRM),
)


def canonicalize_actor_type(label: str) -> ActorType | None:
    """Bucket a free-form actor label onto a canonical :class:`ActorType`.

    The LLM-driven scenario parser sometimes emits actor labels like
    ``"European central banks"`` or ``"OPEC+"`` that don't match any of
    the four canonical types. Strict downstream validation
    (:meth:`EconomicWorldFactory._spawn_actors`) rejects those and the
    whole run fails. This helper does a keyword-based bucketing, returning
    the canonical type or ``None`` when nothing matches so callers can
    decide whether to drop the hint or fold it into a default category.

    The mapping is intentionally generous and biased toward the FIRM bucket
    for ambiguous business-like nouns — see ``_ACTOR_TYPE_KEYWORDS``.
    """
    if not label:
        return None
    s = " " + str(label).lower().strip() + " "
    # Exact canonical match wins immediately.
    try:
        return ActorType(s.strip())
    except ValueError:
        pass
    for kw, atype in _ACTOR_TYPE_KEYWORDS:
        if kw in s:
            return atype
    return None
