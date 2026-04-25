"""LLM client wrappers.

Used only by the *setup* and *reporting* phases of Clyde. This subpackage
must **never** be imported from :mod:`clyde.simulation` or any module under
it -- the simulation phase is strictly rule-based (Requirement 15). A
static import check in ``tests/test_llm_boundary.py`` enforces the rule.
"""

from clyde.llm.client import LLMClient, LLMMessage, LLMResponse
from clyde.llm.factory import (
    available_providers,
    make_llm_client,
    resolve_provider,
)
from clyde.llm.mock import MockLLMClient

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "MockLLMClient",
    "available_providers",
    "make_llm_client",
    "resolve_provider",
]
