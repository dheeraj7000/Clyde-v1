"""Static import-boundary enforcement for the LLM / simulation split.

Requirement 15 (especially 15.3 and 15.5) mandates that the simulation
phase be strictly rule-based: zero LLM imports, zero LLM calls. This file
walks the source tree under ``clyde/simulation/`` and parses each module
with :mod:`ast` (no imports are ever executed), then verifies that no
module imports any LLM facility -- neither the in-tree ``clyde.llm``
package nor any third-party provider SDK.

The check is purely static so that even a lazy / conditional import would
be caught.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# Absolute paths so tests work regardless of the pytest cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
CLYDE_ROOT = REPO_ROOT / "clyde"
SIMULATION_ROOT = CLYDE_ROOT / "simulation"
LLM_ROOT = CLYDE_ROOT / "llm"


# Forbidden root modules. A module name is considered forbidden if it is
# exactly one of these or if it starts with ``<name>.``. ``clyde.llm``
# covers the in-tree setup-phase client; the remainder are external provider
# SDKs that would pull model inference into the simulation path.
FORBIDDEN_ROOTS: tuple[str, ...] = (
    "clyde.llm",
    "anthropic",
    "openai",
    "langchain",
    "llama_index",
    "google.generativeai",
    "cohere",
    "mistralai",
    "groq",
)


def _iter_python_files(root: Path):
    """Yield every ``.py`` file under ``root``, skipping ``__pycache__``."""
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _is_forbidden(module_name: str) -> bool:
    """Return True if ``module_name`` is or sits beneath a forbidden root."""
    if not module_name:
        return False
    for root in FORBIDDEN_ROOTS:
        if module_name == root or module_name.startswith(root + "."):
            return True
    return False


def _collect_forbidden_imports(path: Path) -> list[tuple[str, int, str]]:
    """Return ``(file, lineno, imported_name)`` tuples for forbidden imports.

    Parses ``path`` with :func:`ast.parse`; never executes any code.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    offenders: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    offenders.append((str(path), node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # ``from X import Y`` - ``node.module`` is ``X`` (may be None for
            # ``from . import foo``; relative imports inside ``clyde.simulation``
            # can never reach ``clyde.llm`` so we ignore None safely).
            module = node.module or ""
            if _is_forbidden(module):
                for alias in node.names:
                    imported = f"{module}.{alias.name}"
                    offenders.append((str(path), node.lineno, imported))
    return offenders


# ---------------------------------------------------------------- tests


def test_simulation_package_has_no_llm_imports():
    """No module under ``clyde/simulation/`` may import any LLM facility."""
    assert SIMULATION_ROOT.is_dir(), (
        f"Expected simulation package at {SIMULATION_ROOT}"
    )

    offenders: list[tuple[str, int, str]] = []
    files_scanned = 0
    for path in _iter_python_files(SIMULATION_ROOT):
        files_scanned += 1
        offenders.extend(_collect_forbidden_imports(path))

    assert files_scanned > 0, (
        f"No Python files found under {SIMULATION_ROOT}; boundary check would "
        "be vacuously true."
    )

    if offenders:
        formatted = "\n".join(
            f"  {file}:{line} imports {name!r}" for file, line, name in offenders
        )
        pytest.fail(
            "clyde.simulation is LLM-free by contract (Requirement 15.3/15.5).\n"
            "Found forbidden imports:\n" + formatted
        )


def test_world_factory_does_not_import_llm_providers_directly():
    """Positive control: the setup phase uses DI, not direct provider imports.

    ``clyde.setup.world_factory`` is *allowed* to import anything (including
    ``clyde.llm``), but the current design injects the LLM client rather than
    instantiating provider SDKs inline. Guarding against accidental direct
    imports of provider SDKs keeps the dependency-injection contract honest.

    ``clyde.llm`` itself is permitted here -- only raw provider SDKs are not.
    """
    target = CLYDE_ROOT / "setup" / "world_factory.py"
    assert target.is_file(), f"Expected world factory at {target}"

    # Exclude ``clyde.llm`` from the forbidden set for this positive control.
    provider_only_roots = tuple(r for r in FORBIDDEN_ROOTS if r != "clyde.llm")

    def _is_provider(name: str) -> bool:
        return any(
            name == r or name.startswith(r + ".") for r in provider_only_roots
        )

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_provider(alias.name):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_provider(module):
                for alias in node.names:
                    offenders.append((node.lineno, f"{module}.{alias.name}"))

    assert not offenders, (
        f"{target} imports LLM provider SDKs directly; inject them via "
        f"clyde.llm.LLMClient instead. Offenders: {offenders}"
    )


def test_llm_package_exists_with_expected_modules():
    """Sanity check: the LLM package and its core module are in place."""
    assert LLM_ROOT.is_dir(), f"Expected clyde/llm package at {LLM_ROOT}"
    assert (LLM_ROOT / "client.py").is_file(), (
        f"Expected client protocol module at {LLM_ROOT / 'client.py'}"
    )
    assert (LLM_ROOT / "mock.py").is_file(), (
        f"Expected mock client module at {LLM_ROOT / 'mock.py'}"
    )
    assert (LLM_ROOT / "__init__.py").is_file(), (
        f"Expected package init at {LLM_ROOT / '__init__.py'}"
    )


def test_llm_client_module_does_not_import_from_simulation():
    """Symmetric boundary: the protocol module must not depend on simulation.

    If ``clyde.llm.client`` ever pulled in simulation types, the dependency
    arrow would reverse and the boundary would be useless.
    """
    target = LLM_ROOT / "client.py"
    assert target.is_file(), f"Expected {target} to exist"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "clyde.simulation" or alias.name.startswith(
                    "clyde.simulation."
                ):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "clyde.simulation" or module.startswith(
                "clyde.simulation."
            ):
                for alias in node.names:
                    offenders.append((node.lineno, f"{module}.{alias.name}"))

    assert not offenders, (
        "clyde.llm.client must not import from clyde.simulation "
        f"(offenders: {offenders})."
    )
