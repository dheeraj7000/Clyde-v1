"""Uvicorn entry point: ``python -m clyde.web``.

Reads :envvar:`CLYDE_HOST` (default ``0.0.0.0``), :envvar:`CLYDE_PORT`
(default ``8000``), and :envvar:`CLYDE_RELOAD` (truthy enables auto-reload).
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    host = os.environ.get("CLYDE_HOST", "0.0.0.0")
    port = int(os.environ.get("CLYDE_PORT", "8000"))
    reload_enabled = _truthy(os.environ.get("CLYDE_RELOAD"))
    uvicorn.run(
        "clyde.web.server:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
