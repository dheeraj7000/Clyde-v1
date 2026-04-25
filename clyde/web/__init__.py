"""Clyde FastAPI web layer.

This subpackage exposes the Clyde pipeline over HTTP for the hackathon
demo. It is intentionally thin — all heavy lifting lives in
:mod:`clyde.pipeline` and below. The web layer only:

* runs the pipeline as an asyncio background task,
* serialises the result to JSON-friendly dicts,
* keeps a small in-memory job store for polling / branching, and
* serves the frontend bundle from ``clyde/web/static``.
"""

from clyde.web.server import app, create_app

__all__ = ["app", "create_app"]
