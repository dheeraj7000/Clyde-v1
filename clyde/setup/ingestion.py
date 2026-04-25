"""Document ingestion: load PDF / Markdown / plain-text files into `Document`s.

This module exposes :class:`DocumentIngester`, which reads a file from disk,
extracts its text content, and packages it into a
:class:`clyde.models.input.Document`.

Supported extensions (case-insensitive):

* ``.pdf`` -> extracted via ``pypdf`` (page text joined with double-newlines)
* ``.md`` / ``.markdown`` -> read as UTF-8 (Markdown syntax preserved)
* ``.txt`` -> read as UTF-8

Errors raised by this module:

* :class:`UnsupportedFormatError` -- extension is not in the supported set.
* :class:`DocumentIngestionError` -- I/O or extraction failure. The underlying
  exception is always chained via ``raise ... from e``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from clyde.models.input import Document


class UnsupportedFormatError(ValueError):
    """Raised when a file has an extension outside the supported set."""


class DocumentIngestionError(RuntimeError):
    """Raised when a supported file cannot be read or parsed."""


# Map of accepted extension (including leading dot, lowercased) -> canonical format label
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}

# Set of canonical format labels reported by `supported_formats()`.
_SUPPORTED_FORMATS: set[str] = {"pdf", "md", "txt"}


class DocumentIngester:
    """Load PDF / Markdown / plain-text files into `Document` instances."""

    def __init__(self) -> None:
        # No mutable state; constructor kept for future extensibility (e.g.
        # configurable encodings or pluggable extractors).
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def supported_formats(self) -> set[str]:
        """Return the set of canonical format labels this ingester accepts."""
        return set(_SUPPORTED_FORMATS)

    def ingest(self, file_path: str | Path) -> Document:
        """Extract text from a single PDF, Markdown, or plain-text file.

        Parameters
        ----------
        file_path:
            Path to the file on disk. Accepts ``str`` or ``Path``.

        Returns
        -------
        Document
            A populated :class:`Document` with ``format``, ``content``, and
            a ``metadata`` dict containing at least ``size_bytes`` and
            ``empty`` (plus format-specific fields such as ``pages`` or
            ``line_count``).

        Raises
        ------
        UnsupportedFormatError
            If the file's extension is not in ``{.pdf, .md, .markdown, .txt}``.
        DocumentIngestionError
            If the file cannot be read or its content cannot be extracted.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in _EXTENSION_MAP:
            raise UnsupportedFormatError(
                f"unsupported format: {ext or '<no extension>'}; "
                f"supported: {sorted(_SUPPORTED_FORMATS)!r}"
            )

        fmt = _EXTENSION_MAP[ext]
        if fmt == "pdf":
            return self._ingest_pdf(path)
        if fmt == "md":
            return self._ingest_md(path)
        if fmt == "txt":
            return self._ingest_txt(path)

        # Defensive: should never happen given the extension map.
        raise UnsupportedFormatError(f"unsupported format: {ext}")  # pragma: no cover

    def ingest_many(self, paths: Iterable[str | Path]) -> list[Document]:
        """Ingest multiple documents, silently skipping any that fail.

        This delegates to :meth:`ingest_many_with_errors` and discards the
        error list. Callers that need to know which files failed should use
        :meth:`ingest_many_with_errors` directly.
        """
        documents, _errors = self.ingest_many_with_errors(paths)
        return documents

    def ingest_many_with_errors(
        self, paths: Iterable[str | Path]
    ) -> tuple[list[Document], list[tuple[Path, Exception]]]:
        """Ingest multiple documents, returning both successes and failures.

        Returns a tuple ``(documents, errors)`` where:

        * ``documents`` is the list of successfully ingested documents in the
          same order they appeared in the input (failed entries omitted).
        * ``errors`` is a list of ``(Path, Exception)`` tuples describing
          every per-file failure. The exceptions are instances of
          :class:`UnsupportedFormatError` or :class:`DocumentIngestionError`.
        """
        documents: list[Document] = []
        errors: list[tuple[Path, Exception]] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                documents.append(self.ingest(path))
            except (UnsupportedFormatError, DocumentIngestionError) as exc:
                errors.append((path, exc))
        return documents, errors

    # ------------------------------------------------------------------ #
    # Per-format helpers
    # ------------------------------------------------------------------ #
    def _ingest_txt(self, path: Path) -> Document:
        content = self._read_text(path)
        stat = self._stat(path)
        metadata = {
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "line_count": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
            "empty": len(content) == 0,
        }
        return Document.from_path(path, content=content, fmt="txt", metadata=metadata)

    def _ingest_md(self, path: Path) -> Document:
        content = self._read_text(path)
        stat = self._stat(path)
        metadata = {
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "line_count": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
            "empty": len(content) == 0,
        }
        return Document.from_path(path, content=content, fmt="md", metadata=metadata)

    def _ingest_pdf(self, path: Path) -> Document:
        # Import locally so that missing `pypdf` only breaks PDF ingestion,
        # not the entire module.
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as e:  # pragma: no cover
            raise DocumentIngestionError(
                "pypdf is required for PDF ingestion; install with `pip install pypdf`"
            ) from e

        try:
            stat = self._stat(path)
            reader = PdfReader(str(path))
            pages_text: list[str] = []
            for page in reader.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception as e:  # noqa: BLE001 -- pypdf can raise many types
                    raise DocumentIngestionError(
                        f"failed to extract text from a page of {path}"
                    ) from e
            content = "\n\n".join(pages_text)
            metadata = {
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "pages": len(reader.pages),
                "empty": len(content) == 0,
            }
            return Document.from_path(path, content=content, fmt="pdf", metadata=metadata)
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            raise DocumentIngestionError(f"could not read PDF at {path}: {e}") from e
        except PdfReadError as e:
            raise DocumentIngestionError(f"malformed PDF at {path}: {e}") from e
        except DocumentIngestionError:
            raise
        except Exception as e:  # noqa: BLE001 -- surface any unexpected failure as ingestion error
            raise DocumentIngestionError(f"failed to ingest PDF at {path}: {e}") from e

    # ------------------------------------------------------------------ #
    # I/O helpers (wrapped to translate OSError -> DocumentIngestionError)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            raise DocumentIngestionError(f"could not read {path}: {e}") from e
        except UnicodeDecodeError as e:
            raise DocumentIngestionError(
                f"could not decode {path} as UTF-8: {e}"
            ) from e
        except OSError as e:
            raise DocumentIngestionError(f"OS error while reading {path}: {e}") from e

    @staticmethod
    def _stat(path: Path):
        try:
            return path.stat()
        except OSError as e:
            raise DocumentIngestionError(f"could not stat {path}: {e}") from e
