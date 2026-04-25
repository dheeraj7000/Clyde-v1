"""Tests for :mod:`clyde.setup.ingestion`.

Covers the :class:`DocumentIngester` public API against each of the three
supported formats (PDF / MD / TXT), error paths (unsupported extension,
missing file, empty file), and the batch helpers.

The PDF fixture at ``tests/fixtures/sample.pdf`` is a tiny hand-crafted
valid PDF (~600 bytes). For robustness the tests also build a PDF
programmatically via ``pypdf.PdfWriter`` in ``tmp_path`` to guarantee a
deterministic page count on any platform.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clyde.models.input import Document
from clyde.setup.ingestion import (
    DocumentIngester,
    DocumentIngestionError,
    UnsupportedFormatError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _make_blank_pdf(path: Path, num_pages: int = 1) -> None:
    """Create a valid multi-page blank PDF at ``path`` using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as fh:
        writer.write(fh)


# --------------------------------------------------------------------- #
# supported_formats
# --------------------------------------------------------------------- #
def test_supported_formats_is_pdf_md_txt() -> None:
    ingester = DocumentIngester()
    assert ingester.supported_formats() == {"pdf", "md", "txt"}


def test_supported_formats_returns_fresh_set() -> None:
    # Mutating the returned set must not affect subsequent calls.
    ingester = DocumentIngester()
    formats = ingester.supported_formats()
    formats.add("docx")
    assert ingester.supported_formats() == {"pdf", "md", "txt"}


# --------------------------------------------------------------------- #
# ingest: .txt
# --------------------------------------------------------------------- #
def test_ingest_txt_returns_document_with_content(tmp_path: Path) -> None:
    txt = tmp_path / "note.txt"
    payload = "line one\nline two\nline three\n"
    txt.write_text(payload, encoding="utf-8", newline="\n")

    ingester = DocumentIngester()
    doc = ingester.ingest(txt)

    assert isinstance(doc, Document)
    assert doc.format == "txt"
    assert doc.content == payload
    assert doc.path == str(txt)
    assert doc.metadata["empty"] is False
    assert doc.metadata["size_bytes"] == len(payload.encode("utf-8"))


def test_ingest_txt_fixture() -> None:
    ingester = DocumentIngester()
    doc = ingester.ingest(FIXTURES_DIR / "sample.txt")
    assert doc.format == "txt"
    assert "sample plain-text document" in doc.content


def test_ingest_txt_accepts_string_path(tmp_path: Path) -> None:
    txt = tmp_path / "string_path.txt"
    txt.write_text("hello", encoding="utf-8")
    ingester = DocumentIngester()
    doc = ingester.ingest(str(txt))
    assert doc.content == "hello"
    assert doc.format == "txt"


# --------------------------------------------------------------------- #
# ingest: .md / .markdown
# --------------------------------------------------------------------- #
def test_ingest_md_preserves_markdown_syntax(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    payload = "# Heading\n\n- **bold** item\n- `code` item\n"
    md.write_text(payload, encoding="utf-8")

    ingester = DocumentIngester()
    doc = ingester.ingest(md)

    assert doc.format == "md"
    assert doc.content == payload
    assert "# Heading" in doc.content
    assert "**bold**" in doc.content
    assert doc.metadata["line_count"] >= 3


def test_ingest_markdown_extension_produces_md_format(tmp_path: Path) -> None:
    md = tmp_path / "note.markdown"
    md.write_text("# Hi\n", encoding="utf-8")

    ingester = DocumentIngester()
    doc = ingester.ingest(md)
    assert doc.format == "md"
    assert doc.content == "# Hi\n"


def test_ingest_md_fixture() -> None:
    ingester = DocumentIngester()
    doc = ingester.ingest(FIXTURES_DIR / "sample.md")
    assert doc.format == "md"
    # Markdown syntax must survive intact.
    assert "# Sample Markdown Document" in doc.content
    assert "[A link](https://example.com)" in doc.content


def test_ingest_extension_is_case_insensitive(tmp_path: Path) -> None:
    md_upper = tmp_path / "upper.MD"
    md_upper.write_text("# upper\n", encoding="utf-8")
    txt_upper = tmp_path / "upper.TXT"
    txt_upper.write_text("plain\n", encoding="utf-8")

    ingester = DocumentIngester()
    assert ingester.ingest(md_upper).format == "md"
    assert ingester.ingest(txt_upper).format == "txt"


# --------------------------------------------------------------------- #
# ingest: .pdf
# --------------------------------------------------------------------- #
def test_ingest_pdf_fixture_reports_pages_and_format() -> None:
    ingester = DocumentIngester()
    doc = ingester.ingest(FIXTURES_DIR / "sample.pdf")

    assert doc.format == "pdf"
    assert doc.metadata["pages"] == 1
    # Our hand-crafted PDF embeds the literal phrase "Hello Clyde PDF".
    assert "Hello" in doc.content
    assert "Clyde" in doc.content


def test_ingest_pdf_multipage_from_pypdf(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    _make_blank_pdf(pdf, num_pages=3)

    ingester = DocumentIngester()
    doc = ingester.ingest(pdf)

    assert doc.format == "pdf"
    assert doc.metadata["pages"] == 3
    # Blank pages yield empty extracted text; ensure extraction didn't crash.
    assert isinstance(doc.content, str)


def test_ingest_pdf_corrupted_raises_ingestion_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a real pdf at all")

    ingester = DocumentIngester()
    with pytest.raises(DocumentIngestionError) as exc_info:
        ingester.ingest(bad)
    # The underlying cause must be chained.
    assert exc_info.value.__cause__ is not None


# --------------------------------------------------------------------- #
# ingest: error paths
# --------------------------------------------------------------------- #
def test_ingest_unsupported_extension_raises(tmp_path: Path) -> None:
    weird = tmp_path / "file.xyz"
    weird.write_text("whatever", encoding="utf-8")

    ingester = DocumentIngester()
    with pytest.raises(UnsupportedFormatError) as exc_info:
        ingester.ingest(weird)
    assert ".xyz" in str(exc_info.value)


def test_ingest_docx_raises_unsupported(tmp_path: Path) -> None:
    docx = tmp_path / "resume.docx"
    docx.write_bytes(b"PK\x03\x04...")  # zip magic, but we only check ext
    ingester = DocumentIngester()
    with pytest.raises(UnsupportedFormatError):
        ingester.ingest(docx)


def test_ingest_no_extension_raises(tmp_path: Path) -> None:
    noext = tmp_path / "README"
    noext.write_text("body\n", encoding="utf-8")
    ingester = DocumentIngester()
    with pytest.raises(UnsupportedFormatError):
        ingester.ingest(noext)


def test_ingest_missing_file_raises_document_ingestion_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"
    ingester = DocumentIngester()
    with pytest.raises(DocumentIngestionError) as exc_info:
        ingester.ingest(missing)
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_ingest_missing_pdf_raises_document_ingestion_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    ingester = DocumentIngester()
    with pytest.raises(DocumentIngestionError):
        ingester.ingest(missing)


# --------------------------------------------------------------------- #
# ingest: empty files
# --------------------------------------------------------------------- #
def test_ingest_empty_txt_flags_empty_metadata(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    ingester = DocumentIngester()
    doc = ingester.ingest(empty)

    assert doc.format == "txt"
    assert doc.content == ""
    assert doc.metadata["empty"] is True
    assert doc.metadata["size_bytes"] == 0


def test_ingest_empty_md_flags_empty_metadata(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")

    ingester = DocumentIngester()
    doc = ingester.ingest(empty)

    assert doc.format == "md"
    assert doc.metadata["empty"] is True


# --------------------------------------------------------------------- #
# ingest_many / ingest_many_with_errors
# --------------------------------------------------------------------- #
def test_ingest_many_returns_documents_in_input_order(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("A", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("# B", encoding="utf-8")
    c = tmp_path / "c.txt"
    c.write_text("C", encoding="utf-8")

    ingester = DocumentIngester()
    docs = ingester.ingest_many([a, b, c])

    assert len(docs) == 3
    assert [d.content for d in docs] == ["A", "# B", "C"]
    assert [d.format for d in docs] == ["txt", "md", "txt"]


def test_ingest_many_skips_failures_silently(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("ok", encoding="utf-8")
    bad_ext = tmp_path / "bad.xyz"
    bad_ext.write_text("nope", encoding="utf-8")
    missing = tmp_path / "missing.txt"  # not created

    ingester = DocumentIngester()
    docs = ingester.ingest_many([good, bad_ext, missing])

    assert len(docs) == 1
    assert docs[0].content == "ok"


def test_ingest_many_with_errors_reports_both(tmp_path: Path) -> None:
    good = tmp_path / "good.md"
    good.write_text("# hi", encoding="utf-8")
    bad_ext = tmp_path / "bad.xyz"
    bad_ext.write_text("nope", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    ingester = DocumentIngester()
    docs, errors = ingester.ingest_many_with_errors([good, bad_ext, missing])

    assert len(docs) == 1
    assert docs[0].format == "md"

    assert len(errors) == 2
    error_paths = {p for p, _ in errors}
    assert bad_ext in error_paths
    assert missing in error_paths

    error_map = dict(errors)
    assert isinstance(error_map[bad_ext], UnsupportedFormatError)
    assert isinstance(error_map[missing], DocumentIngestionError)


def test_ingest_many_with_errors_on_all_success(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("A", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("B", encoding="utf-8")

    ingester = DocumentIngester()
    docs, errors = ingester.ingest_many_with_errors([a, b])

    assert len(docs) == 2
    assert errors == []


def test_ingest_many_empty_iterable() -> None:
    ingester = DocumentIngester()
    assert ingester.ingest_many([]) == []
    docs, errors = ingester.ingest_many_with_errors([])
    assert docs == []
    assert errors == []
