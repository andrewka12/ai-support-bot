"""Knowledge-base loading and chunking.

Markdown (or plain-text) documents are parsed section by section: a heading
starts a new section, the text that follows belongs to it. Sections are then
packed into chunks of a bounded size with a small overlap, so a retrieved
chunk carries enough context to be quoted back to a customer.

Every chunk keeps its provenance -- business folder, source file, document
title and section heading. Those fields are shown to the customer as the
source of an answer and are asserted in the tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

#: Supported knowledge-base file extensions.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".md", ".txt")

#: Default upper bound of a chunk body, in characters.
DEFAULT_MAX_CHARS: int = 900

#: Default overlap copied from the tail of a previous chunk.
DEFAULT_OVERLAP_CHARS: int = 150

#: Sections shorter than this are merged with the following section.
DEFAULT_MIN_CHARS: int = 180

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of the knowledge base.

    Attributes:
        chunk_id: Stable identifier, ``"<source>#<n>"``.
        text: Chunk body without the heading line.
        source: POSIX path relative to the knowledge-base root.
        business: Top-level folder name, used to scope a dialogue.
        title: Document title (first level-1 heading).
        heading: Section heading this chunk came from.
        order: 1-based position of the chunk inside its document.
    """

    chunk_id: str
    text: str
    source: str
    business: str
    title: str
    heading: str
    order: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the chunk."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "business": self.business,
            "title": self.title,
            "heading": self.heading,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Chunk":
        """Rebuild a chunk from :meth:`as_dict` output."""
        return cls(
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            source=str(payload["source"]),
            business=str(payload["business"]),
            title=str(payload["title"]),
            heading=str(payload["heading"]),
            order=int(str(payload["order"])),
        )


@dataclass(frozen=True)
class Section:
    """A heading and the text that follows it."""

    heading: str
    body: str


@dataclass(frozen=True)
class Document:
    """A parsed knowledge-base document."""

    source: str
    business: str
    title: str
    sections: tuple[Section, ...]


def discover_documents(kb_dir: Path) -> list[Path]:
    """Return every supported knowledge-base file under *kb_dir*, sorted."""
    if not kb_dir.exists():
        return []
    files = [
        path
        for path in kb_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.as_posix())


def parse_document(path: Path, kb_dir: Path) -> Document:
    """Parse one Markdown/plain-text file into a :class:`Document`.

    The first level-1 heading becomes the document title; every heading of
    any level starts a new section. Text before the first heading becomes a
    section titled after the document title.
    """
    raw = path.read_text(encoding="utf-8")
    source = path.relative_to(kb_dir).as_posix()
    business = source.split("/", 1)[0] if "/" in source else "."

    lines = raw.splitlines()
    title = ""
    blocks: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            level, heading_text = len(match.group(1)), match.group(2).strip()
            if level == 1 and not title:
                title = heading_text
                continue
            blocks.append((current_heading or title or path.stem, current_lines))
            current_heading = heading_text
            current_lines = []
            continue
        current_lines.append(line)
    blocks.append((current_heading or title or path.stem, current_lines))

    sections: list[Section] = []
    for heading, body_lines in blocks:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(Section(heading=heading, body=body))

    return Document(
        source=source,
        business=business,
        title=title or path.stem,
        sections=tuple(sections),
    )


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Break a single oversized paragraph into sentence-sized pieces."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > max_chars:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip() if buffer else sentence
    if buffer:
        pieces.append(buffer)
    return pieces or [paragraph]


def _tail_overlap(text: str, overlap_chars: int) -> str:
    """Return the trailing part of *text* used as an overlap prefix.

    The cut is moved to the nearest sentence boundary when a reasonably
    close one exists, otherwise to the next word boundary.
    """
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text
    tail = text[-overlap_chars:]
    boundary = tail.find(". ")
    if boundary != -1 and boundary + 2 < len(tail):
        return tail[boundary + 2 :].strip()
    space = tail.find(" ")
    if space != -1:
        return tail[space + 1 :].strip()
    return tail.strip()


def split_section(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split a section body into overlapping pieces of at most *max_chars*.

    Paragraphs are kept intact where possible; the tail of a finished piece
    is repeated as the head of the next one so a sentence cut by the
    boundary is still retrievable from either side.
    """
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            paragraphs.extend(_split_long_paragraph(block, max_chars))
    if not paragraphs:
        return []

    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current)
            overlap = _tail_overlap(current, overlap_chars)
            current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    document: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Turn a parsed document into chunks, merging very small sections."""
    groups: list[tuple[str, str]] = []
    pending_headings: list[str] = []
    pending_body = ""
    for section in document.sections:
        pending_headings.append(section.heading)
        pending_body = f"{pending_body}\n\n{section.body}".strip() if pending_body else section.body
        if len(pending_body) >= min_chars:
            groups.append((" \u00b7 ".join(pending_headings), pending_body))
            pending_headings = []
            pending_body = ""
    if pending_body:
        groups.append((" \u00b7 ".join(pending_headings), pending_body))

    chunks: list[Chunk] = []
    order = 0
    for heading, body in groups:
        for piece in split_section(body, max_chars=max_chars, overlap_chars=overlap_chars):
            order += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}#{order}",
                    text=piece,
                    source=document.source,
                    business=document.business,
                    title=document.title,
                    heading=heading,
                    order=order,
                )
            )
    return chunks


def load_chunks(
    kb_dir: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Load and chunk every document in *kb_dir*.

    The result is deterministic: files are visited in sorted order and
    chunk identifiers depend only on the file path and position.
    """
    chunks: list[Chunk] = []
    for path in discover_documents(kb_dir):
        document = parse_document(path, kb_dir)
        chunks.extend(
            chunk_document(
                document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )
        )
    return chunks


def business_names(chunks: Iterable[Chunk]) -> list[str]:
    """Return the sorted list of business folders present in *chunks*."""
    return sorted({chunk.business for chunk in chunks})
