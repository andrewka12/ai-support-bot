"""Shared fixtures for the test suite."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_support_bot.chunking import Chunk, load_chunks  # noqa: E402
from ai_support_bot.retrieval import TfidfIndex, build_index  # noqa: E402

KB_DIR = PROJECT_ROOT / "ai_support_bot" / "kb"

#: Scratch directory for tests, git-ignored. Tests do not use the operating
#: system temp directory because restricted Windows environments may forbid
#: deleting files there.
WORKSPACE_TMP = PROJECT_ROOT / "tests" / "_tmp"

#: Fixed reference moment used by the analytics tests.
FIXED_NOW = datetime(2026, 1, 10, 18, 0, 0, tzinfo=timezone.utc)


def make_temp_dir(case: unittest.TestCase) -> Path:
    """Create a fresh scratch directory and remove it after the test.

    The directory is created with :meth:`Path.mkdir` rather than
    :func:`tempfile.mkdtemp` so that it inherits the permissions of the
    project directory; restricted Windows environments may refuse writes
    into a directory created with a private mode.
    """
    WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE_TMP / f"case-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    case.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


@lru_cache(maxsize=1)
def bundled_chunks() -> tuple[Chunk, ...]:
    """Chunks of the bundled knowledge base (parsed once per test run)."""
    return tuple(load_chunks(KB_DIR))


@lru_cache(maxsize=1)
def bundled_index() -> TfidfIndex:
    """Retrieval index of the bundled knowledge base (built once)."""
    return build_index(bundled_chunks())


class ProjectTestCase(unittest.TestCase):
    """Base class exposing the bundled knowledge base and small helpers."""

    kb_dir: Path = KB_DIR

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Chunks of the bundled knowledge base."""
        return bundled_chunks()

    @property
    def index(self) -> TfidfIndex:
        """Retrieval index of the bundled knowledge base."""
        return bundled_index()

    def top_business(self, question: str, business: str | None = None) -> str:
        """Business folder of the best hit for *question*."""
        hits = self.index.search(question, top_k=1, businesses=[business] if business else None)
        self.assertTrue(hits, f"Нет результатов для вопроса: {question}")
        return hits[0].chunk.business

    def top_source(self, question: str, business: str | None = None) -> str:
        """Source file of the best hit for *question*."""
        hits = self.index.search(question, top_k=1, businesses=[business] if business else None)
        self.assertTrue(hits, f"Нет результатов для вопроса: {question}")
        return hits[0].chunk.source
