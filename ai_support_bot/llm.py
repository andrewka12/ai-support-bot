"""Answer providers: an offline template composer and an LLM client.

:class:`EchoProvider` is the default. It runs without a network, without an
API key and without tokens: it composes an answer from the retrieved
knowledge-base fragments using Russian templates, and reports a confidence
that is derived from the retrieval score and the query coverage. The demo
in ``chat`` mode therefore produces genuinely useful answers offline.

:class:`OpenAICompatProvider` talks to any OpenAI-compatible
``/chat/completions`` endpoint (OpenAI, DeepSeek, OpenRouter, a local Ollama)
using only :mod:`urllib`. It is selected automatically when ``LLM_BASE_URL``
and ``LLM_MODEL`` are present in the environment, or explicitly with
``--provider openai``.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .prompts import DEFAULT_CONTEXT_CHUNKS, NO_ANSWER_RU, build_messages
from .retrieval import ScoredChunk, tokenize

#: Cosine similarity that is treated as a "very good" match when scaling
#: confidence into the 0..1 range for display.
REFERENCE_SCORE: float = 0.5

#: Extra fragments are only cited when they are nearly as relevant as the best.
SUPPORTING_SCORE_RATIO: float = 0.8

#: Maximum number of knowledge-base sentences quoted in one offline answer.
MAX_QUOTED_SENTENCES: int = 3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_PRICE_HINTS = frozenset({"цена", "цен", "стоимость", "прайс", "сколько"})
_TIME_HINTS = frozenset({"срок", "когда", "быстро", "ждать", "время"})


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce an answer."""


@dataclass(frozen=True)
class AnswerSource:
    """A knowledge-base fragment cited in an answer."""

    source: str
    heading: str
    score: float

    def as_label(self) -> str:
        """Human-readable Russian label used in the answer footer."""
        return f"{self.source} \u2192 \u00ab{self.heading}\u00bb"


@dataclass(frozen=True)
class Answer:
    """An answer produced by a provider."""

    text: str
    confidence: float
    provider: str
    sources: tuple[AnswerSource, ...] = field(default=())

    @property
    def has_sources(self) -> bool:
        """Whether the answer quotes the knowledge base."""
        return bool(self.sources)

    def source_label(self) -> str:
        """All cited fragments joined into a single line."""
        return "; ".join(item.as_label() for item in self.sources)


class LLMProvider(Protocol):
    """Minimal interface every answer provider implements."""

    name: str

    def answer(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        *,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        """Produce an answer for *question* given retrieved *hits*."""


def _split_units(text: str) -> list[str]:
    """Split chunk text into quotable units (lines, then sentences)."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def _unit_score(
    unit_tokens: Sequence[str],
    query_tokens: set[str],
    idf: Mapping[str, float] | None = None,
) -> float:
    """Score a unit by the weight of the query terms it contains.

    Terms are weighted by their inverse document frequency when the index is
    available, so a rare word such as «Казань» counts for much more than the
    word «доставка», which appears in the whole section. The sum is
    normalised by the square root of the unit length to avoid favouring very
    long sentences.
    """
    if not unit_tokens:
        return 0.0
    matched = query_tokens & set(unit_tokens)
    if not matched:
        return 0.0
    if idf is None:
        weight = float(len(matched))
    else:
        default = max(idf.values(), default=1.0)
        weight = sum(idf.get(term, default) for term in matched)
    return weight / math.sqrt(len(unit_tokens))


def _pick_sentences(
    hits: Sequence[ScoredChunk],
    query_tokens: set[str],
    limit: int,
    idf: Mapping[str, float] | None = None,
) -> list[tuple[ScoredChunk, str]]:
    """Choose the most relevant sentences across the retrieved fragments.

    Sentences are pooled from every fragment and ranked by query-term
    overlap normalised by length, so a sentence from the second-best
    fragment can still be quoted when it answers the question better. The
    selected sentences are returned in document order.
    """
    candidates: list[tuple[float, int, int, ScoredChunk, str]] = []
    for rank, hit in enumerate(hits):
        for position, unit in enumerate(_split_units(hit.chunk.text)):
            score = _unit_score(tokenize(unit), query_tokens, idf)
            if score > 0.0:
                candidates.append((score, rank, position, hit, unit))
    if not candidates:
        first = hits[0]
        return [(first, unit) for unit in _split_units(first.chunk.text)[: min(2, limit)]]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = sorted(candidates[:limit], key=lambda item: (item[1], item[2]))
    return [(hit, unit) for _, _, _, hit, unit in selected]


def _intro(query_tokens: set[str], *, found: bool) -> str:
    """Pick a Russian lead-in for the composed answer."""
    if not found:
        return "В базе знаний нет точного ответа, но вот что удалось найти:"
    if query_tokens & _PRICE_HINTS:
        return "По ценам из базы знаний:"
    if query_tokens & _TIME_HINTS:
        return "По срокам из базы знаний:"
    return "Вот что есть в базе знаний по вашему вопросу:"


class EchoProvider:
    """Offline provider that answers by quoting the knowledge base.

    The provider never invents content: every sentence of the answer is a
    quote from a retrieved chunk. Confidence combines how well the query
    matched (cosine similarity) with how much of the query the matched
    chunk actually covers.

    Args:
        max_sentences: How many knowledge-base sentences one answer may
            quote.
        idf: Term weights of the retrieval index. They are used to prefer
            sentences containing rare query words; without them every
            matched word counts the same.
    """

    name = "echo"

    def __init__(
        self,
        *,
        max_sentences: int = MAX_QUOTED_SENTENCES,
        idf: Mapping[str, float] | None = None,
    ) -> None:
        self._max_sentences = max_sentences
        self._idf = idf

    def answer(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        *,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        """Compose an answer from *hits*; fall back to the hand-off text."""
        del history  # The offline composer answers the current question only.
        if not hits:
            return Answer(text=NO_ANSWER_RU, confidence=0.0, provider=self.name)

        best = hits[0]
        query_tokens = set(tokenize(question))
        best_tokens = set(tokenize(best.chunk.text))
        best_heading_tokens = set(tokenize(best.chunk.heading))
        coverage = (
            len(query_tokens & best_tokens) / len(query_tokens) if query_tokens else 0.0
        )

        # Fragments that are nearly as relevant as the best one are quoted too:
        # for a keyword model a question is often split across two sections.
        supporting = [
            hit for hit in hits[1:] if hit.score >= best.score * SUPPORTING_SCORE_RATIO
        ][: DEFAULT_CONTEXT_CHUNKS - 1]
        cited = [best, *supporting]

        picked = _pick_sentences(cited, query_tokens, self._max_sentences, self._idf)
        found = bool(picked) and bool(query_tokens & (best_tokens | best_heading_tokens))
        body = " ".join(unit for _, unit in picked).strip()
        if not found or not body:
            return Answer(text=NO_ANSWER_RU, confidence=0.0, provider=self.name)

        used: list[ScoredChunk] = []
        for hit, _ in picked:
            if all(existing.chunk.chunk_id != hit.chunk.chunk_id for existing in used):
                used.append(hit)
        sources = tuple(
            AnswerSource(source=hit.chunk.source, heading=hit.chunk.heading, score=hit.score)
            for hit in used
        )

        text = f"{_intro(query_tokens, found=True)}\n\n{body}\n\nИсточник: {sources[0].as_label()}"
        if len(sources) > 1:
            extra = "; ".join(item.as_label() for item in sources[1:])
            text += f"\nДополнительно: {extra}"

        confidence = 0.5 * coverage + 0.5 * min(1.0, best.score / REFERENCE_SCORE)
        return Answer(
            text=text,
            confidence=max(0.0, min(0.99, confidence)),
            provider=self.name,
            sources=sources,
        )


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible chat-completions endpoint.

    Only :mod:`urllib` is used, so the project keeps a zero-dependency
    install. The system prompt is the same guardrail text that documents the
    offline behaviour; retrieved fragments are injected into the user turn.
    """

    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        if not base_url or not model:
            raise ProviderError("Не заданы LLM_BASE_URL и LLM_MODEL.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._temperature = temperature

    @classmethod
    def from_env(cls) -> "OpenAICompatProvider | None":
        """Build a provider from environment variables, or return ``None``."""
        base_url = os.environ.get("LLM_BASE_URL", "").strip()
        model = os.environ.get("LLM_MODEL", "").strip()
        if not base_url or not model:
            return None
        return cls(
            base_url=base_url,
            api_key=os.environ.get("LLM_API_KEY", "").strip(),
            model=model,
        )

    def answer(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        *,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        """Call the endpoint and return its answer with retrieval-based confidence."""
        if not hits:
            return Answer(text=NO_ANSWER_RU, confidence=0.0, provider=self.name)

        messages = build_messages(question, hits, history=history)
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": False,
        }
        request = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(f"LLM вернул ошибку {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ProviderError(f"Не удалось связаться с LLM: {error}") from error

        try:
            data = json.loads(raw)
            text = data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, AttributeError) as error:
            raise ProviderError("Неожиданный ответ LLM, не удалось разобрать JSON.") from error
        if not text:
            raise ProviderError("LLM вернул пустой ответ.")

        sources = tuple(
            AnswerSource(source=hit.chunk.source, heading=hit.chunk.heading, score=hit.score)
            for hit in hits[:DEFAULT_CONTEXT_CHUNKS]
        )
        confidence = 0.6 + 0.4 * min(1.0, hits[0].score / REFERENCE_SCORE)
        return Answer(
            text=text,
            confidence=max(0.0, min(0.99, confidence)),
            provider=self.name,
            sources=sources,
        )

    def describe(self) -> str:
        """Short Russian description used by the CLI banner."""
        return f"OpenAI-совместимый endpoint: {self._base_url}, модель {self._model}"


def select_provider(
    name: str = "auto",
    *,
    idf: Mapping[str, float] | None = None,
) -> tuple[LLMProvider, str | None]:
    """Choose a provider.

    Args:
        name: ``"auto"``, ``"echo"`` or ``"openai"``.
        idf: Term weights of the retrieval index, passed to the offline
            provider so it can weight rare query words.

    Returns:
        A ``(provider, warning)`` pair. The warning is a Russian message
        explaining a fallback, or ``None`` when the requested provider was
        created successfully.
    """
    requested = name.strip().lower()
    if requested in {"echo", "offline"}:
        return EchoProvider(idf=idf), None
    if requested in {"openai", "llm"}:
        provider = OpenAICompatProvider.from_env()
        if provider is None:
            return (
                EchoProvider(idf=idf),
                "Не заданы переменные LLM_BASE_URL и LLM_MODEL — "
                "использую офлайн-режим (EchoProvider).",
            )
        return provider, None
    provider = OpenAICompatProvider.from_env()
    if provider is not None:
        return provider, None
    return EchoProvider(idf=idf), None


__all__ = [
    "Answer",
    "AnswerSource",
    "EchoProvider",
    "LLMProvider",
    "OpenAICompatProvider",
    "ProviderError",
    "select_provider",
]
