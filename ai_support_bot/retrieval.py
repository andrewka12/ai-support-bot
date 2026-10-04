"""Pure-standard-library retrieval over the knowledge base.

The implementation is a classic TF-IDF vector-space model with cosine
similarity. No external service, no embeddings and no network access are
involved, so the demo runs fully offline and deterministically.

Russian text processing consists of four honest heuristics:

1. **Normalisation** -- lower-casing, ``ё`` folded to ``е``, everything but
   Cyrillic/Latin letters dropped (digits are treated as noise).
2. **A hand-written stop-word list** for Russian function words.
3. **Light stemming** -- trimming the most common Russian endings with a
   minimum stem length. This is a heuristic, not linguistics: it reduces
   ``цена/цены/цене`` to one form, but it cannot relate ``вернуть`` and
   ``возвращают``, because they differ in the prefix, not the ending.
4. **Domain synonym groups** (:data:`SYNONYM_GROUPS`) -- a hand-written map
   from a canonical business term (``возврат``, ``доставка``, ``гарантия``,
   ``цена``, ...) to the word forms that mean it. :func:`expand_terms`
   appends the canonical term to a text's tokens, so a question phrased with
   ``вернуть`` still reaches the document called «Возврат и обмен».

Step 4 is a deliberate compromise: it is a few dozen curated pairs instead of
a trained embedding model, and it only covers the vocabulary of this
knowledge base. It is documented in the README («Ограничения»), and the
hand-off threshold compensates for the cases it still misses.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .chunking import Chunk

#: Index cache format version; bump when the on-disk layout changes.
INDEX_FORMAT_VERSION: int = 2

_TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"^[а-яё]+$")

#: Russian function words. Written by hand; deliberately kept short enough
#: to be readable but long enough to remove the noise from short questions.
STOPWORDS: frozenset[str] = frozenset(
    """
    а без бы был была были было быть в вам вас ведь весь вo вот все всего всех вы
    где да даже для до его ее ей ему если есть еще ж же за и из или им их к как
    какая какие который кто ли либо мне может мои мой мы на над надо нам нас не
    него нее ней нет ни них но ну о об обо он она они оно от очень по под пока
    после потом потому при про раз с сам свою себе себя сейчас со т так также
    такой там тебе тебя тем то тобою тогда того тоже той только том тот тут ты у
    уж уже чего чем через что чтоб чтобы эта эти это этой этом этот эту я
    является который которые которая
    вопрос вопросы вопросов скажите подскажите пожалуйста спасибо здравствуйте
    можно нужно нужен нужна нужны даете даёт дай сколько скольк
    когда почему зачем куда откуда чей чья чьё чье какие какое каком какую
    """.split()
)

#: Minimal endings trimmed by the light stemmer, longest first.
#: Verb endings are included on purpose: without them ``работа``, ``работы``
#: and ``работаете`` never meet, which loses questions like «вы работаете в
#: субботу?». They are written with their stem vowel (``-ает``, ``-ать``,
#: ``-ить``) so short nouns such as ``билет`` are not damaged.
_ENDINGS: tuple[str, ...] = tuple(
    sorted(
        (
            "аться", "иться", "ться", "аете", "аешь", "иями", "ями", "ами",
            "ией", "иях", "ость", "ости", "остью", "ает", "аем", "ают",
            "ать", "ить", "ила", "или",
            "ыми", "ими", "ого", "его", "ому", "ему", "ых", "их", "ой", "ей",
            "ам", "ям", "ах", "ях", "ом", "ем", "ов", "ев", "ую", "юю",
            "ию", "ью", "ия", "ья", "ий", "ый", "ая", "яя", "ое", "ее",
            "ые", "ие", "у", "ю", "а", "я", "о", "е", "ы", "и", "ь", "й",
        ),
        key=len,
        reverse=True,
    )
)

#: Domain synonym groups: canonical term -> word forms that mean it.
#: Written by hand for the vocabulary of the bundled knowledge bases.
#: A form may belong to several groups; :func:`expand_terms` adds them all.
SYNONYM_GROUPS: Mapping[str, tuple[str, ...]] = {
    "возврат": (
        "возврат", "вернуть", "верну", "вернёт", "вернет", "вернёте", "вернете",
        "вернут", "верните", "возвращают", "возвращает", "возвращаем",
        "возвращение", "возвращаться", "обмен", "обменять", "обмену", "обмена",
        "обменом", "отказ", "откажусь", "отказаться", "отказываюсь",
    ),
    "оплата": (
        "оплата", "оплатить", "оплачивается", "заплатить", "платить", "платёж",
        "платеж", "предоплата", "счёт", "счет", "наличные", "сбп", "рассрочка",
        "рассрочку", "кредит", "квитанция",
    ),
    "деньги": ("деньги", "денег", "деньгам", "деньгами", "деньгах"),
    "цена": (
        "цена", "цены", "цену", "цене", "цен", "стоимость", "стоит", "стоят",
        "стоить", "прайс", "расценки", "расценка", "тариф", "тарифы",
    ),
    "срок": (
        "срок", "сроки", "сроков", "срока", "сроку", "длительность", "долго",
        "быстро", "ждать", "занимает",
    ),
    "гарантия": (
        "гарантия", "гарантии", "гарантию", "гарантийный", "гарантийная",
        "гарантийное", "гарантийные", "гарантийного",
    ),
    "доставка": (
        "доставка", "доставки", "доставку", "доставить", "доставлять",
        "доставляем", "доставляет", "доставляют", "привезти", "привезёте",
        "привезете", "привезут", "отправка", "отправить", "курьер", "курьером",
        "пересылка", "транспортная",
    ),
    "запись": (
        "запись", "записи", "записью", "записаться", "записываться",
        "записаться", "забронировать", "бронь", "записывает", "приём", "прием",
        "приёма", "приема",
    ),
    "брак": (
        "брак", "брака", "бракованный", "дефект", "дефекты", "неисправность",
        "неисправности", "неисправный", "поломка", "поломки", "сломался",
        "сломан", "повреждение", "повреждения", "повреждённый",
        "поврежденный", "поврежденным", "повреждённым",
    ),
    "имплант": (
        "имплант", "импланты", "импланта", "имплантов", "имплантация",
        "имплантации", "имплантацию", "имплантацией", "имплантировать",
    ),
    "противопоказание": (
        "противопоказание", "противопоказания", "противопоказаний",
        "противопоказаниям", "ограничение", "ограничения", "ограничений",
    ),
    "лечение": (
        "лечение", "лечения", "лечению", "лечить", "лечу", "полечить",
        "терапия", "терапии", "пломба", "пломбы", "пломбу",
    ),
    "анестезия": ("анестезия", "анестезии", "наркоз", "обезболивание"),
    "отмена": ("отмена", "отменить", "отмену", "перенос", "перенести", "переносить"),
}

#: Tokens shorter than this are dropped after normalisation.
MIN_TOKEN_LENGTH: int = 3

#: Additive bonus for a chunk whose *document title* matches the question.
#: TF-IDF alone cannot see that «вернуть товар» is the topic of the document
#: called «возврат и обмен», while a single section of another file may
#: mention the same words by accident. Measured on the bundled knowledge
#: base: any value in 0.10-0.30 gives the same ranking, 0.15 is the middle.
TITLE_PRIOR_WEIGHT: float = 0.15

INDEX_CACHE_NAME: str = "index.json"


def fold(text: str) -> str:
    """Lower-case *text* and fold ``ё`` to ``е``."""
    return text.lower().replace("ё", "е")


def stem(word: str) -> str:
    """Trim Russian endings from *word* until it stops changing.

    Words shorter than four letters are returned unchanged, and every stem
    keeps at least three letters. Trimming repeats while an ending still
    applies, which makes the function idempotent: ``stem(stem(word))`` is
    always ``stem(word)``. Without that property a token could be shortened a
    second time when it is normalised again (``России`` -> ``росси`` ->
    ``росс``), and equal words would stop matching each other.

    This is a deliberately simple heuristic, not a morphological analyser --
    see the module docstring.
    """
    if len(word) <= 3:
        return word
    current = word
    for _ in range(4):  # A safety bound; real stems stabilise after 1-2 passes.
        for ending in _ENDINGS:
            if current.endswith(ending) and len(current) - len(ending) >= 3:
                current = current[: len(current) - len(ending)]
                break
        else:
            return current
    return current


def _build_synonym_lookup() -> dict[str, tuple[str, ...]]:
    """Invert :data:`SYNONYM_GROUPS` into ``stem -> canonical stems``."""
    lookup: dict[str, list[str]] = {}
    for canonical, forms in SYNONYM_GROUPS.items():
        target = stem(fold(canonical))
        for form in forms:
            key = stem(fold(form))
            bucket = lookup.setdefault(key, [])
            if target not in bucket:
                bucket.append(target)
    return {key: tuple(values) for key, values in lookup.items()}


_SYNONYM_LOOKUP: Mapping[str, tuple[str, ...]] = _build_synonym_lookup()


def base_token(word: str) -> str:
    """Return the stem of *word*, mapped to its canonical domain term."""
    key = stem(fold(word))
    if not _CYRILLIC_RE.match(key):
        return key
    canonicals = _SYNONYM_LOOKUP.get(key)
    return canonicals[0] if canonicals else key


def expand_terms(tokens: Sequence[str]) -> list[str]:
    """Append the canonical domain terms of *tokens*.

    The light stemmer only removes endings, so it cannot relate ``вернуть``
    to ``возвращают``. This function maps every token to its canonical
    domain term and appends the extra canonical forms, which lets a question
    reach a document that uses different wording. It is applied to both
    questions and documents, so the mapping is symmetric, and it is
    idempotent: normalised tokens pass through unchanged.

    Examples:
        >>> expand_terms(["вернуть", "товар"])
        ['возврат', 'товар']
        >>> expand_terms(["товар", "деньги"])
        ['товар', 'деньг']
    """
    normalised = [base_token(token) for token in tokens]
    expanded = list(normalised)
    present = set(normalised)
    for token in normalised:
        for canonical in _SYNONYM_LOOKUP.get(token, ()):
            if canonical not in present:
                present.add(canonical)
                expanded.append(canonical)
    return expanded


def tokenize(text: str) -> list[str]:
    """Split *text* into normalised content tokens.

    Punctuation and digits are treated as noise and removed, stop words are
    dropped, and the remaining words are stemmed, mapped to their canonical
    domain term and finally expanded with :func:`expand_terms`.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(fold(text)):
        if raw in STOPWORDS or len(raw) < MIN_TOKEN_LENGTH:
            continue
        token = base_token(raw)
        if token and token not in STOPWORDS and len(token) >= MIN_TOKEN_LENGTH:
            tokens.append(token)
    return expand_terms(tokens)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk together with its cosine similarity to the query."""

    chunk: Chunk
    score: float


def index_text(chunk: Chunk) -> str:
    """Return the searchable text of a chunk.

    The section heading and the document title are repeated so that they
    weigh more than the body: they are short and highly discriminative.
    """
    return f"{chunk.heading} {chunk.heading} {chunk.title} {chunk.text}"


@dataclass
class TfidfIndex:
    """A TF-IDF vector-space index over a list of chunks."""

    chunks: list[Chunk]
    idf: dict[str, float]
    vectors: list[dict[str, float]]
    #: Cache of tokenised document titles, used by the topic prior.
    _title_tokens: dict[str, frozenset[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunks)

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct terms in the index."""
        return len(self.idf)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        businesses: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        """Return the best matching chunks for *query*, highest score first.

        The score is a cosine similarity plus a small *topic prior*::

            score = cosine(chunk, query)
                  + TITLE_PRIOR_WEIGHT * |query terms in document title| / |query terms|

        The prior exists because a keyword model cannot tell that «как вернуть
        товар» is the subject of the document called «возврат и обмен», while
        a single section of another file («Когда товар меняют или возвращают
        деньги») merely shares its words. On the bundled knowledge base the
        prior lifts exact-document accuracy from 11/12 to 12/12 and is stable
        for any weight between 0.10 and 0.30.

        Args:
            query: Free-form customer question in Russian.
            top_k: Maximum number of results to return.
            businesses: Optional whitelist of business folders; chunks of
                other businesses are ignored.

        Returns:
            Chunks with a positive score, sorted by descending score and
            then by chunk id, so the order is fully deterministic.
        """
        query_terms = tokenize(query)
        query_vector = self._vectorize(query_terms)
        if not query_vector:
            return []
        allowed = set(businesses) if businesses is not None else None
        title_sets = self._title_token_sets()
        distinct_query_terms = set(query_terms)

        scored: list[ScoredChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if allowed is not None and chunk.business not in allowed:
                continue
            score = _dot(query_vector, vector)
            if score <= 0.0:
                continue
            matches = len(distinct_query_terms & title_sets.get(chunk.source, frozenset()))
            prior = TITLE_PRIOR_WEIGHT * matches / len(distinct_query_terms)
            scored.append(ScoredChunk(chunk=chunk, score=score + prior))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]

    def _title_token_sets(self) -> dict[str, frozenset[str]]:
        """Return the tokenised document titles, computed once per index."""
        if not self._title_tokens:
            self._title_tokens.update(
                {
                    chunk.source: frozenset(tokenize(chunk.title))
                    for chunk in self.chunks
                }
            )
        return self._title_tokens

    def _vectorize(self, tokens: Sequence[str]) -> dict[str, float]:
        """Build an L2-normalised TF-IDF vector for *tokens*."""
        if not tokens:
            return {}
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        weights = {
            term: (1.0 + math.log(count)) * self.idf.get(term, self._default_idf())
            for term, count in counts.items()
        }
        return _normalize(weights)

    def _default_idf(self) -> float:
        """IDF assigned to a query term missing from the index vocabulary."""
        return math.log((self.size + 1) / 1.0) + 1.0


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    """Return *weights* scaled to unit L2 norm."""
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if norm == 0.0:
        return dict(weights)
    return {term: value / norm for term, value in weights.items()}


def _dot(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine similarity of two L2-normalised sparse vectors."""
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def build_index(chunks: Sequence[Chunk]) -> TfidfIndex:
    """Build a TF-IDF index over *chunks*.

    Term frequency is sub-linear (``1 + log(tf)``) and inverse document
    frequency is smoothed (``log((N + 1) / (df + 1)) + 1``), which keeps
    the weights stable for the small corpus of a single knowledge base.
    """
    chunk_list = list(chunks)
    document_tokens = [tokenize(index_text(chunk)) for chunk in chunk_list]
    total = len(chunk_list)

    document_frequency: dict[str, int] = {}
    for tokens in document_tokens:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    idf = {
        term: math.log((total + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }

    vectors: list[dict[str, float]] = []
    for tokens in document_tokens:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        weights = {term: (1.0 + math.log(count)) * idf[term] for term, count in counts.items()}
        vectors.append(_normalize(weights))

    return TfidfIndex(chunks=chunk_list, idf=idf, vectors=vectors)


def search(
    index: TfidfIndex,
    query: str,
    *,
    top_k: int = 3,
    businesses: Iterable[str] | None = None,
) -> list[ScoredChunk]:
    """Convenience wrapper around :meth:`TfidfIndex.search`."""
    return index.search(query, top_k=top_k, businesses=businesses)


def kb_signature(kb_dir: Path) -> dict[str, list[int]]:
    """Return a cheap fingerprint of the knowledge-base files.

    Used to decide whether a cached index is still valid: the mapping from
    relative path to ``[mtime_ns, size]`` changes whenever a file is edited.
    """
    from .chunking import discover_documents

    signature: dict[str, list[int]] = {}
    for path in discover_documents(kb_dir):
        stat = path.stat()
        signature[path.relative_to(kb_dir).as_posix()] = [stat.st_mtime_ns, stat.st_size]
    return signature


def save_index(index: TfidfIndex, path: Path, *, signature: Mapping[str, object] | None = None) -> None:
    """Persist *index* to *path* as JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": INDEX_FORMAT_VERSION,
        "signature": dict(signature) if signature is not None else {},
        "chunks": [chunk.as_dict() for chunk in index.chunks],
        "idf": index.idf,
        "vectors": index.vectors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(path: Path, *, expected_signature: Mapping[str, object] | None = None) -> TfidfIndex | None:
    """Load a cached index, or return ``None`` when it is unusable.

    The cache is rejected when the format version differs or when
    *expected_signature* was supplied and does not match the stored one.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("format") != INDEX_FORMAT_VERSION:
        return None
    if expected_signature is not None and payload.get("signature") != dict(expected_signature):
        return None
    try:
        chunks = [Chunk.from_dict(item) for item in payload["chunks"]]
        idf = {str(term): float(value) for term, value in payload["idf"].items()}
        vectors = [
            {str(term): float(value) for term, value in vector.items()}
            for vector in payload["vectors"]
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if len(chunks) != len(vectors):
        return None
    return TfidfIndex(chunks=chunks, idf=idf, vectors=vectors)
