"""Hand-off rules: when a question must go to a human operator.

Escalation happens for three independent reasons:

* a **trigger keyword** -- the customer asks for an operator or raises a
  complaint, a refund, a court or Rospotrebnadzor;
* a **low retrieval score** -- the knowledge base does not contain a
  relevant fragment, so the bot must not invent an answer;
* **too many unresolved questions in a row** in the same chat.

The reason is stored with the ticket so ``stats`` can show why customers end
up with a human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .retrieval import ScoredChunk, fold

#: Cosine similarity below which an answer is considered unsupported.
#: Calibrated on the bundled knowledge base: relevant questions score
#: roughly 0.20-0.55, unrelated ones stay under 0.12.
DEFAULT_SCORE_THRESHOLD: float = 0.16

#: How many unresolved questions in a row trigger a hand-off.
DEFAULT_MAX_UNRESOLVED_STREAK: int = 3

#: Trigger words, matched at the beginning of any word of the question, so
#: that «оператора», «оператору» and «оператор» all match while «посуда»
#: does not match «суд».
TRIGGER_WORDS: tuple[str, ...] = (
    "оператор",
    "претенз",
    "роспотреб",
    "юрист",
    "адвокат",
    "суд",
    "руководител",
    "человек",
)

#: Stems matched anywhere in the question, because a Russian verb can carry a
#: prefix («по-жаловаться») that word-start matching would miss. Kept short
#: and unambiguous on purpose.
TRIGGER_STEMS: tuple[str, ...] = (
    "жалоб",
    "жалов",
    "жалу",
    "обман",
    "мошенн",
    "развод",
)

#: Multi-word phrases, matched as-is.
TRIGGER_PHRASES: tuple[str, ...] = (
    "верните деньги",
    "вернуть деньги",
    "возврат денег",
    "живой человек",
    "позовите человека",
)

#: Every trigger, in the order they are reported.
TRIGGER_KEYWORDS: tuple[str, ...] = TRIGGER_WORDS + TRIGGER_STEMS + TRIGGER_PHRASES

_WORD_RE = re.compile(r"[а-яёa-z]+")


def _match_keywords(
    question: str,
    words: Sequence[str],
    stems: Sequence[str],
    phrases: Sequence[str],
) -> tuple[str, ...]:
    """Return the trigger patterns found in *question*.

    Matching is done on the folded text (lower case, ``ё`` -> ``е``), which
    is why the patterns are written without ``ё``.
    """
    folded = fold(question)
    tokens = _WORD_RE.findall(folded)
    matched: list[str] = []
    for word in words:
        if any(token.startswith(word) for token in tokens):
            matched.append(word)
    for stem in stems:
        if stem in folded:
            matched.append(stem)
    for phrase in phrases:
        if phrase in folded:
            matched.append(phrase)
    return tuple(matched)

REASON_KEYWORD: str = "trigger_keyword"
REASON_LOW_SCORE: str = "low_score"
REASON_REPEAT: str = "repeated_unresolved"
REASON_NONE: str = "none"

#: Russian labels for the stored reason codes.
REASON_LABELS_RU: dict[str, str] = {
    REASON_KEYWORD: "триггерные слова (оператор, жалоба, суд)",
    REASON_LOW_SCORE: "нет ответа в базе знаний",
    REASON_REPEAT: "несколько непонятных вопросов подряд",
    REASON_NONE: "не требуется",
}


@dataclass(frozen=True)
class EscalationDecision:
    """Outcome of the hand-off check for a single question."""

    needed: bool
    reason: str
    reason_ru: str
    matched_keywords: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "needed": self.needed,
            "reason": self.reason,
            "reason_ru": self.reason_ru,
            "matched_keywords": list(self.matched_keywords),
        }


def matched_keywords(question: str) -> tuple[str, ...]:
    """Return the default trigger keywords found in *question*."""
    return _match_keywords(question, TRIGGER_WORDS, TRIGGER_STEMS, TRIGGER_PHRASES)


class EscalationPolicy:
    """Decides whether a question should be handed to an operator."""

    def __init__(
        self,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_unresolved_streak: int = DEFAULT_MAX_UNRESOLVED_STREAK,
        keywords: Sequence[str] = TRIGGER_KEYWORDS,
    ) -> None:
        self.score_threshold = score_threshold
        self.max_unresolved_streak = max_unresolved_streak
        self.keywords = tuple(keywords)
        self.trigger_words = tuple(word for word in self.keywords if " " not in word)
        self.trigger_phrases = tuple(keyword for keyword in self.keywords if " " in keyword)

    def key_matches(self, question: str) -> tuple[str, ...]:
        """Return the configured trigger keywords present in *question*."""
        return _match_keywords(question, self.trigger_words, self.trigger_words, self.trigger_phrases)

    def check(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        *,
        unresolved_streak: int = 0,
    ) -> EscalationDecision:
        """Decide whether *question* must be handed to a human.

        Args:
            question: The customer question.
            hits: Retrieval results for that question (best first).
            unresolved_streak: Number of consecutive unresolved questions in
                the same chat *before* this one.

        Returns:
            The decision, with a Russian explanation of the reason.
        """
        keywords = self.key_matches(question)
        if keywords:
            return EscalationDecision(
                needed=True,
                reason=REASON_KEYWORD,
                reason_ru=REASON_LABELS_RU[REASON_KEYWORD],
                matched_keywords=keywords,
            )
        top_score = hits[0].score if hits else 0.0
        if not hits or top_score < self.score_threshold:
            return EscalationDecision(
                needed=True,
                reason=REASON_LOW_SCORE,
                reason_ru=REASON_LABELS_RU[REASON_LOW_SCORE],
            )
        if unresolved_streak + 1 >= self.max_unresolved_streak:
            return EscalationDecision(
                needed=True,
                reason=REASON_REPEAT,
                reason_ru=REASON_LABELS_RU[REASON_REPEAT],
            )
        return EscalationDecision(
            needed=False,
            reason=REASON_NONE,
            reason_ru=REASON_LABELS_RU[REASON_NONE],
        )


def build_operator_message(
    question: str,
    hits: Sequence[ScoredChunk],
    decision: EscalationDecision,
    *,
    business: str | None = None,
    chat_id: str = "0",
    ticket_id: int | None = None,
) -> str:
    """Compose the message an operator receives with a ticket.

    The operator sees the question, the reason for the hand-off and the best
    knowledge-base fragments with their scores, so no context is lost.
    """
    lines = ["Новый вопрос от клиента — нужен оператор."]
    if ticket_id is not None:
        lines.append(f"Тикет: #{ticket_id}")
    if business and business != ".":
        lines.append(f"Направление: {business}")
    lines.append(f"Чат: {chat_id}")
    lines.append(f"Причина: {decision.reason_ru}")
    if decision.matched_keywords:
        lines.append("Слова-триггеры: " + ", ".join(decision.matched_keywords))
    lines.append("")
    lines.append(f"Вопрос: {question}")
    if hits:
        lines.append("")
        lines.append("Лучшие фрагменты базы знаний:")
        for number, hit in enumerate(hits[:3], start=1):
            lines.append(
                f"  {number}. {hit.chunk.source} -> «{hit.chunk.heading}» "
                f"(релевантность {hit.score:.2f})"
            )
    else:
        lines.append("")
        lines.append("В базе знаний нет ни одного релевантного фрагмента.")
    return "\n".join(lines)


def customer_handoff_text(ticket_id: int, *, business: str | None = None) -> str:
    """Russian message shown to the customer when the question is handed off."""
    lines = [
        "Передаю ваш вопрос оператору — он ответит в рабочее время.",
        f"Номер обращения: #{ticket_id}.",
    ]
    if business and business != ".":
        lines.append("Чтобы ускорить ответ, не закрывайте этот чат: оператор увидит историю переписки.")
    return "\n".join(lines)
