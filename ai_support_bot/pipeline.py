"""The dialogue pipeline tying retrieval, guardrails and storage together.

Both front-ends -- the offline console chat and the real Telegram bot -- call
:class:`SupportPipeline`. Keeping the flow in one place means the console demo
and the production bot behave identically, which is what makes the offline
demo honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .escalation import (
    REASON_LOW_SCORE,
    EscalationDecision,
    EscalationPolicy,
    build_operator_message,
    customer_handoff_text,
)
from .llm import Answer, AnswerSource, EchoProvider, LLMProvider, ProviderError
from .prompts import build_retrieval_query
from .retrieval import ScoredChunk, TfidfIndex
from .storage import DialogueStore

#: How many previous turns are kept per chat for follow-up questions.
HISTORY_LIMIT: int = 6


@dataclass(frozen=True)
class BotReply:
    """What the bot decided to answer, and why."""

    text: str
    confidence: float
    resolved: bool
    provider: str
    hits: tuple[ScoredChunk, ...] = ()
    sources: tuple[AnswerSource, ...] = ()
    escalation_reason: str = ""
    escalation_reason_ru: str = ""
    ticket_id: int | None = None
    operator_message: str = ""
    matched_keywords: tuple[str, ...] = ()

    @property
    def top_score(self) -> float:
        """Retrieval score of the best matching fragment (0 when none)."""
        return self.hits[0].score if self.hits else 0.0


@dataclass
class _ChatState:
    """Per-chat conversation state."""

    history: list[tuple[str, str]] = field(default_factory=list)
    unresolved_streak: int = 0


class SupportPipeline:
    """Answer customer questions with retrieval, guardrails and hand-off."""

    def __init__(
        self,
        index: TfidfIndex,
        provider: LLMProvider | None = None,
        *,
        store: DialogueStore | None = None,
        policy: EscalationPolicy | None = None,
        top_k: int = 3,
        businesses: Iterable[str] | None = None,
    ) -> None:
        self.index = index
        self.provider = provider or EchoProvider(idf=index.idf)
        self.store = store
        self.policy = policy or EscalationPolicy()
        self.top_k = top_k
        self.businesses = tuple(businesses) if businesses is not None else None
        self._states: dict[str, _ChatState] = {}

    # ------------------------------------------------------------------ public

    def answer(
        self,
        question: str,
        *,
        chat_id: str = "local",
        business: str | None = None,
    ) -> BotReply:
        """Answer *question* for *chat_id*.

        The method retrieves fragments, applies the hand-off rules, asks the
        provider for an answer, logs everything to the dialogue store and
        updates the per-chat state.
        """
        state = self._states.setdefault(str(chat_id), _ChatState())
        scoped = (business,) if business else self.businesses
        retrieval_query = build_retrieval_query(question, state.history)
        hits = tuple(
            self.index.search(retrieval_query, top_k=self.top_k, businesses=scoped)
        )

        decision = self.policy.check(
            question, hits, unresolved_streak=state.unresolved_streak
        )
        if decision.needed:
            return self._hand_off(question, hits, decision, chat_id=str(chat_id))

        answer = self._ask_provider(question, hits, state)
        if not answer.has_sources:
            # Retrieval matched only the heading: nothing to quote honestly.
            fallback_decision = EscalationDecision(
                needed=True,
                reason=REASON_LOW_SCORE,
                reason_ru="в базе знаний нет готового ответа на этот вопрос",
            )
            return self._hand_off(question, hits, fallback_decision, chat_id=str(chat_id))

        state.unresolved_streak = 0
        self._remember(state, question, answer.text)
        if self.store is not None:
            self.store.log_message(str(chat_id), "user", question)
            self.store.log_answer(
                str(chat_id),
                question,
                answer.text,
                answer.confidence,
                answer.provider,
                source=answer.source_label(),
                resolved=True,
            )
            self.store.log_message(str(chat_id), "bot", answer.text)
        return BotReply(
            text=answer.text,
            confidence=answer.confidence,
            resolved=True,
            provider=answer.provider,
            hits=hits,
            sources=answer.sources,
        )

    def reset(self, chat_id: str) -> None:
        """Forget the conversation state of one chat."""
        self._states.pop(str(chat_id), None)

    def unresolved_streak(self, chat_id: str) -> int:
        """Number of consecutive unresolved questions in a chat."""
        state = self._states.get(str(chat_id))
        return state.unresolved_streak if state else 0

    def history(self, chat_id: str) -> tuple[tuple[str, str], ...]:
        """Recent ``(role, text)`` pairs of a chat."""
        state = self._states.get(str(chat_id))
        return tuple(state.history) if state else ()

    def describe(self) -> str:
        """Short Russian description of the pipeline configuration."""
        return (
            f"провайдер {self.provider.name}, фрагментов в индексе {self.index.size}, "
            f"порог эскалации {self.policy.score_threshold:.2f}"
        )

    # ----------------------------------------------------------------- private

    def _ask_provider(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        state: _ChatState,
    ) -> Answer:
        """Ask the configured provider, falling back to the offline one.

        A failing LLM endpoint must never leave the customer without an
        answer, so any :class:`ProviderError` degrades to the offline
        composer; the degradation is visible in the answer text and in the
        ``echo-fallback`` provider name stored with it.
        """
        try:
            return self.provider.answer(question, hits, history=state.history)
        except ProviderError as error:
            offline = EchoProvider(idf=self.index.idf)
            fallback = offline.answer(question, hits, history=state.history)
            return Answer(
                text=(
                    f"{fallback.text}\n\n"
                    f"(LLM недоступен: {error}. Ответ собран офлайн по базе знаний.)"
                ),
                confidence=fallback.confidence,
                provider="echo-fallback",
                sources=fallback.sources,
            )

    def _hand_off(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        decision: EscalationDecision,
        *,
        chat_id: str,
    ) -> BotReply:
        """Create a ticket, log the unresolved question and notify the operator."""
        state = self._states.setdefault(chat_id, _ChatState())
        state.unresolved_streak += 1
        ticket_id: int | None = None
        if self.store is not None:
            ticket_id = self.store.create_ticket(chat_id, decision.reason, question)
            self.store.log_unresolved(chat_id, question, hits[0].score if hits else 0.0, decision.reason)
            self.store.log_message(chat_id, "user", question)
            text = customer_handoff_text(ticket_id, business=hits[0].chunk.business if hits else None)
            self.store.log_answer(
                chat_id,
                question,
                text,
                hits[0].score if hits else 0.0,
                self.provider.name,
                source="",
                resolved=False,
            )
            self.store.log_message(chat_id, "bot", text)
        else:
            text = customer_handoff_text(0, business=hits[0].chunk.business if hits else None)

        business = hits[0].chunk.business if hits else None
        operator_message = build_operator_message(
            question,
            hits,
            decision,
            business=business,
            chat_id=chat_id,
            ticket_id=ticket_id,
        )
        self._remember(state, question, text)
        return BotReply(
            text=text,
            confidence=hits[0].score if hits else 0.0,
            resolved=False,
            provider=self.provider.name,
            hits=tuple(hits),
            escalation_reason=decision.reason,
            escalation_reason_ru=decision.reason_ru,
            ticket_id=ticket_id,
            operator_message=operator_message,
            matched_keywords=decision.matched_keywords,
        )

    @staticmethod
    def _remember(state: _ChatState, question: str, answer_text: str) -> None:
        """Append a turn to the chat history, keeping only the last few."""
        state.history.append(("user", question))
        state.history.append(("assistant", answer_text))
        if len(state.history) > HISTORY_LIMIT:
            del state.history[: len(state.history) - HISTORY_LIMIT]


__all__ = ["BotReply", "SupportPipeline"]
