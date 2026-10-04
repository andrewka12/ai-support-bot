"""End-to-end tests of the offline dialogue pipeline.

These tests prove the promise made to the client: the bot answers questions
from the knowledge base, hands over what it cannot answer, and does all of it
without a network, a token or an API key.
"""

from __future__ import annotations

import unittest
from typing import Sequence
from unittest import mock

from tests.base import ProjectTestCase

from ai_support_bot.escalation import (
    REASON_KEYWORD,
    REASON_LOW_SCORE,
    REASON_REPEAT,
    EscalationPolicy,
)
from ai_support_bot.llm import Answer
from ai_support_bot.pipeline import SupportPipeline
from ai_support_bot.retrieval import ScoredChunk
from ai_support_bot.storage import DialogueStore


class SourceLessProvider:
    """Provider stub that never returns a quotable fragment."""

    name = "stub"

    def answer(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        *,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        """Return an answer without sources, forcing the hand-off path."""
        del question, hits, history
        return Answer(text="Не знаю.", confidence=0.0, provider=self.name)


class PipelineTestCase(ProjectTestCase):
    """Base class building a pipeline on an in-memory dialogue log."""

    def build_pipeline(self, **kwargs) -> SupportPipeline:
        """Create a pipeline with an in-memory store and the default provider.

        No provider is passed on purpose: this is exactly how the CLI and the
        Telegram bot build it, including the term weights handed to the
        offline provider.
        """
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        self.store = store
        return SupportPipeline(self.index, store=store, **kwargs)


class OfflineAnswerTest(PipelineTestCase):
    """Questions from the knowledge base are answered offline."""

    def test_known_question_is_answered_with_a_source(self) -> None:
        pipeline = self.build_pipeline()
        reply = pipeline.answer("Сколько стоит замена масла в двигателе?", business="avtoservice")
        self.assertTrue(reply.resolved)
        self.assertEqual(reply.provider, "echo")
        self.assertEqual(reply.sources[0].source, "avtoservice/02-uslugi-i-ceny.md")
        self.assertIn("1 200 ₽", reply.text)
        self.assertIn("Источник:", reply.text)
        self.assertGreater(reply.confidence, 0.3)

    def test_answer_text_comes_from_the_knowledge_base(self) -> None:
        pipeline = self.build_pipeline()
        reply = pipeline.answer("Сколько стоит доставка по Москве?", business="shop")
        self.assertTrue(reply.resolved)
        source_text = (self.kb_dir / reply.sources[0].source).read_text(encoding="utf-8")
        quoted = reply.text.split("\n\n")[1]
        self.assertIn(quoted[:40], source_text)

    def test_rare_query_words_drive_the_quoted_sentence(self) -> None:
        # «Казань» is rare in the corpus and must outweigh the word
        # «доставка», which occurs in the whole section.
        pipeline = self.build_pipeline()
        reply = pipeline.answer("Сколько идёт доставка в Казань?", business="shop")
        self.assertTrue(reply.resolved)
        self.assertIn("Казань", reply.text)
        self.assertIn("3–5 рабочих дней", reply.text)

    def test_no_network_is_used(self) -> None:
        pipeline = self.build_pipeline()
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
            reply = pipeline.answer("Какая гарантия на ремонт двигателя?", business="avtoservice")
        self.assertTrue(reply.resolved)

    def test_dialogue_is_logged(self) -> None:
        pipeline = self.build_pipeline()
        pipeline.answer("Сколько стоит лечение кариеса?", chat_id="7", business="stomatologiya")
        self.assertEqual(self.store.count("answers"), 1)
        self.assertEqual(self.store.count("messages"), 2)
        self.assertEqual(self.store.count("tickets"), 0)
        self.assertTrue(self.store.fetch_answers()[0].resolved)

    def test_business_scope_is_respected(self) -> None:
        pipeline = self.build_pipeline(businesses=["shop"])
        reply = pipeline.answer("Сколько стоит доставка по Москве?")
        self.assertTrue(reply.resolved)
        self.assertEqual(reply.hits[0].chunk.business, "shop")

    def test_history_is_kept_per_chat(self) -> None:
        pipeline = self.build_pipeline()
        pipeline.answer("Сколько стоит доставка по Москве?", chat_id="1", business="shop")
        pipeline.answer("А в Казань?", chat_id="1", business="shop")
        self.assertEqual(len(pipeline.history("1")), 4)
        self.assertEqual(pipeline.history("2"), ())
        pipeline.reset("1")
        self.assertEqual(pipeline.history("1"), ())


class HandOffTest(PipelineTestCase):
    """Questions the bot cannot answer create a ticket for a human."""

    def test_unknown_question_is_handed_over(self) -> None:
        pipeline = self.build_pipeline()
        reply = pipeline.answer("Есть ли у вас скидка на годовой абонемент?", business="avtoservice")
        self.assertFalse(reply.resolved)
        self.assertEqual(reply.escalation_reason, REASON_LOW_SCORE)
        self.assertIsNotNone(reply.ticket_id)
        self.assertEqual(self.store.count("tickets"), 1)
        self.assertEqual(self.store.count("unresolved"), 1)
        self.assertIn("#", reply.text)

    def test_trigger_keyword_is_handed_over(self) -> None:
        pipeline = self.build_pipeline()
        reply = pipeline.answer("Позовите оператора, пожалуйста", business="shop")
        self.assertFalse(reply.resolved)
        self.assertEqual(reply.escalation_reason, REASON_KEYWORD)
        self.assertIn("оператор", reply.matched_keywords)

    def test_operator_message_contains_the_question(self) -> None:
        pipeline = self.build_pipeline()
        question = "Верните деньги за некачественный товар"
        reply = pipeline.answer(question, business="shop")
        self.assertFalse(reply.resolved)
        self.assertIn(question, reply.operator_message)
        self.assertIn(f"#{reply.ticket_id}", reply.operator_message)

    def test_repeated_unclear_questions_trigger_the_streak_rule(self) -> None:
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        policy = EscalationPolicy(score_threshold=0.0, max_unresolved_streak=2)
        pipeline = SupportPipeline(self.index, SourceLessProvider(), store=store, policy=policy)

        first = pipeline.answer("Расскажите про доставку", chat_id="9", business="shop")
        second = pipeline.answer("Расскажите про доставку", chat_id="9", business="shop")

        self.assertFalse(first.resolved)
        self.assertEqual(first.escalation_reason, REASON_LOW_SCORE)
        self.assertEqual(second.escalation_reason, REASON_REPEAT)
        self.assertEqual(store.count("tickets"), 2)

    def test_streak_resets_after_a_good_answer(self) -> None:
        pipeline = self.build_pipeline()
        pipeline.answer("Есть ли у вас скидка на абонемент?", chat_id="5", business="avtoservice")
        self.assertEqual(pipeline.unresolved_streak("5"), 1)
        pipeline.answer("Сколько стоит замена масла?", chat_id="5", business="avtoservice")
        self.assertEqual(pipeline.unresolved_streak("5"), 0)

    def test_fallback_when_the_provider_fails(self) -> None:
        from ai_support_bot.llm import ProviderError

        class BrokenProvider:
            name = "broken"

            def answer(self, question, hits, *, history=None):  # noqa: ANN001, ANN202
                raise ProviderError("endpoint недоступен")

        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        pipeline = SupportPipeline(self.index, BrokenProvider(), store=store)
        reply = pipeline.answer("Сколько стоит доставка по Москве?", business="shop")
        self.assertTrue(reply.resolved)
        self.assertEqual(reply.provider, "echo-fallback")
        self.assertIn("LLM недоступен", reply.text)


if __name__ == "__main__":
    unittest.main()
