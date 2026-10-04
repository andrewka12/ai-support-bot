"""Tests for the hand-off rules."""

from __future__ import annotations

import unittest

from ai_support_bot.chunking import Chunk
from ai_support_bot.escalation import (
    DEFAULT_MAX_UNRESOLVED_STREAK,
    REASON_KEYWORD,
    REASON_LOW_SCORE,
    REASON_NONE,
    REASON_REPEAT,
    EscalationPolicy,
    build_operator_message,
    customer_handoff_text,
    matched_keywords,
)
from ai_support_bot.retrieval import ScoredChunk


def make_hit(score: float, source: str = "shop/04-vozvrat-i-obmen.md") -> ScoredChunk:
    """Build a scored chunk for the policy tests."""
    chunk = Chunk(
        chunk_id=f"{source}#1",
        text="Текст фрагмента базы знаний.",
        source=source,
        business=source.split("/")[0],
        title="Документ",
        heading="Раздел",
        order=1,
    )
    return ScoredChunk(chunk=chunk, score=score)


class TriggerKeywordTest(unittest.TestCase):
    """Keywords that always send the question to a human."""

    def setUp(self) -> None:
        self.policy = EscalationPolicy()

    def test_operator_request_triggers(self) -> None:
        decision = self.policy.check("Позовите оператора, пожалуйста", [make_hit(0.9)])
        self.assertTrue(decision.needed)
        self.assertEqual(decision.reason, REASON_KEYWORD)

    def test_complaint_and_refund_trigger(self) -> None:
        questions = (
            "Хочу пожаловаться на курьера",
            "Верните деньги за заказ",
            "Это обман, я напишу в Роспотребнадзор",
            "Подам в суд, если не решите вопрос",
            "Нужен юрист, вы нарушили договор",
            "Дайте жалобную книгу",
        )
        for question in questions:
            with self.subTest(question=question):
                decision = self.policy.check(question, [make_hit(0.9)])
                self.assertTrue(decision.needed)
                self.assertEqual(decision.reason, REASON_KEYWORD)

    def test_keywords_are_case_and_yo_insensitive(self) -> None:
        self.assertIn("роспотреб", matched_keywords("НАПИШУ В РОСПОТРЕБНАДЗОР"))
        self.assertTrue(self.policy.check("ВЕРНИТЕ ДЕНЬГИ", [make_hit(0.9)]).needed)

    def test_keyword_wins_over_a_high_score(self) -> None:
        decision = self.policy.check("Верните деньги за доставку", [make_hit(0.95)])
        self.assertEqual(decision.reason, REASON_KEYWORD)

    def test_matched_keywords_are_reported(self) -> None:
        decision = self.policy.check("Позовите оператора и юриста", [make_hit(0.9)])
        self.assertIn("оператор", decision.matched_keywords)
        self.assertIn("юрист", decision.matched_keywords)


class ScoreThresholdTest(unittest.TestCase):
    """The retrieval score decides whether an answer is trustworthy."""

    def setUp(self) -> None:
        self.policy = EscalationPolicy(score_threshold=0.16)

    def test_low_score_triggers_hand_off(self) -> None:
        decision = self.policy.check("Есть ли у вас скидка на абонемент?", [make_hit(0.05)])
        self.assertTrue(decision.needed)
        self.assertEqual(decision.reason, REASON_LOW_SCORE)

    def test_no_hits_trigger_hand_off(self) -> None:
        decision = self.policy.check("Какая погода в Москве?", [])
        self.assertTrue(decision.needed)
        self.assertEqual(decision.reason, REASON_LOW_SCORE)

    def test_confident_answer_does_not_trigger(self) -> None:
        decision = self.policy.check("Сколько стоит доставка по Москве?", [make_hit(0.42)])
        self.assertFalse(decision.needed)
        self.assertEqual(decision.reason, REASON_NONE)

    def test_score_exactly_on_the_threshold_is_accepted(self) -> None:
        decision = self.policy.check("Сколько стоит доставка по Москве?", [make_hit(0.16)])
        self.assertFalse(decision.needed)

    def test_custom_threshold_is_respected(self) -> None:
        strict = EscalationPolicy(score_threshold=0.5)
        self.assertTrue(strict.check("Сколько стоит доставка?", [make_hit(0.4)]).needed)


class UnresolvedStreakTest(unittest.TestCase):
    """Several unclear questions in a row call a human."""

    def setUp(self) -> None:
        self.policy = EscalationPolicy(score_threshold=0.16, max_unresolved_streak=3)

    def test_streak_below_limit_does_not_trigger(self) -> None:
        decision = self.policy.check("Сколько стоит доставка?", [make_hit(0.4)], unresolved_streak=1)
        self.assertFalse(decision.needed)

    def test_third_unclear_question_triggers(self) -> None:
        decision = self.policy.check(
            "Сколько стоит доставка?",
            [make_hit(0.4)],
            unresolved_streak=DEFAULT_MAX_UNRESOLVED_STREAK - 1,
        )
        self.assertTrue(decision.needed)
        self.assertEqual(decision.reason, REASON_REPEAT)

    def test_low_score_reason_wins_over_streak(self) -> None:
        decision = self.policy.check("Абонемент?", [make_hit(0.01)], unresolved_streak=5)
        self.assertEqual(decision.reason, REASON_LOW_SCORE)


class OperatorMessageTest(unittest.TestCase):
    """The message an operator receives must be self-contained."""

    def test_message_contains_question_and_reason(self) -> None:
        policy = EscalationPolicy()
        hits = [make_hit(0.05, "shop/04-vozvrat-i-obmen.md")]
        decision = policy.check("Верните деньги", hits)
        message = build_operator_message(
            "Верните деньги",
            hits,
            decision,
            business="shop",
            chat_id="42",
            ticket_id=7,
        )
        self.assertIn("#7", message)
        self.assertIn("Верните деньги", message)
        self.assertIn("shop", message)
        self.assertIn("42", message)
        self.assertIn(decision.reason_ru, message)
        self.assertIn("shop/04-vozvrat-i-obmen.md", message)

    def test_message_without_hits_says_so(self) -> None:
        policy = EscalationPolicy()
        decision = policy.check("Вопрос вне базы", [])
        message = build_operator_message("Вопрос вне базы", [], decision)
        self.assertIn("нет ни одного релевантного фрагмента", message)

    def test_customer_text_contains_ticket_number(self) -> None:
        text = customer_handoff_text(15, business="shop")
        self.assertIn("#15", text)
        self.assertIn("оператор", text)


if __name__ == "__main__":
    unittest.main()
