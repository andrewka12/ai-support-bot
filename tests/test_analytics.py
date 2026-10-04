"""Tests for the analytics over a seeded dialogue log."""

from __future__ import annotations

import unittest
from datetime import timedelta

from tests.base import FIXED_NOW

from ai_support_bot.analytics import collect_stats, render_report
from ai_support_bot.escalation import REASON_KEYWORD, REASON_LOW_SCORE
from ai_support_bot.storage import DialogueStore, demo_dialogue_count, seed_demo_dialogues


def seeded_store() -> DialogueStore:
    """Build an in-memory log with a known shape.

    Five questions in the window: three resolved (confidence 0.80, 0.60,
    0.50) and two handed to an operator (0.10 with ``low_score`` and 0.30
    with ``trigger_keyword``).
    """
    store = DialogueStore(":memory:")
    records = (
        ("Сколько стоит доставка по Москве?", True, 0.80, None, 0),
        ("Можно ли вернуть товар?", True, 0.60, None, 0),
        ("Есть ли скидка на абонемент?", False, 0.10, REASON_LOW_SCORE, 1),
        ("Верните деньги за заказ", False, 0.30, REASON_KEYWORD, 2),
        ("Как оплатить заказ картой?", True, 0.50, None, 3),
    )
    for index, (question, resolved, confidence, reason, days_ago) in enumerate(records):
        stamp = (FIXED_NOW - timedelta(days=days_ago)).replace(hour=10 + index).isoformat()
        store.log_message("test", "user", question, created_at=stamp)
        store.log_answer(
            "test",
            question,
            "Ответ бота." if resolved else "Передаю оператору.",
            confidence,
            "echo",
            resolved=resolved,
            created_at=stamp,
        )
        if not resolved:
            assert reason is not None
            store.log_unresolved("test", question, confidence - 0.05, reason, created_at=stamp)
            store.create_ticket("test", reason, question, created_at=stamp)
    return store


class CollectStatsTest(unittest.TestCase):
    """Aggregation of the seeded fixture."""

    def setUp(self) -> None:
        self.store = seeded_store()
        self.addCleanup(self.store.close)
        self.report = collect_stats(self.store, days=7, now=FIXED_NOW)

    def test_totals_and_percentage(self) -> None:
        self.assertEqual(self.report.total_questions, 5)
        self.assertEqual(self.report.resolved, 3)
        self.assertEqual(self.report.escalated, 2)
        self.assertAlmostEqual(self.report.auto_resolved_pct, 60.0, places=6)

    def test_average_confidence(self) -> None:
        self.assertAlmostEqual(self.report.average_confidence, 0.46, places=6)
        self.assertAlmostEqual(self.report.average_confidence_resolved, 1.9 / 3, places=6)

    def test_tickets_by_reason(self) -> None:
        self.assertEqual(self.report.tickets_total, 2)
        self.assertEqual(self.report.tickets_open, 2)
        self.assertEqual(
            self.report.tickets_by_reason,
            {REASON_LOW_SCORE: 1, REASON_KEYWORD: 1},
        )

    def test_weak_questions_are_ranked_by_frequency_then_score(self) -> None:
        self.assertEqual(len(self.report.weak_questions), 2)
        first = self.report.weak_questions[0]
        self.assertEqual(first.question, "Есть ли скидка на абонемент?")
        self.assertEqual(first.count, 1)
        self.assertAlmostEqual(first.average_score, 0.05, places=6)

    def test_per_day_buckets(self) -> None:
        self.assertEqual(len(self.report.per_day), 7)
        self.assertEqual(self.report.per_day[0].day, "2026-01-04")
        self.assertEqual(self.report.per_day[-1].day, "2026-01-10")
        by_day = {item.day: item for item in self.report.per_day}
        self.assertEqual(by_day["2026-01-10"].questions, 2)
        self.assertEqual(by_day["2026-01-10"].resolved, 2)
        self.assertEqual(by_day["2026-01-09"].escalated, 1)
        self.assertEqual(by_day["2026-01-06"].questions, 0)

    def test_short_window_excludes_old_rows(self) -> None:
        report = collect_stats(self.store, days=2, now=FIXED_NOW)
        self.assertEqual(report.total_questions, 3)
        self.assertEqual(report.resolved, 2)
        self.assertEqual(report.escalated, 1)

    def test_render_contains_key_numbers(self) -> None:
        text = render_report(self.report)
        self.assertIn("Всего вопросов:                5", text)
        self.assertIn("60.0 %", text)
        self.assertIn("Есть ли скидка на абонемент?", text)
        self.assertIn(REASON_LOW_SCORE, text)
        self.assertIn("Динамика по дням", text)


class EmptyLogTest(unittest.TestCase):
    """An empty log renders a helpful message instead of dividing by zero."""

    def test_empty_report(self) -> None:
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        report = collect_stats(store, days=7, now=FIXED_NOW)
        self.assertEqual(report.total_questions, 0)
        self.assertEqual(report.auto_resolved_pct, 0.0)
        self.assertEqual(report.average_confidence, 0.0)
        self.assertIn("нет ни одного диалога", render_report(report))

    def test_invalid_window_is_rejected(self) -> None:
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            collect_stats(store, days=0)


class DemoSeedTest(unittest.TestCase):
    """The demo seeder used by a fresh clone."""

    def test_seeding_fills_the_log(self) -> None:
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        self.assertFalse(store.has_data())
        count = seed_demo_dialogues(store, days=7, now=FIXED_NOW)
        self.assertEqual(count, demo_dialogue_count())
        self.assertTrue(store.has_data())

    def test_seeded_log_produces_a_lively_report(self) -> None:
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        seed_demo_dialogues(store, days=7, now=FIXED_NOW)
        report = collect_stats(store, days=7, now=FIXED_NOW)
        self.assertEqual(report.total_questions, demo_dialogue_count())
        self.assertGreater(report.auto_resolved_pct, 40.0)
        self.assertLess(report.auto_resolved_pct, 70.0)
        self.assertGreater(report.average_confidence, 0.2)
        self.assertEqual(report.tickets_total, report.escalated)
        self.assertTrue(all(item.questions > 0 for item in report.per_day))
        self.assertTrue(report.weak_questions)
        self.assertGreaterEqual(report.weak_questions[0].count, 1)

    def test_seeding_respects_the_window(self) -> None:
        store = DialogueStore(":memory:")
        self.addCleanup(store.close)
        seed_demo_dialogues(store, days=3, now=FIXED_NOW)
        report = collect_stats(store, days=3, now=FIXED_NOW)
        self.assertEqual(report.total_questions, demo_dialogue_count())


if __name__ == "__main__":
    unittest.main()
