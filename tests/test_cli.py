"""Tests for the command line interface.

The CLI is what a reviewer runs first, so its four commands are exercised
here exactly as the README describes them.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from tests.base import PROJECT_ROOT, make_temp_dir

from ai_support_bot.cli import main


class CLITestCase(unittest.TestCase):
    """Base class providing a scratch directory and output capture."""

    def setUp(self) -> None:
        self.tmp = make_temp_dir(self)
        self.db = str(self.tmp / "dialogues.db")
        self.index = str(self.tmp / "index.json")

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        """Run the CLI and return ``(exit_code, stdout, stderr)``."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()


class ChatCommandTest(CLITestCase):
    """``chat --ask`` answers questions non-interactively."""

    def test_ask_answers_a_question_from_the_knowledge_base(self) -> None:
        code, out, _ = self.run_cli(
            [
                "chat",
                "--db",
                self.db,
                "--index",
                self.index,
                "--no-log",
                "--ask",
                "Сколько стоит замена масла в двигателе?",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("Вы: Сколько стоит замена масла в двигателе?", out)
        self.assertIn("Бот:", out)
        self.assertIn("1 200 ₽", out)
        self.assertIn("Источник: avtoservice/02-uslugi-i-ceny.md", out)

    def test_ask_hands_over_an_unknown_question(self) -> None:
        code, out, _ = self.run_cli(
            [
                "chat",
                "--index",
                self.index,
                "--db",
                self.db,
                "--no-log",
                "--ask",
                "Есть ли у вас скидка на абонемент?",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("Передаю ваш вопрос оператору", out)

    def test_ask_handles_the_operator_keyword(self) -> None:
        code, out, _ = self.run_cli(
            ["chat", "--index", self.index, "--db", self.db, "--no-log", "--ask", "позовите оператора"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Передаю ваш вопрос оператору", out)

    def test_show_operator_prints_the_ticket_text(self) -> None:
        code, out, _ = self.run_cli(
            [
                "chat",
                "--index",
                self.index,
                "--db",
                self.db,
                "--no-log",
                "--show-operator",
                "--ask",
                "позовите оператора",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("сообщение оператору", out)

    def test_business_scope_limits_the_search(self) -> None:
        code, out, _ = self.run_cli(
            [
                "chat",
                "--index",
                self.index,
                "--db",
                self.db,
                "--no-log",
                "--business",
                "shop",
                "--ask",
                "Сколько стоит доставка по Москве?",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("shop/02-dostavka.md", out)

    def test_unknown_business_is_reported(self) -> None:
        code, _, err = self.run_cli(
            [
                "chat",
                "--index",
                self.index,
                "--db",
                self.db,
                "--no-log",
                "--business",
                "cafe",
                "--ask",
                "Сколько стоит кофе?",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("не найдено", err)

    def test_dialogue_is_written_to_the_database(self) -> None:
        code, _, _ = self.run_cli(
            ["chat", "--index", self.index, "--db", self.db, "--ask", "Сколько стоит доставка по Москве?"]
        )
        self.assertEqual(code, 0)
        self.assertTrue(Path(self.db).exists())


class IngestCommandTest(CLITestCase):
    """``ingest`` builds and caches the index."""

    def test_ingest_writes_the_cache_and_reports_counts(self) -> None:
        code, out, _ = self.run_cli(["ingest", "--index", self.index])
        self.assertEqual(code, 0)
        self.assertIn("avtoservice", out)
        self.assertIn("Всего:", out)
        self.assertTrue(Path(self.index).exists())

    def test_ingest_quiet_prints_nothing(self) -> None:
        code, out, _ = self.run_cli(["ingest", "--index", self.index, "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "")

    def test_missing_knowledge_base_is_reported(self) -> None:
        code, _, err = self.run_cli(["ingest", "--kb-dir", str(self.tmp / "nope")])
        self.assertEqual(code, 1)
        self.assertIn("не найдена", err)


class StatsCommandTest(CLITestCase):
    """``stats`` works on a fresh clone thanks to the demo seeder."""

    def test_stats_seeds_an_empty_database(self) -> None:
        code, out, _ = self.run_cli(["stats", "--db", self.db, "--days", "7"])
        self.assertEqual(code, 0)
        self.assertIn("Журнал диалогов пуст", out)
        self.assertIn("Всего вопросов:", out)
        self.assertIn("Динамика по дням", out)

    def test_second_run_does_not_seed_again(self) -> None:
        self.run_cli(["stats", "--db", self.db, "--days", "7"])
        code, out, _ = self.run_cli(["stats", "--db", self.db, "--days", "7"])
        self.assertEqual(code, 0)
        self.assertNotIn("Журнал диалогов пуст", out)
        self.assertIn("Всего вопросов:", out)

    def test_no_seed_flag_keeps_the_database_empty(self) -> None:
        code, out, _ = self.run_cli(["stats", "--db", self.db, "--no-seed"])
        self.assertEqual(code, 0)
        self.assertNotIn("Журнал диалогов пуст", out)
        self.assertIn("нет ни одного диалога", out)


class ServeCommandTest(CLITestCase):
    """``serve`` fails with a clear Russian message when the token is missing."""

    def test_missing_token_returns_exit_code_two(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            code, _, err = self.run_cli(["serve", "--db", self.db, "--index", self.index])
        self.assertEqual(code, 2)
        self.assertIn("Не задан токен Telegram-бота", err)
        self.assertIn("BotFather", err)


class ParserTest(unittest.TestCase):
    """Command line parsing basics."""

    def test_no_command_prints_help(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main([])
        self.assertEqual(code, 2)
        self.assertIn("chat", out.getvalue())
        self.assertIn("Укажите команду", err.getvalue())

    def test_project_root_contains_the_package(self) -> None:
        self.assertTrue((PROJECT_ROOT / "ai_support_bot" / "cli.py").exists())


if __name__ == "__main__":
    unittest.main()
