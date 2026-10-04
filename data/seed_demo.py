"""Fill the dialogue log with a demonstration history.

``python -m ai_support_bot stats`` seeds an empty database automatically, but
this script does the same thing explicitly, for example to reset the log
before a demonstration::

    python data/seed_demo.py --force
    python -m ai_support_bot stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_support_bot.analytics import collect_stats, render_report  # noqa: E402
from ai_support_bot.storage import (  # noqa: E402
    DEFAULT_DB_PATH,
    DialogueStore,
    seed_demo_dialogues,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python data/seed_demo.py",
        description="Создать демонстрационный журнал диалогов для команды stats.",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / DEFAULT_DB_PATH),
        help="файл журнала SQLite (по умолчанию data/dialogues.db)",
    )
    parser.add_argument("--days", type=int, default=7, help="за сколько дней разложить диалоги")
    parser.add_argument(
        "--force",
        action="store_true",
        help="добавить демо-диалоги, даже если журнал уже не пуст",
    )
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Seed the database and optionally print the resulting report."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass

    args = build_parser().parse_args(argv)
    store = DialogueStore(args.db)
    try:
        if store.has_data() and not args.force:
            print(
                f"В журнале {args.db} уже есть данные ({store.count('answers')} ответов).\n"
                "Добавить демо-диалоги поверх: python data/seed_demo.py --force"
            )
            return 0
        count = seed_demo_dialogues(store, days=args.days)
        print(
            f"Демонстрационный журнал создан: {count} вопросов за {args.days} дн. "
            f"Файл: {args.db}"
        )
        if not args.quiet:
            print()
            print(f"Источник данных: {args.db} (время UTC)")
            print(render_report(collect_stats(store, days=args.days)))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
