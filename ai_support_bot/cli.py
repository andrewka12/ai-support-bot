"""Command line interface: ``python -m ai_support_bot <command>``.

Commands:
    chat   -- offline console chat (the primary demo, no token required)
    ingest -- build the retrieval index from the ``kb/`` folder
    stats  -- analytics over the dialogue log
    serve  -- real Telegram bot via long polling (needs a token)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .analytics import collect_stats, render_report
from .bot import BotConfigurationError, TOKEN_ENV_VAR, run_bot
from .chunking import business_names, load_chunks
from .escalation import DEFAULT_SCORE_THRESHOLD, EscalationPolicy
from .llm import select_provider
from .pipeline import BotReply, SupportPipeline
from .retrieval import (
    TfidfIndex,
    build_index,
    kb_signature,
    load_index,
    save_index,
)
from .storage import DEFAULT_DB_PATH, DialogueStore, seed_demo_dialogues

#: Repository root, derived from this file's location.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Bundled knowledge base.
DEFAULT_KB_DIR: Path = PROJECT_ROOT / "ai_support_bot" / "kb"

#: Cached retrieval index (git-ignored, rebuilt by ``ingest``).
DEFAULT_INDEX_PATH: Path = PROJECT_ROOT / "data" / "index.json"

_SEPARATOR = "-" * 60

_EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Сколько стоит замена масла в двигателе?",
    "Есть ли противопоказания к имплантации зубов?",
    "Сколько стоит доставка по Москве?",
    "позовите оператора",
)


@dataclass(frozen=True)
class CLIContext:
    """Resolved paths for one CLI invocation."""

    kb_dir: Path
    db_path: str
    index_path: Path


@dataclass(frozen=True)
class IngestSummary:
    """Result of building the index, used by ``ingest`` and by the tests."""

    chunks_total: int
    files_total: int
    vocabulary_size: int
    per_business: dict[str, tuple[int, int]]


def _configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr so Russian text survives Windows pipes."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _add_common_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """Add the options shared by the top level and every subcommand.

    Subcommands receive the same options with ``SUPPRESS`` defaults, so both
    ``--db X stats`` and ``stats --db X`` work and neither form overwrites a
    value given in the other position.
    """
    def value(fallback):
        return argparse.SUPPRESS if suppress_defaults else fallback

    parser.add_argument(
        "--kb-dir",
        default=value(str(DEFAULT_KB_DIR)),
        help="папка с базой знаний (по умолчанию ai_support_bot/kb)",
    )
    parser.add_argument(
        "--db",
        default=value(str(DEFAULT_DB_PATH)),
        help="файл журнала диалогов SQLite (по умолчанию data/dialogues.db)",
    )
    parser.add_argument(
        "--index",
        default=value(str(DEFAULT_INDEX_PATH)),
        help="файл кэша поискового индекса (по умолчанию data/index.json)",
    )
    parser.add_argument(
        "--provider",
        default=value("auto"),
        choices=("auto", "echo", "openai"),
        help="поставщик ответов: auto (по переменным окружения), echo (офлайн), openai",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=value(3),
        help="сколько фрагментов базы знаний искать под вопрос (по умолчанию 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=value(DEFAULT_SCORE_THRESHOLD),
        help=(
            "порог релевантности: ниже него вопрос уходит оператору "
            f"(по умолчанию {DEFAULT_SCORE_THRESHOLD})"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m ai_support_bot",
        description=(
            "ИИ-бот поддержки клиентов по базе знаний: поиск по документам, "
            "защита от выдумывания ответов и передача сложных вопросов оператору."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python -m ai_support_bot chat\n"
            "  python -m ai_support_bot chat --ask \"Сколько стоит замена масла?\"\n"
            "  python -m ai_support_bot ingest\n"
            "  python -m ai_support_bot stats\n"
            "  python -m ai_support_bot serve\n"
        ),
    )
    _add_common_options(parser, suppress_defaults=False)
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser(
        "chat",
        help="офлайн-чат в консоли: главное демо, токен не нужен",
        description="Диалог с ботом в консоли. Работает офлайн, без ключей и интернета.",
    )
    _add_common_options(chat, suppress_defaults=True)
    chat.add_argument(
        "--ask",
        action="append",
        metavar="ВОПРОС",
        help="задать один вопрос и выйти (можно указать несколько раз)",
    )
    chat.add_argument(
        "--business",
        default=None,
        help="ограничить поиск одним направлением (папкой базы знаний)",
    )
    chat.add_argument("--chat-id", default="console", help="идентификатор диалога в журнале")
    chat.add_argument(
        "--show-operator",
        action="store_true",
        help="показывать сообщение, которое уходит оператору",
    )
    chat.add_argument(
        "--no-log",
        action="store_true",
        help="не записывать диалог в базу данных",
    )

    ingest = subparsers.add_parser(
        "ingest",
        help="построить поисковый индекс по папке kb/",
        description="Читает базу знаний, нарезает её на фрагменты и сохраняет индекс.",
    )
    _add_common_options(ingest, suppress_defaults=True)
    ingest.add_argument(
        "--quiet",
        action="store_true",
        help="не печатать подробности",
    )

    stats = subparsers.add_parser(
        "stats",
        help="аналитика по журналу диалогов",
        description="Показывает долю автоответов, слабые вопросы, тикеты и динамику по дням.",
    )
    _add_common_options(stats, suppress_defaults=True)
    stats.add_argument("--days", type=int, default=7, help="период отчёта в днях")
    stats.add_argument(
        "--no-seed",
        action="store_true",
        help="не создавать демонстрационный журнал, если база пуста",
    )

    serve = subparsers.add_parser(
        "serve",
        help="запустить настоящего Telegram-бота (нужен TELEGRAM_BOT_TOKEN)",
        description=(
            "Длинный опрос Telegram Bot API. Токен берётся из переменной "
            f"окружения {TOKEN_ENV_VAR}."
        ),
    )
    _add_common_options(serve, suppress_defaults=True)
    serve.add_argument("--token", default=None, help=f"токен бота (или {TOKEN_ENV_VAR})")
    serve.add_argument(
        "--operator-chat-id",
        default=None,
        help="чат оператора, куда пересылать обращения (или OPERATOR_CHAT_ID)",
    )
    serve.add_argument(
        "--once",
        action="store_true",
        help="обработать текущие сообщения и выйти (для проверки настройки)",
    )
    return parser


def resolve_context(args: argparse.Namespace) -> CLIContext:
    """Turn parsed arguments into concrete paths."""
    return CLIContext(
        kb_dir=Path(args.kb_dir).expanduser(),
        db_path=args.db,
        index_path=Path(args.index).expanduser(),
    )


def ingest_knowledge_base(kb_dir: Path) -> tuple[TfidfIndex, IngestSummary]:
    """Load, chunk and index the knowledge base under *kb_dir*.

    Raises:
        FileNotFoundError: when the folder does not exist.
        ValueError: when it contains no documents.
    """
    if not kb_dir.exists():
        raise FileNotFoundError(f"Папка базы знаний не найдена: {kb_dir}")
    chunks = load_chunks(kb_dir)
    if not chunks:
        raise ValueError(
            f"В папке {kb_dir} нет файлов .md или .txt. Добавьте документы и повторите."
        )
    index = build_index(chunks)
    per_business: dict[str, tuple[int, int]] = {}
    for name in business_names(chunks):
        scoped = [chunk for chunk in chunks if chunk.business == name]
        per_business[name] = (len({chunk.source for chunk in scoped}), len(scoped))
    summary = IngestSummary(
        chunks_total=len(chunks),
        files_total=len({chunk.source for chunk in chunks}),
        vocabulary_size=index.vocabulary_size,
        per_business=per_business,
    )
    return index, summary


def load_or_build_index(ctx: CLIContext, *, rebuild: bool = False) -> tuple[TfidfIndex, bool]:
    """Return the retrieval index, using the cache when it is still valid.

    Returns:
        A ``(index, was_cached)`` pair.
    """
    signature = kb_signature(ctx.kb_dir)
    if not rebuild:
        cached = load_index(ctx.index_path, expected_signature=signature)
        if cached is not None:
            return cached, True
    index, _ = ingest_knowledge_base(ctx.kb_dir)
    save_index(index, ctx.index_path, signature=signature)
    return index, False


def _build_pipeline(
    args: argparse.Namespace,
    ctx: CLIContext,
    *,
    store: DialogueStore | None,
    rebuild_index: bool = False,
) -> SupportPipeline:
    """Create the dialogue pipeline used by ``chat`` and ``serve``."""
    index, _ = load_or_build_index(ctx, rebuild=rebuild_index)
    provider, warning = select_provider(args.provider, idf=index.idf)
    if warning:
        print(f"Внимание: {warning}")
    policy = EscalationPolicy(score_threshold=args.threshold)
    businesses = [args.business] if getattr(args, "business", None) else None
    if businesses and businesses[0] not in business_names(index.chunks):
        raise ValueError(
            f"Направление «{businesses[0]}» не найдено. Доступные: "
            + ", ".join(business_names(index.chunks))
        )
    return SupportPipeline(
        index,
        provider,
        store=store,
        policy=policy,
        top_k=args.top_k,
        businesses=businesses,
    )


def _print_reply(
    reply: BotReply,
    *,
    show_operator: bool = False,
    out=print,
) -> None:
    """Print one bot answer with a compact metadata footer."""
    out(f"Бот: {reply.text}")
    if reply.resolved:
        meta = f"уверенность {reply.confidence:.2f} · ответ по базе знаний"
    else:
        meta = (
            f"уверенность {reply.confidence:.2f} · передан оператору "
            f"· тикет #{reply.ticket_id} · причина: {reply.escalation_reason_ru}"
        )
    if reply.hits:
        best = reply.hits[0].chunk
        meta += f" · лучший фрагмент: {best.source} → «{best.heading}» ({reply.top_score:.2f})"
    out(f"\n[ {meta} ]\n")
    if show_operator and reply.operator_message:
        out("--- сообщение оператору ---")
        out(reply.operator_message)
        out("---------------------------\n")


def _banner(index: TfidfIndex, pipeline: SupportPipeline, *, interactive: bool) -> None:
    """Print the opening banner of the console chat."""
    names = ", ".join(business_names(index.chunks))
    print("=" * 60)
    print(" Демо-бот поддержки клиентов по базе знаний")
    print(" Офлайн-режим: без интернета, токенов и API-ключей")
    print("=" * 60)
    print(f"База знаний: {index.size} фрагментов, направления: {names}")
    print(
        f"Провайдер ответов: {pipeline.provider.name} · "
        f"порог передачи оператору: {pipeline.policy.score_threshold:.2f}"
    )
    if interactive:
        print("Команды: /kb <направление>, /stats, /tickets, /help, /exit")
        print("Примеры вопросов: " + " | ".join(_EXAMPLE_QUESTIONS[:3]))
    print(_SEPARATOR)


def _print_stats(store: DialogueStore | None, *, days: int, out=print) -> None:
    """Print the full statistics report, or explain why it is unavailable."""
    if store is None:
        out("Статистика недоступна: журнал диалогов отключён (--no-log).")
        return
    out(render_report(collect_stats(store, days=days)))


def _print_tickets(store: DialogueStore | None, *, out=print) -> None:
    """Print the latest operator tickets."""
    if store is None:
        out("Тикеты недоступны: журнал диалогов отключён (--no-log).")
        return
    tickets = store.fetch_tickets()
    if not tickets:
        out("Тикетов пока нет — сложных вопросов не было.")
        return
    out(f"Последние тикеты (всего {len(tickets)}):")
    for ticket in tickets[-10:]:
        out(
            f"  #{ticket.id} [{ticket.status}] {ticket.created_at} · "
            f"{ticket.reason} · {ticket.question}"
        )


def command_chat(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Run the offline console chat, interactively or with ``--ask``."""
    store = None if args.no_log else DialogueStore(ctx.db_path)
    try:
        pipeline = _build_pipeline(args, ctx, store=store)
        chat_id = args.chat_id
        interactive = not args.ask
        _banner(pipeline.index, pipeline, interactive=interactive)

        if args.ask:
            for question in args.ask:
                print(f"Вы: {question}")
                reply = pipeline.answer(question, chat_id=chat_id)
                _print_reply(reply, show_operator=args.show_operator)
            return 0

        current_business = args.business
        while True:
            try:
                text = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nДо связи!")
                return 0
            if not text:
                continue
            if text.startswith("/"):
                command = text[1:].strip()
                name, _, argument = command.partition(" ")
                name = name.lower()
                argument = argument.strip()
                if name in {"exit", "quit", "выход"}:
                    print("До связи!")
                    return 0
                if name == "help":
                    print(
                        "Команды: /kb <направление|all>, /stats, /tickets, /exit.\n"
                        "Просто напишите вопрос: «Сколько стоит замена масла?»"
                    )
                    continue
                if name == "kb":
                    if not argument:
                        print(
                            "Направления: "
                            + ", ".join(business_names(pipeline.index.chunks))
                            + f". Текущее: {current_business or 'все'}"
                        )
                        continue
                    if argument.lower() in {"all", "все"}:
                        current_business = None
                        print("Отвечаю по всем направлениям.")
                        continue
                    if argument not in business_names(pipeline.index.chunks):
                        print(
                            "Не знаю направление «"
                            + argument
                            + "». Доступные: "
                            + ", ".join(business_names(pipeline.index.chunks))
                        )
                        continue
                    current_business = argument
                    print(f"Переключился на направление «{argument}».")
                    continue
                if name == "stats":
                    _print_stats(store, days=7)
                    continue
                if name == "tickets":
                    _print_tickets(store)
                    continue
                print("Неизвестная команда. Напишите /help.")
                continue

            reply = pipeline.answer(text, chat_id=chat_id, business=current_business)
            _print_reply(reply, show_operator=args.show_operator)
    finally:
        if store is not None:
            store.close()


def command_ingest(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Build the index and store it in the cache, printing a summary."""
    index, summary = ingest_knowledge_base(ctx.kb_dir)
    if not args.quiet:
        print(f"Индексирую базу знаний: {ctx.kb_dir}")
        for name, (files, chunks) in sorted(summary.per_business.items()):
            print(f"  {name:<16} файлов: {files}, фрагментов: {chunks}")
        print(
            f"Всего: {summary.files_total} файлов, {summary.chunks_total} фрагментов, "
            f"словарь: {summary.vocabulary_size} терминов"
        )
    save_index(index, ctx.index_path, signature=kb_signature(ctx.kb_dir))
    if not args.quiet:
        print(f"Индекс сохранён: {ctx.index_path}")
        print('Проверить поиск: python -m ai_support_bot chat --ask "Сколько стоит замена масла?"')
    return 0


def command_stats(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Print the analytics report, seeding a demo log when the DB is empty."""
    store = DialogueStore(ctx.db_path)
    try:
        if not store.has_data() and not args.no_seed:
            count = seed_demo_dialogues(store, days=args.days)
            print(
                "Журнал диалогов пуст — создаю демонстрационный журнал за "
                f"{args.days} дн. ({count} вопросов).\n"
                "Он нужен, чтобы отчёт был наглядным на свежем клоне проекта.\n"
                "Отключить: python -m ai_support_bot stats --no-seed\n"
            )
        print(f"Источник данных: {ctx.db_path} (время UTC)")
        print(render_report(collect_stats(store, days=args.days)))
    finally:
        store.close()
    return 0


def command_serve(args: argparse.Namespace, ctx: CLIContext) -> int:
    """Start the real Telegram bot, or explain clearly why it cannot start."""
    store = DialogueStore(ctx.db_path)
    try:
        pipeline = _build_pipeline(args, ctx, store=store)
        print(
            f"Запускаю Telegram-бота: {pipeline.describe()}. "
            "Остановить — Ctrl+C."
        )
        run_bot(
            pipeline,
            token=args.token,
            operator_chat_id=args.operator_chat_id,
            store=store,
            once=args.once,
        )
    except BotConfigurationError as error:
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nБот остановлен.")
        return 0
    finally:
        store.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m ai_support_bot``."""
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        print(
            "\nУкажите команду: chat (демо в консоли), ingest, stats или serve.",
            file=sys.stderr,
        )
        return 2

    ctx = resolve_context(args)
    handlers = {
        "chat": command_chat,
        "ingest": command_ingest,
        "stats": command_stats,
        "serve": command_serve,
    }
    try:
        return handlers[args.command](args, ctx)
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130


__all__ = [
    "CLIContext",
    "IngestSummary",
    "build_parser",
    "command_chat",
    "command_ingest",
    "command_serve",
    "command_stats",
    "ingest_knowledge_base",
    "load_or_build_index",
    "main",
    "resolve_context",
]
