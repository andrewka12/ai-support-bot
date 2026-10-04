"""Telegram Bot API client based on :mod:`urllib` long polling.

The real bot is optional: the offline console chat ships the same pipeline
without a token. When ``TELEGRAM_BOT_TOKEN`` is missing the client refuses to
start with an actionable Russian message instead of a traceback.

Only the standard library is used, so the demo can be deployed with a bare
Python interpreter.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .analytics import collect_stats
from .pipeline import SupportPipeline
from .retrieval import TfidfIndex
from .storage import DialogueStore

#: Telegram API root.
TELEGRAM_API_BASE: str = "https://api.telegram.org"

#: Telegram hard limit is 4096 characters; a margin keeps formatting safe.
MAX_MESSAGE_LENGTH: int = 4000

#: Long-polling timeout in seconds.
DEFAULT_POLL_TIMEOUT: int = 25

#: Environment variable holding the bot token.
TOKEN_ENV_VAR: str = "TELEGRAM_BOT_TOKEN"

#: Environment variable holding the operator chat id.
OPERATOR_ENV_VAR: str = "OPERATOR_CHAT_ID"

TOKEN_HELP_RU: str = (
    "Не задан токен Telegram-бота.\n"
    "1. Откройте @BotFather в Telegram и создайте бота командой /newbot.\n"
    "2. Скопируйте токен вида 123456:AA...\n"
    "3. Запустите: set TELEGRAM_BOT_TOKEN=<токен>  (Windows)\n"
    "             export TELEGRAM_BOT_TOKEN=<токен>  (Linux/macOS)\n"
    "   и повторите: python -m ai_support_bot serve\n"
    "Демо без токена: python -m ai_support_bot chat"
)

START_RU: str = (
    "Здравствуйте! Я ассистент службы поддержки и отвечаю по базе знаний компании.\n"
    "Спросите, что вас интересует: цены, сроки, гарантия, запись, доставка, возврат.\n"
    "Если я не найду ответ — передам вопрос оператору.\n\n"
    "Доступные команды: /help, /kb, /stats, /operator"
)

HELP_RU: str = (
    "Что я умею:\n"
    "• отвечаю на вопросы по базе знаний компании;\n"
    "• честно говорю, когда ответа нет, и передаю вопрос оператору;\n"
    "• создаю обращение с номером, чтобы вопрос не потерялся.\n\n"
    "Команды:\n"
    "/start — начать сначала\n"
    "/kb — показать или сменить направление (базу знаний)\n"
    "/stats — краткая статистика по диалогам\n"
    "/operator — сразу позвать живого оператора\n"
    "/help — эта справка"
)


class BotConfigurationError(RuntimeError):
    """Raised when the bot cannot start because of its configuration."""


class TelegramApiError(RuntimeError):
    """Raised when Telegram answers with ``ok: false`` or a transport error."""


@dataclass(frozen=True)
class BotConfig:
    """Configuration of the Telegram client."""

    token: str
    api_base: str = TELEGRAM_API_BASE
    poll_timeout: int = DEFAULT_POLL_TIMEOUT
    operator_chat_id: str | None = None

    @classmethod
    def from_env(
        cls,
        token: str | None = None,
        *,
        operator_chat_id: str | None = None,
    ) -> "BotConfig":
        """Build a configuration from the environment.

        Raises:
            BotConfigurationError: when no token is available.
        """
        resolved = (token or os.environ.get(TOKEN_ENV_VAR, "")).strip()
        if not resolved:
            raise BotConfigurationError(TOKEN_HELP_RU)
        return cls(
            token=resolved,
            api_base=os.environ.get("TELEGRAM_API_BASE", TELEGRAM_API_BASE).rstrip("/"),
            poll_timeout=int(os.environ.get("TELEGRAM_POLL_TIMEOUT", DEFAULT_POLL_TIMEOUT)),
            operator_chat_id=(
                operator_chat_id or os.environ.get(OPERATOR_ENV_VAR, "").strip() or None
            ),
        )


def split_message(text: str, *, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split *text* into chunks Telegram accepts, preferring line breaks."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


def parse_command(text: str) -> tuple[str, str]:
    """Split a Telegram message into ``(command, argument)``.

    ``/kb@MyBot shop`` becomes ``("kb", "shop")``; plain text returns
    ``("", text)``. Commands are lower-cased.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "", stripped
    head, _, argument = stripped.partition(" ")
    command = head[1:].split("@", 1)[0].lower()
    return command, argument.strip()


class TelegramBot:
    """Long-polling Telegram bot on top of :class:`SupportPipeline`."""

    def __init__(
        self,
        config: BotConfig,
        pipeline: SupportPipeline,
        *,
        index: TfidfIndex | None = None,
        store: DialogueStore | None = None,
        out: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.index = index or pipeline.index
        self.store = store
        self._out = out
        self._chat_business: dict[str, str] = {}
        self._operator_reported = False

    # ------------------------------------------------------------------ public

    def run(self, *, once: bool = False) -> None:
        """Poll Telegram for updates until interrupted.

        Args:
            once: Fetch one batch of updates (non-blocking) and return.
        """
        me = self._call("getMe")
        username = me.get("username", "?") if isinstance(me, dict) else "?"
        self._out(f"Бот @{username} запущен. Ожидаю сообщения (Ctrl+C для выхода).")
        offset: int | None = None
        failures = 0
        while True:
            try:
                updates = self._get_updates(offset, timeout=0 if once else self.config.poll_timeout)
                failures = 0
            except TelegramApiError as error:
                failures += 1
                delay = min(60, 5 * failures)
                self._out(f"Сбой связи с Telegram: {error}. Повтор через {delay} с.")
                if once:
                    raise
                time.sleep(delay)
                continue
            for update in updates:
                offset = int(update["update_id"]) + 1
                self.handle_update(update)
            if once:
                return

    def handle_update(self, update: dict[str, Any]) -> None:
        """Process a single Telegram update."""
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if not text or not chat_id:
            return
        self.handle_text(chat_id, str(text))

    def handle_text(self, chat_id: str, text: str) -> None:
        """Route one incoming message to a command handler or the pipeline."""
        command, argument = parse_command(text)
        if command in {"start", "help"}:
            self.send_message(chat_id, START_RU if command == "start" else HELP_RU)
            return
        if command == "kb":
            self._handle_kb(chat_id, argument)
            return
        if command == "stats":
            self.send_message(chat_id, self._stats_text())
            return
        if command == "operator":
            reply = self.pipeline.answer("позовите оператора", chat_id=chat_id)
            self._deliver(chat_id, reply)
            return
        if command:
            self.send_message(chat_id, "Неизвестная команда. " + HELP_RU)
            return
        reply = self.pipeline.answer(
            text,
            chat_id=chat_id,
            business=self._chat_business.get(chat_id),
        )
        self._deliver(chat_id, reply)

    # ---------------------------------------------------------------- internals

    def _deliver(self, chat_id: str, reply: Any) -> None:
        """Send a pipeline reply and forward escalations to the operator."""
        self.send_message(chat_id, reply.text)
        if reply.resolved:
            return
        operator_message = reply.operator_message
        if self.config.operator_chat_id:
            self.send_message(self.config.operator_chat_id, operator_message)
            return
        if not self._operator_reported:
            self._out(
                "Оператор не настроен: задайте OPERATOR_CHAT_ID, чтобы получать "
                "такие обращения в Telegram. Пока они только пишутся в базу."
            )
            self._operator_reported = True
        self._out(operator_message)

    def _handle_kb(self, chat_id: str, argument: str) -> None:
        """Show or switch the knowledge-base scope of a chat."""
        businesses = sorted({chunk.business for chunk in self.index.chunks})
        if not argument:
            current = self._chat_business.get(chat_id, "все направления")
            self.send_message(
                chat_id,
                "Доступные направления: " + ", ".join(businesses) + ".\n"
                f"Текущее: {current}.\n"
                "Сменить: /kb <направление>. Вернуться ко всем: /kb all",
            )
            return
        choice = argument.strip().lower()
        if choice in {"all", "все", "все направления"}:
            self._chat_business.pop(chat_id, None)
            self.send_message(chat_id, "Хорошо, отвечаю по всем направлениям.")
            return
        if choice not in businesses:
            self.send_message(
                chat_id,
                f"Не знаю направление «{argument}». Доступные: " + ", ".join(businesses),
            )
            return
        self._chat_business[chat_id] = choice
        self.send_message(chat_id, f"Переключился на направление «{choice}».")

    def _stats_text(self) -> str:
        """Short statistics message for the ``/stats`` command."""
        if self.store is None:
            return "Статистика недоступна: база диалогов не подключена."
        report = collect_stats(self.store, days=7)
        if report.total_questions == 0:
            return "За последние 7 дней диалогов не было."
        return (
            "Статистика за 7 дней:\n"
            f"• вопросов: {report.total_questions}\n"
            f"• решено без оператора: {report.resolved} ({report.auto_resolved_pct:.1f} %)\n"
            f"• передано оператору: {report.escalated}\n"
            f"• средняя уверенность: {report.average_confidence:.2f}\n"
            "Подробный отчёт: python -m ai_support_bot stats"
        )

    def _get_updates(self, offset: int | None, *, timeout: int) -> list[dict[str, Any]]:
        """Fetch pending updates, optionally blocking up to *timeout* seconds."""
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message", "edited_message"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload, timeout=timeout + 10)
        return list(result) if isinstance(result, list) else []

    def _call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> Any:
        """Call a Telegram Bot API method and return its ``result`` field."""
        request = urllib.request.Request(
            url=f"{self.config.api_base}/bot{self.config.token}/{method}",
            data=json.dumps(payload or {}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 401:
                raise BotConfigurationError(
                    "Telegram отклонил токен (401 Unauthorized). " + TOKEN_HELP_RU
                ) from error
            raise TelegramApiError(f"HTTP {error.code}: {body[:200]}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TelegramApiError(str(error)) from error

        try:
            data = json.loads(raw)
        except ValueError as error:
            raise TelegramApiError("Telegram вернул не-JSON ответ.") from error
        if not data.get("ok"):
            raise TelegramApiError(str(data.get("description", "неизвестная ошибка")))
        return data.get("result")

    def send_message(self, chat_id: str, text: str) -> None:
        """Send *text* to *chat_id*, splitting it when it is too long."""
        for part in split_message(text):
            if not part:
                continue
            self._call("sendMessage", {"chat_id": chat_id, "text": part})


def run_bot(
    pipeline: SupportPipeline,
    *,
    token: str | None = None,
    operator_chat_id: str | None = None,
    store: DialogueStore | None = None,
    once: bool = False,
    out: Callable[[str], None] = print,
) -> None:
    """Entry point used by the CLI: configure and run the Telegram bot."""
    config = BotConfig.from_env(token, operator_chat_id=operator_chat_id)
    bot = TelegramBot(config, pipeline, store=store, out=out)
    bot.run(once=once)


__all__ = [
    "BotConfig",
    "BotConfigurationError",
    "HELP_RU",
    "START_RU",
    "TOKEN_ENV_VAR",
    "TOKEN_HELP_RU",
    "TelegramApiError",
    "TelegramBot",
    "parse_command",
    "run_bot",
    "split_message",
]
