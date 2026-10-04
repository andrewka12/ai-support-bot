"""SQLite storage for dialogues, answers, unresolved questions and tickets.

The database lives in ``data/dialogues.db`` and is git-ignored. Every write
method accepts an explicit ``created_at`` timestamp so tests and the demo
seeder can build a deterministic history.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: Default location of the database, relative to the project root.
DEFAULT_DB_PATH: str = "data/dialogues.db"

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'bot', 'operator')),
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    answer     TEXT    NOT NULL,
    confidence REAL    NOT NULL,
    provider   TEXT    NOT NULL,
    source     TEXT    NOT NULL DEFAULT '',
    resolved   INTEGER NOT NULL DEFAULT 1,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS unresolved (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    top_score  REAL    NOT NULL,
    reason     TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT    NOT NULL,
    reason     TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'open',
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_answers_created ON answers (created_at);
CREATE INDEX IF NOT EXISTS idx_unresolved_created ON unresolved (created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets (created_at);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages (chat_id, id);
"""


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(moment: datetime) -> str:
    """Format *moment* as an ISO-8601 string used across the database."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Ticket:
    """An operator ticket."""

    id: int
    chat_id: str
    reason: str
    question: str
    status: str
    created_at: str


@dataclass(frozen=True)
class AnswerRecord:
    """A stored bot answer."""

    id: int
    chat_id: str
    question: str
    answer: str
    confidence: float
    provider: str
    source: str
    resolved: bool
    created_at: str


@dataclass(frozen=True)
class UnresolvedRecord:
    """A stored question the bot could not answer."""

    id: int
    chat_id: str
    question: str
    top_score: float
    reason: str
    created_at: str


class DialogueStore:
    """Thin, typed wrapper around the SQLite dialogue log."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ writes

    def log_message(
        self,
        chat_id: str,
        role: str,
        text: str,
        *,
        created_at: str | None = None,
    ) -> int:
        """Append a chat message and return its row id."""
        cursor = self._conn.execute(
            "INSERT INTO messages (chat_id, role, text, created_at) VALUES (?, ?, ?, ?)",
            (str(chat_id), role, text, created_at or iso(utc_now())),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def log_answer(
        self,
        chat_id: str,
        question: str,
        answer: str,
        confidence: float,
        provider: str,
        *,
        source: str = "",
        resolved: bool = True,
        created_at: str | None = None,
    ) -> int:
        """Store a bot answer and return its row id."""
        cursor = self._conn.execute(
            """
            INSERT INTO answers
                (chat_id, question, answer, confidence, provider, source, resolved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(chat_id),
                question,
                answer,
                float(confidence),
                provider,
                source,
                1 if resolved else 0,
                created_at or iso(utc_now()),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def log_unresolved(
        self,
        chat_id: str,
        question: str,
        top_score: float,
        reason: str,
        *,
        created_at: str | None = None,
    ) -> int:
        """Store a question the bot could not answer on its own."""
        cursor = self._conn.execute(
            """
            INSERT INTO unresolved (chat_id, question, top_score, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(chat_id), question, float(top_score), reason, created_at or iso(utc_now())),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def create_ticket(
        self,
        chat_id: str,
        reason: str,
        question: str,
        *,
        status: str = "open",
        created_at: str | None = None,
    ) -> int:
        """Create an operator ticket and return its id."""
        cursor = self._conn.execute(
            """
            INSERT INTO tickets (chat_id, reason, question, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(chat_id), reason, question, status, created_at or iso(utc_now())),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def close_ticket(self, ticket_id: int) -> bool:
        """Mark a ticket as closed; return ``True`` when a row changed."""
        cursor = self._conn.execute(
            "UPDATE tickets SET status = 'closed' WHERE id = ? AND status != 'closed'",
            (int(ticket_id),),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------- reads

    def has_data(self) -> bool:
        """Whether the dialogue log contains at least one message."""
        row = self._conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
        return row is not None

    def count(self, table: str) -> int:
        """Return the number of rows in one of the known tables."""
        if table not in {"messages", "answers", "unresolved", "tickets"}:
            raise ValueError(f"Unknown table: {table!r}")
        row = self._conn.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()
        return int(row["total"])

    def fetch_answers(self, *, since: datetime | None = None) -> list[AnswerRecord]:
        """Return stored answers, optionally limited to a time window."""
        query = "SELECT * FROM answers"
        params: list[Any] = []
        if since is not None:
            query += " WHERE created_at >= ?"
            params.append(iso(since))
        query += " ORDER BY created_at, id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            AnswerRecord(
                id=int(row["id"]),
                chat_id=str(row["chat_id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                confidence=float(row["confidence"]),
                provider=str(row["provider"]),
                source=str(row["source"]),
                resolved=bool(row["resolved"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def fetch_unresolved(self, *, since: datetime | None = None) -> list[UnresolvedRecord]:
        """Return stored unresolved questions, optionally in a time window."""
        query = "SELECT * FROM unresolved"
        params: list[Any] = []
        if since is not None:
            query += " WHERE created_at >= ?"
            params.append(iso(since))
        query += " ORDER BY created_at, id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            UnresolvedRecord(
                id=int(row["id"]),
                chat_id=str(row["chat_id"]),
                question=str(row["question"]),
                top_score=float(row["top_score"]),
                reason=str(row["reason"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def fetch_tickets(self, *, since: datetime | None = None) -> list[Ticket]:
        """Return tickets, optionally limited to a time window."""
        query = "SELECT * FROM tickets"
        params: list[Any] = []
        if since is not None:
            query += " WHERE created_at >= ?"
            params.append(iso(since))
        query += " ORDER BY created_at, id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            Ticket(
                id=int(row["id"]),
                chat_id=str(row["chat_id"]),
                reason=str(row["reason"]),
                question=str(row["question"]),
                status=str(row["status"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def fetch_messages(self, chat_id: str, *, limit: int = 20) -> list[tuple[str, str]]:
        """Return the last *limit* ``(role, text)`` pairs of a chat."""
        rows = self._conn.execute(
            """
            SELECT role, text FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (str(chat_id), int(limit)),
        ).fetchall()
        return [(str(row["role"]), str(row["text"])) for row in reversed(rows)]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# --------------------------------------------------------------------- demo log

#: Curated demonstration dialogues: question, business folder, resolved flag,
#: confidence, escalation reason and the knowledge-base file the answer came from.
_DEMO_DIALOGUES: tuple[tuple[str, str, bool, float, str, str], ...] = (
    ("Сколько стоит замена масла в двигателе?", "avtoservice", True, 0.71, "none", "02-uslugi-i-ceny.md"),
    ("Вы работаете в воскресенье?", "avtoservice", True, 0.66, "none", "01-o-nas.md"),
    ("Как записаться на ремонт и можно ли перенести запись?", "avtoservice", True, 0.58, "none", "05-zapis-i-oplata.md"),
    ("Какая гарантия на ремонт двигателя?", "avtoservice", True, 0.63, "none", "04-garantiya.md"),
    ("Сколько времени занимает замена ремня ГРМ?", "avtoservice", True, 0.55, "none", "03-sroki-rabot.md"),
    ("Вы обслуживаете Hyundai Solaris?", "avtoservice", True, 0.61, "none", "01-o-nas.md"),
    ("Сколько стоит лечение кариеса?", "stomatologiya", True, 0.68, "none", "02-uslugi-i-ceny.md"),
    ("Есть ли противопоказания к имплантации зубов?", "stomatologiya", True, 0.57, "none", "03-protivopokazaniya.md"),
    ("С какого возраста вы принимаете детей?", "stomatologiya", True, 0.52, "none", "04-zapis-i-priem.md"),
    ("Даёте ли гарантию на коронки?", "stomatologiya", True, 0.60, "none", "05-garantiya-i-vozvrat.md"),
    ("Можно ли лечить зубы при беременности?", "stomatologiya", True, 0.49, "none", "03-protivopokazaniya.md"),
    ("Сколько стоит доставка по Москве?", "shop", True, 0.70, "none", "02-dostavka.md"),
    ("Можно вернуть товар, если он не подошёл?", "shop", True, 0.64, "none", "04-vozvrat-i-obmen.md"),
    ("Как оплатить заказ картой или в рассрочку?", "shop", True, 0.59, "none", "03-oplata.md"),
    ("Сколько идёт доставка в Казань?", "shop", True, 0.66, "none", "02-dostavka.md"),
    ("Что делать, если товар пришёл повреждённым?", "shop", True, 0.54, "none", "05-garantiya-i-brak.md"),
    ("А вы делаете скидку на абонемент?", "avtoservice", False, 0.21, "low_score", ""),
    ("Есть ли у вас программа лояльности для постоянных клиентов?", "shop", False, 0.19, "low_score", ""),
    ("Можно ли оплатить лечение материнским капиталом?", "stomatologiya", False, 0.14, "low_score", ""),
    ("Позовите оператора, пожалуйста", "avtoservice", False, 0.42, "trigger_keyword", ""),
    ("Верните деньги за некачественный ремонт, иначе пойду в суд", "avtoservice", False, 0.38, "trigger_keyword", ""),
    ("Хочу пожаловаться на грубость администратора", "shop", False, 0.33, "trigger_keyword", ""),
    ("Обман! Вы обещали доставить вчера, а курьера нет", "shop", False, 0.29, "trigger_keyword", ""),
    ("А если я не согласен с заключением врача?", "stomatologiya", False, 0.17, "low_score", ""),
    ("Есть ли у вас лечение зубов за границей?", "stomatologiya", False, 0.16, "low_score", ""),
    ("Сколько стоит абонемент на год?", "avtoservice", False, 0.20, "low_score", ""),
    ("А можно дешевле?", "avtoservice", False, 0.11, "repeated_unresolved", ""),
    ("А можно дешевле?", "avtoservice", False, 0.10, "repeated_unresolved", ""),
    ("А есть что-то похожее, но недорого?", "shop", False, 0.13, "repeated_unresolved", ""),
)


def seed_demo_dialogues(
    store: DialogueStore,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> int:
    """Fill *store* with a realistic demonstration dialogue log.

    The data is deterministic apart from the timestamps, which are spread
    over the last *days* days so that ``stats`` shows a lively per-day
    chart on a fresh clone. Returns the number of recorded questions.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    reference = (now or utc_now()).astimezone(timezone.utc).replace(microsecond=0)
    rng = random.Random(20240517)

    # Spread the dialogues over the window: every day gets at least one
    # question, the rest are distributed with uneven weights so the per-day
    # chart of ``stats`` looks like a real week instead of a flat line.
    total = len(_DEMO_DIALOGUES)
    first_days = list(range(days))
    rng.shuffle(first_days)
    weights = [0.5 + 0.3 * ((day * 3) % 5) for day in range(days)]

    for index, (question, business, resolved, confidence, reason, source) in enumerate(
        _DEMO_DIALOGUES
    ):
        if index < days:
            days_ago = first_days[index]
        else:
            days_ago = rng.choices(range(days), weights=weights, k=1)[0]
        moment = (reference - timedelta(days=days_ago)).replace(
            hour=9 + (index % 9),
            minute=(index * 7) % 60,
            second=0,
        )
        if moment > reference:
            moment = reference
        stamp = iso(moment)
        chat_id = f"demo-{business}-{index % 4 + 1}"
        store.log_message(chat_id, "user", question, created_at=stamp)
        if resolved:
            answer_text = (
                f"Ответ по базе знаний: {business}/{source}. "
                "Полный текст фрагмента есть в файле базы знаний."
            )
            store.log_answer(
                chat_id,
                question,
                answer_text,
                confidence,
                "echo",
                source=f"{business}/{source}",
                resolved=True,
                created_at=stamp,
            )
        else:
            jitter = round(rng.uniform(0.0, 0.01), 3)
            store.log_answer(
                chat_id,
                question,
                "Передаю ваш вопрос оператору.",
                confidence,
                "echo",
                source="",
                resolved=False,
                created_at=stamp,
            )
            store.log_unresolved(
                chat_id,
                question,
                top_score=round(max(0.0, confidence - 0.05 + jitter), 3),
                reason=reason,
                created_at=stamp,
            )
            store.create_ticket(chat_id, reason, question, created_at=stamp)
        store.log_message(chat_id, "bot", "Ответ отправлен клиенту.", created_at=stamp)
    return len(_DEMO_DIALOGUES)


def demo_dialogue_count() -> int:
    """Number of dialogues in the bundled demonstration log."""
    return len(_DEMO_DIALOGUES)


__all__ = [
    "AnswerRecord",
    "DEFAULT_DB_PATH",
    "DialogueStore",
    "SCHEMA",
    "Ticket",
    "UnresolvedRecord",
    "demo_dialogue_count",
    "iso",
    "seed_demo_dialogues",
    "utc_now",
]
