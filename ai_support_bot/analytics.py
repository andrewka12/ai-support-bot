"""Analytics over the stored dialogue log.

The report answers the questions a business owner actually asks: how many
customers the bot solved on its own, what it failed to answer, how confident
the answers were, why tickets were created and how the load is distributed
across days.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .escalation import REASON_LABELS_RU
from .storage import AnswerRecord, DialogueStore, Ticket, UnresolvedRecord

#: How many weak questions the report lists by default.
DEFAULT_TOP_WEAK: int = 5

#: Width of the per-day bar chart, in characters.
BAR_WIDTH: int = 20


@dataclass(frozen=True)
class WeakQuestion:
    """A question the bot could not answer, aggregated over the period."""

    question: str
    count: int
    average_score: float


@dataclass(frozen=True)
class DayStat:
    """Per-day dialogue counters."""

    day: str
    questions: int
    resolved: int
    escalated: int


@dataclass(frozen=True)
class StatsReport:
    """Everything ``python -m ai_support_bot stats`` prints."""

    period_days: int
    period_start: str
    period_end: str
    total_questions: int
    resolved: int
    escalated: int
    auto_resolved_pct: float
    average_confidence: float
    average_confidence_resolved: float
    tickets_total: int
    tickets_open: int
    tickets_by_reason: dict[str, int] = field(default_factory=dict)
    weak_questions: tuple[WeakQuestion, ...] = ()
    per_day: tuple[DayStat, ...] = ()


def _day_of(stamp: str) -> str:
    """Return the UTC calendar day of an ISO-8601 timestamp."""
    try:
        return datetime.fromisoformat(stamp).astimezone(timezone.utc).date().isoformat()
    except ValueError:
        return stamp[:10]


def collect_stats(
    store: DialogueStore,
    *,
    days: int = 7,
    now: datetime | None = None,
    top_weak: int = DEFAULT_TOP_WEAK,
) -> StatsReport:
    """Aggregate the dialogue log into a :class:`StatsReport`.

    Args:
        store: Open dialogue store.
        days: Length of the reporting window, in days (today included).
        now: Reference moment, defaults to the current UTC time.
        top_weak: How many unanswered questions to list.

    Returns:
        The aggregated report; all percentages are computed from the rows
        inside the window only.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end_day = reference.date()
    start_day = end_day - timedelta(days=days - 1)
    since = datetime.combine(start_day, datetime.min.time(), tzinfo=timezone.utc)

    answers: list[AnswerRecord] = store.fetch_answers(since=since)
    unresolved: list[UnresolvedRecord] = store.fetch_unresolved(since=since)
    tickets: list[Ticket] = store.fetch_tickets(since=since)

    total = len(answers)
    resolved_records = [record for record in answers if record.resolved]
    resolved = len(resolved_records)
    escalated = total - resolved

    average_confidence = (
        sum(record.confidence for record in answers) / total if total else 0.0
    )
    average_confidence_resolved = (
        sum(record.confidence for record in resolved_records) / resolved if resolved else 0.0
    )

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in unresolved:
        grouped[record.question].append(record.top_score)
    weak = [
        WeakQuestion(
            question=question,
            count=len(scores),
            average_score=sum(scores) / len(scores),
        )
        for question, scores in grouped.items()
    ]
    weak.sort(key=lambda item: (-item.count, item.average_score, item.question))

    per_day_counts: dict[str, list[int]] = {
        (start_day + timedelta(days=offset)).isoformat(): [0, 0, 0] for offset in range(days)
    }
    for record in answers:
        bucket = per_day_counts.get(_day_of(record.created_at))
        if bucket is None:
            continue
        bucket[0] += 1
        bucket[1 if record.resolved else 2] += 1

    per_day = tuple(
        DayStat(day=day, questions=values[0], resolved=values[1], escalated=values[2])
        for day, values in sorted(per_day_counts.items())
    )

    return StatsReport(
        period_days=days,
        period_start=start_day.isoformat(),
        period_end=end_day.isoformat(),
        total_questions=total,
        resolved=resolved,
        escalated=escalated,
        auto_resolved_pct=(100.0 * resolved / total) if total else 0.0,
        average_confidence=average_confidence,
        average_confidence_resolved=average_confidence_resolved,
        tickets_total=len(tickets),
        tickets_open=sum(1 for ticket in tickets if ticket.status == "open"),
        tickets_by_reason=dict(Counter(ticket.reason for ticket in tickets)),
        weak_questions=tuple(weak[:top_weak]),
        per_day=per_day,
    )


def _format_day(day: str) -> str:
    """Render an ISO day as ``дд.мм``."""
    try:
        return datetime.fromisoformat(day).strftime("%d.%m")
    except ValueError:
        return day


def _bar(value: int, maximum: int) -> str:
    """Draw a text bar scaled to *maximum*."""
    if value <= 0:
        return "\u00b7"
    filled = max(1, round(BAR_WIDTH * value / maximum)) if maximum else 0
    return "\u2588" * filled


def _reason_label(reason: str) -> str:
    """Russian label for a stored escalation reason code."""
    label = REASON_LABELS_RU.get(reason)
    if label is None:
        return reason
    return f"{label} [{reason}]"


def render_report(report: StatsReport) -> str:
    """Render *report* as the Russian text printed by ``stats``."""
    line = "=" * 64
    lines: list[str] = [
        line,
        " Аналитика диалогов ИИ-бота поддержки",
        f" Период: последние {report.period_days} дн. "
        f"({_format_day(report.period_start)} \u2014 {_format_day(report.period_end)})",
        line,
    ]

    if report.total_questions == 0:
        lines.append("За выбранный период нет ни одного диалога.")
        lines.append("Запустите демо-чат: python -m ai_support_bot chat")
        lines.append(line)
        return "\n".join(lines)

    lines.extend(
        [
            f"Всего вопросов:                {report.total_questions}",
            f"Решено без оператора:          {report.resolved} "
            f"({report.auto_resolved_pct:.1f} %)",
            f"Передано оператору:            {report.escalated} "
            f"({100.0 - report.auto_resolved_pct:.1f} %)",
            f"Средняя уверенность ответа:    {report.average_confidence:.2f}"
            f"  (по автоответам: {report.average_confidence_resolved:.2f})",
            f"Тикетов создано:               {report.tickets_total} "
            f"(открытых: {report.tickets_open})",
            "",
            "Топ вопросов без ответа:",
        ]
    )
    if report.weak_questions:
        for number, item in enumerate(report.weak_questions, start=1):
            lines.append(
                f"  {number}. \u00ab{item.question}\u00bb \u2014 {item.count} раз(а), "
                f"средняя релевантность {item.average_score:.2f}"
            )
    else:
        lines.append("  \u00b7 таких вопросов не было")

    lines.extend(["", "Тикеты по причинам:"])
    if report.tickets_by_reason:
        for reason, count in sorted(
            report.tickets_by_reason.items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"  {_reason_label(reason)}: {count}")
    else:
        lines.append("  \u00b7 тикетов не было")

    lines.extend(["", "Динамика по дням (вопросов в день):"])
    maximum = max((item.questions for item in report.per_day), default=0)
    for item in report.per_day:
        lines.append(
            f"  {_format_day(item.day)}  {_bar(item.questions, maximum):<{BAR_WIDTH}} "
            f"{item.questions:>3}  (авто: {item.resolved}, оператор: {item.escalated})"
        )

    lines.extend(
        [
            line,
            "Как улучшить показатели: добавьте в базу знаний ответы на вопросы из топа",
            "и повторно запустите: python -m ai_support_bot ingest",
            line,
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DayStat",
    "StatsReport",
    "WeakQuestion",
    "collect_stats",
    "render_report",
]
