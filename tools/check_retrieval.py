"""Retrieval quality checker for the bundled knowledge base.

Answers two questions a client will ask about a keyword-based bot:

1. does each realistic customer question find the *right document*?
2. do questions that are outside the business stay below the hand-off
   threshold instead of producing a confident wrong answer?

Usage::

    python tools/check_retrieval.py             # human-readable report
    python tools/check_retrieval.py --verbose   # also show the top-3 hits

The script exits with status 1 when any case fails, so it can be wired into
a build pipeline. It never touches the database and never writes files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_support_bot.chunking import load_chunks  # noqa: E402
from ai_support_bot.escalation import DEFAULT_SCORE_THRESHOLD  # noqa: E402
from ai_support_bot.retrieval import TfidfIndex, build_index, search  # noqa: E402

KB_DIR = PROJECT_ROOT / "ai_support_bot" / "kb"

#: (business folder, question, expected source document)
POSITIVE_CASES: tuple[tuple[str, str, str], ...] = (
    ("avtoservice", "Сколько стоит замена масла на Toyota Camry?", "avtoservice/02-uslugi-i-ceny.md"),
    ("avtoservice", "Есть ли гарантия на ремонт и сколько она действует?", "avtoservice/04-garantiya.md"),
    ("avtoservice", "Работаете ли вы с юридическими лицами по безналу?", "avtoservice/05-zapis-i-oplata.md"),
    ("avtoservice", "Сколько по времени занимает кузовной ремонт?", "avtoservice/03-sroki-rabot.md"),
    ("avtoservice", "Как перенести запись на ремонт?", "avtoservice/05-zapis-i-oplata.md"),
    ("avtoservice", "Почему ремонт может задержаться и есть ли компенсация?", "avtoservice/03-sroki-rabot.md"),
    ("avtoservice", "Сколько стоит замена тормозных колодок?", "avtoservice/02-uslugi-i-ceny.md"),
    ("avtoservice", "Вы работаете в субботу?", "avtoservice/01-o-nas.md"),
    ("stomatologiya", "Сколько стоит лечение кариеса под ключ?", "stomatologiya/02-uslugi-i-ceny.md"),
    ("stomatologiya", "Можно ли ставить имплант при сахарном диабете?", "stomatologiya/03-protivopokazaniya.md"),
    ("stomatologiya", "Как записаться на приём и можно ли перенести время?", "stomatologiya/04-zapis-i-priem.md"),
    ("stomatologiya", "Вернут ли деньги, если я откажусь от лечения?", "stomatologiya/05-garantiya-i-vozvrat.md"),
    ("stomatologiya", "Есть ли гарантия на пломбы и коронки?", "stomatologiya/05-garantiya-i-vozvrat.md"),
    ("stomatologiya", "Сколько стоит имплант зуба?", "stomatologiya/02-uslugi-i-ceny.md"),
    ("stomatologiya", "Даёте ли вы гарантию на лечение за границей?", "stomatologiya/05-garantiya-i-vozvrat.md"),
    ("shop", "Сколько идёт доставка в другой город и сколько стоит?", "shop/02-dostavka.md"),
    ("shop", "Как вернуть товар и когда придут деньги?", "shop/04-vozvrat-i-obmen.md"),
    ("shop", "Можно ли обменять товар, который не подошёл?", "shop/04-vozvrat-i-obmen.md"),
    ("shop", "Какие способы оплаты вы принимаете?", "shop/03-oplata.md"),
    ("shop", "Что делать, если товар пришёл с браком?", "shop/05-garantiya-i-brak.md"),
    ("shop", "Есть ли у вас рассрочка на технику?", "shop/03-oplata.md"),
)

#: (business folder, question) -- must stay below the hand-off threshold.
NEGATIVE_CASES: tuple[tuple[str, str], ...] = (
    ("avtoservice", "А вы делаете скидку на абонемент на год?"),
    ("avtoservice", "Какая погода завтра в Москве?"),
    ("avtoservice", "Ремонтируете ли вы мотоциклы?"),
    ("stomatologiya", "Можно ли оплатить лечение материнским капиталом?"),
    ("shop", "Есть ли у вас программа лояльности для постоянных клиентов?"),
    ("shop", "Сколько стоит билет на самолёт до Сочи?"),
)

#: Questions only *partly* covered by the knowledge base. TF-IDF matches the
#: words it knows and ignores the qualifier it does not, so the bot answers
#: from a formally similar section instead of handing over. A coverage rule
#: cannot fix this: measured coverage of correct answers goes as low as 0.50,
#: while this case sits at 0.67, so no threshold separates the two without
#: breaking dozens of good questions. Reported here on purpose, honestly.
LIMITATION_CASES: tuple[tuple[str, str], ...] = (
    ("shop", "Вы принимаете карты UnionPay?"),
    ("avtoservice", "Делаете ли вы чип-тюнинг двигателя?"),
)


def _top_source(index: TfidfIndex, question: str, business: str) -> tuple[str, float]:
    """Return the best-matching document of *business* and its score."""
    hits = search(index, question, top_k=3, businesses=[business])
    if not hits:
        return "<нет совпадений>", 0.0
    return hits[0].chunk.source, hits[0].score


def run(verbose: bool = False, threshold: float = DEFAULT_SCORE_THRESHOLD) -> int:
    """Check every case and print a Russian report; return the exit code."""
    chunks = load_chunks(KB_DIR)
    index = build_index(chunks)
    print("Проверка качества поиска по базе знаний")
    print(f"Фрагментов: {index.size}, терминов в словаре: {index.vocabulary_size}")
    print(f"Порог передачи оператору: {threshold:.2f}\n")

    print(f"Вопросы по базе знаний ({len(POSITIVE_CASES)} шт.):")
    positive_failures = 0
    for business, question, expected in POSITIVE_CASES:
        source, score = _top_source(index, question, business)
        ok = source == expected
        positive_failures += not ok
        mark = "OK  " if ok else "МИМО"
        print(f"  [{mark}] {business}: {question}")
        print(f"          найдено: {source} ({score:.3f}), ожидалось: {expected}")
        if verbose and not ok:
            for hit in index.search(question, top_k=3, businesses=[business]):
                print(f"            {hit.score:.3f} {hit.chunk.source} «{hit.chunk.heading}»")

    total_positive = len(POSITIVE_CASES)
    print(
        f"\nНайдено верных документов: {total_positive - positive_failures}/{total_positive} "
        f"({100 * (total_positive - positive_failures) / total_positive:.0f} %)"
    )

    print(f"\nВопросы вне базы знаний ({len(NEGATIVE_CASES)} шт., должны уходить оператору):")
    negative_failures = 0
    for business, question in NEGATIVE_CASES:
        source, score = _top_source(index, question, business)
        ok = score < threshold
        negative_failures += not ok
        mark = "OK  " if ok else "РИСК"
        print(f"  [{mark}] {business}: {question}")
        print(f"          лучший результат {score:.3f} < {threshold:.2f}? → {'да' if ok else 'нет'}")
        if verbose and not ok:
            for hit in index.search(question, top_k=3, businesses=[business]):
                print(f"            {hit.score:.3f} {hit.chunk.source} «{hit.chunk.heading}»")

    total_negative = len(NEGATIVE_CASES)
    print(f"\nОсталось за порогом: {total_negative - negative_failures}/{total_negative}")

    print(f"\nИзвестные ограничения TF-IDF ({len(LIMITATION_CASES)} шт., не считаются ошибкой):")
    for business, question in LIMITATION_CASES:
        source, score = _top_source(index, question, business)
        verdict = "ушёл бы оператору" if score < threshold else "ответит по формально похожему разделу"
        print(f"  · {business}: {question}")
        print(f"    {source} ({score:.3f}) — {verdict}")

    failures = positive_failures + negative_failures
    print("\nИТОГ: " + ("проверка пройдена" if failures == 0 else f"проблем: {failures}"))
    return 0 if failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the checker."""
    parser = argparse.ArgumentParser(description="Проверка качества поиска по базе знаний")
    parser.add_argument("--verbose", action="store_true", help="показывать топ-3 фрагмента для промахов")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD, help="порог передачи оператору")
    args = parser.parse_args(argv)
    return run(verbose=args.verbose, threshold=args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
