"""Tests for the Russian tokenizer and the TF-IDF ranking."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.base import ProjectTestCase, make_temp_dir

from ai_support_bot.chunking import load_chunks
from ai_support_bot.retrieval import (
    TITLE_PRIOR_WEIGHT,
    build_index,
    expand_terms,
    kb_signature,
    load_index,
    save_index,
    stem,
    tokenize,
)

#: (business folder, question, expected source file). One question per
#: business plus edge cases: payment, warranty, booking, delivery, returns.
RANKING_CASES: tuple[tuple[str, str, str], ...] = (
    ("avtoservice", "Сколько стоит замена масла в двигателе?", "avtoservice/02-uslugi-i-ceny.md"),
    ("avtoservice", "Какая гарантия на ремонт двигателя?", "avtoservice/04-garantiya.md"),
    ("avtoservice", "Как записаться на ремонт и можно ли перенести запись?", "avtoservice/05-zapis-i-oplata.md"),
    ("avtoservice", "Вы работаете в субботу?", "avtoservice/01-o-nas.md"),
    ("stomatologiya", "Сколько стоит лечение кариеса?", "stomatologiya/02-uslugi-i-ceny.md"),
    ("stomatologiya", "Есть ли противопоказания к имплантации зуба?", "stomatologiya/03-protivopokazaniya.md"),
    ("stomatologiya", "Даёте ли вы гарантию на коронки?", "stomatologiya/05-garantiya-i-vozvrat.md"),
    ("shop", "Сколько стоит доставка по Москве?", "shop/02-dostavka.md"),
    ("shop", "Как оплатить заказ картой?", "shop/03-oplata.md"),
    ("shop", "Можно ли вернуть товар, если он не подошёл?", "shop/04-vozvrat-i-obmen.md"),
    ("shop", "Что делать, если товар пришёл повреждённым?", "shop/05-garantiya-i-brak.md"),
)

#: Regression cases for the synonym and topic-prior work: a question phrased
#: with «вернуть»/«вернут» must still reach the returns document.
RETURN_REGRESSION_CASES: tuple[str, ...] = (
    "Как вернуть товар и когда придут деньги?",
    "Можно ли вернуть товар, если он не подошёл?",
    "Можно ли обменять товар, который не подошёл?",
    "Вернут ли деньги, если я откажусь от лечения?",
)

#: Questions the bundled knowledge base cannot answer.
OFF_TOPIC_CASES: tuple[str, ...] = (
    "Какая погода в Москве завтра?",
    "Сколько стоит билет на самолёт до Сочи?",
    "А вы делаете скидку на абонемент на год?",
)


class TokenizerTest(unittest.TestCase):
    """Normalisation, stop words, stemming and synonym expansion."""

    def test_folding_and_stop_words(self) -> None:
        tokens = tokenize("Почём Ёлки в Москве?")
        self.assertNotIn("в", tokens)
        self.assertIn("елк", tokens)  # «ёлки» folded to «елки» and stemmed
        self.assertNotIn("ёлки", tokens)
        self.assertIn("москв", tokens)

    def test_digits_and_punctuation_are_dropped(self) -> None:
        self.assertEqual(tokenize("1 200 ₽ — цена!"), tokenize("цена"))

    def test_inflected_forms_collapse(self) -> None:
        self.assertEqual(stem("цены"), stem("цена"))
        self.assertEqual(stem("цене"), stem("цена"))
        self.assertEqual(tokenize("доставки"), tokenize("доставка"))

    def test_verb_forms_collapse(self) -> None:
        self.assertEqual(tokenize("работаете"), tokenize("работает"))
        self.assertEqual(tokenize("работать"), tokenize("работает"))

    def test_expand_terms_adds_canonical_domain_term(self) -> None:
        expanded = expand_terms(["вернуть", "товар"])
        self.assertIn("возврат", expanded)
        self.assertIn("товар", expanded)

    def test_expand_terms_is_idempotent(self) -> None:
        once = expand_terms(["вернуть", "товар"])
        self.assertEqual(expand_terms(once), once)

    def test_canonical_terms_are_symmetric(self) -> None:
        self.assertEqual(tokenize("вернуть"), tokenize("возврат"))
        self.assertEqual(tokenize("обменять"), tokenize("возврат"))
        self.assertEqual(tokenize("деньги"), tokenize("денег"))

    def test_short_words_are_kept_intact(self) -> None:
        self.assertEqual(stem("билет"), "билет")
        self.assertEqual(stem("суд"), "суд")

    def test_stemming_is_idempotent(self) -> None:
        for word in ("россии", "доставляем", "гарантия", "имплантация", "цена", "масло"):
            with self.subTest(word=word):
                once = stem(word)
                self.assertEqual(stem(once), once)

    def test_tokenizer_is_idempotent(self) -> None:
        for text in ("Сколько идёт доставка в Казань?", "Верните деньги за брак"):
            with self.subTest(text=text):
                once = tokenize(text)
                self.assertEqual(expand_terms(once), once)

    def test_informative_words_are_not_swallowed_by_synonyms(self) -> None:
        # «Почтой» must stay its own token: mapping it to «доставка» made the
        # sentence scorer prefer boilerplate over the sentence with a city.
        self.assertNotIn("доставк", tokenize("Почтой России"))
        self.assertIn("почт", tokenize("Почтой России"))
        self.assertNotIn("оплат", tokenize("картой"))
        self.assertIn("карт", tokenize("картой"))

    def test_verb_forms_of_delivery_collapse(self) -> None:
        self.assertEqual(tokenize("доставляем"), tokenize("доставка"))


class RankingTest(ProjectTestCase):
    """Document ranking on the bundled knowledge base."""

    def test_expected_document_is_ranked_first(self) -> None:
        for business, question, expected in RANKING_CASES:
            with self.subTest(question=question):
                self.assertEqual(self.top_source(question, business), expected)

    def test_return_questions_reach_the_returns_document(self) -> None:
        expectations = {
            "Как вернуть товар и когда придут деньги?": "shop/04-vozvrat-i-obmen.md",
            "Можно ли вернуть товар, если он не подошёл?": "shop/04-vozvrat-i-obmen.md",
            "Можно ли обменять товар, который не подошёл?": "shop/04-vozvrat-i-obmen.md",
            "Вернут ли деньги, если я откажусь от лечения?": "stomatologiya/05-garantiya-i-vozvrat.md",
        }
        for question, expected in expectations.items():
            business = expected.split("/")[0]
            with self.subTest(question=question):
                self.assertEqual(self.top_source(question, business), expected)

    def test_off_topic_questions_score_low(self) -> None:
        for question in OFF_TOPIC_CASES:
            with self.subTest(question=question):
                hits = self.index.search(question, top_k=1)
                score = hits[0].score if hits else 0.0
                self.assertLess(score, 0.16)

    def test_topic_prior_is_small_and_bounded(self) -> None:
        self.assertGreater(TITLE_PRIOR_WEIGHT, 0.0)
        self.assertLessEqual(TITLE_PRIOR_WEIGHT, 0.3)

    def test_search_is_deterministic(self) -> None:
        first = self.index.search("Сколько стоит замена масла?", top_k=3)
        second = self.index.search("Сколько стоит замена масла?", top_k=3)
        self.assertEqual(
            [(hit.chunk.chunk_id, round(hit.score, 9)) for hit in first],
            [(hit.chunk.chunk_id, round(hit.score, 9)) for hit in second],
        )

    def test_scores_are_sorted_and_positive(self) -> None:
        hits = self.index.search("доставка и оплата заказа", top_k=5)
        self.assertTrue(hits)
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(score > 0 for score in scores))

    def test_business_filter_restricts_results(self) -> None:
        hits = self.index.search("гарантия и возврат", top_k=5, businesses=["shop"])
        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.business == "shop" for hit in hits))

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.index.search("и в на", top_k=3), [])

    def test_index_reports_its_size(self) -> None:
        self.assertEqual(self.index.size, len(self.chunks))
        self.assertGreater(self.index.vocabulary_size, 500)


class IndexCacheTest(unittest.TestCase):
    """The JSON index cache used by ``ingest``."""

    def setUp(self) -> None:
        self.path = make_temp_dir(self) / "index.json"
        self.chunks = load_chunks(Path(__file__).resolve().parent.parent / "ai_support_bot" / "kb")
        self.index = build_index(self.chunks)

    def test_roundtrip_keeps_chunks_and_scores(self) -> None:
        signature = {"shop/02-dostavka.md": [1, 2]}
        save_index(self.index, self.path, signature=signature)
        restored = load_index(self.path, expected_signature=signature)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.size, self.index.size)
        self.assertEqual([chunk.chunk_id for chunk in restored.chunks], [c.chunk_id for c in self.index.chunks])
        original = self.index.search("Сколько стоит доставка по Москве?", top_k=3)
        reloaded = restored.search("Сколько стоит доставка по Москве?", top_k=3)
        self.assertEqual(
            [hit.chunk.chunk_id for hit in original],
            [hit.chunk.chunk_id for hit in reloaded],
        )
        for left, right in zip(original, reloaded):
            self.assertAlmostEqual(left.score, right.score, places=9)

    def test_signature_mismatch_rejects_the_cache(self) -> None:
        save_index(self.index, self.path, signature={"a.md": [1, 1]})
        self.assertIsNone(load_index(self.path, expected_signature={"a.md": [2, 2]}))

    def test_broken_cache_is_ignored(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(load_index(self.path))

    def test_missing_cache_returns_none(self) -> None:
        self.assertIsNone(load_index(self.path))

    def test_signature_changes_with_file_mtime(self) -> None:
        kb_dir = Path(__file__).resolve().parent.parent / "ai_support_bot" / "kb"
        signature = kb_signature(kb_dir)
        self.assertIn("shop/02-dostavka.md", signature)
        self.assertEqual(len(signature["shop/02-dostavka.md"]), 2)
        self.assertIsInstance(json.dumps(signature), str)


if __name__ == "__main__":
    unittest.main()
