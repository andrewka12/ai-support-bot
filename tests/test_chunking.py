"""Tests for the knowledge-base loader and the chunk splitter."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from tests.base import KB_DIR, PROJECT_ROOT, ProjectTestCase, make_temp_dir

from ai_support_bot.chunking import (
    DEFAULT_MAX_CHARS,
    Document,
    Section,
    chunk_document,
    discover_documents,
    load_chunks,
    parse_document,
    split_section,
)

EXPECTED_BUSINESSES = ("avtoservice", "shop", "stomatologiya")


class BundledKnowledgeBaseTest(ProjectTestCase):
    """The three bundled businesses must load with usable metadata."""

    def test_three_businesses_are_present(self) -> None:
        counts = Counter(chunk.business for chunk in self.chunks)
        self.assertEqual(sorted(counts), list(EXPECTED_BUSINESSES))

    def test_every_business_has_enough_chunks(self) -> None:
        counts = Counter(chunk.business for chunk in self.chunks)
        for business, count in counts.items():
            with self.subTest(business=business):
                self.assertGreaterEqual(count, 15)
                self.assertLessEqual(count, 30)

    def test_total_chunk_count(self) -> None:
        self.assertGreaterEqual(len(self.chunks), 45)
        self.assertLessEqual(len(self.chunks), 90)

    def test_chunk_metadata_is_complete(self) -> None:
        for chunk in self.chunks:
            with self.subTest(chunk=chunk.chunk_id):
                self.assertTrue(chunk.text.strip())
                self.assertTrue(chunk.heading.strip())
                self.assertTrue(chunk.title.strip())
                self.assertEqual(chunk.business, chunk.source.split("/")[0])
                self.assertTrue(chunk.source.endswith((".md", ".txt")))
                self.assertGreaterEqual(chunk.order, 1)
                self.assertTrue((KB_DIR / chunk.source).exists())

    def test_chunk_ids_are_unique_and_stable(self) -> None:
        ids = [chunk.chunk_id for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [chunk.chunk_id for chunk in load_chunks(KB_DIR)])

    def test_documents_are_discovered_per_business(self) -> None:
        documents = discover_documents(KB_DIR)
        self.assertEqual(len(documents), 15)
        for path in documents:
            with self.subTest(path=path.name):
                self.assertIn(path.suffix, {".md", ".txt"})


class DocumentParsingTest(unittest.TestCase):
    """Parsing and splitting behaviour on synthetic documents."""

    def setUp(self) -> None:
        self.root = make_temp_dir(self)

    def _write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_titles_and_headings_are_extracted(self) -> None:
        path = self._write(
            "shop/prices.md",
            "# Магазин — цены\n\n## Доставка\n\nКурьер по Москве — 490 ₽.\n\n## Оплата\n\nКартой или наличными.\n",
        )
        document = parse_document(path, self.root)
        self.assertEqual(document.title, "Магазин — цены")
        self.assertEqual(document.business, "shop")
        self.assertEqual([section.heading for section in document.sections], ["Доставка", "Оплата"])

    def test_short_sections_are_merged_into_one_chunk(self) -> None:
        document = Document(
            source="shop/short.md",
            business="shop",
            title="Короткий документ",
            sections=(
                Section(heading="Первое", body="Короткий текст про доставку."),
                Section(heading="Второе", body="Ещё короткий текст про оплату."),
            ),
        )
        chunks = chunk_document(document, min_chars=80)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Первое", chunks[0].heading)
        self.assertIn("Второе", chunks[0].heading)

    def test_long_section_is_split_with_overlap(self) -> None:
        first = "Первое предложение про доставку по Москве. " * 12
        second = "Второе предложение про оплату заказа картой. " * 12
        text = f"{first.strip()}\n\n{second.strip()}"
        pieces = split_section(text, max_chars=400, overlap_chars=120)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            with self.subTest(piece=piece[:40]):
                self.assertLessEqual(len(piece), 640)
        tail = pieces[0][-60:].split(".")[-2].strip()
        self.assertIn(tail[:25], pieces[1])

    def test_chunk_count_for_a_long_document(self) -> None:
        body = "\n\n".join(
            f"Раздел номер {index}. " + "Текст про услуги и цены. " * 10 for index in range(4)
        )
        document = Document(
            source="avtoservice/long.md",
            business="avtoservice",
            title="Длинный документ",
            sections=(Section(heading="Услуги", body=body),),
        )
        chunks = chunk_document(document)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual([chunk.order for chunk in chunks], list(range(1, len(chunks) + 1)))
        for chunk in chunks:
            with self.subTest(chunk=chunk.chunk_id):
                self.assertEqual(chunk.source, "avtoservice/long.md")
                self.assertEqual(chunk.heading, "Услуги")
                self.assertGreaterEqual(len(chunk.text), 1)
                self.assertLessEqual(len(chunk.text), DEFAULT_MAX_CHARS + 260)

    def test_text_files_are_supported(self) -> None:
        self._write("shop/notes.txt", "Заметка про доставку и оплату заказа.")
        chunks = load_chunks(self.root)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].business, "shop")

    def test_empty_directory_returns_no_chunks(self) -> None:
        self.assertEqual(load_chunks(self.root), [])

    def test_project_root_is_a_package_directory(self) -> None:
        self.assertTrue((PROJECT_ROOT / "ai_support_bot" / "__main__.py").exists())


if __name__ == "__main__":
    unittest.main()
