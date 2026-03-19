"""Unit tests for multi-prompt scoring integration in searcher."""

from __future__ import annotations

import unittest

import numpy as np

from searcher import PhotoSearcher


class DummyStore:
    def load_embeddings_matrix(self):
        paths = ["/a.jpg", "/b.jpg"]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        nans = np.array([np.nan, np.nan], dtype=np.float64)
        none_texts = [None, None]
        media = ["image", "image"]
        source = paths.copy()
        frame = np.array([np.nan, np.nan], dtype=np.float64)
        return paths, matrix, nans, nans, nans, none_texts, none_texts, none_texts, media, source, frame


class DummyEmbedder:
    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t == "first":
                out.append([1.0, 0.0])
            elif t == "second":
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return np.asarray(out, dtype=np.float32)


class SearcherPromptTests(unittest.TestCase):
    def test_prompt_ensemble_changes_scores(self) -> None:
        searcher = PhotoSearcher(DummyStore())
        searcher.load_index()
        embedder = DummyEmbedder()

        baseline = searcher.search(query="anything", topk=2, embedder=embedder)
        self.assertEqual(baseline[0].file_path, "/a.jpg")

        mixed = searcher.search(
            query="anything",
            topk=2,
            embedder=embedder,
            text_prompts=["first", "second"],
        )
        self.assertEqual(len(mixed), 2)
        self.assertAlmostEqual(mixed[0].score, 0.5, places=6)
        self.assertAlmostEqual(mixed[1].score, 0.5, places=6)


class LocationDummyStore:
    def load_embeddings_matrix(self):
        paths = ["/turkey.jpg", "/nl.jpg", "/none.jpg"]
        matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        nans = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        country_codes = ["tr", "nl", None]
        country_names = ["turkey", "netherlands", None]
        city_names = ["istanbul", "amsterdam", None]
        media = ["image", "image", "image"]
        source = paths.copy()
        frame = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        return paths, matrix, nans, nans, nans, country_codes, country_names, city_names, media, source, frame


class LocationSearchTests(unittest.TestCase):
    def test_hybrid_mode_restricts_when_location_matches_exist(self) -> None:
        searcher = PhotoSearcher(LocationDummyStore())
        searcher.load_index()
        embedder = DummyEmbedder()

        results = searcher.search(
            query="anything",
            topk=3,
            embedder=embedder,
            location_query="turkey",
            location_mode="hybrid",
        )
        self.assertEqual([r.file_path for r in results], ["/turkey.jpg"])
        self.assertEqual(searcher.last_location_status, "hybrid_applied")
        self.assertEqual(searcher.last_location_match_count, 1)

    def test_hybrid_mode_falls_back_when_no_location_matches_exist(self) -> None:
        searcher = PhotoSearcher(LocationDummyStore())
        searcher.load_index()
        embedder = DummyEmbedder()

        results = searcher.search(
            query="anything",
            topk=3,
            embedder=embedder,
            location_query="japan",
            location_mode="hybrid",
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(searcher.last_location_status, "hybrid_fallback_no_match")
        self.assertEqual(searcher.last_location_match_count, 0)

    def test_strict_mode_returns_no_results_without_matches(self) -> None:
        searcher = PhotoSearcher(LocationDummyStore())
        searcher.load_index()
        embedder = DummyEmbedder()

        results = searcher.search(
            query="anything",
            topk=3,
            embedder=embedder,
            location_query="japan",
            location_mode="strict",
        )
        self.assertEqual(results, [])
        self.assertEqual(searcher.last_location_status, "strict_no_match")


if __name__ == "__main__":
    unittest.main()
