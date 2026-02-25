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
        media = ["image", "image"]
        source = paths.copy()
        frame = np.array([np.nan, np.nan], dtype=np.float64)
        return paths, matrix, nans, nans, nans, media, source, frame


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


if __name__ == "__main__":
    unittest.main()
