"""In-memory numpy search over stored CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indexer import CLIPEmbedder
from store import PhotoStore


@dataclass(slots=True)
class SearchResult:
    """One ranked search result."""

    rank: int
    score: float
    file_path: str


class PhotoSearcher:
    """Loads DB embeddings once and performs fast dot-product retrieval."""

    def __init__(self, store: PhotoStore) -> None:
        self.store = store
        self.paths: list[str] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)

    def load_index(self) -> None:
        """Load all stored embeddings into memory."""
        self.paths, self.matrix = self.store.load_embeddings_matrix()

    def search(self, query: str, topk: int, embedder: CLIPEmbedder) -> list[SearchResult]:
        """Return top-K results for a text query."""
        if self.matrix.size == 0 or not self.paths:
            return []

        text_emb = embedder.encode_text(query)
        scores = self.matrix @ text_emb

        k = min(max(1, topk), scores.shape[0])
        top_idx = np.argpartition(scores, -k)[-k:]
        sorted_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results: list[SearchResult] = []
        for rank, row_idx in enumerate(sorted_idx, start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[row_idx]),
                    file_path=self.paths[int(row_idx)],
                )
            )
        return results
