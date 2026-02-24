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
    taken_ts: int | None
    latitude: float | None
    longitude: float | None
    media_type: str
    source_path: str
    frame_ts: float | None


class PhotoSearcher:
    """Loads DB embeddings once and performs fast dot-product retrieval."""

    def __init__(self, store: PhotoStore) -> None:
        self.store = store
        self.paths: list[str] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)
        self.taken_ts = np.empty((0,), dtype=np.float64)
        self.latitude = np.empty((0,), dtype=np.float64)
        self.longitude = np.empty((0,), dtype=np.float64)
        self.media_types: list[str] = []
        self.source_paths: list[str] = []
        self.frame_ts = np.empty((0,), dtype=np.float64)

    def load_index(self) -> None:
        """Load all stored embeddings into memory."""
        (
            self.paths,
            self.matrix,
            self.taken_ts,
            self.latitude,
            self.longitude,
            self.media_types,
            self.source_paths,
            self.frame_ts,
        ) = self.store.load_embeddings_matrix()

    def search(
        self,
        query: str,
        topk: int,
        embedder: CLIPEmbedder,
        min_taken_ts: int | None = None,
        max_taken_ts: int | None = None,
        has_gps: bool = False,
        min_score: float | None = None,
        relative_to_best: float | None = None,
    ) -> list[SearchResult]:
        """Return top-K results for a text query."""
        if self.matrix.size == 0 or not self.paths:
            return []

        text_emb = embedder.encode_text(query)
        scores = self.matrix @ text_emb
        mask = np.ones(scores.shape[0], dtype=bool)
        if min_taken_ts is not None:
            mask &= np.isfinite(self.taken_ts) & (self.taken_ts >= float(min_taken_ts))
        if max_taken_ts is not None:
            mask &= np.isfinite(self.taken_ts) & (self.taken_ts <= float(max_taken_ts))
        if has_gps:
            mask &= np.isfinite(self.latitude) & np.isfinite(self.longitude)

        valid_idx = np.where(mask)[0]
        if valid_idx.size == 0:
            return []
        valid_scores = scores[valid_idx]

        if min_score is not None:
            score_mask = valid_scores >= float(min_score)
            valid_idx = valid_idx[score_mask]
            valid_scores = valid_scores[score_mask]
            if valid_idx.size == 0:
                return []

        if relative_to_best is not None:
            best = float(np.max(valid_scores))
            cutoff = best - float(relative_to_best)
            rel_mask = valid_scores >= cutoff
            valid_idx = valid_idx[rel_mask]
            valid_scores = valid_scores[rel_mask]
            if valid_idx.size == 0:
                return []

        k = min(max(1, topk), valid_scores.shape[0])
        top_rel_idx = np.argpartition(valid_scores, -k)[-k:]
        sorted_rel_idx = top_rel_idx[np.argsort(valid_scores[top_rel_idx])[::-1]]
        sorted_idx = valid_idx[sorted_rel_idx]

        results: list[SearchResult] = []
        for rank, row_idx in enumerate(sorted_idx, start=1):
            taken_val = self.taken_ts[int(row_idx)]
            lat_val = self.latitude[int(row_idx)]
            lon_val = self.longitude[int(row_idx)]
            frame_val = self.frame_ts[int(row_idx)]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[row_idx]),
                    file_path=self.paths[int(row_idx)],
                    taken_ts=int(taken_val) if np.isfinite(taken_val) else None,
                    latitude=float(lat_val) if np.isfinite(lat_val) else None,
                    longitude=float(lon_val) if np.isfinite(lon_val) else None,
                    media_type=self.media_types[int(row_idx)] if self.media_types else "image",
                    source_path=self.source_paths[int(row_idx)] if self.source_paths else self.paths[int(row_idx)],
                    frame_ts=float(frame_val) if np.isfinite(frame_val) else None,
                )
            )
        return results
