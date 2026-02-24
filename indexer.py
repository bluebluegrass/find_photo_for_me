"""Indexing pipeline: scan files, embed images, and upsert into SQLite."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from store import PhotoRecord, PhotoStore
from utils import is_supported_image, iter_files_recursive, load_image_rgb

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexSummary:
    """Summary metrics returned after indexing."""

    total_files_seen: int = 0
    total_indexed: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    total_unchanged: int = 0


class CLIPEmbedder:
    """Wrapper around OpenCLIP model for image/text embedding."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image into model input tensor."""
        return self.preprocess(image)

    @torch.inference_mode()
    def encode_image_tensors(self, image_tensors: list[torch.Tensor]) -> np.ndarray:
        """Encode a batch of preprocessed images and return normalized float32 embeddings."""
        if not image_tensors:
            return np.empty((0, 0), dtype=np.float32)

        batch = torch.stack(image_tensors).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode query text into a normalized float32 embedding vector."""
        tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().numpy().astype(np.float32)


ProgressCallback = Callable[[int, int], None]


class PhotoIndexer:
    """Incremental photo indexer using CLIP image embeddings."""

    def __init__(self, store: PhotoStore, embedder: CLIPEmbedder, batch_size: int = 32) -> None:
        self.store = store
        self.embedder = embedder
        self.batch_size = batch_size

    def index_folder(self, root_path: str | Path, progress_callback: ProgressCallback | None = None) -> IndexSummary:
        """Index all images under ``root_path`` recursively.

        Incremental behavior:
        - unchanged mtime -> skipped unchanged
        - new or changed -> re-embedded + upsert
        """
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid folder path: {root}")

        mtime_map = self.store.load_mtime_map()
        files = list(iter_files_recursive(root))

        summary = IndexSummary(total_files_seen=len(files))
        batch_tensors: list[torch.Tensor] = []
        batch_meta: list[tuple[str, int, int, int]] = []

        def flush_batch() -> None:
            nonlocal batch_tensors, batch_meta, summary
            if not batch_tensors:
                return
            embs = self.embedder.encode_image_tensors(batch_tensors)
            for i, (file_path, mtime, width, height) in enumerate(batch_meta):
                record = PhotoRecord(
                    file_path=file_path,
                    mtime=mtime,
                    width=width,
                    height=height,
                    embedding=embs[i],
                )
                self.store.upsert_photo(record)
            self.store.commit()
            summary.total_indexed += len(batch_meta)
            batch_tensors = []
            batch_meta = []

        pbar = tqdm(total=len(files), desc="Indexing files", unit="file")
        last_commit_ts = time.time()

        for idx, file_path in enumerate(files, start=1):
            pbar.update(1)
            if progress_callback:
                progress_callback(idx, len(files))

            if not is_supported_image(file_path):
                summary.total_skipped += 1
                continue

            try:
                path_str = str(file_path.resolve())
                mtime = int(file_path.stat().st_mtime)
            except Exception:
                LOGGER.exception("Failed to stat file: %s", file_path)
                summary.total_errors += 1
                continue

            prev_mtime = mtime_map.get(path_str)
            if prev_mtime is not None and prev_mtime == mtime:
                summary.total_unchanged += 1
                continue

            try:
                image = load_image_rgb(file_path)
                width, height = image.size
                tensor = self.embedder.preprocess_image(image)
                batch_tensors.append(tensor)
                batch_meta.append((path_str, mtime, width, height))
            except Exception:
                LOGGER.warning("Skipping decode failure: %s", file_path, exc_info=True)
                summary.total_skipped += 1
                continue

            if len(batch_tensors) >= self.batch_size:
                flush_batch()

            # Safety commit cadence for long runs.
            if time.time() - last_commit_ts > 30:
                self.store.commit()
                last_commit_ts = time.time()

        flush_batch()
        pbar.close()

        self.store.upsert_stat("last_total_files_seen", summary.total_files_seen)
        self.store.upsert_stat("last_total_indexed", summary.total_indexed)
        self.store.upsert_stat("last_total_skipped", summary.total_skipped)
        self.store.upsert_stat("last_total_errors", summary.total_errors)
        self.store.upsert_stat("last_total_unchanged", summary.total_unchanged)
        self.store.commit()

        return summary
