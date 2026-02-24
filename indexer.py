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
from utils import is_supported_image, iter_files_recursive, load_image_rgb_with_metadata

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexSummary:
    """Summary metrics returned after indexing."""

    total_files_seen: int = 0
    total_indexed: int = 0
    total_skipped: int = 0
    skipped_non_image: int = 0
    skipped_decode_failure: int = 0
    total_errors: int = 0
    total_unchanged: int = 0
    total_pruned: int = 0


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
        self.text_templates = [
            "{}",
            "a photo of {}",
            "a picture of {}",
            "an image of {}",
            "close-up photo of {}",
            "object: {}",
        ]

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
        """Encode query text into a normalized float32 embedding vector.

        Uses lightweight prompt ensembling to improve retrieval stability for short queries.
        """
        text = text.strip()
        prompts = [tmpl.format(text) for tmpl in self.text_templates]
        tokens = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean_feat = feats.mean(dim=0, keepdim=True)
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
        return mean_feat[0].detach().cpu().numpy().astype(np.float32)


ProgressCallback = Callable[[int, int], None]


class PhotoIndexer:
    """Incremental photo indexer using CLIP image embeddings."""

    def __init__(self, store: PhotoStore, embedder: CLIPEmbedder, batch_size: int = 32) -> None:
        self.store = store
        self.embedder = embedder
        self.batch_size = batch_size

    def index_folder(
        self,
        root_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        skip_log_path: str | Path | None = None,
        force_reindex: bool = False,
        prune_deleted: bool = False,
    ) -> IndexSummary:
        """Index all images under ``root_path`` recursively.

        Incremental behavior:
        - unchanged mtime -> skipped unchanged
        - new or changed -> re-embedded + upsert
        """
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid folder path: {root}")

        mtime_map = self.store.load_mtime_map()
        total_files = sum(1 for _ in iter_files_recursive(root))

        summary = IndexSummary(total_files_seen=total_files)
        batch_tensors: list[torch.Tensor] = []
        batch_meta: list[tuple[str, int, int, int, int | None, float | None, float | None]] = []
        skip_log_file = None
        if skip_log_path is not None:
            skip_path = Path(skip_log_path).expanduser().resolve()
            skip_path.parent.mkdir(parents=True, exist_ok=True)
            skip_log_file = skip_path.open("w", encoding="utf-8")
            skip_log_file.write("reason\tpath\tdetail\n")

        def log_skip(reason: str, path: Path, detail: str = "") -> None:
            if skip_log_file is None:
                return
            safe_detail = detail.replace("\n", " ").replace("\t", " ")
            skip_log_file.write(f"{reason}\t{path}\t{safe_detail}\n")

        def flush_batch() -> None:
            nonlocal batch_tensors, batch_meta, summary
            if not batch_tensors:
                return
            embs = self.embedder.encode_image_tensors(batch_tensors)
            for i, (file_path, mtime, width, height, taken_ts, latitude, longitude) in enumerate(batch_meta):
                record = PhotoRecord(
                    file_path=file_path,
                    mtime=mtime,
                    width=width,
                    height=height,
                    taken_ts=taken_ts,
                    latitude=latitude,
                    longitude=longitude,
                    embedding=embs[i],
                )
                self.store.upsert_photo(record)
            self.store.commit()
            summary.total_indexed += len(batch_meta)
            batch_tensors = []
            batch_meta = []

        pbar = tqdm(total=total_files, desc="Indexing files", unit="file")
        last_commit_ts = time.time()

        for idx, file_path in enumerate(iter_files_recursive(root), start=1):
            pbar.update(1)
            if progress_callback:
                progress_callback(idx, total_files)

            if not is_supported_image(file_path):
                summary.total_skipped += 1
                summary.skipped_non_image += 1
                log_skip("non_image_or_unsupported_extension", file_path)
                continue

            try:
                path_str = str(file_path.resolve())
                mtime = int(file_path.stat().st_mtime)
            except Exception:
                LOGGER.exception("Failed to stat file: %s", file_path)
                summary.total_errors += 1
                log_skip("stat_error", file_path)
                continue

            prev_mtime = mtime_map.get(path_str)
            if (not force_reindex) and prev_mtime is not None and prev_mtime == mtime:
                summary.total_unchanged += 1
                continue

            try:
                image, metadata = load_image_rgb_with_metadata(file_path)
                width, height = image.size
                tensor = self.embedder.preprocess_image(image)
                batch_tensors.append(tensor)
                batch_meta.append((path_str, mtime, width, height, metadata.taken_ts, metadata.latitude, metadata.longitude))
            except Exception:
                LOGGER.warning("Skipping decode failure: %s", file_path, exc_info=True)
                summary.total_skipped += 1
                summary.skipped_decode_failure += 1
                log_skip("decode_failure", file_path)
                continue

            if len(batch_tensors) >= self.batch_size:
                flush_batch()

            # Safety commit cadence for long runs.
            if time.time() - last_commit_ts > 30:
                self.store.commit()
                last_commit_ts = time.time()

        flush_batch()
        pbar.close()
        if skip_log_file is not None:
            skip_log_file.close()
        if prune_deleted:
            summary.total_pruned = self._prune_deleted_under_root(root)

        self.store.upsert_stat("last_total_files_seen", summary.total_files_seen)
        self.store.upsert_stat("last_total_indexed", summary.total_indexed)
        self.store.upsert_stat("last_total_skipped", summary.total_skipped)
        self.store.upsert_stat("last_skipped_non_image", summary.skipped_non_image)
        self.store.upsert_stat("last_skipped_decode_failure", summary.skipped_decode_failure)
        self.store.upsert_stat("last_total_errors", summary.total_errors)
        self.store.upsert_stat("last_total_unchanged", summary.total_unchanged)
        self.store.upsert_stat("last_total_pruned", summary.total_pruned)
        self.store.commit()

        return summary

    def _prune_deleted_under_root(self, root: Path) -> int:
        """Delete DB records for files under root that no longer exist on disk."""
        indexed_paths = self.store.load_paths_under_root(root)
        to_delete: list[str] = []
        for path_str in indexed_paths:
            if not Path(path_str).exists():
                to_delete.append(path_str)
        deleted = self.store.delete_paths(to_delete)
        if deleted:
            self.store.commit()
        return deleted
