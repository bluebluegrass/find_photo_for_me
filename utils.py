"""Utility helpers for image handling, HEIC support, and macOS file opening."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

_HEIF_REGISTERED = False


def configure_heif_support() -> None:
    """Register HEIC/HEIF opener for Pillow exactly once."""
    global _HEIF_REGISTERED
    if not _HEIF_REGISTERED:
        register_heif_opener()
        _HEIF_REGISTERED = True


def iter_files_recursive(root: Path) -> Iterator[Path]:
    """Yield all files recursively under ``root``."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def is_supported_image(path: Path) -> bool:
    """Return True when path extension is one of supported image types."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image using Pillow and convert to RGB.

    Raises:
        OSError: For decoding/opening failures.
        ValueError: For invalid image files.
    """
    configure_heif_support()
    with Image.open(path) as img:
        return img.convert("RGB")


def load_thumbnail_array(path: Path, max_size: tuple[int, int] = (320, 320)) -> np.ndarray | None:
    """Load a thumbnail as a numpy array for UI display.

    Returns None if decoding fails.
    """
    try:
        image = load_image_rgb(path)
        image.thumbnail(max_size)
        return np.array(image)
    except Exception:
        logging.exception("Thumbnail load failed: %s", path)
        return None


def open_in_finder(path: Path) -> bool:
    """Open file in Finder using macOS ``open`` command."""
    try:
        completed = subprocess.run(["open", str(path)], check=False, capture_output=True)
        return completed.returncode == 0
    except Exception:
        logging.exception("Failed to open file in Finder: %s", path)
        return False
