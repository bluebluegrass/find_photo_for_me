"""Utility helpers for image handling, HEIC support, and macOS file opening."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
DEFAULT_APP_DB_PATH = Path.home() / "Library" / "Application Support" / "LocalPix" / "photo_index.db"

_HEIF_REGISTERED = False
_GPS_TAG = 34853
_GPS_INFO_TAG = "GPSInfo"


@dataclass(slots=True)
class ImageMetadata:
    """Metadata extracted from image EXIF fields."""

    taken_ts: int | None = None
    latitude: float | None = None
    longitude: float | None = None


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
    name = path.name
    # macOS AppleDouble sidecar files (e.g. "._IMG_1234.JPG") are metadata, not real images.
    if name.startswith("._"):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_supported_video(path: Path) -> bool:
    """Return True when path extension is one of supported video types."""
    name = path.name
    if name.startswith("._"):
        return False
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image using Pillow and convert to RGB.

    Raises:
        OSError: For decoding/opening failures.
        ValueError: For invalid image files.
    """
    configure_heif_support()
    with Image.open(path) as img:
        return img.convert("RGB")


def load_image_rgb_with_metadata(path: Path) -> tuple[Image.Image, ImageMetadata]:
    """Load image, convert to RGB, and extract basic EXIF metadata."""
    configure_heif_support()
    with Image.open(path) as img:
        metadata = extract_image_metadata(img)
        rgb = img.convert("RGB")
    return rgb, metadata


def extract_image_metadata(image: Image.Image) -> ImageMetadata:
    """Extract capture timestamp and GPS coordinates from EXIF if available."""
    metadata = ImageMetadata()

    exif_obj = None
    try:
        exif_obj = image.getexif()
    except Exception:
        return metadata
    if exif_obj is None:
        return metadata

    # Taken time priority: DateTimeOriginal (36867), then DateTimeDigitized (36868), then DateTime (306).
    for key in (36867, 36868, 306):
        value = exif_obj.get(key)
        ts = _parse_exif_datetime(value)
        if ts is not None:
            metadata.taken_ts = ts
            break

    gps_info = None
    try:
        gps_info = exif_obj.get_ifd(_GPS_TAG)
    except Exception:
        # Fallback for formats where GPS info is already embedded at the tag.
        gps_info = exif_obj.get(_GPS_TAG) or exif_obj.get(_GPS_INFO_TAG)
    lat, lon = _parse_gps(gps_info)
    metadata.latitude = lat
    metadata.longitude = lon
    return metadata


def _parse_exif_datetime(value: Any) -> int | None:
    """Parse EXIF datetime string into Unix timestamp."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    return None


def _parse_gps(gps_info: Any) -> tuple[float | None, float | None]:
    """Parse EXIF GPS data into decimal latitude/longitude."""
    if gps_info is None:
        return None, None

    if not isinstance(gps_info, dict):
        try:
            gps_info = dict(gps_info)
        except Exception:
            return None, None

    lat_ref = gps_info.get(1) or gps_info.get("GPSLatitudeRef")
    lat_val = gps_info.get(2) or gps_info.get("GPSLatitude")
    lon_ref = gps_info.get(3) or gps_info.get("GPSLongitudeRef")
    lon_val = gps_info.get(4) or gps_info.get("GPSLongitude")

    lat = _gps_to_decimal(lat_val, lat_ref)
    lon = _gps_to_decimal(lon_val, lon_ref)
    return lat, lon


def _gps_to_decimal(value: Any, ref: Any) -> float | None:
    """Convert EXIF DMS tuple to decimal coordinates."""
    if value is None:
        return None
    try:
        d, m, s = value
        deg = _to_float(d)
        minutes = _to_float(m)
        seconds = _to_float(s)
        dec = deg + (minutes / 60.0) + (seconds / 3600.0)
        ref_text = ref.decode("utf-8", errors="ignore") if isinstance(ref, (bytes, bytearray)) else str(ref)
        if ref and ref_text.upper() in {"S", "W"}:
            dec = -dec
        return float(dec)
    except Exception:
        return None


def _to_float(x: Any) -> float:
    """Convert EXIF rational/number-like value to float."""
    try:
        return float(x)
    except Exception:
        pass
    try:
        # PIL TiffImagePlugin.IFDRational may expose numerator/denominator.
        return float(x.numerator) / float(x.denominator)
    except Exception:
        raise ValueError(f"Cannot convert EXIF numeric value to float: {x!r}")


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


def choose_folder_dialog_macos(prompt: str = "Select photo folder to index") -> str | None:
    """Open native macOS folder picker and return selected folder path.

    Returns:
        Absolute POSIX path with trailing slash removed, or None if cancelled/failed.
    """
    script = [
        "-e",
        f'set chosenFolder to choose folder with prompt "{prompt}"',
        "-e",
        "POSIX path of chosenFolder",
    ]
    try:
        completed = subprocess.run(["osascript", *script], check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            return None
        selected = completed.stdout.strip()
        if not selected:
            return None
        return selected.rstrip("/")
    except Exception:
        logging.exception("Failed to open macOS folder picker.")
        return None


def default_db_path() -> str:
    """Return recommended persistent DB path for macOS app data."""
    return str(DEFAULT_APP_DB_PATH)


def ffmpeg_available() -> bool:
    """Return True when ffmpeg binary is available in PATH."""
    return shutil.which("ffmpeg") is not None


def extract_video_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = 1.0,
    max_frames: int = 300,
) -> list[tuple[Path, float]]:
    """Extract sampled video frames to JPEG files and return (frame_path, timestamp_sec)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found. Install ffmpeg to index videos.")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be > 0")

    output_pattern = output_dir / "frame_%06d.jpg"
    fps_expr = f"fps=1/{interval_sec}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        fps_expr,
        "-q:v",
        "2",
        "-frames:v",
        str(max_frames),
        str(output_pattern),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "ffmpeg frame extraction failed").strip())

    frames = sorted(output_dir.glob("frame_*.jpg"))
    out: list[tuple[Path, float]] = []
    for idx, frame_path in enumerate(frames):
        ts = float(idx) * interval_sec
        out.append((frame_path.resolve(), ts))
    return out
