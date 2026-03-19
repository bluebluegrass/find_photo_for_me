"""Utility helpers for media handling, metadata extraction, and app defaults."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
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
DEFAULT_LLM_MODEL = "qwen2.5:3b-instruct"
DEFAULT_LLM_TIMEOUT_SEC = 2.0
DEFAULT_LLM_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_LOCATION_MODE = "hybrid"

COUNTRY_CODE_TO_NAME = {
    "ad": "andorra",
    "ae": "united arab emirates",
    "af": "afghanistan",
    "ag": "antigua and barbuda",
    "ai": "anguilla",
    "al": "albania",
    "am": "armenia",
    "ao": "angola",
    "aq": "antarctica",
    "ar": "argentina",
    "as": "american samoa",
    "at": "austria",
    "au": "australia",
    "aw": "aruba",
    "ax": "aland islands",
    "az": "azerbaijan",
    "ba": "bosnia and herzegovina",
    "bb": "barbados",
    "bd": "bangladesh",
    "be": "belgium",
    "bf": "burkina faso",
    "bg": "bulgaria",
    "bh": "bahrain",
    "bi": "burundi",
    "bj": "benin",
    "bl": "saint barthelemy",
    "bm": "bermuda",
    "bn": "brunei",
    "bo": "bolivia",
    "bq": "caribbean netherlands",
    "br": "brazil",
    "bs": "bahamas",
    "bt": "bhutan",
    "bv": "bouvet island",
    "bw": "botswana",
    "by": "belarus",
    "bz": "belize",
    "ca": "canada",
    "cc": "cocos islands",
    "cd": "democratic republic of the congo",
    "cf": "central african republic",
    "cg": "republic of the congo",
    "ch": "switzerland",
    "ci": "ivory coast",
    "ck": "cook islands",
    "cl": "chile",
    "cm": "cameroon",
    "cn": "china",
    "co": "colombia",
    "cr": "costa rica",
    "cu": "cuba",
    "cv": "cape verde",
    "cw": "curacao",
    "cx": "christmas island",
    "cy": "cyprus",
    "cz": "czechia",
    "de": "germany",
    "dj": "djibouti",
    "dk": "denmark",
    "dm": "dominica",
    "do": "dominican republic",
    "dz": "algeria",
    "ec": "ecuador",
    "ee": "estonia",
    "eg": "egypt",
    "eh": "western sahara",
    "er": "eritrea",
    "es": "spain",
    "et": "ethiopia",
    "fi": "finland",
    "fj": "fiji",
    "fk": "falkland islands",
    "fm": "micronesia",
    "fo": "faroe islands",
    "fr": "france",
    "ga": "gabon",
    "gb": "united kingdom",
    "gd": "grenada",
    "ge": "georgia",
    "gf": "french guiana",
    "gg": "guernsey",
    "gh": "ghana",
    "gi": "gibraltar",
    "gl": "greenland",
    "gm": "gambia",
    "gn": "guinea",
    "gp": "guadeloupe",
    "gq": "equatorial guinea",
    "gr": "greece",
    "gs": "south georgia and the south sandwich islands",
    "gt": "guatemala",
    "gu": "guam",
    "gw": "guinea-bissau",
    "gy": "guyana",
    "hk": "hong kong",
    "hm": "heard island and mcdonald islands",
    "hn": "honduras",
    "hr": "croatia",
    "ht": "haiti",
    "hu": "hungary",
    "id": "indonesia",
    "ie": "ireland",
    "il": "israel",
    "im": "isle of man",
    "in": "india",
    "io": "british indian ocean territory",
    "iq": "iraq",
    "ir": "iran",
    "is": "iceland",
    "it": "italy",
    "je": "jersey",
    "jm": "jamaica",
    "jo": "jordan",
    "jp": "japan",
    "ke": "kenya",
    "kg": "kyrgyzstan",
    "kh": "cambodia",
    "ki": "kiribati",
    "km": "comoros",
    "kn": "saint kitts and nevis",
    "kp": "north korea",
    "kr": "south korea",
    "kw": "kuwait",
    "ky": "cayman islands",
    "kz": "kazakhstan",
    "la": "laos",
    "lb": "lebanon",
    "lc": "saint lucia",
    "li": "liechtenstein",
    "lk": "sri lanka",
    "lr": "liberia",
    "ls": "lesotho",
    "lt": "lithuania",
    "lu": "luxembourg",
    "lv": "latvia",
    "ly": "libya",
    "ma": "morocco",
    "mc": "monaco",
    "md": "moldova",
    "me": "montenegro",
    "mf": "saint martin",
    "mg": "madagascar",
    "mh": "marshall islands",
    "mk": "north macedonia",
    "ml": "mali",
    "mm": "myanmar",
    "mn": "mongolia",
    "mo": "macao",
    "mp": "northern mariana islands",
    "mq": "martinique",
    "mr": "mauritania",
    "ms": "montserrat",
    "mt": "malta",
    "mu": "mauritius",
    "mv": "maldives",
    "mw": "malawi",
    "mx": "mexico",
    "my": "malaysia",
    "mz": "mozambique",
    "na": "namibia",
    "nc": "new caledonia",
    "ne": "niger",
    "nf": "norfolk island",
    "ng": "nigeria",
    "ni": "nicaragua",
    "nl": "netherlands",
    "no": "norway",
    "np": "nepal",
    "nr": "nauru",
    "nu": "niue",
    "nz": "new zealand",
    "om": "oman",
    "pa": "panama",
    "pe": "peru",
    "pf": "french polynesia",
    "pg": "papua new guinea",
    "ph": "philippines",
    "pk": "pakistan",
    "pl": "poland",
    "pm": "saint pierre and miquelon",
    "pn": "pitcairn islands",
    "pr": "puerto rico",
    "ps": "palestine",
    "pt": "portugal",
    "pw": "palau",
    "py": "paraguay",
    "qa": "qatar",
    "re": "reunion",
    "ro": "romania",
    "rs": "serbia",
    "ru": "russia",
    "rw": "rwanda",
    "sa": "saudi arabia",
    "sb": "solomon islands",
    "sc": "seychelles",
    "sd": "sudan",
    "se": "sweden",
    "sg": "singapore",
    "sh": "saint helena",
    "si": "slovenia",
    "sj": "svalbard and jan mayen",
    "sk": "slovakia",
    "sl": "sierra leone",
    "sm": "san marino",
    "sn": "senegal",
    "so": "somalia",
    "sr": "suriname",
    "ss": "south sudan",
    "st": "sao tome and principe",
    "sv": "el salvador",
    "sx": "sint maarten",
    "sy": "syria",
    "sz": "eswatini",
    "tc": "turks and caicos islands",
    "td": "chad",
    "tf": "french southern territories",
    "tg": "togo",
    "th": "thailand",
    "tj": "tajikistan",
    "tk": "tokelau",
    "tl": "timor-leste",
    "tm": "turkmenistan",
    "tn": "tunisia",
    "to": "tonga",
    "tr": "turkey",
    "tt": "trinidad and tobago",
    "tv": "tuvalu",
    "tw": "taiwan",
    "tz": "tanzania",
    "ua": "ukraine",
    "ug": "uganda",
    "um": "u.s. minor outlying islands",
    "us": "united states",
    "uy": "uruguay",
    "uz": "uzbekistan",
    "va": "vatican city",
    "vc": "saint vincent and the grenadines",
    "ve": "venezuela",
    "vg": "british virgin islands",
    "vi": "u.s. virgin islands",
    "vn": "vietnam",
    "vu": "vanuatu",
    "wf": "wallis and futuna",
    "ws": "samoa",
    "xk": "kosovo",
    "ye": "yemen",
    "yt": "mayotte",
    "za": "south africa",
    "zm": "zambia",
    "zw": "zimbabwe",
}

_HEIF_REGISTERED = False
_RG_IMPORT_WARNED = False
_GPS_TAG = 34853
_GPS_INFO_TAG = "GPSInfo"


@dataclass(slots=True)
class ImageMetadata:
    """Metadata extracted from image EXIF fields."""

    taken_ts: int | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass(slots=True)
class LocationMetadata:
    """Normalized location metadata derived from GPS coordinates."""

    country_code: str | None = None
    country_name: str | None = None
    city_name: str | None = None


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


def default_llm_model() -> str:
    """Return default local LLM model name for smart-query parsing."""
    return os.getenv("LOCALPIX_LLM_MODEL", DEFAULT_LLM_MODEL)


def default_llm_timeout() -> float:
    """Return default timeout seconds for local smart-query parsing."""
    raw = os.getenv("LOCALPIX_LLM_TIMEOUT")
    if raw is None:
        return DEFAULT_LLM_TIMEOUT_SEC
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SEC


def default_llm_endpoint() -> str:
    """Return default local LLM endpoint URL."""
    return os.getenv("LOCALPIX_LLM_ENDPOINT", DEFAULT_LLM_ENDPOINT)


def default_location_mode() -> str:
    """Return default location filter mode for Smart Query."""
    value = os.getenv("LOCALPIX_LOCATION_MODE", DEFAULT_LOCATION_MODE).strip().lower()
    if value in {"hybrid", "strict", "off"}:
        return value
    return DEFAULT_LOCATION_MODE


def normalize_location_text(text: str | None) -> str | None:
    """Normalize location text for exact in-memory matching."""
    if text is None:
        return None
    normalized = " ".join(text.strip().lower().split())
    return normalized or None


def reverse_geocode_location(latitude: float | None, longitude: float | None) -> LocationMetadata:
    """Resolve GPS coordinates into country/city metadata using offline reverse geocoding."""
    global _RG_IMPORT_WARNED
    if latitude is None or longitude is None:
        return LocationMetadata()
    try:
        import reverse_geocoder as rg
    except ImportError:
        if not _RG_IMPORT_WARNED:
            logging.info("reverse_geocoder is not installed; skipping location enrichment.")
            _RG_IMPORT_WARNED = True
        return LocationMetadata()

    try:
        matches = rg.search((latitude, longitude), mode=1)
    except Exception:
        logging.exception("Reverse geocoding failed for coordinates (%s, %s)", latitude, longitude)
        return LocationMetadata()

    if not matches:
        return LocationMetadata()

    match = matches[0]
    country_code = normalize_location_text(str(match.get("cc", "")) or None)
    country_name = COUNTRY_CODE_TO_NAME.get(country_code or "")
    city_name = normalize_location_text(str(match.get("name", "")) or None)

    return LocationMetadata(
        country_code=country_code,
        country_name=country_name,
        city_name=city_name,
    )


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
