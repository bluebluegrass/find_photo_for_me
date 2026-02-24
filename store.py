"""SQLite storage layer for image metadata and embeddings."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PhotoRecord:
    """One indexed photo row."""

    file_path: str
    mtime: int
    width: int | None
    height: int | None
    taken_ts: int | None
    latitude: float | None
    longitude: float | None
    media_type: str
    source_path: str
    frame_ts: float | None
    embedding: np.ndarray


class PhotoStore:
    """SQLite-backed store for photo embeddings."""

    def __init__(self, db_path: str | Path = "photo_index.db") -> None:
        db_path_obj = Path(db_path).expanduser()
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path_obj.resolve())
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.init_schema()

    def init_schema(self) -> None:
        """Create required tables if absent."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                file_path TEXT PRIMARY KEY,
                mtime INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                taken_ts INTEGER,
                latitude REAL,
                longitude REAL,
                media_type TEXT NOT NULL DEFAULT 'image',
                source_path TEXT,
                frame_ts REAL,
                embedding BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Add missing columns for older DB versions."""
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(photos)").fetchall()}
        if "taken_ts" not in existing:
            self.conn.execute("ALTER TABLE photos ADD COLUMN taken_ts INTEGER")
        if "latitude" not in existing:
            self.conn.execute("ALTER TABLE photos ADD COLUMN latitude REAL")
        if "longitude" not in existing:
            self.conn.execute("ALTER TABLE photos ADD COLUMN longitude REAL")
        if "media_type" not in existing:
            self.conn.execute("ALTER TABLE photos ADD COLUMN media_type TEXT NOT NULL DEFAULT 'image'")
        if "source_path" not in existing:
            self.conn.execute("ALTER TABLE photos ADD COLUMN source_path TEXT")
            self.conn.execute("UPDATE photos SET source_path=file_path WHERE source_path IS NULL")
        if "frame_ts" not in existing:
            self.conn.execute("ALTER TABLE photos ADD COLUMN frame_ts REAL")

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def load_mtime_map(self) -> dict[str, int]:
        """Return mapping of file_path -> mtime for incremental indexing."""
        rows = self.conn.execute("SELECT file_path, mtime FROM photos").fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def upsert_photo(self, record: PhotoRecord) -> None:
        """Insert/update one photo record."""
        embedding = np.asarray(record.embedding, dtype=np.float32)
        self.conn.execute(
            """
            INSERT INTO photos (
                file_path, mtime, width, height, taken_ts, latitude, longitude, media_type, source_path, frame_ts, embedding, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                mtime=excluded.mtime,
                width=excluded.width,
                height=excluded.height,
                taken_ts=excluded.taken_ts,
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                media_type=excluded.media_type,
                source_path=excluded.source_path,
                frame_ts=excluded.frame_ts,
                embedding=excluded.embedding,
                updated_at=excluded.updated_at
            """,
            (
                record.file_path,
                record.mtime,
                record.width,
                record.height,
                record.taken_ts,
                record.latitude,
                record.longitude,
                record.media_type,
                record.source_path,
                record.frame_ts,
                embedding.tobytes(),
                int(time.time()),
            ),
        )

    def commit(self) -> None:
        """Commit current transaction."""
        self.conn.commit()

    def load_embeddings_matrix(
        self,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
        """Load embeddings + metadata as aligned arrays."""
        rows = self.conn.execute(
            """
            SELECT file_path, embedding, taken_ts, latitude, longitude, media_type, source_path, frame_ts
            FROM photos
            ORDER BY file_path ASC
            """
        ).fetchall()
        if not rows:
            empty = np.empty((0,), dtype=np.float64)
            return [], np.empty((0, 0), dtype=np.float32), empty, empty, empty, [], [], empty

        first_emb = np.frombuffer(rows[0][1], dtype=np.float32)
        dim = int(first_emb.shape[0])
        matrix = np.empty((len(rows), dim), dtype=np.float32)
        paths: list[str] = []
        taken_ts = np.full((len(rows),), np.nan, dtype=np.float64)
        lat = np.full((len(rows),), np.nan, dtype=np.float64)
        lon = np.full((len(rows),), np.nan, dtype=np.float64)
        media_types: list[str] = []
        source_paths: list[str] = []
        frame_ts = np.full((len(rows),), np.nan, dtype=np.float64)

        for idx, (file_path, blob, taken_val, lat_val, lon_val, media_type, source_path, frame_val) in enumerate(rows):
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.shape[0] != dim:
                raise ValueError(
                    f"Embedding dimension mismatch for '{file_path}'. "
                    f"Expected {dim}, got {emb.shape[0]}."
                )
            matrix[idx] = emb
            paths.append(file_path)
            if taken_val is not None:
                taken_ts[idx] = float(taken_val)
            if lat_val is not None:
                lat[idx] = float(lat_val)
            if lon_val is not None:
                lon[idx] = float(lon_val)
            media_types.append(media_type or "image")
            source_paths.append(source_path or file_path)
            if frame_val is not None:
                frame_ts[idx] = float(frame_val)

        return paths, matrix, taken_ts, lat, lon, media_types, source_paths, frame_ts

    def get_total_count(self) -> int:
        """Total number of indexed photos."""
        row = self.conn.execute("SELECT COUNT(*) FROM photos").fetchone()
        return int(row[0]) if row else 0

    def upsert_stat(self, key: str, value: int) -> None:
        """Store a numeric stat key/value."""
        self.conn.execute(
            """
            INSERT INTO stats (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, int(value)),
        )

    def load_paths_under_root(self, root_path: str | Path) -> list[str]:
        """Return indexed file paths that are under a specific root path."""
        root = str(Path(root_path).expanduser().resolve())
        if not root.endswith("/"):
            root = root + "/"
        rows = self.conn.execute("SELECT file_path FROM photos WHERE file_path LIKE ?", (f"{root}%",)).fetchall()
        return [r[0] for r in rows]

    def load_entries_under_root(self, root_path: str | Path) -> list[tuple[str, str, str]]:
        """Return (file_path, source_path, media_type) rows where source_path is under root."""
        root = str(Path(root_path).expanduser().resolve())
        if not root.endswith("/"):
            root = root + "/"
        rows = self.conn.execute(
            """
            SELECT file_path, COALESCE(source_path, file_path), COALESCE(media_type, 'image')
            FROM photos
            WHERE COALESCE(source_path, file_path) LIKE ?
            """,
            (f"{root}%",),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def delete_paths(self, file_paths: list[str]) -> int:
        """Delete a batch of paths from photos table and return deleted count."""
        if not file_paths:
            return 0
        self.conn.executemany("DELETE FROM photos WHERE file_path = ?", [(p,) for p in file_paths])
        return len(file_paths)

    def load_source_mtime_map(self, media_type: str) -> dict[str, int]:
        """Return mapping source_path -> max mtime for a media type."""
        rows = self.conn.execute(
            """
            SELECT source_path, MAX(mtime)
            FROM photos
            WHERE media_type = ? AND source_path IS NOT NULL
            GROUP BY source_path
            """,
            (media_type,),
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows if r[0] is not None}

    def load_paths_by_source(self, source_path: str, media_type: str) -> list[str]:
        """Return file_path rows belonging to source_path/media_type."""
        rows = self.conn.execute(
            "SELECT file_path FROM photos WHERE source_path = ? AND media_type = ?",
            (source_path, media_type),
        ).fetchall()
        return [r[0] for r in rows]
