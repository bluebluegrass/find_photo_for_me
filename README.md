# Local Private Photo Search (OpenCLIP + SQLite)

A fully offline photo search tool for macOS that indexes your image files with CLIP embeddings, then lets you search using natural language queries like:

- `cat`
- `old kitchen`
- `whiteboard`
- `receipt`
- `fridge`

This project is designed for local/private use. No cloud APIs are required.

## Why This Exists

Traditional folder search depends on filenames and metadata. This tool uses **visual-semantic embeddings** so you can search by what appears in the image, even if filenames are random (e.g. `IMG_4219.HEIC`).

## Features

- Fully local and offline
- Supports Apple Silicon (MPS acceleration when available)
- HEIC/HEIF support via `pillow-heif`
- Incremental indexing (mtime-based)
- Fast search using in-memory NumPy matrix + dot-product similarity
- SQLite-backed index (`photo_index.db`)
- CLI and Streamlit UI
- One-click open in Finder (`open`)

## Supported File Types

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.heic`
- `.heif`

## System Requirements

- macOS
- Python 3.11+
- Sufficient RAM to hold embeddings matrix (for ~20k images this is typically manageable)
- Sufficient storage for SQLite DB

## Installation

### Recommended (venv)

```bash
cd /path/to/photo_search
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Important: Use One Interpreter Consistently

If you index with `python3.11`, also search with `python3.11` (or run both inside the same activated venv).

Bad (mixed interpreters):

```bash
python3.11 app.py index ...
python app.py search ...   # may fail if this is a different environment
```

Good:

```bash
python app.py index ...
python app.py search ...
```

(when venv is active)

## Apple Silicon Notes (M1/M2/M3)

- The code auto-selects `mps` if available: `torch.backends.mps.is_available()`
- Falls back to CPU if MPS is unavailable
- `open_clip_torch` uses your local PyTorch installation

If you need to force CPU for debugging:

```bash
python app.py --device cpu search --query "cat"
```

## Quick Start

### 1) Build index

```bash
python app.py index --path "/Volumes/ExternalDrive/Photos"
```

To backfill newly added metadata (taken time/GPS) for already indexed files:

```bash
python app.py index --path "/Volumes/ExternalDrive/Photos" --force
```

### 2) Search from CLI

```bash
python app.py search --query "cat" --topk 30
```

Filter by taken date and GPS availability:

```bash
python app.py search --query "trip" --from-date 2023-01-01 --to-date 2023-12-31 --has-gps
```

### 3) Open top result(s) in Finder

```bash
python app.py search --query "receipt" --topk 30 --open
python app.py search --query "receipt" --topk 30 --open 3
```

### 4) Run full Streamlit UI

```bash
streamlit run ui.py
```

### 5) Run minimal Streamlit UI (query + preview grid)

```bash
streamlit run simple_ui.py
```

## CLI Reference

### `index`

```bash
python app.py index --path "/Volumes/ExternalDrive/Photos"
```

Options:

- `--path` (required): root folder to scan recursively
- `--batch-size` (default `32`): image embedding batch size
- `--skip-log` (optional): write skip diagnostics TSV (`reason`, `path`, `detail`)
- `--force` (optional): force re-index all supported files (ignore unchanged mtime)
- global `--db` (default `photo_index.db`): SQLite file path
- global `--model` (default `ViT-B-32`)
- global `--pretrained` (default `laion2b_s34b_b79k`)
- global `--device` (optional: `mps`, `cpu`)
- global `--verbose`

### `search`

```bash
python app.py search --query "cat" --topk 30
```

Options:

- `--query` (required): text prompt
- `--topk` (default `30`)
- `--from-date` (optional): taken date lower bound (`YYYY-MM-DD`)
- `--to-date` (optional): taken date upper bound (`YYYY-MM-DD`)
- `--has-gps` (optional): only return results with GPS coordinates
- `--min-score` (default `0.22`): suppress low-confidence matches
- `--relative-to-best` (default `0.10`): keep only results close to best match score
- `--open` (optional): open top result(s)
  - `--open` => open top 1
  - `--open 5` => open top 5

Output format:

- rank
- similarity score
- absolute file path

Relevance tuning example (stricter results):

```bash
python app.py search --query "cameras" --topk 50 --min-score 0.27 --relative-to-best 0.07
```

## How Indexing Works

1. Recursively scans the folder.
2. Keeps only supported image extensions.
3. For each file:
   - reads `mtime`
   - compares with DB
   - unchanged `mtime` => skip
   - new/changed => decode image, generate embedding, upsert row
4. Embeddings are stored as `float32` blobs.
5. Width/height are stored when decode succeeds.

HEIC/HEIF decode failures are skipped gracefully and logged.

## How Search Works

1. Load all embeddings from SQLite into memory once.
2. Encode query text into CLIP embedding.
3. L2-normalized dot product = similarity score.
4. Use `np.argpartition` for efficient top-K selection.
5. Return ranked results.

## Database Schema

SQLite DB: `photo_index.db`

Table `photos`:

- `file_path TEXT PRIMARY KEY`
- `mtime INTEGER NOT NULL`
- `width INTEGER`
- `height INTEGER`
- `taken_ts INTEGER` (EXIF capture timestamp, unix seconds)
- `latitude REAL` (EXIF GPS latitude)
- `longitude REAL` (EXIF GPS longitude)
- `embedding BLOB NOT NULL` (`float32` array bytes)
- `updated_at INTEGER NOT NULL`

Table `stats`:

- key/value counters from latest indexing run

## Project Structure

- `app.py` - CLI entrypoint
- `indexer.py` - OpenCLIP wrapper + incremental index pipeline
- `store.py` - SQLite schema/data access
- `searcher.py` - in-memory retrieval
- `utils.py` - HEIC registration, image loading, Finder open helper
- `ui.py` - full Streamlit interface (index + search + result grid)
- `simple_ui.py` - minimal Streamlit search UI (query + preview)
- `requirements.txt`
- `.gitignore`

## Performance Notes

- `ViT-B-32` is a good speed/quality tradeoff.
- External drive throughput can dominate indexing time.
- Larger `--batch-size` can improve throughput (until memory constraints).
- Search latency is usually low after in-memory matrix is loaded.

## Privacy

- All computation is local.
- No external inference APIs.
- Data remains on your machine/external drive unless you manually copy it.

## Limitations

1. Quality is semantic, not exact object detection.
- CLIP retrieval is approximate and can miss edge cases.
- Short or ambiguous queries can return visually related but unwanted results.

2. No face recognition or identity labeling.
- This is not a biometric or person-ID system.

3. No EXIF/date/location filters yet.
- Search is embedding-based only; metadata filtering is not implemented.

4. Incremental logic depends on `mtime`.
- If a file changes without `mtime` update (rare), it may not re-index automatically.

5. No automatic deletion cleanup.
- If files are removed from disk, old DB rows remain until you add cleanup logic or rebuild index.

6. macOS-specific open behavior.
- `--open` and UI open buttons use macOS `open` command.

7. First model load can be slow.
- Initial startup may take longer due to model weight loading.

8. HEIC compatibility depends on local codec stack.
- Most HEIC files should work via `pillow-heif`, but some malformed files will still fail and be skipped.

9. Not optimized for millions of images.
- Current design targets small-to-medium libraries (~10k-100k range depending on hardware).

10. No auth/multi-user support.
- This is a local single-user tool.

## Troubleshooting

### Many files were skipped (e.g. 1290)

`Skipped` includes two categories:

- non-image / unsupported extension
- decode failure (corrupt/unsupported image payload, including some HEIC edge cases)

Re-run indexing with skip diagnostics:

```bash
python app.py index --path "/Volumes/ExternalDrive/Photos" --skip-log skipped.tsv
```

Then inspect:

- CLI summary breakdown (`Non-image`, `Decode failures`)
- `skipped.tsv` for exact file paths and reasons

Quick counts by reason:

```bash
cut -f1 skipped.tsv | sort | uniq -c
```

### `ModuleNotFoundError: No module named 'open_clip'`

Install dependencies in the same interpreter you use to run the app:

```bash
python -m pip install -r requirements.txt
python app.py search --query "cat"
```

### Streamlit command not found

Use module invocation:

```bash
python -m streamlit run ui.py
```

### HEIC files skipped

Reinstall imaging stack in active environment:

```bash
python -m pip install --upgrade Pillow pillow-heif
```

### Slow indexing

- Use local SSD if possible
- Increase `--batch-size`
- Confirm MPS is available

## Development

### Lint/sanity check

```bash
python -m compileall app.py indexer.py searcher.py store.py ui.py simple_ui.py utils.py
```

### Typical contribution flow

1. Fork repo
2. Create feature branch
3. Add tests or validation notes
4. Open PR with before/after behavior

## License

Add a license file before public release (e.g. MIT).

## Roadmap Ideas

- Metadata filters (date range, camera model, folder)
- Delete-sync cleanup command
- Re-ranking and query expansion
- Optional duplicate clustering
- Export search results to CSV
