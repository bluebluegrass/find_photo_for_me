"""CLI for offline photo indexing and CLIP-based text search."""

from __future__ import annotations

import argparse
from datetime import datetime, time as dtime
import logging
from pathlib import Path

from indexer import CLIPEmbedder, PhotoIndexer
from searcher import PhotoSearcher
from store import PhotoStore
from utils import default_db_path, open_in_finder


def configure_logging(verbose: bool = False) -> None:
    """Configure global logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_index(args: argparse.Namespace) -> int:
    """Handle ``index`` command."""
    store = PhotoStore(args.db)
    try:
        embedder = CLIPEmbedder(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        )
        indexer = PhotoIndexer(store=store, embedder=embedder, batch_size=args.batch_size)

        summary = indexer.index_folder(
            args.path,
            skip_log_path=args.skip_log,
            force_reindex=args.force,
            prune_deleted=args.prune,
        )

        print(f"Indexed:   {summary.total_indexed}")
        print(f"Skipped:   {summary.total_skipped}")
        print(f"  - non-image/unsupported: {summary.skipped_non_image}")
        print(f"  - decode failures:       {summary.skipped_decode_failure}")
        print(f"Unchanged: {summary.total_unchanged}")
        print(f"Errors:    {summary.total_errors}")
        print(f"Pruned:    {summary.total_pruned}")
        print(f"DB count:  {store.get_total_count()}")
        print(f"DB file:   {Path(store.db_path)}")
        if args.skip_log:
            print(f"Skip log:  {Path(args.skip_log).expanduser().resolve()}")
        return 0
    finally:
        store.close()


def cmd_search(args: argparse.Namespace) -> int:
    """Handle ``search`` command."""
    store = PhotoStore(args.db)
    try:
        embedder = CLIPEmbedder(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        )
        searcher = PhotoSearcher(store)
        searcher.load_index()

        if not searcher.paths:
            print("No indexed photos found. Run the index command first.")
            return 1

        try:
            min_ts, max_ts = _parse_date_filters(args.from_date, args.to_date)
        except ValueError as exc:
            print(f"Invalid date filter: {exc}")
            print("Use format YYYY-MM-DD, e.g. --from-date 2020-01-01 --to-date 2020-12-31")
            return 2
        results = searcher.search(
            query=args.query,
            topk=args.topk,
            embedder=embedder,
            min_taken_ts=min_ts,
            max_taken_ts=max_ts,
            has_gps=args.has_gps,
            min_score=args.min_score,
            relative_to_best=args.relative_to_best,
        )
        if not results:
            print("No results.")
            return 0

        for res in results:
            taken_text = "-"
            if res.taken_ts is not None:
                taken_text = datetime.fromtimestamp(res.taken_ts).strftime("%Y-%m-%d %H:%M:%S")
            gps_text = "-"
            if res.latitude is not None and res.longitude is not None:
                gps_text = f"{res.latitude:.6f},{res.longitude:.6f}"
            print(f"{res.rank:>3}  {res.score:>8.4f}  {taken_text:>19}  {gps_text:>23}  {res.file_path}")

        if args.open is not None:
            open_count = min(args.open, len(results))
            for item in results[:open_count]:
                open_in_finder(Path(item.file_path))

        return 0
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    """Build root argparse parser."""
    parser = argparse.ArgumentParser(description="Local private photo search using OpenCLIP embeddings")
    parser.add_argument("--db", default=default_db_path(), help="Path to SQLite database file")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained checkpoint tag",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. mps, cpu). Default: auto-detect",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Index a folder recursively")
    p_index.add_argument("--path", required=True, help="Root folder containing photos")
    p_index.add_argument("--batch-size", type=int, default=32, help="Image embedding batch size")
    p_index.add_argument(
        "--skip-log",
        default=None,
        help="Optional path to write TSV skip diagnostics (reason, path, detail)",
    )
    p_index.add_argument(
        "--force",
        action="store_true",
        help="Re-index all supported files even when mtime is unchanged (useful to backfill new metadata fields)",
    )
    p_index.add_argument(
        "--prune",
        action="store_true",
        help="Delete DB rows for files under --path that no longer exist on disk",
    )
    p_index.set_defaults(func=cmd_index)

    p_search = subparsers.add_parser("search", help="Search indexed photos")
    p_search.add_argument("--query", required=True, help="Text query")
    p_search.add_argument("--topk", type=int, default=30, help="Number of results")
    p_search.add_argument("--from-date", default=None, help="Filter taken date start (YYYY-MM-DD)")
    p_search.add_argument("--to-date", default=None, help="Filter taken date end (YYYY-MM-DD)")
    p_search.add_argument("--has-gps", action="store_true", help="Only return photos with GPS coordinates")
    p_search.add_argument(
        "--min-score",
        type=float,
        default=0.22,
        help="Absolute similarity cutoff to suppress weak matches (recommended 0.20-0.28)",
    )
    p_search.add_argument(
        "--relative-to-best",
        type=float,
        default=0.10,
        help="Keep results within this score gap from the best result (recommended 0.06-0.14)",
    )
    p_search.add_argument(
        "--open",
        nargs="?",
        const=1,
        type=int,
        default=None,
        help="Open top result(s) in Finder. Pass N to open top N (default 1).",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    return int(args.func(args))


def _parse_date_filters(from_date: str | None, to_date: str | None) -> tuple[int | None, int | None]:
    """Parse optional YYYY-MM-DD date filters into inclusive Unix timestamp bounds."""
    min_ts = None
    max_ts = None
    if from_date:
        start_date = datetime.strptime(from_date, "%Y-%m-%d").date()
        min_ts = int(datetime.combine(start_date, dtime.min).timestamp())
    if to_date:
        end_date = datetime.strptime(to_date, "%Y-%m-%d").date()
        max_ts = int(datetime.combine(end_date, dtime.max).timestamp())
    return min_ts, max_ts


if __name__ == "__main__":
    raise SystemExit(main())
