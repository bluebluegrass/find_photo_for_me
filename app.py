"""CLI for offline photo indexing and CLIP-based text search."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from indexer import CLIPEmbedder, PhotoIndexer
from searcher import PhotoSearcher
from store import PhotoStore
from utils import open_in_finder


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

        summary = indexer.index_folder(args.path)

        print(f"Indexed:   {summary.total_indexed}")
        print(f"Skipped:   {summary.total_skipped}")
        print(f"Unchanged: {summary.total_unchanged}")
        print(f"Errors:    {summary.total_errors}")
        print(f"DB count:  {store.get_total_count()}")
        print(f"DB file:   {Path(store.db_path)}")
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

        results = searcher.search(query=args.query, topk=args.topk, embedder=embedder)
        if not results:
            print("No results.")
            return 0

        for res in results:
            print(f"{res.rank:>3}  {res.score:>8.4f}  {res.file_path}")

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
    parser.add_argument("--db", default="photo_index.db", help="Path to SQLite database file")
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
    p_index.set_defaults(func=cmd_index)

    p_search = subparsers.add_parser("search", help="Search indexed photos")
    p_search.add_argument("--query", required=True, help="Text query")
    p_search.add_argument("--topk", type=int, default=30, help="Number of results")
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


if __name__ == "__main__":
    raise SystemExit(main())
