"""Streamlit UI for local photo indexing and CLIP search."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from indexer import CLIPEmbedder, PhotoIndexer
from searcher import PhotoSearcher, SearchResult
from store import PhotoStore
from utils import load_thumbnail_array, open_in_finder

st.set_page_config(page_title="Local Photo Search", layout="wide")
st.title("Local Private Photo Search")
st.caption("OpenCLIP + SQLite | Offline | HEIC compatible")


def _db_key(db_path: str) -> str:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        return f"{path}:missing"
    return f"{path}:{int(path.stat().st_mtime)}"


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str, pretrained: str) -> CLIPEmbedder:
    """Cache loaded CLIP model in Streamlit session."""
    return CLIPEmbedder(model_name=model_name, pretrained=pretrained)


def get_searcher(db_path: str) -> PhotoSearcher:
    """Get cached searcher and reload if DB changed."""
    key = _db_key(db_path)
    if st.session_state.get("searcher_key") != key:
        store = PhotoStore(db_path)
        searcher = PhotoSearcher(store)
        searcher.load_index()
        st.session_state["searcher"] = searcher
        st.session_state["searcher_key"] = key
    return st.session_state["searcher"]


def run_indexing(folder: str, db_path: str, model_name: str, pretrained: str, batch_size: int) -> None:
    """Execute indexing and update Streamlit widgets."""
    root = Path(folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        st.error(f"Invalid folder: {root}")
        return

    progress = st.progress(0.0, text="Starting...")
    status = st.empty()

    store = PhotoStore(db_path)
    try:
        embedder = get_embedder(model_name=model_name, pretrained=pretrained)
        indexer = PhotoIndexer(store=store, embedder=embedder, batch_size=batch_size)

        def cb(done: int, total: int) -> None:
            pct = (done / total) if total else 1.0
            progress.progress(min(1.0, pct), text=f"Processed {done}/{total} files")

        with st.spinner("Indexing photos..."):
            summary = indexer.index_folder(root, progress_callback=cb)

        progress.progress(1.0, text="Completed")
        status.success(
            " | ".join(
                [
                    f"Indexed: {summary.total_indexed}",
                    f"Skipped: {summary.total_skipped}",
                    f"Unchanged: {summary.total_unchanged}",
                    f"Errors: {summary.total_errors}",
                ]
            )
        )

        # Force search cache refresh.
        st.session_state.pop("searcher", None)
        st.session_state.pop("searcher_key", None)
    finally:
        store.close()


def render_results(results: list[SearchResult], columns: int = 4) -> None:
    """Render search results in thumbnail grid."""
    if not results:
        st.info("No results found.")
        return

    cols = st.columns(columns)
    for idx, item in enumerate(results):
        col = cols[idx % columns]
        with col:
            path = Path(item.file_path)
            thumb = load_thumbnail_array(path)
            if thumb is not None:
                st.image(thumb, use_container_width=True)
            else:
                st.warning("Preview unavailable")

            st.caption(f"Score: {item.score:.4f}")
            st.caption(path.name)
            st.code(str(path), language=None)
            if st.button("Open", key=f"open_{idx}_{item.file_path}"):
                ok = open_in_finder(path)
                if not ok:
                    st.error(f"Failed to open: {path}")


with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("DB Path", value="photo_index.db")
    model_name = st.text_input("Model", value="ViT-B-32")
    pretrained = st.text_input("Pretrained", value="laion2b_s34b_b79k")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=32, step=1)

st.subheader("Index")
folder = st.text_input("Photo Folder", placeholder="/Volumes/ExternalDrive/Photos")
if st.button("Index Folder", type="primary"):
    run_indexing(
        folder=folder,
        db_path=db_path,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=int(batch_size),
    )

st.subheader("Search")
query = st.text_input("Text Query", placeholder="fridge")
topk = st.slider("Top K", min_value=1, max_value=200, value=30, step=1)

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a query.")
    else:
        searcher = get_searcher(db_path)
        embedder = get_embedder(model_name=model_name, pretrained=pretrained)
        results = searcher.search(query=query.strip(), topk=topk, embedder=embedder)
        st.session_state["results"] = results

render_results(st.session_state.get("results", []))
