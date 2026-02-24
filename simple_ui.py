"""Minimal Streamlit UI: text query + preview grid from existing photo index."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from indexer import CLIPEmbedder
from searcher import PhotoSearcher, SearchResult
from store import PhotoStore
from utils import load_thumbnail_array, open_in_finder

st.set_page_config(page_title="Photo Search", layout="wide")
st.title("Photo Search")

DB_PATH = "photo_index.db"


@st.cache_resource(show_spinner=False)
def get_embedder() -> CLIPEmbedder:
    """Load CLIP model once."""
    return CLIPEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")


@st.cache_resource(show_spinner=False)
def get_searcher(db_path: str) -> PhotoSearcher:
    """Load embeddings matrix once per DB path."""
    store = PhotoStore(db_path)
    searcher = PhotoSearcher(store)
    searcher.load_index()
    return searcher


def render_results(results: list[SearchResult], columns: int = 4) -> None:
    """Render search results in a simple thumbnail grid."""
    if not results:
        st.info("No results.")
        return

    cols = st.columns(columns)
    for i, item in enumerate(results):
        with cols[i % columns]:
            path = Path(item.file_path)
            thumb = load_thumbnail_array(path)
            if thumb is not None:
                st.image(thumb, use_container_width=True)
            else:
                st.warning("Preview unavailable")
            st.caption(f"{item.score:.4f}")
            st.caption(path.name)
            if st.button("Open", key=f"open_{i}_{item.file_path}"):
                open_in_finder(path)


query = st.text_input("Query", placeholder="cat")
topk = st.slider("Top K", min_value=1, max_value=100, value=24, step=1)

if st.button("Search", type="primary"):
    db_file = Path(DB_PATH).resolve()
    if not db_file.exists():
        st.error(f"DB not found: {db_file}")
    else:
        with st.spinner("Searching..."):
            searcher = get_searcher(DB_PATH)
            embedder = get_embedder()
            results = searcher.search(query=query.strip(), topk=topk, embedder=embedder)
            st.session_state["results"] = results

render_results(st.session_state.get("results", []))
