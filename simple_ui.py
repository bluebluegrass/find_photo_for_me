"""Minimal Streamlit UI: text query + preview grid from existing photo index."""

from __future__ import annotations

from datetime import datetime, time as dtime
from pathlib import Path

import streamlit as st

from indexer import CLIPEmbedder
from searcher import PhotoSearcher, SearchResult
from store import PhotoStore
from utils import default_db_path, load_thumbnail_array, open_in_finder

st.set_page_config(page_title="Photo Search", layout="wide")
st.title("Photo Search")

DB_PATH = default_db_path()


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
            if item.taken_ts is not None:
                st.caption(f"Taken: {datetime.fromtimestamp(item.taken_ts).strftime('%Y-%m-%d %H:%M:%S')}")
            if item.latitude is not None and item.longitude is not None:
                st.caption(f"GPS: {item.latitude:.6f}, {item.longitude:.6f}")
            st.caption(path.name)
            if st.button("Open", key=f"open_{i}_{item.file_path}"):
                open_in_finder(path)


query = st.text_input("Query", placeholder="cat")
topk = st.slider("Top K", min_value=1, max_value=100, value=24, step=1)
col1, col2, col3 = st.columns(3)
with col1:
    from_date = st.text_input("From date (YYYY-MM-DD)", value="")
with col2:
    to_date = st.text_input("To date (YYYY-MM-DD)", value="")
with col3:
    has_gps = st.checkbox("Only with GPS", value=False)
col4, col5 = st.columns(2)
with col4:
    min_score = st.slider("Min similarity score", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
with col5:
    relative_to_best = st.slider("Max gap from best score", min_value=0.0, max_value=0.5, value=0.10, step=0.01)

if st.button("Search", type="primary"):
    db_file = Path(DB_PATH).resolve()
    if not db_file.exists():
        st.error(f"DB not found: {db_file}")
    else:
        min_ts: int | None = None
        max_ts: int | None = None
        if from_date.strip():
            try:
                d = datetime.strptime(from_date.strip(), "%Y-%m-%d").date()
                min_ts = int(datetime.combine(d, dtime.min).timestamp())
            except ValueError:
                st.error("Invalid from-date format. Use YYYY-MM-DD.")
                st.stop()
        if to_date.strip():
            try:
                d = datetime.strptime(to_date.strip(), "%Y-%m-%d").date()
                max_ts = int(datetime.combine(d, dtime.max).timestamp())
            except ValueError:
                st.error("Invalid to-date format. Use YYYY-MM-DD.")
                st.stop()
        with st.spinner("Searching..."):
            searcher = get_searcher(DB_PATH)
            embedder = get_embedder()
            results = searcher.search(
                query=query.strip(),
                topk=topk,
                embedder=embedder,
                min_taken_ts=min_ts,
                max_taken_ts=max_ts,
                has_gps=has_gps,
                min_score=float(min_score),
                relative_to_best=float(relative_to_best),
            )
            st.session_state["results"] = results

render_results(st.session_state.get("results", []))
