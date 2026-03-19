"""Minimal Streamlit UI: text query + preview grid from existing photo index."""

from __future__ import annotations

from datetime import datetime, time as dtime
from pathlib import Path

import streamlit as st

from indexer import CLIPEmbedder
from llm_parser import SmartQueryParser
from searcher import PhotoSearcher, SearchResult
from store import PhotoStore
from utils import default_db_path, default_llm_model, default_location_mode, load_thumbnail_array, open_in_finder

st.set_page_config(page_title="LocalPix", layout="wide")
st.title("LocalPix")
st.caption("Search your photos and videos offline.")

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


def get_index_count(db_path: str) -> int:
    """Return current indexed row count for header display."""
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        return 0
    store = PhotoStore(path)
    try:
        return store.get_total_count()
    finally:
        store.close()


def render_results(results: list[SearchResult], columns: int = 4) -> None:
    """Render search results in a simple thumbnail grid."""
    if not results:
        st.info("No matches yet. Try a simpler description or relax the filters in More Search Options.")
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
            st.markdown(f"**{path.name}**")
            if item.taken_ts is not None:
                st.caption(f"Taken: {datetime.fromtimestamp(item.taken_ts).strftime('%b %d, %Y %H:%M')}")
            if item.city_name or item.country_name or item.country_code:
                city = item.city_name or "-"
                country = item.country_name or (item.country_code or "-")
                st.caption(f"Location: {city}, {country}")
            if item.media_type == "video_frame":
                ts_text = f"{item.frame_ts:.1f}s" if item.frame_ts is not None else "-"
                st.caption(f"Video frame @ {ts_text}")
            with st.expander("Details", expanded=False):
                st.caption(f"Score: {item.score:.4f}")
                if item.latitude is not None and item.longitude is not None:
                    st.caption(f"GPS: {item.latitude:.6f}, {item.longitude:.6f}")
                if item.media_type == "video_frame":
                    st.caption(f"Source: {Path(item.source_path).name}")
                st.code(str(path), language=None)
            if st.button("Open", key=f"open_{i}_{item.file_path}"):
                open_target = Path(item.source_path) if item.media_type == "video_frame" else path
                open_in_finder(open_target)

index_count = get_index_count(DB_PATH)
if index_count > 0:
    st.info(f"Library ready: {index_count:,} items indexed")
else:
    st.info("No library indexed yet.")

search_col, button_col = st.columns([5, 1])
with search_col:
    query = st.text_input(
        "Search your library",
        placeholder="Try: cat, black cat in turkey, receipt, whiteboard in office",
        label_visibility="collapsed",
    )
with button_col:
    st.write("")
    st.write("")
    do_search = st.button("Search", type="primary", use_container_width=True)

topk = st.slider("How many results", min_value=1, max_value=100, value=24, step=1)
smart_query_enabled = st.checkbox("Understand natural descriptions", value=False, help="Use a local model to better understand natural-language queries.")
llm_model = default_llm_model()
if smart_query_enabled:
    st.caption("Examples: `black cat in turkey`, `receipt from 2023`, `whiteboard in office`")
    with st.expander("Technical Settings", expanded=False):
        llm_model = st.text_input("LLM Model", value=llm_model)
quick_col1, quick_col2 = st.columns(2)
with quick_col1:
    media_option = st.selectbox("Search in", options=["Photos", "Videos", "Both"], index=0)
with quick_col2:
    has_gps = st.checkbox("Only show items with location", value=False)

with st.expander("More Search Options", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.text_input("From date (YYYY-MM-DD)", value="")
    with col2:
        to_date = st.text_input("To date (YYYY-MM-DD)", value="")
    location_mode_label = st.selectbox(
        "Location matching",
        options=["Hybrid", "Strict", "Off"],
        index={"hybrid": 0, "strict": 1, "off": 2}.get(default_location_mode(), 0),
    )
    col4, col5 = st.columns(2)
    with col4:
        min_score = st.slider("Minimum match confidence", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
    with col5:
        relative_to_best = st.slider("Keep results close to the best match", min_value=0.0, max_value=0.5, value=0.10, step=0.01)

if do_search:
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
            media_filter = {"Photos": "photo", "Videos": "video", "Both": "both"}[media_option]
            query_text = query.strip()
            prompt_list: list[str] | None = None
            location_query: str | None = None
            location_mode = location_mode_label.lower()
            if smart_query_enabled:
                parser = SmartQueryParser(model=llm_model)
                intent = parser.parse(query_text)
                st.session_state["smart_intent"] = intent
                query_text = intent.visual_query or query_text
                prompt_list = intent.expanded_queries or None
                location_query = intent.normalized_location_text
            else:
                st.session_state.pop("smart_intent", None)
                st.session_state.pop("location_status", None)
                location_mode = "off"
            results = searcher.search(
                query=query_text,
                topk=topk,
                embedder=embedder,
                min_taken_ts=min_ts,
                max_taken_ts=max_ts,
                has_gps=has_gps,
                min_score=float(min_score),
                relative_to_best=float(relative_to_best),
                media_filter=media_filter,
                text_prompts=prompt_list,
                location_query=location_query,
                location_mode=location_mode,
            )
            st.session_state["last_query"] = query.strip()
            st.session_state["results"] = results
            st.session_state["location_status"] = {
                "status": searcher.last_location_status,
                "query": searcher.last_location_query,
                "mode": searcher.last_location_mode,
                "matches": searcher.last_location_match_count,
            }

intent = st.session_state.get("smart_intent")
if intent is not None:
    with st.expander("How LocalPix understood your search", expanded=False):
        st.write(f"Looking for: `{intent.visual_query}`")
        if intent.location_text:
            st.write(f"Place: `{intent.location_text}`")
        if intent.time_text:
            st.write(f"Time: `{intent.time_text}`")

location_status = st.session_state.get("location_status")
if location_status and location_status.get("query"):
    status = location_status.get("status")
    query_text = location_status.get("query")
    if status == "hybrid_applied":
        st.info(f"Using location match: `{query_text}`.")
    elif status == "hybrid_fallback_no_match":
        st.info(f"No indexed items matched `{query_text}`, so LocalPix showed the closest visual results instead.")
    elif status == "strict_applied":
        st.info(f"Showing only results that match `{query_text}`.")
    elif status == "strict_no_match":
        st.warning(f"No indexed items matched `{query_text}` with strict location matching.")

results = st.session_state.get("results")
last_query = st.session_state.get("last_query")
if results:
    st.subheader("Results")
    st.caption(f"{len(results)} matches for “{last_query}”")
elif last_query:
    st.subheader("Results")
    st.info(f"No matches for “{last_query}”. Try a simpler description or relax the search options.")

if results is not None:
    render_results(results)
