"""Streamlit UI for local photo indexing and CLIP search."""

from __future__ import annotations

from datetime import datetime, time as dtime
from pathlib import Path

import streamlit as st

from indexer import CLIPEmbedder, PhotoIndexer
from llm_parser import SmartQueryParser
from searcher import PhotoSearcher, SearchResult
from store import PhotoStore
from utils import (
    choose_folder_dialog_macos,
    default_db_path,
    default_llm_endpoint,
    default_llm_model,
    default_llm_timeout,
    default_location_mode,
    load_thumbnail_array,
    open_in_finder,
)

st.set_page_config(page_title="LocalPix", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --lp-bg: #f4efe6;
        --lp-paper: #fbf7f0;
        --lp-paper-2: #f0e6d8;
        --lp-ink: #232018;
        --lp-muted: #6b6458;
        --lp-accent: #b55d3d;
        --lp-accent-soft: #e8c5b5;
        --lp-line: rgba(94, 78, 60, 0.16);
        --lp-shadow: 0 24px 60px rgba(71, 53, 34, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(181, 93, 61, 0.10), transparent 28rem),
            linear-gradient(180deg, #f7f2ea 0%, var(--lp-bg) 100%);
        color: var(--lp-ink);
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(246, 241, 233, 0.98) 0%, rgba(239, 230, 218, 0.98) 100%);
        border-right: 1px solid var(--lp-line);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    html, body, [class*="css"] {
        color: var(--lp-ink);
        font-family: "Avenir Next", "Segoe UI", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        letter-spacing: -0.02em;
        color: var(--lp-ink);
    }

    h2 {
        font-size: clamp(1.8rem, 2.8vw, 2.5rem);
        margin-top: 0.4rem;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--lp-muted);
    }

    .lp-hero {
        background:
            linear-gradient(135deg, rgba(251, 247, 240, 0.92), rgba(240, 230, 216, 0.88)),
            var(--lp-paper);
        border: 1px solid rgba(181, 93, 61, 0.14);
        border-radius: 28px;
        padding: 1.5rem 1.6rem 1.3rem 1.6rem;
        box-shadow: var(--lp-shadow);
        margin: 0 0 1.6rem 0;
        position: relative;
        overflow: hidden;
    }

    .lp-hero::after {
        content: "";
        position: absolute;
        inset: auto -3rem -4rem auto;
        width: 11rem;
        height: 11rem;
        background: radial-gradient(circle, rgba(181, 93, 61, 0.18), transparent 65%);
        pointer-events: none;
    }

    .lp-kicker {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.74rem;
        color: var(--lp-accent);
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .lp-title {
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        font-size: clamp(2.3rem, 5vw, 4.2rem);
        line-height: 0.92;
        letter-spacing: -0.04em;
        color: var(--lp-ink);
        margin: 0;
        max-width: 10ch;
    }

    .lp-subtitle {
        max-width: 42rem;
        font-size: 1rem;
        line-height: 1.55;
        color: var(--lp-muted);
        margin: 0.95rem 0 1.2rem 0;
    }

    .lp-pillrow {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
    }

    .lp-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        background: rgba(255, 250, 244, 0.92);
        border: 1px solid rgba(94, 78, 60, 0.12);
        color: var(--lp-ink);
        font-size: 0.92rem;
    }

    .lp-section-note {
        margin: -0.2rem 0 1rem 0;
        color: var(--lp-muted);
    }

    .lp-results-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 0.5rem;
    }

    .lp-results-copy {
        color: var(--lp-muted);
        font-size: 0.98rem;
    }

    .lp-filename {
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.28;
        color: var(--lp-ink);
        margin: 0.72rem 0 0.45rem 0;
        min-height: 2.6em;
    }

    .lp-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0 0 0.6rem 0;
    }

    .lp-meta-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.3rem 0.62rem;
        background: rgba(240, 230, 216, 0.64);
        border: 1px solid rgba(94, 78, 60, 0.1);
        color: #584f43;
        font-size: 0.77rem;
        line-height: 1.2;
        white-space: nowrap;
    }

    .lp-card-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(94, 78, 60, 0.14), rgba(94, 78, 60, 0.02));
        margin: 0.55rem 0 0.8rem 0;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--lp-line);
        border-radius: 18px;
        background: rgba(251, 247, 240, 0.82);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stDateInput"] input {
        background: rgba(255, 251, 246, 0.96);
        border: 1px solid rgba(94, 78, 60, 0.14);
        border-radius: 16px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stSlider"] label {
        font-weight: 600;
        color: var(--lp-ink);
    }

    .stButton > button,
    [data-testid="baseButton-secondary"] {
        border-radius: 999px;
        min-height: 2.8rem;
        border: 1px solid rgba(94, 78, 60, 0.12);
        background: rgba(255, 251, 246, 0.98);
        color: var(--lp-ink);
        transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }

    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(71, 53, 34, 0.08);
        border-color: rgba(181, 93, 61, 0.26);
    }

    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #c96c49 0%, var(--lp-accent) 100%);
        color: #fff7f1;
        border-color: rgba(181, 93, 61, 0.22);
        box-shadow: 0 14px 28px rgba(181, 93, 61, 0.22);
    }

    [data-testid="stInfo"] {
        background: rgba(255, 251, 246, 0.92);
        border: 1px solid rgba(181, 93, 61, 0.14);
        border-radius: 18px;
        color: var(--lp-ink);
    }

    [data-testid="stAlert"] {
        border-radius: 18px;
    }

    [data-testid="stImage"] img {
        border-radius: 18px;
    }

    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1rem;
        }

        .lp-hero {
            padding: 1.15rem 1rem;
            border-radius: 22px;
        }

        .lp-title {
            max-width: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def run_indexing(
    folder: str,
    db_path: str,
    model_name: str,
    pretrained: str,
    batch_size: int,
    force_reindex: bool = False,
    prune_deleted: bool = False,
    video_interval_sec: float = 1.5,
    video_max_frames: int = 300,
    video_cache_dir: str = ".video_frame_cache",
) -> None:
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
            summary = indexer.index_folder(
                root,
                progress_callback=cb,
                force_reindex=force_reindex,
                prune_deleted=prune_deleted,
                video_interval_sec=video_interval_sec,
                video_max_frames=video_max_frames,
                video_frame_cache_dir=video_cache_dir,
            )

        progress.progress(1.0, text="Completed")
        status.success(
            " | ".join(
                [
                    f"Indexed: {summary.total_indexed}",
                    f"Skipped: {summary.total_skipped}",
                    f"Non-image: {summary.skipped_non_image}",
                    f"Decode failures: {summary.skipped_decode_failure}",
                    f"Unchanged: {summary.total_unchanged}",
                    f"Errors: {summary.total_errors}",
                    f"Pruned: {summary.total_pruned}",
                ]
            )
        )

        # Force search cache refresh.
        st.session_state.pop("searcher", None)
        st.session_state.pop("searcher_key", None)
    finally:
        store.close()


def delete_photo_and_index(file_path: str, db_path: str) -> tuple[bool, str]:
    """Delete photo file and corresponding DB row."""
    path = Path(file_path)
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        return False, f"Failed to delete file: {exc}"

    try:
        store = PhotoStore(db_path)
        deleted = store.delete_paths([str(path.resolve())])
        store.commit()
        store.close()
        if deleted == 0:
            return False, "File deleted, but no DB row was removed."
    except Exception as exc:
        return False, f"File deleted, but DB update failed: {exc}"

    # Refresh cached search matrix after destructive DB update.
    st.session_state.pop("searcher", None)
    st.session_state.pop("searcher_key", None)
    return True, "Deleted file and removed from index."


def delete_media_result(item: SearchResult, db_path: str) -> tuple[bool, str]:
    """Delete media source and associated index rows."""
    if item.media_type == "video_frame":
        source = Path(item.source_path)
        try:
            if source.exists():
                source.unlink()
        except Exception as exc:
            return False, f"Failed to delete video file: {exc}"

        try:
            store = PhotoStore(db_path)
            frame_paths = store.load_paths_by_source(item.source_path, "video_frame")
            for p in frame_paths:
                pp = Path(p)
                if pp.exists():
                    try:
                        pp.unlink()
                    except Exception:
                        pass
            if frame_paths:
                store.delete_paths(frame_paths)
                store.commit()
            store.close()
        except Exception as exc:
            return False, f"Video deleted, but DB cleanup failed: {exc}"

        st.session_state.pop("searcher", None)
        st.session_state.pop("searcher_key", None)
        return True, "Deleted video and removed indexed frames."

    return delete_photo_and_index(item.file_path, db_path)


def render_results(results: list[SearchResult], db_path: str, columns: int = 3) -> None:
    """Render search results in thumbnail grid."""
    if not results:
        st.info("No matches yet. Try a simpler description or relax the filters in More Search Options.")
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

            st.markdown(f'<div class="lp-filename">{path.name}</div>', unsafe_allow_html=True)

            meta_chips: list[str] = []
            if item.taken_ts is not None:
                meta_chips.append(datetime.fromtimestamp(item.taken_ts).strftime("%b %d, %Y"))
            if item.city_name or item.country_name or item.country_code:
                city = item.city_name or None
                country = item.country_name or item.country_code
                place = ", ".join([p for p in [city, country] if p])
                if place:
                    meta_chips.append(place)
            if item.media_type == "video_frame":
                ts_text = f"{item.frame_ts:.1f}s" if item.frame_ts is not None else "-"
                meta_chips.append(f"Video @ {ts_text}")

            if meta_chips:
                chips_html = "".join(f'<span class="lp-meta-chip">{chip}</span>' for chip in meta_chips)
                st.markdown(f'<div class="lp-meta">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="lp-meta"><span class="lp-meta-chip">No date or location</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="lp-card-divider"></div>', unsafe_allow_html=True)

            with st.expander("View details", expanded=False):
                st.caption(f"Score: {item.score:.4f}")
                if item.latitude is not None and item.longitude is not None:
                    st.caption(f"GPS: {item.latitude:.6f}, {item.longitude:.6f}")
                if item.media_type == "video_frame":
                    st.caption(f"Source: {Path(item.source_path).name}")
                st.code(str(path), language=None)
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Open", key=f"open_{idx}_{item.file_path}", use_container_width=True):
                    open_target = Path(item.source_path) if item.media_type == "video_frame" else path
                    ok = open_in_finder(open_target)
                    if not ok:
                        st.error(f"Failed to open: {open_target}")
            with action_cols[1]:
                if st.session_state.get("delete_mode", False):
                    if st.button(
                        "Delete",
                        key=f"delete_{idx}_{item.file_path}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        ok, msg = delete_media_result(item, db_path)
                        if ok:
                            st.success(msg)
                            st.session_state["results"] = [
                                r for r in st.session_state.get("results", []) if r.file_path != item.file_path
                            ]
                            st.rerun()
                        else:
                            st.error(msg)


with st.sidebar:
    st.caption("Everything stays on this Mac.")
    smart_query_enabled = st.checkbox("Understand natural descriptions", value=False, help="Use a local model to better understand natural-language queries.")
    if smart_query_enabled:
        st.caption("Examples: `black cat in turkey`, `receipt from 2023`, `whiteboard in office`")

    db_path = default_db_path()
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    batch_size = 32
    video_interval_sec = 1.5
    video_max_frames = 300
    video_cache_dir = ".video_frame_cache"
    llm_model = default_llm_model()
    llm_timeout = float(default_llm_timeout())
    llm_endpoint = default_llm_endpoint()

    with st.expander("Technical Settings", expanded=False):
        st.caption("Most users do not need to change these.")
        db_path = st.text_input("Index Database Path", value=db_path)
        model_name = st.text_input("Search Model", value=model_name)
        pretrained = st.text_input("Model Weights", value=pretrained)
        batch_size = st.number_input("Index Batch Size", min_value=1, max_value=256, value=batch_size, step=1)
        video_interval_sec = st.number_input(
            "Video sampling interval (sec)",
            min_value=0.1,
            max_value=30.0,
            value=video_interval_sec,
            step=0.1,
        )
        video_max_frames = st.number_input("Max sampled video frames", min_value=1, max_value=5000, value=video_max_frames, step=10)
        video_cache_dir = st.text_input("Video frame cache folder", value=video_cache_dir)
        if smart_query_enabled:
            st.divider()
            st.caption("Smart Search runtime")
            llm_model = st.text_input("LLM Model", value=llm_model)
            llm_timeout = st.number_input(
                "LLM Timeout (sec)",
                min_value=0.2,
                max_value=20.0,
                value=llm_timeout,
                step=0.1,
            )
            llm_endpoint = st.text_input("LLM Endpoint", value=llm_endpoint)
index_count = get_index_count(db_path)
library_status = f"{index_count:,} items indexed" if index_count > 0 else "No library indexed yet"
st.markdown(
    f"""
    <section class="lp-hero">
        <div class="lp-kicker">Private Visual Search</div>
        <h1 class="lp-title">Find the photo you meant.</h1>
        <p class="lp-subtitle">
            Search your personal library with natural language, location hints, and on-device indexing.
            Nothing leaves this Mac.
        </p>
        <div class="lp-pillrow">
            <span class="lp-pill">{library_status}</span>
            <span class="lp-pill">Photos and videos</span>
            <span class="lp-pill">Offline by default</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.subheader("Library")
st.markdown('<p class="lp-section-note">Choose the folder where your photos and videos live.</p>', unsafe_allow_html=True)
if "index_folder" not in st.session_state:
    st.session_state["index_folder"] = ""
if "pending_index_folder" in st.session_state:
    st.session_state["index_folder"] = st.session_state.pop("pending_index_folder")

folder_col, actions_col = st.columns([5, 2])
with folder_col:
    folder = st.text_input("Photo or Video Folder", placeholder="/Volumes/ExternalDrive/Photos", key="index_folder")
with actions_col:
    st.write("")
    st.write("")
    if st.button("Browse Finder"):
        selected = choose_folder_dialog_macos()
        if selected:
            st.session_state["pending_index_folder"] = selected
            st.rerun()
    if st.button("Open Folder"):
        current = Path(st.session_state.get("index_folder", "")).expanduser()
        if current.exists():
            open_in_finder(current)
        else:
            st.warning("Current folder path does not exist.")

with st.expander("More Indexing Options", expanded=False):
    force_reindex = st.checkbox("Recheck every file", value=False)
    prune_deleted = st.checkbox("Remove missing files from index", value=False)
if st.button("Index Library", type="primary"):
    run_indexing(
        folder=st.session_state.get("index_folder", folder),
        db_path=db_path,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=int(batch_size),
        force_reindex=force_reindex,
        prune_deleted=prune_deleted,
        video_interval_sec=float(video_interval_sec),
        video_max_frames=int(video_max_frames),
        video_cache_dir=video_cache_dir,
    )

st.subheader("Search")
st.markdown('<p class="lp-section-note">Describe what you want to find.</p>', unsafe_allow_html=True)
search_bar_col, search_button_col = st.columns([5, 1])
with search_bar_col:
    query = st.text_input(
        "Search your library",
        placeholder="Try: cat, black cat in turkey, receipt, whiteboard in office",
        label_visibility="collapsed",
    )
with search_button_col:
    st.write("")
    st.write("")
    do_search = st.button("Search", type="primary", use_container_width=True)

topk = st.slider("How many results", min_value=1, max_value=200, value=30, step=1)
quick_col1, quick_col2, quick_col3 = st.columns(3)
with quick_col1:
    media_option = st.selectbox("Search in", options=["Photos", "Videos", "Both"], index=0)
with quick_col2:
    has_gps = st.checkbox("Only show items with location", value=False)
with quick_col3:
    delete_mode = st.checkbox("Show delete buttons", value=False, key="delete_mode")
st.caption("Delete removes the original file from your disk.")

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
    if not query.strip():
        st.warning("Enter a query.")
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
        searcher = get_searcher(db_path)
        embedder = get_embedder(model_name=model_name, pretrained=pretrained)
        media_filter = {"Photos": "photo", "Videos": "video", "Both": "both"}[media_option]
        query_text = query.strip()
        prompt_list: list[str] | None = None
        location_query: str | None = None
        location_mode = location_mode_label.lower()
        if smart_query_enabled:
            parser = SmartQueryParser(model=llm_model, timeout_sec=float(llm_timeout), endpoint=llm_endpoint)
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
    st.markdown(
        f"""
        <div class="lp-results-head">
            <h2>Results</h2>
            <div class="lp-results-copy">{len(results)} matches for “{last_query}”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif last_query:
    st.markdown('<div class="lp-results-head"><h2>Results</h2></div>', unsafe_allow_html=True)
    st.info(f"No matches for “{last_query}”. Try a simpler description or relax the search options.")

if results is not None:
    render_results(results, db_path=db_path)
