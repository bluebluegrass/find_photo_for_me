"""Streamlit UI for local photo indexing and CLIP search."""

from __future__ import annotations

from datetime import datetime, time as dtime
from pathlib import Path

import streamlit as st

from indexer import CLIPEmbedder, PhotoIndexer
from searcher import PhotoSearcher, SearchResult
from store import PhotoStore
from utils import choose_folder_dialog_macos, default_db_path, load_thumbnail_array, open_in_finder

st.set_page_config(page_title="LocalPix", layout="wide")
st.title("LocalPix")
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


def render_results(results: list[SearchResult], db_path: str, columns: int = 4) -> None:
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
            if item.taken_ts is not None:
                st.caption(f"Taken: {datetime.fromtimestamp(item.taken_ts).strftime('%Y-%m-%d %H:%M:%S')}")
            if item.latitude is not None and item.longitude is not None:
                st.caption(f"GPS: {item.latitude:.6f}, {item.longitude:.6f}")
            if item.media_type == "video_frame":
                ts_text = f"{item.frame_ts:.1f}s" if item.frame_ts is not None else "-"
                st.caption(f"Video frame @ {ts_text}")
                st.caption(f"Source: {Path(item.source_path).name}")
            st.caption(path.name)
            st.code(str(path), language=None)
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Open", key=f"open_{idx}_{item.file_path}"):
                    open_target = Path(item.source_path) if item.media_type == "video_frame" else path
                    ok = open_in_finder(open_target)
                    if not ok:
                        st.error(f"Failed to open: {open_target}")
            with action_cols[1]:
                if st.session_state.get("delete_mode", False):
                    if st.button("Delete", key=f"delete_{idx}_{item.file_path}", type="secondary"):
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
    st.header("Settings")
    db_path = st.text_input("DB Path", value=default_db_path())
    model_name = st.text_input("Model", value="ViT-B-32")
    pretrained = st.text_input("Pretrained", value="laion2b_s34b_b79k")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=32, step=1)
    video_interval_sec = st.number_input("Video frame interval (sec)", min_value=0.1, max_value=30.0, value=1.5, step=0.1)
    video_max_frames = st.number_input("Video max frames", min_value=1, max_value=5000, value=300, step=10)
    video_cache_dir = st.text_input("Video frame cache dir", value=".video_frame_cache")

st.subheader("Index")
if "index_folder" not in st.session_state:
    st.session_state["index_folder"] = ""
if "pending_index_folder" in st.session_state:
    st.session_state["index_folder"] = st.session_state.pop("pending_index_folder")

folder_col, actions_col = st.columns([5, 2])
with folder_col:
    folder = st.text_input("Photo Folder", placeholder="/Volumes/ExternalDrive/Photos", key="index_folder")
with actions_col:
    st.write("")
    st.write("")
    if st.button("Browse Finder"):
        selected = choose_folder_dialog_macos()
        if selected:
            st.session_state["pending_index_folder"] = selected
            st.rerun()
    if st.button("Open in Finder"):
        current = Path(st.session_state.get("index_folder", "")).expanduser()
        if current.exists():
            open_in_finder(current)
        else:
            st.warning("Current folder path does not exist.")

force_reindex = st.checkbox("Force reindex all files (backfill metadata)", value=False)
prune_deleted = st.checkbox("Prune missing files from DB", value=False)
if st.button("Index Folder", type="primary"):
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
query = st.text_input("Text Query", placeholder="fridge")
topk = st.slider("Top K", min_value=1, max_value=200, value=30, step=1)
col1, col2, col3 = st.columns(3)
with col1:
    from_date = st.text_input("From date (YYYY-MM-DD)", value="")
with col2:
    to_date = st.text_input("To date (YYYY-MM-DD)", value="")
with col3:
    has_gps = st.checkbox("Only with GPS", value=False)
col_media = st.columns(1)[0]
with col_media:
    media_option = st.selectbox("Media type", options=["Photo", "Video", "Both"], index=0)
col4, col5 = st.columns(2)
with col4:
    min_score = st.slider("Min similarity score", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
with col5:
    relative_to_best = st.slider("Max gap from best score", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
delete_mode = st.checkbox("Enable delete buttons (permanent)", value=False, key="delete_mode")

if st.button("Search"):
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
        media_filter = {"Photo": "photo", "Video": "video", "Both": "both"}[media_option]
        results = searcher.search(
            query=query.strip(),
            topk=topk,
            embedder=embedder,
            min_taken_ts=min_ts,
            max_taken_ts=max_ts,
            has_gps=has_gps,
            min_score=float(min_score),
            relative_to_best=float(relative_to_best),
            media_filter=media_filter,
        )
        st.session_state["results"] = results

render_results(st.session_state.get("results", []), db_path=db_path)
