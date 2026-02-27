"""
Batch inference service.

Processes a list of (pre_image, post_image) pairs, calling predict_damage()
on each. Errors on individual tiles are caught so the batch continues.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image

from .inference import predict_damage
from .priority import compute_priority, compute_aggregate_stats
from .report_generator import generate_report

logger = logging.getLogger(__name__)

# In-memory session store: session_id → AnalysisResult dict
# For a production system, replace with Redis or SQLite.
_sessions: dict = {}

# Grid layout: if no coords supplied, space tiles by this many degrees
GRID_SPACING_DEG = 0.005   # ~550 m between tile centres at equator
DEFAULT_CENTER   = (0.0, 0.0)


def _grid_coords(index: int, total: int, center: Tuple[float, float] = DEFAULT_CENTER):
    """
    Assign synthetic lat/lon for a tile using a square grid layout.
    Tiles are arranged row-by-row, centred on `center`.
    """
    import math
    cols = math.ceil(math.sqrt(total))
    row, col = divmod(index, cols)
    lat = center[0] + row * GRID_SPACING_DEG
    lon = center[1] + col * GRID_SPACING_DEG
    return round(lat, 6), round(lon, 6)


def create_session() -> str:
    """Create a new analysis session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "status": "processing",
        "total_tiles": 0,
        "completed": 0,
        "failed": 0,
        "tiles": [],
        "stats": None,
        "top_priority": None,
        "situation_report": None,
    }
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Retrieve a session by ID."""
    return _sessions.get(session_id)


def run_batch(
    session_id: str,
    pairs: List[Tuple[Optional[bytes], bytes, Optional[str], str]],  # (pre_bytes, post_bytes, pre_name, post_name)
    coords: Optional[List[Tuple[float, float]]] = None,  # optional [(lat, lon), ...]
    center: Optional[Tuple[float, float]] = None,
):
    """
    Process all image pairs for a session.

    Args:
        session_id: Active session.
        pairs:      List of (pre_bytes, post_bytes, pre_filename, post_filename).
        coords:     Optional per-tile (lat, lon). Falls back to grid layout.
        center:     Grid centre point for coordinate fallback.
    """
    session = _sessions[session_id]
    session["total_tiles"] = len(pairs)

    tile_results = []

    for idx, (pre_bytes, post_bytes, pre_name, post_name) in enumerate(pairs):
        tile_id = f"tile_{idx:03d}"
        
        lat, lon = None, None
        if coords and idx < len(coords):
            lat, lon = coords[idx]
        elif center is not None:
            lat, lon = _grid_coords(idx, len(pairs), center)

        try:
            pre_img = None
            if pre_bytes:
                pre_img = Image.open(__import__("io").BytesIO(pre_bytes)).convert("RGB")
            
            post_img = Image.open(__import__("io").BytesIO(post_bytes)).convert("RGB")

            result = predict_damage(pre_img, post_img)
            priority = compute_priority(result["class"], result["confidence"])

            tile_result = {
                "tile_id": tile_id,
                "filename_pre": pre_name,
                "filename_post": post_name,
                "damage_class": result["class"],
                "confidence": round(result["confidence"], 4),
                "probability_distribution": result.get("probabilities", {}),
                "priority_score": priority,
                "gradcam": result.get("gradcam"),
                "lat": lat,
                "lon": lon,
                "status": "ok",
                "error_message": None,
            }
            session["completed"] += 1

        except Exception as e:
            import traceback
            err_msg = f"{e}\n{traceback.format_exc()}"
            logger.error(f"[{tile_id}] Inference failed: {err_msg}")
            tile_result = {
                "tile_id": tile_id,
                "filename_pre": pre_name,
                "filename_post": post_name,
                "damage_class": "unknown",
                "confidence": 0.0,
                "probability_distribution": {},
                "priority_score": 0.0,
                "lat": lat,
                "lon": lon,
                "status": "error",
                "error_message": err_msg,
            }
            session["failed"] += 1

        tile_results.append(tile_result)
        session["tiles"] = tile_results  # stream partial updates

    # ── Aggregate stats ────────────────────────────────────────────────────────
    stats = compute_aggregate_stats(tile_results)
    top_tiles = sorted(
        [t for t in tile_results if t["status"] == "ok"],
        key=lambda t: t["priority_score"],
        reverse=True,
    )[:5]

    session["stats"] = stats
    session["top_priority"] = top_tiles
    session["situation_report"] = generate_report(stats, top_tiles)
    session["status"] = "done"

    logger.info(f"Session {session_id}: {session['completed']} ok, {session['failed']} failed.")
