"""
DisasterVision FastAPI backend.

Endpoints:
  POST  /api/analyze          — upload pre+post image pairs, start batch inference
  GET   /api/status/{sid}     — poll session status and partial results
  GET   /api/results/{sid}    — full results once status == "done"
  GET   /api/report/{sid}     — download situation report as .txt
  GET   /healthz              — health check
"""

import io
import logging
import threading
from pathlib import Path
from typing import List, Optional, Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

from .batch_service import create_session, get_session, run_batch
from .inference import is_mock_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Max upload limits
MAX_FILES      = 50   # max tile pairs per request
MAX_BYTES      = 20 * 1024 * 1024  # 20 MB per file

app = FastAPI(
    title="DisasterVision API",
    description="Building damage assessment from pre/post satellite imagery.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/healthz", tags=["System"])
def health():
    from .inference import is_mock_mode, INFERENCE_VERSION
    return {
        "status": "ok",
        "version": "1.0.2",
        "inference_version": INFERENCE_VERSION,
        "inference_mode": "mock" if is_mock_mode() else "real",
    }


# ── Analysis endpoint ─────────────────────────────────────────────────────────

@app.post("/api/analyze", tags=["Analysis"])
async def analyze(
    background_tasks: BackgroundTasks,
    pre_images: Optional[List[UploadFile]] = File(None),
    post_images: List[UploadFile] = File(...),
    mode: str = Form("dual"),
    lats: Optional[str] = Form(None, description="Comma-separated latitudes (one per pair)"),
    lons: Optional[str] = Form(None, description="Comma-separated longitudes (one per pair)"),
    center_lat: Optional[float] = Form(None, description="Grid centre latitude (used when no coords given)"),
    center_lon: Optional[float] = Form(None, description="Grid centre longitude"),
):
    """
    Accept paired pre/post image uploads and start batch inference.

    Returns a session_id immediately; poll /api/status/{session_id} for progress.
    """
    # ── Validation ──────────────────────────────────────────────────────────
    is_dual = (mode == "dual")
    
    if is_dual and (not pre_images or len(pre_images) != len(post_images)):
        raise HTTPException(
            status_code=400,
            detail=f"In Dual mode, # of pre-images must equal post-images ({len(post_images)}).",
        )
    if not post_images or len(post_images) == 0:
        raise HTTPException(status_code=400, detail="No post-disaster images uploaded.")
    if len(post_images) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} tile pairs per request.")

    # ── Read image bytes eagerly (before background task) ───────────────────
    pairs = []
    for idx, post_f in enumerate(post_images):
        _validate_image_file(post_f)
        post_bytes = await post_f.read()
        _validate_size(post_bytes, post_f.filename)

        pre_bytes = None
        pre_filename = None
        if is_dual and pre_images:
            pre_f = pre_images[idx]
            _validate_image_file(pre_f)
            pre_bytes = await pre_f.read()
            _validate_size(pre_bytes, pre_f.filename)
            pre_filename = pre_f.filename

        pairs.append((pre_bytes, post_bytes, pre_filename, post_f.filename))

    # ── Parse optional coordinates ───────────────────────────────────────────
    coords = None
    if lats and lons:
        try:
            lat_list = [float(v.strip()) for v in lats.split(",") if v.strip()]
            lon_list = [float(v.strip()) for v in lons.split(",") if v.strip()]
            if len(lat_list) == len(pairs) and len(lon_list) == len(pairs):
                coords = list(zip(lat_list, lon_list))
        except ValueError:
            logger.warning("Could not parse coordinate strings; using grid layout.")

    # ── Create session and dispatch background job ───────────────────────────
    session_id = create_session()
    center = (center_lat, center_lon) if center_lat is not None and center_lon is not None else None

    # Run in a daemon thread so the endpoint returns immediately.
    t = threading.Thread(
        target=run_batch,
        args=(session_id, pairs, coords, center),
        daemon=True,
    )
    t.start()

    return {"session_id": session_id, "total_tiles": len(pairs), "status": "processing"}


# ── Status / results endpoints ────────────────────────────────────────────────

@app.get("/api/status/{session_id}", tags=["Analysis"])
def get_status(session_id: str):
    """Returns current processing status and partial tile list."""
    session = _require_session(session_id)
    return {
        "session_id": session_id,
        "status": session["status"],
        "total_tiles": session["total_tiles"],
        "completed": session["completed"],
        "failed": session["failed"],
        "tiles_ready": len(session["tiles"]),
    }


@app.get("/api/results/{session_id}", tags=["Analysis"])
def get_results(session_id: str):
    """Returns full results including tiles, stats, and situation report."""
    session = _require_session(session_id)
    return session


@app.get("/api/report/{session_id}", tags=["Analysis"], response_class=PlainTextResponse)
def download_report(session_id: str):
    """Download the situation report as plain text."""
    session = _require_session(session_id)
    report = session.get("situation_report") or "Report not yet available."
    return PlainTextResponse(
        content=report,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=situation_report_{session_id[:8]}.txt"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_session(session_id: str) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


def _validate_image_file(f: UploadFile):
    allowed = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
    if f.content_type and f.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"File '{f.filename}' has unsupported type '{f.content_type}'. Allowed: JPEG, PNG, TIFF, WEBP.",
        )


def _validate_size(data: bytes, filename: str):
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' exceeds the 20 MB size limit.",
        )
