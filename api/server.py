"""
Sports Analytics API Server

Endpoints:
  POST /jobs                   Upload video + config JSON → start processing job
  GET  /jobs/{job_id}          Poll status, progress (0-1), current step
  GET  /jobs/{job_id}/video    Stream the annotated MP4
  GET  /jobs/{job_id}/metrics  Player metrics JSON
  GET  /jobs/{job_id}/events   Match events JSON
  GET  /jobs                   List all jobs (newest first)
  DELETE /jobs/{job_id}        Cancel or remove a job

Run with:
  uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import shutil
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Ensure project root is on sys.path so MatchProcessor can be imported from a thread
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sports Analytics API",
    version="1.0.0",
    description="Upload match video + config, get annotated video + player metrics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Job store (in-memory; replace with Redis/SQLite for multi-process deployments)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_or_404(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


def _set(job_id: str, **kwargs):
    """Thread-safe partial update of job record."""
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_job(job_id: str, config_path: str):
    """
    Runs MatchProcessor in a background thread.
    Updates job dict with progress, then final results or error.
    """
    _set(job_id, status="processing", started_at=datetime.utcnow().isoformat())

    def progress_callback(step: str, fraction: float):
        _set(job_id, step=step, progress=round(fraction, 3))

    try:
        # Import here so each thread gets a fresh import state
        from main_v3 import MatchProcessor

        processor = MatchProcessor(config_path)
        results = processor.process(progress_callback=progress_callback)

        _set(
            job_id,
            status="done",
            progress=1.0,
            step="Complete",
            finished_at=datetime.utcnow().isoformat(),
            output_video=results["output_video"],
            metrics=results["metrics"],
            events=results["events"],
        )

    except Exception as exc:
        _set(
            job_id,
            status="failed",
            finished_at=datetime.utcnow().isoformat(),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(..., description="Match video file (MP4 / AVI / MKV)"),
    config: str = Form(
        ...,
        description=(
            "Match config JSON string. "
            "At minimum: {sport, match_id, competition, team_home, team_away, "
            "processing_options}. video_path is ignored — the uploaded file is used."
        ),
    ),
):
    """
    Upload a video file + match config JSON to start a new analysis job.

    Returns a job_id you can use to poll status and fetch results.

    **Example curl:**
    ```
    curl -X POST http://localhost:8000/jobs \\
      -F "video=@input_videos/match.mp4" \\
      -F 'config={"sport":"soccer","match_id":"test001","competition":"Test",
                  "team_home":{"id":"t1","name":"Home","color_primary":"red",
                               "color_secondary":"white","players":[]},
                  "team_away":{"id":"t2","name":"Away","color_primary":"blue",
                               "color_secondary":"white","players":[]},
                  "processing_options":{"use_gpu":false,"enable_jersey_ocr":false,
                                        "enable_radar":false}}'
    ```
    """
    # Parse + validate config
    try:
        config_data = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config JSON: {exc}")

    required = ["sport", "match_id", "competition", "team_home", "team_away"]
    missing = [k for k in required if k not in config_data]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Config missing required keys: {missing}"
        )

    job_id = str(uuid.uuid4())
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    safe_name = Path(video.filename).name if video.filename else "video.mp4"
    video_path = job_dir / safe_name
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Point config at the saved video and use job_id as match_id for output naming
    config_data["video_path"] = str(video_path)
    config_data["match_id"] = job_id

    config_path = job_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # Register job
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "step": "Queued",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "video_filename": safe_name,
            "output_video": None,
            "metrics": None,
            "events": None,
            "error": None,
        }

    # Launch background thread (daemon so it doesn't block server shutdown)
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, str(config_path)),
        name=f"job-{job_id[:8]}",
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs")
def list_jobs():
    """Return all jobs, newest first."""
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    # Strip large internal fields from list view
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "progress": j["progress"],
            "step": j["step"],
            "created_at": j["created_at"],
            "finished_at": j["finished_at"],
            "video_filename": j["video_filename"],
        }
        for j in jobs
    ]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """
    Poll job status.

    Returns:
        status:   queued | processing | done | failed
        progress: 0.0 – 1.0
        step:     current pipeline step name
        error:    error message if status == failed
    """
    job = _job_or_404(job_id)
    # Don't expose internal paths or full traceback in normal response
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "step": job["step"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "video_filename": job["video_filename"],
        "error": job.get("error"),
    }


@app.get("/jobs/{job_id}/video")
def get_job_video(job_id: str):
    """
    Stream the annotated output MP4.
    Only available when status == done.
    """
    job = _job_or_404(job_id)

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Video not ready — job status is '{job['status']}' "
                   f"(progress: {job['progress']:.0%})",
        )

    video_path = job.get("output_video")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Output video file not found on disk")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=Path(video_path).name,
    )


@app.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str):
    """
    Return player metrics JSON.
    Only available when status == done.
    """
    job = _job_or_404(job_id)

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Metrics not ready — job status is '{job['status']}'",
        )

    metrics_path = job.get("metrics")
    if not metrics_path or not Path(metrics_path).exists():
        raise HTTPException(status_code=404, detail="Metrics file not found on disk")

    with open(metrics_path) as f:
        return json.load(f)


@app.get("/jobs/{job_id}/events")
def get_job_events(job_id: str):
    """
    Return match events (passes, shots, possession changes) JSON.
    Only available when status == done.
    """
    job = _job_or_404(job_id)

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Events not ready — job status is '{job['status']}'",
        )

    events_path = job.get("events")
    if not events_path or not Path(events_path).exists():
        raise HTTPException(status_code=404, detail="Events file not found on disk")

    with open(events_path) as f:
        return json.load(f)


@app.delete("/jobs/{job_id}", status_code=200)
def delete_job(job_id: str):
    """
    Remove a job from the store and delete its upload directory.
    Processing jobs that are still running will finish but results won't be kept.
    """
    _job_or_404(job_id)

    with _jobs_lock:
        _jobs.pop(job_id, None)

    job_dir = UPLOADS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"deleted": job_id}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    with _jobs_lock:
        counts = {}
        for j in _jobs.values():
            counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {"status": "ok", "jobs": counts}
