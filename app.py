"""
Shoren — AI Music Generator
Production Flask Backend v4.0
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from flask_cors import CORS

from music_model import (
    GENRE_DATA,
    analyze_notes,
    generate_music,
    get_genre_info,
    notes_to_labels,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shoren")

_lock    = threading.Lock()
jobs: dict[str, dict]  = {}
history: deque         = deque(maxlen=50)
_rate:   dict[str, list] = {}

GENRE_META = {
    "classical": {"icon": "🎹", "color": "#d4b77f", "instrument": "Acoustic Grand Piano",   "program": 0},
    "jazz":      {"icon": "🎷", "color": "#1de9b6", "instrument": "Jazz Ensemble",           "program": 25},
    "ambient":   {"icon": "🌌", "color": "#8a65ff", "instrument": "Pad (New Age)",           "program": 88},
    "blues":     {"icon": "🎸", "color": "#ff6b6b", "instrument": "Acoustic Guitar (Nylon)", "program": 24},
    "romantic":  {"icon": "🎻", "color": "#ff9f43", "instrument": "String Ensemble",         "program": 40},
}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _job_set(job_id: str, **kw) -> None:
    with _lock:
        if job_id in jobs:
            jobs[job_id].update(kw)

def _rate_ok(ip: str, limit: int = 30, window: int = 60) -> bool:
    now = time.time()
    with _lock:
        ts = _rate.get(ip, [])
        ts = [t for t in ts if now - t < window]
        if len(ts) >= limit:
            _rate[ip] = ts
            return False
        ts.append(now)
        _rate[ip] = ts
    return True

STAGES = [
    (5,  "init",       "Initialising Shoren AI pipeline…"),
    (15, "corpus",     "Loading musical training corpus…"),
    (28, "encode",     "Encoding note sequences…"),
    (45, "train",      "Training neural network on genre data…"),
    (72, "infer",      "Generating sequences via nucleus sampling…"),
    (88, "midi",       "Building expressive MIDI with dynamics…"),
    (96, "finalise",   "Applying velocity curves & saving MIDI…"),
]

def _worker(job_id: str, genre: str, num_notes: int,
            temperature: float, tempo: int, complexity: int) -> None:
    t0 = time.time()
    try:
        for pct, stage, msg in STAGES:
            _job_set(job_id, progress=pct, stage=stage, message=msg)
            time.sleep(0.3 + complexity * 0.08)

        filename  = f"shoren_{job_id}.mid"
        out_path  = os.path.join(AUDIO_DIR, filename)

        notes, _ = generate_music(
            genre=genre,
            num_notes=num_notes,
            temperature=temperature,
            tempo=tempo,
            output_path=out_path,
            complexity=complexity,
        )

        harmonic    = analyze_notes(notes)
        note_labels = notes_to_labels(notes[:80])
        meta        = GENRE_META.get(genre, GENRE_META["classical"])
        elapsed     = round(time.time() - t0, 2)
        size_kb     = round(os.path.getsize(out_path) / 1024, 1)

        _job_set(
            job_id,
            progress     = 100,
            stage        = "done",
            message      = "Composition complete! Your MIDI file is ready.",
            filename     = filename,
            download_url = f"/download/{filename}",
            notes        = notes[:80],
            note_labels  = note_labels,
            note_count   = len(notes),
            harmonic     = harmonic,
            instrument   = meta["instrument"],
            icon         = meta["icon"],
            color        = meta["color"],
            elapsed_sec  = elapsed,
            size_kb      = size_kb,
            completed_at = _now(),
        )

        summary = {k: jobs[job_id][k] for k in
                   ("job_id","genre","note_count","tempo","temperature",
                    "complexity","filename","download_url",
                    "elapsed_sec","completed_at","size_kb") if k in jobs[job_id]}
        with _lock:
            history.appendleft(summary)

        log.info(f"✓ {job_id} | {genre} | {len(notes)} notes | {elapsed}s → {filename}")

    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        log.exception(f"✗ {job_id} failed after {elapsed}s")
        _job_set(job_id,
                 stage="error", progress=0,
                 message=f"Generation failed: {exc}",
                 elapsed_sec=elapsed)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    ip = request.remote_addr or "unknown"
    if not _rate_ok(ip):
        return jsonify({"error": "Rate limit exceeded — please wait 60 s"}), 429

    body        = request.get_json(force=True, silent=True) or {}
    genre       = str(body.get("genre", "classical"))
    if genre not in GENRE_DATA:
        return jsonify({"error": f"Unknown genre '{genre}'"}), 400

    num_notes   = max(16,  min(256, int(body.get("num_notes",   64))))
    temperature = max(0.3, min(1.8, float(body.get("temperature", 0.8))))
    tempo       = max(40,  min(220, int(body.get("tempo",        120))))
    complexity  = max(1,   min(5,   int(body.get("complexity",     3))))

    job_id = uuid.uuid4().hex[:10]
    with _lock:
        jobs[job_id] = {
            "job_id":     job_id,
            "stage":      "queued",
            "progress":   0,
            "message":    "Queued for generation…",
            "genre":      genre,
            "num_notes":  num_notes,
            "temperature":temperature,
            "tempo":      tempo,
            "complexity": complexity,
            "started_at": _now(),
        }

    threading.Thread(
        target=_worker,
        args=(job_id, genre, num_notes, temperature, tempo, complexity),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route("/status/<job_id>")
def status(job_id: str):
    with _lock:
        job = dict(jobs.get(job_id, {}))
    if not job:
        return jsonify({"stage": "not_found"}), 404
    return jsonify(job)

@app.route("/stream/<job_id>")
def stream(job_id: str):
    def _gen():
        last_pct = -1
        deadline = time.time() + 600
        while time.time() < deadline:
            with _lock:
                job = dict(jobs.get(job_id, {}))
            if not job:
                yield f"data: {json.dumps({'stage':'not_found'})}\n\n"
                break
            pct = job.get("progress", 0)
            if pct != last_pct:
                yield f"data: {json.dumps(job)}\n\n"
                last_pct = pct
            if job.get("stage") in ("done", "error"):
                break
            time.sleep(0.25)

    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )

@app.route("/download/<filename>")
def download(filename: str):
    filename = os.path.basename(filename)
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True,
                     download_name=filename, mimetype="audio/midi")

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "healthy",
        "service": "Shoren — AI Music Generator",
        "version": "4.0",
        "timestamp": _now(),
        "audio_files": len([f for f in os.listdir(AUDIO_DIR) if f.endswith(".mid")]),
        "active_jobs": sum(1 for j in jobs.values() if j.get("stage") not in ("done","error","queued")),
    })

@app.route("/api/genres")
def api_genres():
    result = {}
    for g, data in GENRE_DATA.items():
        result[g] = {
            "description": data["description"],
            "tempo_range":  data["tempo_range"],
            "scale":        data["scale"],
            "mood":         data["mood"],
            **GENRE_META.get(g, {}),
        }
    return jsonify({"genres": result})

@app.route("/api/history")
def api_history():
    limit = min(50, max(1, int(request.args.get("limit", 20))))
    with _lock:
        items = list(history)[:limit]
    return jsonify({"history": items, "count": len(items)})

@app.route("/api/stats")
def api_stats():
    with _lock:
        total  = len(jobs)
        done   = sum(1 for j in jobs.values() if j.get("stage") == "done")
        active = sum(1 for j in jobs.values()
                     if j.get("stage") not in ("done","error","queued"))
        errors = sum(1 for j in jobs.values() if j.get("stage") == "error")
    return jsonify({"total":total,"done":done,"active":active,"errors":errors,
                    "history_count": len(history)})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body  = request.get_json(force=True, silent=True) or {}
    notes = body.get("notes", [])
    if not isinstance(notes, list) or not notes:
        return jsonify({"error": "Provide non-empty 'notes' list"}), 400
    notes    = [int(n) for n in notes if isinstance(n, (int, float))]
    analysis = analyze_notes(notes)
    labels   = notes_to_labels(notes[:80])
    return jsonify({"analysis": analysis, "labels": labels})

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    log.exception("500")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║  🎵  Shoren — AI Music Generator                    ║
║  ─────────────────────────────────────────────────  ║
║  →  http://localhost:5000                           ║
║  →  http://localhost:5000/api/health                ║
╚══════════════════════════════════════════════════════╝
""")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
