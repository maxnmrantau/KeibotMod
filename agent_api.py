"""Agent API untuk KeibotMod — akses API-key (X-API-Key) tanpa login PIN web.

Dipakai Max (agen) untuk kontrol penuh KeibotMod dari luar panel web.
Panel web + login PIN tetap berfungsi seperti biasa; modul ini hanya
menambah endpoint /api/agent/* yang diamankan API key dari env
KEIBOT_AGENT_KEY (di-load systemd via EnvironmentFile).

Dependensi (state aplikasi) diinjeksikan dari app.py lewat
init_agent_api(...) — modul ini TIDAK mengimpor app, karena app.py
jalan sebagai __main__ dan impor ulang akan menduplikasi state
(antrean, task, worker) menjadi dua salinan.
"""

import os
import re
import secrets
import shutil
import time
from flask import Blueprint, jsonify, request

agent_api = Blueprint("agent_api", __name__)

_STARTED = time.time()
_DEPS = {}


def init_agent_api(deps):
    """Terima dependensi dari app.py (fungsi + akses state). Dipanggil
    sekali saat registrasi blueprint, sebelum server menerima request."""
    _DEPS.update(deps)


def _dep(name):
    fn = _DEPS.get(name)
    if fn is None:
        raise RuntimeError(f"Agent API belum diinisialisasi: {name}")
    return fn


# ── Auth: X-API-Key ──────────────────────────────────────────────
@agent_api.before_request
def _require_agent_key():
    expected = os.environ.get("KEIBOT_AGENT_KEY", "")
    if not expected:
        return jsonify({"status": "error",
                        "error": "Agent API belum dikonfigurasi (KEIBOT_AGENT_KEY kosong)"}), 503
    provided = request.headers.get("X-API-Key", "")
    if not provided or not secrets.compare_digest(provided, expected):
        return jsonify({"status": "error",
                        "error": "API key salah atau tidak ada"}), 401


# ── Helpers ──────────────────────────────────────────────────────
def _strip_html(text):
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


def _video_url(status):
    m = re.search(r"https?://youtu\.be/[A-Za-z0-9_-]+", str(status or ""))
    return m.group(0) if m else None


def _progress(status, finished=False):
    """Perkiraan progres 0-100 dari string status worker. None = status
    akhir tanpa progres (gagal/dibatalkan)."""
    s = str(status or "")
    if finished:
        if s.startswith("Tayang") or s.startswith("Render Selesai"):
            return 100
        return None
    if s.startswith("Gagal Upload, Antre Ulang"):
        return 0
    if "Gagal" in s or "Dibatalkan" in s:
        return None
    if s.startswith("In Factory Queue"):
        return 0
    if s.startswith("Antrean Ditunda") or s.startswith("Menunggu RAM"):
        return 0
    if s.startswith("Meracik Aset"):
        return 10
    if s.startswith("Rendering Visual & Background"):
        return 40
    if s.startswith("Rendering Visual"):
        return 50
    if s.startswith("Smart Cut"):
        return 60
    if s.startswith("Transisi"):
        return 65
    if s.startswith("Auto-Looping"):
        return 70
    if s.startswith("Render Selesai"):
        return 90
    if s.startswith("Mengunggah"):
        m = re.search(r"(\d+)%", s)
        pct = int(m.group(1)) if m else 0
        return 90 + pct // 10
    if s.startswith("Memasang Thumbnail"):
        return 98
    if "Retry" in s or "Koneksi" in s:
        return 85
    return None


def _task_view(t, finished=False):
    st = t.get("status", "")
    view = {
        "id": t.get("id"),
        "name": t.get("title"),
        "title": t.get("title"),
        "status": _strip_html(st),
        "status_raw": st,
        "progress": _progress(st, finished=finished),
        "type": t.get("type"),
        "publish_date": t.get("time"),
        "yt_id": (t.get("blueprint") or {}).get("yt_id"),
        "duration": t.get("duration"),
        "finish_time": t.get("finish_time"),
    }
    url = _video_url(st)
    if url:
        view["video_url"] = url
    return view


# ── Endpoints ────────────────────────────────────────────────────
@agent_api.route("/api/agent/health", methods=["GET"])
def agent_health():
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    ffmpeg_ok = bool(ffmpeg and os.path.exists(ffmpeg))
    return jsonify({
        "status": "ok",
        "service": "keibot",
        "ffmpeg": ffmpeg_ok,
        "ffmpeg_path": ffmpeg if ffmpeg_ok else None,
        "queue_size": _dep("get_queue_size")(),
        "uptime_seconds": int(time.time() - _STARTED),
    })


@agent_api.route("/api/agent/status", methods=["GET"])
def agent_status():
    active = _dep("get_active_tasks")()
    history = _dep("get_history_tasks")()
    channels = _dep("get_channels")()
    queued = sum(1 for t in active
                 if str(t.get("status", "")).startswith("In Factory Queue"))
    uploaded = sum(1 for t in history
                   if str(t.get("status", "")).startswith("Tayang"))
    failed = sum(1 for t in history if "Gagal" in str(t.get("status", "")))
    yt_connected = any("Connected" in str(c.get("status", ""))
                       for c in channels)
    return jsonify({
        "status": "ok",
        "counts": {
            "batches_active": len(active),
            "queued": queued,
            "rendering_or_uploading": len(active) - queued,
            "queue_size": _dep("get_queue_size")(),
            "uploaded": uploaded,
            "failed": failed,
            "history_total": len(history),
            "channels": len(channels),
        },
        "youtube_connected": yt_connected,
        "channels": [{"id": c.get("id"), "name": c.get("name"),
                      "yt_id": c.get("yt_id"), "status": c.get("status")}
                     for c in channels],
        "recent_jobs": [_task_view(t, finished=True) for t in history[:5]],
    })


@agent_api.route("/api/agent/batches", methods=["GET"])
def agent_batches():
    include_history = request.args.get("include_history", "0") == "1"
    active = _dep("get_active_tasks")()
    out = [_task_view(t) for t in active]
    if include_history:
        out += [_task_view(t, finished=True)
                for t in _dep("get_history_tasks")()]
    return jsonify({"status": "ok", "count": len(out), "batches": out})


@agent_api.route("/api/agent/jobs", methods=["GET"])
def agent_jobs():
    try:
        limit = int(request.args.get("limit", "20"))
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 50))
    jobs = [_task_view(t) for t in _dep("get_active_tasks")()]
    jobs += [_task_view(t, finished=True)
             for t in _dep("get_history_tasks")()]
    return jsonify({"status": "ok", "count": len(jobs[:limit]),
                    "jobs": jobs[:limit]})


@agent_api.route("/api/agent/batch", methods=["POST"])
def agent_batch():
    data = request.get_json(silent=True) or {}
    payload, code = _dep("create_batch_tasks")(data)
    return jsonify(payload), code


@agent_api.route("/api/agent/upload", methods=["POST"])
def agent_upload():
    data = request.get_json(silent=True) or {}
    batch_id = data.get("batch_id", data.get("id"))
    if batch_id is None:
        return jsonify({"status": "error",
                        "error": "batch_id wajib diisi"}), 400
    try:
        batch_id = int(batch_id)
    except (TypeError, ValueError):
        return jsonify({"status": "error",
                        "error": "batch_id harus angka (id task)"}), 400
    payload, code = _dep("requeue_task")(batch_id)
    return jsonify(payload), code
