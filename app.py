import os, time, queue, threading, subprocess, random, json, shutil, math, gc
import sys

# Ensure UTF-8 encoding on console output for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import cv2, librosa, imageio
import datetime as dt
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import secrets
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session
import requests

# 🎨 PIL untuk font TTF (agar teks render mirip preview browser).
# Fallback aman jika PIL tidak terinstall.
try:
    from PIL import ImageFont, Image as PILImage, ImageDraw as PILDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ── Resolver path font TTF cross-platform ──
# Prioritas: Linux DejaVu (mirip Arial) → Windows Arial → DejaVuSans.
def _resolve_ttf(font_key='M'):
    """Kembalikan path file .ttf yang cocok untuk key font (M/S/I/C).
    Mengembalikan None bila tidak ada font TTF yang tersedia (caller fallback ke cv2)."""
    key = str(font_key).upper() if font_key else 'M'
    # kandidat per OS (Linux VPS & Windows dev)
    linux_bold = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ]
    linux_reg = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]
    win_bold = ['C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/segoeuib.ttf']
    win_reg  = ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf']
    bold = linux_bold + win_bold
    reg  = linux_reg + win_reg
    # 'I' (Impact) & 'C' (Condensed) → pakai bold untuk nuansa tebal
    pool = bold if key in ('I', 'C', 'M') else reg
    for p in pool:
        if os.path.exists(p):
            return p
    return None

def _put_text_pil(frame, text, org, font_ttf, size, color_bgr, thickness=1):
    """Gambar teks TTF ke frame OpenCV (BGR) via PIL (ROI-based, cepat).
    org = (x, y) = BASELINE kiri-bawah (kompatibel dengan konvensi cv2.putText),
    lalu dikonversi ke top-left untuk PIL. Mengembalikan True bila berhasil."""
    if not _HAS_PIL or font_ttf is None or size <= 0:
        return False
    if frame is None or frame.size == 0:
        return False
    if not text or not str(text).strip():
        return False
    try:
        pil_size = max(8, int(round(size)))
        font = _get_pil_font(font_ttf, pil_size)
        if font is None:
            return False
        # konversi org baseline → top-left PIL (PIL pakai anchor 'la' = left-ascender)
        asc, desc = font.getmetrics()
        # ukur teks untuk menentukan ROI
        bbox = font.getbbox(str(text))
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox = int(org[0]) - bbox[0]
        # oy_top = posisi baseline - ascent
        oy_top = int(org[1]) - asc
        oy_bot = oy_top + th
        fh, fw = frame.shape[:2]
        # padding untuk stroke
        pad = max(thickness + 2, 4)
        rx0 = max(0, ox - pad)
        ry0 = max(0, oy_top - pad)
        rx1 = min(fw, ox + tw + pad)
        ry1 = min(fh, oy_bot + pad)
        if rx1 <= rx0 or ry1 <= ry0:
            return False
        roi = frame[ry0:ry1, rx0:rx1]
        # pastikan array contiguous (cv2.cvtColor butuh C-contiguous)
        if not roi.flags['C_CONTIGUOUS']:
            roi = np.ascontiguousarray(roi)
        img = PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = PILDraw.Draw(img)
        rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        lx, ly = ox - rx0, oy_top - ry0
        if thickness >= 2:
            stroke_w = max(1, int(thickness))
            draw.text((lx, ly), str(text), font=font, fill=rgb,
                      stroke_width=stroke_w, stroke_fill=rgb)
        else:
            draw.text((lx, ly), str(text), font=font, fill=rgb)
        frame[ry0:ry1, rx0:rx1] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return True
    except Exception:
        return False

# ── Cache font PIL agar tidak reload setiap frame ──
_pil_font_cache = {}

def _get_pil_font(font_ttf, size):
    """Return cached PIL ImageFont, atau None."""
    if not _HAS_PIL or font_ttf is None or size <= 0:
        return None
    key = (font_ttf, max(8, int(round(size))))
    if key not in _pil_font_cache:
        try:
            _pil_font_cache[key] = ImageFont.truetype(font_ttf, key[1])
        except Exception:
            return None
    return _pil_font_cache[key]

def _draw_text(frame, text, org, cv_font, fontScale, color, thickness, lineType,
               ttf_path=None, pil_size=None):
    """Gambar teks: coba PIL (TTF) dulu, fallback ke cv2 Hershey font.
    ttf_path + pil_size → gunakan PIL. Jika PIL gagal → cv2.putText fallback."""
    if ttf_path and pil_size and pil_size > 0:
        if _put_text_pil(frame, text, org, ttf_path, pil_size, color, thickness):
            return
    # fallback: cv2 Hershey
    cv2.putText(frame, text, org, cv_font, fontScale, color, thickness, lineType)

def _get_text_size(text, ttf_path, pil_size, cv_font, cv_fontScale, cv_thickness):
    """Return (width, height) teks, pakai TTF bila PIL tersedia.
    Ukuran penting untuk posisi watermark/timestamp (tr, br, center)."""
    if ttf_path and pil_size and pil_size > 0 and _HAS_PIL:
        font = _get_pil_font(ttf_path, pil_size)
        if font is not None:
            try:
                bbox = font.getbbox(str(text))
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                asc, desc = font.getmetrics()
                return (tw, asc)  # asc ≈ tinggi baseline untuk posisi
            except Exception:
                pass
    # fallback: cv2 Hershey
    (w, h), _ = cv2.getTextSize(str(text), cv_font, cv_fontScale, cv_thickness)
    return (w, h)
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build

# ==========================================
# 🛡️ SETUP & MONITORING
# ==========================================
def auto_setup_dependencies():
    ffmpeg_found = shutil.which("ffmpeg") or os.path.exists("/usr/bin/ffmpeg")
    if not ffmpeg_found:
        print("⚙️ KEIBOT: ffmpeg tidak ditemukan, mencoba install otomatis...")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True, capture_output=True, timeout=60)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True, capture_output=True, timeout=120)
            print("✅ ffmpeg berhasil diinstall!")
        except Exception as e:
            print(f"❌ Gagal install ffmpeg otomatis: {e}")
            print("👉 Jalankan manual: apt-get install -y ffmpeg")
    else:
        path = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        print(f"✅ ffmpeg ditemukan: {path}")

auto_setup_dependencies()

last_cpu_idle = 0
last_cpu_total = 0

def get_system_stats():
    global last_cpu_idle, last_cpu_total
    cpu_pct = 0.0
    try:
        with open('/proc/stat', 'r') as f:
            parts = [int(i) for i in f.readline().split()[1:8]]
        idle = parts[3] + parts[4]
        total = sum(parts)
        if last_cpu_total > 0:
            diff_idle = idle - last_cpu_idle
            diff_total = total - last_cpu_total
            if diff_total > 0:
                cpu_pct = round(100.0 * (1.0 - diff_idle / diff_total), 1)
        last_cpu_idle = idle
        last_cpu_total = total
        if cpu_pct < 0.0: cpu_pct = 0.0
        if cpu_pct > 100.0: cpu_pct = 100.0
    except: pass

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return {"cpu": cpu_pct, "ram_pct": mem.percent, "ram_used": round(mem.used / (1024**3), 2), "ram_total": round(mem.total / (1024**3), 2)}
    except: pass

    return {"cpu": cpu_pct, "ram_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0}

# ==========================================
# 💾 DATABASE & FOLDER SYSTEM
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')

def is_configured(): return os.path.exists(CONFIG_FILE)
def load_bot_config():
    if is_configured():
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return {}

bot_config = load_bot_config()
app.secret_key = bot_config.get('secret_key', secrets.token_hex(24))

@app.before_request
def check_security():
    allowed_routes = ['login', 'setup', 'static', 'serve_uploads', 'device_login', 'poll_device_token']
    if request.endpoint in allowed_routes: return
    if not is_configured(): return redirect(url_for('setup'))
    if 'logged_in' not in session: return redirect(url_for('login'))

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if is_configured(): return redirect(url_for('login'))
    error = None
    if request.method == 'POST':
        pin = request.form.get('new_pin'); pin2 = request.form.get('confirm_pin')
        if not pin or len(pin) < 3: error = "PIN minimal 3 karakter."
        elif pin != pin2: error = "PIN tidak cocok!"
        else:
            new_secret = secrets.token_hex(24)
            with open(CONFIG_FILE, 'w') as f: json.dump({"admin_pin": pin, "secret_key": new_secret}, f, indent=4)
            app.secret_key = new_secret; session['logged_in'] = True
            return redirect(url_for('index'))
    return render_template('setup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not is_configured(): return redirect(url_for('setup'))
    error = None
    if request.method == 'POST':
        if request.form.get('password') == load_bot_config().get('admin_pin'):
            session['logged_in'] = True; return redirect(url_for('index'))
        else: error = 'Akses Ditolak! PIN Salah.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None); return redirect(url_for('login'))

BASE_UPLOAD = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "keibot-output") if os.name == 'nt' else '/root/keibot-output'
DB_FILE = os.path.join(BASE_DIR, 'channels_db.json')
TASKS_FILE = os.path.join(BASE_DIR, 'tasks_db.json')
PRESETS_FILE = os.path.join(BASE_DIR, 'presets.json')
METADATA_PRESETS_FILE = os.path.join(BASE_DIR, 'metadata_presets.json')
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, 'client_secret.json')
SCOPES = ['https://www.googleapis.com/auth/youtube', 'https://www.googleapis.com/auth/youtube.upload']

os.makedirs(BASE_UPLOAD, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)

db_lock = threading.Lock()

GALLERY_FOLDER_MAP = {
    'audio':      'audios',
    'audios':     'audios',
    'background': 'backgrounds',
    'backgrounds':'backgrounds',
    'thumbnail':  'thumbnails',
    'thumbnails': 'thumbnails'
}

def resolve_folder(g_type: str) -> str:
    return GALLERY_FOLDER_MAP.get(str(g_type).strip().lower(), 'audios')

def safe_float(val, default=0.0):
    """Konversi string ke float dengan aman tanpa crash jika berisi teks seperti 'original'"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def load_tasks_db():
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, 'r') as f: return json.load(f)
        except: return {"active": [], "history": []}
    return {"active": [], "history": []}

def save_tasks_db():
    with db_lock:
        data = {"active": active_tasks, "history": history_tasks}
        with open(TASKS_FILE, 'w') as f: json.dump(data, f, indent=4)

def load_channels():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f: return json.load(f)
        except: return []
    return []

def save_channels(channels):
    with db_lock:
        with open(DB_FILE, 'w') as f: json.dump(channels, f, indent=4)

task_data = load_tasks_db()
active_tasks = task_data.get("active", [])
history_tasks = task_data.get("history", [])
database_channel = load_channels()

render_queue = queue.Queue()
stop_flags = {}
channel_cooldowns = {}

# 🎵 Cache audio info (durasi) — expire setelah 5 menit
_audio_info_cache = {}
AUDIO_CACHE_TTL = 300

# 🔥 SISTEM NOTIFIKASI LONCENG 🔥
system_notifications = []

def get_ffmpeg_path():
    local_exe = os.path.join(BASE_DIR, "ffmpeg.exe")
    if os.path.exists(local_exe): return local_exe
    found = shutil.which("ffmpeg")
    if found: return found
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"]:
        if os.path.exists(p): return p
    raise FileNotFoundError("ffmpeg tidak ditemukan! Jalankan: apt-get install -y ffmpeg")

def get_ffprobe_path():
    found = shutil.which("ffprobe")
    if found: return found
    for p in ["/usr/bin/ffprobe", "/usr/local/bin/ffprobe", "/bin/ffprobe"]:
        if os.path.exists(p): return p
    return "ffprobe"

def wait_for_resources(task_id, max_ram_pct=85.0):
    while True:
        if stop_flags.get(task_id): return False
        stats = get_system_stats()
        if stats['ram_pct'] < max_ram_pct: return True
        with db_lock:
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = f"Menunggu RAM Turun ({stats['ram_pct']}%) ⏳"
        save_tasks_db()
        time.sleep(10)

def move_to_history(task_id, final_status):
    global active_tasks, history_tasks
    now = time.time()
    with db_lock:
        for t in active_tasks:
            if t['id'] == task_id:
                start_ts = t.get('start_ts')
                if start_ts:
                    el = max(0, int(now - start_ts))
                    t['duration'] = f"{el // 60}m {el % 60}s" if el >= 60 else f"{el}s"
                else:
                    t['duration'] = "—"
                t['finish_time'] = dt.datetime.now().strftime('%H:%M:%S')
                t['status'] = final_status
                t.pop('blueprint', None)
                history_tasks.insert(0, t)
                active_tasks.remove(t)
                if len(history_tasks) > 50: history_tasks.pop()
                break
    save_tasks_db()

def restore_and_resume_queue():
    """Memulihkan antrean yang belum selesai jika server restart/reboot/crash"""
    global active_tasks, history_tasks
    resumed = 0
    with db_lock:
        for t in active_tasks:
            bp = t.get('blueprint')
            if bp:
                # Jika server restart saat task sedang dalam proses render/upload, kembalikan ke antrean
                st = t.get('status', '')
                if st != "In Factory Queue ⚙️":
                    t['status'] = "In Factory Queue ⚙️"
                    t.pop('start_ts', None)
                    t.pop('render_start', None)
                render_queue.put(bp)
                resumed += 1
            else:
                # Task lama yang tidak punya blueprint saat server restart
                if t.get('status') == "In Factory Queue ⚙️" or "Rendering" in t.get('status', ''):
                    t['status'] = "Dibatalkan (Server Restart) ⚠️"
                    history_tasks.insert(0, t)

        active_tasks = [t for t in active_tasks if "Dibatalkan" not in t.get('status', '')]
    save_tasks_db()
    if resumed > 0:
        print(f"[KeiBot] 🔄 Berhasil memulihkan {resumed} task ke antrean setelah restart!")

def get_fresh_credentials(channel_data):
    creds_str = channel_data.get('creds_list', [channel_data.get('creds_json')])[0]
    creds = Credentials.from_authorized_user_info(json.loads(creds_str))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

# ==========================================
# 🚨 SATPAM API KEY (AUTO-CHECKER)
# ==========================================
def api_key_checker_worker():
    global system_notifications, database_channel
    while True:
        time.sleep(10) 
        new_notifs = []
        for c in database_channel:
            creds_list = c.get('creds_list', [c.get('creds_json', '')])
            for idx, cred_str in enumerate(creds_list):
                if not cred_str: continue
                try:
                    creds = Credentials.from_authorized_user_info(json.loads(cred_str))
                    if creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                except Exception as e:
                    msg = f"⚠️ API Key #{idx+1} untuk Channel '{c.get('name','Unknown')}' EXPIRED! Silakan hapus dan tautkan ulang."
                    if not any(n['msg'] == msg for n in system_notifications):
                        new_notifs.append({"msg": msg, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
        
        if new_notifs:
            with db_lock:
                system_notifications.extend(new_notifs)
                
        time.sleep(43200)

threading.Thread(target=api_key_checker_worker, daemon=True).start()

# ==========================================
# 🏭 GALLERY & ASSET MANAGER
# ==========================================
def get_channel_folder(yt_id, sub):
    path = os.path.join(BASE_UPLOAD, yt_id, sub)
    os.makedirs(path, exist_ok=True)
    return path

def get_multi_backgrounds(yt_id, count=1):
    path = get_channel_folder(yt_id, "backgrounds")
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.mp4', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.mov'))]
    if not files: return []
    random.shuffle(files)
    
    selected = []
    while len(selected) < count and files:
        for f in files:
            selected.append(f)
            if len(selected) == count: break
    return selected

def get_all_audios(yt_id):
    path = get_channel_folder(yt_id, "audios")
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.mp3', '.wav'))]
    random.shuffle(files)
    return files

def get_and_consume_thumbnail(yt_id):
    path = get_channel_folder(yt_id, "thumbnails")
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not files: return None
    return os.path.join(path, files[0])

def get_random_preset(allowed_names=None):
    if not os.path.exists(PRESETS_FILE): return None
    try:
        with open(PRESETS_FILE, 'r') as f: presets = json.load(f)
        if not presets: return None
        if allowed_names:
            filtered = {k: v for k, v in presets.items() if k in allowed_names}
            if filtered: return random.choice(list(filtered.values()))
        return random.choice(list(presets.values()))
    except: return None

def get_smart_preset(audio_path):
    """Menganalisis audio dan memilih preset yang cocok secara cerdas"""
    preset = None
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
        # deteksi BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # energi rata-rata
        rms = np.sqrt(np.mean(y**2))
        energy = min(1.0, rms * 5)
        # spectral centroid (brightness)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        brightness = min(1.0, cent / 3000)

        if tempo < 80:
            # slow / chill
            preset = {
                "effect_type": random.choice(["waveform", "mirror", "smooth_blob", "sinusoidal"]),
                "particle_type": random.choice(["golden_dust", "embers", "twinkle", "petals", "smoke", "snow"]),
                "bar_style": "center",
                "reactivity": round(random.uniform(0.4, 0.7), 2),
                "gravity": round(random.uniform(0.04, 0.10), 2),
            }
        elif tempo < 120:
            # medium
            preset = {
                "effect_type": random.choice(["spectrum", "circular", "dots_pixel", "filled_wave"]),
                "particle_type": random.choice(["golden_dust", "embers", "sparkle", "bokeh", "bubbles"]),
                "bar_style": "bottom",
                "reactivity": round(random.uniform(0.6, 0.9), 2),
                "gravity": round(random.uniform(0.06, 0.14), 2),
            }
        else:
            # fast / energetic
            energy_boost = min(1.0, energy * 1.3)
            preset = {
                "effect_type": random.choice(["sunburst", "neon_glow", "halftone", "pixel"]),
                "particle_type": random.choice(["drift_sparks", "fireworks", "trail", "twinkle", "sparkle"]),
                "bar_style": "bottom",
                "reactivity": round(random.uniform(0.8, 1.5), 2),
                "gravity": round(random.uniform(0.10, 0.20), 2),
            }

        if preset:
            # warna berdasarkan brightness
            if brightness > 0.6:
                preset["color_bot"] = random.choice(["#ff6b6b", "#f093fb", "#4facfe", "#fa709a"])
                preset["color_top"] = random.choice(["#00f2fe", "#4facfe", "#f093fb", "#fa709a"])
            else:
                preset["color_bot"] = random.choice(["#10b981", "#00d4ff", "#7c5cfc", "#06d6a0"])
                preset["color_top"] = random.choice(["#00e5ff", "#7c5cfc", "#10b981", "#7209b7"])

            preset["color_part"] = "#ffffff"
            preset["pos_x"] = 50
            preset["pos_y"] = 85
            preset["width_pct"] = 60
            preset["max_height"] = 40
            preset["idle_height"] = 5
            preset["bar_count"] = random.choice([48, 64, 80])
            preset["spacing"] = random.choice([2, 3, 4])
            preset["part_amount"] = random.choice([3, 5, 8])
            preset["part_speed"] = round(random.uniform(0.5, 1.5), 1)
            preset["smoothing"] = 0.90
            preset["use_beat_pulse"] = random.choice([True, False])
            preset["fade_duration"] = 0
            preset["use_watermark"] = False
            preset["wm_text"] = ""
            preset["wm_color"] = "#ffffff"
            preset["wm_font"] = "M"
            preset["wm_size"] = 24
            preset["wm_position"] = "bl"
            preset["wm_move"] = "none"
            preset["use_tracklist"] = False
            preset["tl_font"] = "M"
            preset["tl_size"] = "medium"
            preset["tl_position"] = "tr"
            preset["tl_bg"] = "dark"
            preset["tl_title"] = "PLAYLIST"
            preset["tl_color"] = "#ffffff"
            preset["tl_active"] = "none"

        return preset
    except:
        return None

# ==========================================
# ⚙️ CORE ENGINE (VISUALIZER & FFMPEG)
# ==========================================
class AudioBrain:
    def __init__(self):
        self.y = None; self.sr = None; self.onset_env = None; self.has_audio = False
        self.duration = 0.0
        self.n_fft = 2048
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.cached_bars = -1
        self.tilt = None
        self.bin_edges = None
        self.kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)
        self.wave_mod = None
        self.phase_pattern = None

    def _ensure_bars(self, n_bars):
        if self.cached_bars == n_bars and self.tilt is not None:
            return
        self.cached_bars = n_bars
        sr_ref = self.sr if self.sr else 22050
        self.tilt = (1.0 + 3.2 * (np.linspace(0, 1, n_bars) ** 0.85)).astype(np.float32)
        f_edges = 35.0 + (3800.0 - 35.0) * (np.linspace(0, 1, n_bars + 1) ** 1.6)
        self.bin_edges = np.clip(np.round(f_edges * self.n_fft / sr_ref).astype(int), 3, self.n_fft // 2)
        self.wave_mod = (0.80 + 0.20 * np.sin(np.linspace(0, 4.0 * np.pi, n_bars))).astype(np.float32)
        self.phase_pattern = np.linspace(0, 4.0 * np.pi, n_bars).astype(np.float32)

    def load(self, path, max_duration=None):
        try:
            self.y = None
            self.onset_env = None
            gc.collect()
            self.y, self.sr = librosa.load(path, sr=22050, mono=True, duration=max_duration)
            self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            self.duration = len(self.y) / self.sr
            self.has_audio = True
            self._ensure_bars(64)
        except Exception as e:
            print(f"Audio Error: {e}")

    def get_data(self, t, n_bars=64): 
        if not self.has_audio: return 0.0, False, np.zeros(n_bars, dtype=np.float32)
        self._ensure_bars(n_bars)
        idx = int(t * self.sr)
        if idx >= len(self.y): return 0.0, False, np.zeros(n_bars, dtype=np.float32)

        try:
            chunk = self.y[idx:idx+1024]
            vol = float(np.sqrt(np.mean(chunk**2)) * 10) if len(chunk) > 0 else 0.0
        except:
            vol = 0.0
        
        hit = False
        try:
            onset_idx = int(idx / 512)
            if onset_idx < len(self.onset_env) and self.onset_env[onset_idx] > 2.0: 
                hit = True
        except:
            pass

        final_bars = np.full(n_bars, 0.12, dtype=np.float32)
        try:
            fft_data = self.y[idx:idx+self.n_fft]
            if len(fft_data) == self.n_fft:
                # 1. FFT dengan pre-cached Hanning window
                windowed_data = fft_data * self.window
                spec = np.abs(np.fft.rfft(windowed_data))

                # 2. Ambil energi per band frekuensi musikal (~35 Hz s/d ~3800 Hz)
                raw_bars = np.zeros(n_bars, dtype=np.float32)
                spec_len = len(spec)
                for i in range(n_bars):
                    s = int(self.bin_edges[i])
                    e = int(max(s + 1, self.bin_edges[i + 1]))
                    if e > spec_len: e = spec_len
                    if e > s:
                        # Logarithmic decibel compression dengan treble tilt boost
                        mag = float(np.mean(spec[s:e])) * self.tilt[i]
                        raw_bars[i] = float(np.log1p(mag * 0.18))
                    else:
                        raw_bars[i] = 0.0

                # 3. Spatial smoothing antar bar tetangga
                smooth_bars = np.convolve(raw_bars, self.kernel, mode='same')

                # 4. Modulasi multi-peak wave (meniru wave preview yang berombak indah)
                modulated = smooth_bars * self.wave_mod

                # 5. Lantai aktif ritmis dinamis (mencegah sudut kanan kosong / 0)
                vol_factor = min(1.0, max(0.35, vol * 0.45))
                floor_pattern = (0.15 + 0.08 * np.sin(t * 3.2 + self.phase_pattern)) * vol_factor
                active_bars = np.maximum(modulated, floor_pattern.astype(np.float32))

                # 6. HEADROOM NORMALIZATION (Anti-Ceiling / Anti-Rata):
                # Memastikan puncak tertinggi tidak pernah mentok ke atap (maksimal ~0.65 - 0.68)
                # sehingga selalu ada ruang bebas di atas bar persis seperti di preview!
                peak = float(np.max(active_bars))
                if peak > 0.68:
                    scaled_bars = active_bars * (0.68 / peak)
                else:
                    scaled_bars = active_bars

                final_bars = np.maximum(0.12, scaled_bars)
        except Exception:
            pass

        return vol, hit, final_bars

class BackgroundManager:
    def __init__(self, bg_paths, w, h):
        self.bg_paths = bg_paths; self.w = w; self.h = h; self.idx = 0
        self.cap = None; self.static_bg = None; self.load_current()
        
    def load_current(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if not self.bg_paths:
                self.static_bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                return
            path = self.bg_paths[self.idx]
            if not os.path.exists(path):
                self.static_bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                return
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): 
                img = cv2.imread(path)
                if img is not None:
                    interp = cv2.INTER_AREA if (img.shape[1] > self.w or img.shape[0] > self.h) else cv2.INTER_LINEAR
                    self.static_bg = cv2.resize(img, (self.w, self.h), interpolation=interp)
                else:
                    self.static_bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            else: 
                # ⚡ Gunakan OpenCV native VideoCapture (3x-5x lebih cepat dari imageio)
                self.cap = cv2.VideoCapture(path)
                self.static_bg = None
        except Exception as e:
            print(f"[BackgroundManager] Gagal load {path}: {e}")
            self.static_bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            
    def get_frame(self):
        if self.static_bg is not None:
            return self.static_bg.copy()
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Loop video dari awal atau ganti ke background berikutnya
                if len(self.bg_paths) > 1:
                    self.idx = (self.idx + 1) % len(self.bg_paths)
                    self.load_current()
                    return self.get_frame()
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.w or frame.shape[0] != self.h:
                    frame = cv2.resize(frame, (self.w, self.h))
                return frame
        return np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

class VisualEngine:
    def __init__(self, c_bot, c_top, c_part):
        self.col_bot = (c_bot[2], c_bot[1], c_bot[0])
        self.col_top = (c_top[2], c_top[1], c_top[0])
        self.col_part = (c_part[2], c_part[1], c_part[0])
        self.bar_h = None
        self.zoom = 1.0
        self.flash = 0.0

        self.grad = np.zeros((1000, 1, 3), dtype=np.uint8)
        for c in range(3):
            self.grad[:, 0, c] = np.linspace(self.col_top[c], self.col_bot[c], 1000)

        self.particles = []
        self._glow_sprites = {}
        for sz in (6, 10, 16, 24, 36, 48):
            r = sz // 2
            y, x = np.ogrid[-r:r, -r:r]
            dist = np.sqrt(x * x + y * y)
            alpha = np.clip(1.0 - (dist / max(1, r)) ** 1.3, 0.0, 1.0).astype(np.float32)
            self._glow_sprites[sz] = alpha[:, :, np.newaxis]

    def _draw_glow(self, frame, cx, cy, radius, color_bgr, alpha=1.0):
        h, w = frame.shape[:2]
        sizes = (6, 10, 16, 24, 36, 48)
        target_sz = max(6, min(48, int(radius * 2)))
        chosen_sz = min(sizes, key=lambda s: abs(s - target_sz))
        sprite = self._glow_sprites[chosen_sz]
        r = chosen_sz // 2
        cx, cy = int(cx), int(cy)
        x1 = max(0, cx - r); x2 = min(w, cx + r)
        y1 = max(0, cy - r); y2 = min(h, cy + r)
        if x2 <= x1 or y2 <= y1: return
        sx1 = x1 - (cx - r); sx2 = sx1 + (x2 - x1)
        sy1 = y1 - (cy - r); sy2 = sy1 + (y2 - y1)
        sub_sprite = sprite[sy1:sy2, sx1:sx2]
        eff_alpha = max(0.0, min(1.0, safe_float(alpha, 1.0)))
        tint = (sub_sprite * np.array(color_bgr, dtype=np.float32) * eff_alpha).astype(np.uint8)
        roi = frame[y1:y2, x1:x2]
        cv2.add(roi, tint, dst=roi)

    # ── helper ──
    @staticmethod
    def _sn(val, default):
        try: return float(val) if val != "" and val is not None else default
        except: return default

    # ── helper warna per bar (interpolasi gradasi) ──
    def _bar_color(self, i, n):
        t = i / max(1, n - 1)
        r = int(self.col_top[0] * (1 - t) + self.col_bot[0] * t)
        g = int(self.col_top[1] * (1 - t) + self.col_bot[1] * t)
        b = int(self.col_top[2] * (1 - t) + self.col_bot[2] * t)
        return (r, g, b)

    # ── dispatcher utama ──
    def process(self, frame, vol, is_hit, bars, cfg):
        h, w = frame.shape[:2]
        n = len(bars)
        if self.bar_h is None or len(self.bar_h) != n:
            self.bar_h = np.zeros(n)

        react  = self._sn(cfg.get('reactivity'), 0.66)
        idle   = int(self._sn(cfg.get('idle_height'), 5))
        space  = int(self._sn(cfg.get('spacing'), 3))
        px     = self._sn(cfg.get('pos_x'), 50) / 100
        py     = self._sn(cfg.get('pos_y'), 85) / 100
        wp     = self._sn(cfg.get('width_pct'), 60) / 100
        max_h  = h * (self._sn(cfg.get('max_height'), 40) / 100)
        p_amt  = int(self._sn(cfg.get('part_amount'), 3))
        p_spd  = self._sn(cfg.get('part_speed'), 1.0)
        smooth = self._sn(cfg.get('smoothing'), 0.90)

        # smooth bar heights: fast attack (responsif beat) & smooth liquid decay (mengalir halus)
        for i in range(n):
            target = bars[i] * react
            if target > self.bar_h[i]:
                self.bar_h[i] = self.bar_h[i] * 0.35 + target * 0.65
            else:
                decay = max(0.85, min(0.95, smooth))
                self.bar_h[i] = self.bar_h[i] * decay + target * (1.0 - decay)
            self.bar_h[i] = max(0.0, self.bar_h[i])

        # ── Jedag-Jedug (Screen Zoom Bounce) ──
        jj_mode = str(cfg.get('jj_mode', 'off')).lower()
        if jj_mode not in ('off', 'none') and is_hit and vol > 1.2:
            target_zoom = 1.055 if jj_mode == 'hard' else 1.028
            if target_zoom > self.zoom:
                self.zoom = target_zoom
        self.zoom = self.zoom * 0.72 + 1.0 * 0.28

        # ── Efek Lampu Flash (Strobe / Glow) ──
        flash_mode = str(cfg.get('flash_mode', 'off')).lower()
        if flash_mode not in ('off', 'none') and is_hit and vol > 1.2:
            self.flash = max(self.flash, min(0.38, vol * 0.08))
        self.flash = self.flash * 0.65

        # beat pulse (additive, bisa aktif bersama efek lain)
        if cfg.get('use_beat_pulse', False):
            self._draw_beat_pulse(frame, vol, is_hit, w, h)

        # dispatch ke efek utama
        effect = cfg.get('effect_type', 'spectrum')
        if effect in ('none', 'off', 'disabled'):
            pass  # Efek visualizer dinonaktifkan
        elif effect == 'circular':
            self._draw_circular(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'waveform':
            self._draw_waveform(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'mirror':
            self._draw_mirror(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'neon_glow':
            self._draw_neon_glow(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'sunburst':
            self._draw_sunburst(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'pixel':
            self._draw_pixel(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'double_symmetric':
            self._draw_double_symmetric(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'dots_pixel':
            self._draw_dots_pixel(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'filled_wave':
            self._draw_filled_wave(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'sinusoidal':
            self._draw_sinusoidal(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'smooth_blob':
            self._draw_smooth_blob(frame, n, idle, space, px, py, wp, max_h, w, h)
        elif effect == 'halftone':
            self._draw_halftone(frame, n, idle, space, px, py, wp, max_h, w, h)
        else:
            bar_style = cfg.get('bar_style', 'bottom')
            self._draw_spectrum(frame, n, idle, space, px, py, wp, max_h, w, h, bar_style)

        # sparkle / particles (bisa dinonaktifkan)
        p_type = cfg.get('particle_type', 'sparkle')
        if p_amt > 0 and p_type not in ('none', 'off', 'disabled'):
            self._draw_particles(frame, vol, is_hit, p_amt, p_spd, w, h, p_type, cfg)

        # ── Terapkan Efek Jedag-Jedug (Zoom Layar) ──
        if self.zoom > 1.004 and jj_mode not in ('off', 'none'):
            pad_x = int(w * (self.zoom - 1.0) / 2)
            pad_y = int(h * (self.zoom - 1.0) / 2)
            if pad_x > 0 and pad_y > 0 and (w - 2 * pad_x) > 0 and (h - 2 * pad_y) > 0:
                crop = frame[pad_y:h-pad_y, pad_x:w-pad_x]
                frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Terapkan Efek Lampu (Flash / Strobe) ──
        if self.flash > 0.02 and flash_mode not in ('off', 'none'):
            if flash_mode == 'neon':
                add_b = int(self.col_top[0] * self.flash)
                add_g = int(self.col_top[1] * self.flash)
                add_r = int(self.col_top[2] * self.flash)
                frame = cv2.add(frame, (add_b, add_g, add_r, 0))
            else:  # 'white' / strobe
                add_val = int(255 * self.flash)
                frame = cv2.add(frame, (add_val, add_val, add_val, 0))

        return frame

    # ═══════════════════════════════════════════════════════════
    #  SPECTRUM BARS  (efek asli)
    # ═══════════════════════════════════════════════════════════
    def _draw_spectrum(self, frame, n, idle, space, px, py, wp, max_h, w, h, bar_style):
        bar_w = int(max(1, (w * wp - space * (n - 1)) / n))
        s_x   = int((w * px) - (w * wp / 2))
        b_y   = int(h * py)

        for i in range(n):
            height = int(max(idle, min(max_h, self.bar_h[i] * max_h)))
            if height <= 0: continue
            x1 = s_x + i * (bar_w + space)
            x2 = x1 + bar_w

            if bar_style == 'center':
                y1 = b_y - (height // 2); y2 = b_y + (height // 2)
            else:
                y1 = b_y - height; y2 = b_y

            x1s = max(0, min(w, x1)); x2s = max(0, min(w, x2))
            y1s = max(0, min(h, y1)); y2s = max(0, min(h, y2))
            ws = x2s - x1s; hs = y2s - y1s
            if ws > 0 and hs > 0:
                bg = cv2.resize(self.grad, (bar_w, height))
                frame[y1s:y2s, x1s:x2s] = bg[y1s-y1:y1s-y1+hs, x1s-x1:x1s-x1+ws]
                if bar_w >= 3 and y1s < h:
                    cv2.line(frame, (x1s, y1s), (x2s - 1, y1s), self.col_top, 1, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    #  CIRCULAR SPECTRUM  (bar radial membentuk lingkaran)
    # ═══════════════════════════════════════════════════════════
    def _draw_circular(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        cx = int(w * px)
        cy = int(h * py)
        # 🐛 FIX: radius pakai min(w,h) (bukan w saja) agar konsisten dengan preview
        # canvas. Sebelumnya radius terlalu besar di video landscape (16:9).
        radius = int(min(w, h) * (wp / 2))
        angle_step = (2 * math.pi) / n
        bar_w = max(2, int((2 * math.pi * radius - space * n) / n))

        for i in range(n):
            # 🐛 FIX: value dikalikan 0.5 & clamp max_h*0.5 agar tinggi bar
            # konsisten dengan preview (preview pakai prev_bars[i]*0.5, max_h/200).
            # max_h di sini sudah = h*(max_height/100), jadi max_h*0.5 = h*(max_height/200).
            height = int(max(idle, min(max_h * 0.5, self.bar_h[i] * max_h * 0.5)))
            if height <= 0: continue
            angle = i * angle_step - math.pi / 2

            # inner & outer edge
            ix = int(cx + radius * math.cos(angle))
            iy = int(cy + radius * math.sin(angle))
            ox = int(cx + (radius + height) * math.cos(angle))
            oy = int(cy + (radius + height) * math.sin(angle))

            color = self._bar_color(i, n)
            cv2.line(frame, (ix, iy), (ox, oy), color, bar_w, cv2.LINE_AA)

        # lingkaran dasar tipis
        cv2.circle(frame, (cx, cy), radius, self.col_bot, 1, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    #  WAVEFORM  (gelombang audio klasik)
    # ═══════════════════════════════════════════════════════════
    def _draw_waveform(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        tot_w = w * wp
        s_x   = int((w * px) - (tot_w / 2))
        b_y   = int(h * py)
        step  = tot_w / n

        pts_upper = []
        pts_lower = []
        for i in range(n):
            height = self.bar_h[i] * max_h
            height = max(idle, min(max_h, height))
            x = int(s_x + i * step)
            pts_upper.append((x, int(b_y - height)))
            pts_lower.append((x, int(b_y + height)))

        # fill area antara waveform
        pts_fill = pts_upper + pts_lower[::-1]
        if len(pts_fill) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(pts_fill, dtype=np.int32)], (*self.col_bot,))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # garis utama (atas) tebal
        pts_line_upper = np.array(pts_upper, dtype=np.int32)
        if len(pts_line_upper) >= 2:
            cv2.polylines(frame, [pts_line_upper], False, self.col_top, 2, cv2.LINE_AA)

        # garis bawah tipis (refleksi)
        pts_line_lower = np.array(pts_lower, dtype=np.int32)
        if len(pts_line_lower) >= 2:
            cv2.polylines(frame, [pts_line_lower], False, self.col_bot, 1, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    #  MIRROR SPECTRUM  (bar spectrum + refleksi simetris)
    # ═══════════════════════════════════════════════════════════
    def _draw_mirror(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        bar_w = int(max(1, (w * wp - space * (n - 1)) / n))
        s_x   = int((w * px) - (w * wp / 2))
        b_y   = int(h * py)
        half_h = max_h // 2

        for i in range(n):
            height = int(max(idle, min(half_h, self.bar_h[i] * half_h)))
            if height <= 0: continue
            x1 = s_x + i * (bar_w + space)
            x2 = x1 + bar_w

            # bar ke atas (full opacity)
            y1_up = b_y - height; y2_up = b_y
            x1s = max(0, min(w, x1)); x2s = max(0, min(w, x2))
            y1s = max(0, min(h, y1_up)); y2s = max(0, min(h, y2_up))
            ws = x2s - x1s; hs = y2s - y1s
            if ws > 0 and hs > 0:
                bg = cv2.resize(self.grad, (bar_w, height))
                frame[y1s:y2s, x1s:x2s] = bg[y1s-y1_up:y1s-y1_up+hs, x1s-x1:x1s-x1+ws]

            # bar refleksi ke bawah (lebih redup)
            y1_dn = b_y; y2_dn = b_y + height
            x1s2 = max(0, min(w, x1)); x2s2 = max(0, min(w, x2))
            y1s2 = max(0, min(h, y1_dn)); y2s2 = max(0, min(h, y2_dn))
            ws2 = x2s2 - x1s2; hs2 = y2s2 - y1s2
            if ws2 > 0 and hs2 > 0:
                bg2 = cv2.resize(self.grad, (bar_w, height))
                faded = (bg2 * 0.35).astype(np.uint8)
                frame[y1s2:y2s2, x1s2:x2s2] = faded[y1s2-y1_dn:y1s2-y1_dn+hs2, x1s2-x1:x1s2-x1+ws2]

    # ═══════════════════════════════════════════════════════════
    #  NEON GLOW BARS  (bar dengan efek glow/bloom seperti neon)
    # ═══════════════════════════════════════════════════════════
    def _draw_neon_glow(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        # gambar di overlay terpisah lalu blur untuk efek bloom
        overlay = np.zeros_like(frame)
        bar_w = int(max(1, (w * wp - space * (n - 1)) / n))
        s_x   = int((w * px) - (w * wp / 2))
        b_y   = int(h * py)

        for i in range(n):
            height = int(max(idle, min(max_h, self.bar_h[i] * max_h)))
            if height <= 0: continue
            x1 = s_x + i * (bar_w + space)
            x2 = x1 + bar_w
            y1 = b_y - height; y2 = b_y

            x1s = max(0, min(w, x1)); x2s = max(0, min(w, x2))
            y1s = max(0, min(h, y1)); y2s = max(0, min(h, y2))
            ws = x2s - x1s; hs = y2s - y1s
            if ws > 0 and hs > 0:
                bg = cv2.resize(self.grad, (bar_w, height))
                overlay[y1s:y2s, x1s:x2s] = bg[y1s-y1:y1s-y1+hs, x1s-x1:x1s-x1+ws]

        # bloom layer (blur) — kernel adaptif terhadap resolusi agar dekati
        # Canvas shadowBlur=20 (preview ~450px; di render perlu proporsional).
        blur_k = max(11, int(20 * h / 450) | 1)  # ganjil, min 11
        bloom = cv2.GaussianBlur(overlay, (blur_k, blur_k), 0)
        cv2.addWeighted(bloom, 0.7, frame, 1.0, 0, frame)
        # core layer (tajam)
        cv2.addWeighted(overlay, 0.9, frame, 1.0, 0, frame)

    # ═══════════════════════════════════════════════════════════
    #  RADIAL SUNBURST  (memancar dari pusat ke segala arah)
    # ═══════════════════════════════════════════════════════════
    def _draw_sunburst(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        cx = int(w * px)
        cy = int(h * py)
        # 🐛 FIX: max_r pakai min(w,h) (bukan w saja) agar konsisten dengan preview.
        max_r = int(min(w, h) * (wp / 2))
        angle_step = (2 * math.pi) / n

        for i in range(n):
            # 🐛 FIX: clamp max_h*0.666 + value *0.666 agar konsisten dengan preview
            # (preview: maxBarH = ch*(max_height/150), h_val = prev_bars[i]*0.666).
            # max_h di sini = h*(max_height/100); 0.666*max_h = h*(max_height/150).
            height = int(max(idle, min(max_h * 0.666, self.bar_h[i] * max_h * 0.666)))
            if height <= 0: continue
            angle = i * angle_step - math.pi / 2

            # dari pusat (radius kecil) memancar keluar
            inner_r = max_r * 0.15
            ix = int(cx + inner_r * math.cos(angle))
            iy = int(cy + inner_r * math.sin(angle))
            ox = int(cx + (inner_r + height) * math.cos(angle))
            oy = int(cy + (inner_r + height) * math.sin(angle))

            color = self._bar_color(i, n)
            cv2.line(frame, (ix, iy), (ox, oy), color, 3, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    #  PIXEL BLOCKS  (8-bit retro, blok kotak besar)
    # ═══════════════════════════════════════════════════════════
    def _draw_pixel(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        # kurangi bar count untuk efek blocky, pakai blok besar
        block_size = max(6, int(w * wp / n))
        gap = max(2, space)
        bar_w = block_size - gap
        s_x   = int((w * px) - (w * wp / 2))
        b_y   = int(h * py)
        block_unit = max(4, int(max_h / 16))  # tinggi per pixel block

        for i in range(n):
            height = int(max(idle, min(max_h, self.bar_h[i] * max_h)))
            if height <= 0: continue
            x1 = s_x + i * (block_size)

            blocks = max(1, height // block_unit)
            for b in range(blocks):
                y_block = b_y - (b + 1) * block_unit
                if y_block < 0: break
                # warna gradient per block
                t = b / max(1, blocks)
                color = (
                    int(self.col_top[0] * (1-t) + self.col_bot[0] * t),
                    int(self.col_top[1] * (1-t) + self.col_bot[1] * t),
                    int(self.col_top[2] * (1-t) + self.col_bot[2] * t),
                )
                cv2.rectangle(frame, (x1, y_block), (x1 + bar_w, y_block + block_unit - gap), color, -1)

    # ═══════════════════════════════════════════════════════════
    #  DOUBLE SYMMETRIC  (mirror kiri-kanan dari garis tengah vertikal)
    # ═══════════════════════════════════════════════════════════
    def _draw_double_symmetric(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        half = n // 2
        bar_w = int(max(1, (w * wp - space * (n - 1)) / n))
        s_x   = int((w * px) - (w * wp / 2))
        b_y   = int(h * py)

        for i in range(n):
            height = int(max(idle, min(max_h, self.bar_h[i] * max_h)))
            if height <= 0: continue
            x1 = s_x + i * (bar_w + space)
            y1 = b_y - height; y2 = b_y

            x1s = max(0, min(w, x1)); x2s = max(0, min(w, x1 + bar_w))
            y1s = max(0, min(h, y1)); y2s = max(0, min(h, y2))
            ws = x2s - x1s; hs = y2s - y1s
            if ws > 0 and hs > 0:
                bg = cv2.resize(self.grad, (bar_w, height))
                frame[y1s:y2s, x1s:x2s] = bg[y1s-y1:y1s-y1+hs, x1s-x1:x1s-x1+ws]

            # mirror ke kanan dari garis tengah vertikal
            mid_x = int(w * px)
            m_x1 = 2 * mid_x - x1 - bar_w
            m_x2 = m_x1 + bar_w
            m_x1s = max(0, min(w, m_x1)); m_x2s = max(0, min(w, m_x2))
            mws = m_x2s - m_x1s
            if mws > 0 and hs > 0:
                frame[y1s:y2s, m_x1s:m_x2s] = bg[y1s-y1:y1s-y1+hs, m_x1s-m_x1:m_x1s-m_x1+mws]

    # ═══════════════════════════════════════════════════════════
    #  DOTS PIXEL  (lingkaran di puncak bar)
    # ═══════════════════════════════════════════════════════════
    def _draw_dots_pixel(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        bar_w = max(2, int((w * wp - space * (n - 1)) / n))
        s_x   = int((w * px) - (w * wp / 2))
        b_y   = int(h * py)
        for i in range(n):
            height = int(max(idle, min(max_h, self.bar_h[i] * max_h)))
            if height <= 0: continue
            cx = s_x + i * (bar_w + space) + bar_w // 2
            cy = b_y - height
            r = max(2, int(height * 0.3))
            color = self._bar_color(i, n)
            cv2.circle(frame, (cx, cy), r, color, -1)

    # ═══════════════════════════════════════════════════════════
    #  FILLED WAVE  (gelombang terisi penuh)
    # ═══════════════════════════════════════════════════════════
    def _draw_filled_wave(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        tot_w = w * wp
        s_x   = int((w * px) - (tot_w / 2))
        b_y   = int(h * py)
        step  = tot_w / n

        pts = []
        for i in range(n):
            height = max(idle, min(max_h, self.bar_h[i] * max_h))
            x = int(s_x + i * step)
            pts.append((x, int(b_y - height)))
        # tutup polygon dari kanan bawah ke kiri bawah
        pts.append((int(s_x + tot_w), b_y))
        pts.append((s_x, b_y))

        if len(pts) >= 3:
            overlay_wave = frame.copy()
            cv2.fillPoly(overlay_wave, [np.array(pts, dtype=np.int32)], self.col_bot)
            cv2.addWeighted(overlay_wave, 0.55, frame, 0.45, 0, frame)

        # garis atas
        for i in range(1, n):
            h1 = max(idle, min(max_h, self.bar_h[i-1] * max_h))
            h2 = max(idle, min(max_h, self.bar_h[i] * max_h))
            x1 = int(s_x + (i-1) * step)
            x2 = int(s_x + i * step)
            cv2.line(frame, (x1, int(b_y - h1)), (x2, int(b_y - h2)), self.col_top, 3, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    #  SINUSOIDAL  (gelombang sinusoidal berfrekuensi audio)
    # ═══════════════════════════════════════════════════════════
    def _draw_sinusoidal(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        tot_w = w * wp
        s_x   = int((w * px) - (tot_w / 2))
        b_y   = int(h * py)
        amp   = max_h * 0.4
        freq  = 4.0  # jumlah gelombang

        pts = []
        for i in range(n):
            t = i / max(1, n - 1)
            modulation = self.bar_h[i]  # 0..1 dari audio
            wave = math.sin(t * freq * 2 * math.pi + math.pi * 0.5) * modulation * amp
            height = max(idle, wave + amp * modulation * 0.3)
            x = int(s_x + i * (tot_w / n))
            y = int(b_y - height)
            pts.append((x, y))

        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, self.col_top, 2, cv2.LINE_AA)
            # glow tipis
            overlay_sin = frame.copy()
            for i in range(n):
                # 🐛 FIX: ganti nama variabel 'h' -> 'seg' agar tidak menimpa
                # parameter frame-height 'h' (latent bug: h tertimpa di loop ini).
                seg = max(0, b_y - pts[i][1])
                cv2.line(overlay_sin, (pts[i][0], b_y), (pts[i][0], pts[i][1]), self.col_bot, max(1, int(seg * 0.15)), cv2.LINE_AA)
            cv2.addWeighted(overlay_sin, 0.3, frame, 0.7, 0, frame)

    # ═══════════════════════════════════════════════════════════
    #  SMOOTH BLOB  (gumpalan organik halus)
    # ═══════════════════════════════════════════════════════════
    def _draw_smooth_blob(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        cx = int(w * px)
        cy = int(h * py)
        base_r = min(w, h) * 0.08
        blob_amp = min(w, h) * 0.25

        angle_step = (2 * math.pi) / n
        pts = []
        for i in range(n):
            angle = i * angle_step - math.pi / 2
            r_offset = self.bar_h[i] * blob_amp
            r = base_r + r_offset
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            pts.append([x, y])

        if len(pts) >= 3:
            overlay_blob = frame.copy()
            cv2.fillPoly(overlay_blob, [np.array(pts, dtype=np.int32)], self.col_bot)
            # blur untuk efek halus — kernel adaptif terhadap resolusi
            blur_k = max(11, int(14 * h / 450) | 1)  # ganjil, min 11
            overlay_blob = cv2.GaussianBlur(overlay_blob, (blur_k, blur_k), 0)
            cv2.addWeighted(overlay_blob, 0.6, frame, 1.0, 0, frame)
            # core outline
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, self.col_top, 2, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    #  HALFTONE DOTS  (titik-titik yang membesar oleh audio)
    # ═══════════════════════════════════════════════════════════
    def _draw_halftone(self, frame, n, idle, space, px, py, wp, max_h, w, h):
        cols = int(n * 0.5)
        rows = max(3, int(cols * 0.4))
        cell_w = int(w * wp / cols)
        cell_h = int(max_h * 0.8 / rows)
        start_x = int(w * px) - (cols * cell_w) // 2
        start_y = int(h * py) - (rows * cell_h) // 2
        max_dot = min(cell_w, cell_h) * 0.45

        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c) % n
                # 🐛 FIX: sebelumnya amp = bar_h/max_h (nilai 0..0.66 dibagi ~432px
                # = hampir 0 -> semua dot jadi 1px). self.bar_h sudah bernilai 0..~1
                # (relatif, hasil smoothing), jadi cukup clamp ke 1.0 agar dot
                # membesar sesuai amplitudo audio — konsisten dengan preview.
                amp = min(1.0, max(0.0, self.bar_h[idx]))
                dot_r = max(1, int(1 + amp * max_dot))
                dx = start_x + c * cell_w + cell_w // 2
                dy = start_y + r * cell_h + cell_h // 2
                color = self._bar_color(int(idx), int(n))
                cv2.circle(frame, (dx, dy), dot_r, color, -1)

    # ═══════════════════════════════════════════════════════════
    #  BEAT PULSE / GLOW  (overlay flash saat bass hit)
    # ═══════════════════════════════════════════════════════════
    def _draw_beat_pulse(self, frame, vol, is_hit, w, h):
        if is_hit and vol > 1.2:
            intensity = min(0.18, vol * 0.04)
            overlay = frame.copy()
            overlay[:] = self.col_top
            cv2.addWeighted(overlay, intensity, frame, 1.0 - intensity, 0, frame)

    # ═══════════════════════════════════════════════════════════
    #  SPARKLE / PARTICLES (Modern Soft Glow & Organic Physics)
    # ═══════════════════════════════════════════════════════════
    def _draw_particles(self, frame, vol, is_hit, p_amt, p_spd, w, h, p_type='embers', cfg=None):
        cfg = cfg or {}
        p_type = str(p_type).lower().strip()
        if p_type in ('none', 'off', 'disabled') or p_amt <= 0:
            self.particles = []
            return

        # Handle reset jika data partikel lama masih berbentuk list
        if self.particles and isinstance(self.particles[0], list):
            self.particles = []

        sz_mult = {'small': 0.65, 'large': 1.6}.get(cfg.get('part_size', 'medium'), 1.0)
        life_mult = {'short': 0.6, 'long': 1.8}.get(cfg.get('part_life', 'medium'), 1.0)
        dens_mult = {'low': 0.5, 'high': 1.8}.get(cfg.get('part_density', 'medium'), 1.0)
        part_alpha = max(0.0, min(1.0, safe_float(cfg.get('part_opacity', 1.0), 1.0)))

        spd = max(0.25, (1.0 + (vol * 0.12)) * p_spd)
        beat_boost = 0.40 if (is_hit and vol > 1.2) else 0.0

        def pcol(base, a=1.0):
            factor = max(0.0, min(1.0, part_alpha * a))
            return (int(base[0] * factor), int(base[1] * factor), int(base[2] * factor))

        # ── 1. AMBIENT POPULATION (Layar selalu hidup & estetik, tidak pernah kosong) ──
        target_ambient = max(12, min(95, int((p_amt + 3) * 7 * dens_mult)))
        while len(self.particles) < target_ambient:
            init_y = random.uniform(0, h) if len(self.particles) < (target_ambient // 2) else random.uniform(h * 0.7, h + 20)
            if p_type in ('embers', 'fireflies'):
                self.particles.append({
                    'x': random.uniform(0, w), 'y': init_y,
                    'vx': random.uniform(-0.3, 0.3), 'vy': -random.uniform(0.7, 1.8),
                    'r': random.uniform(3.5, 7.5) * sz_mult,
                    'life': int(random.uniform(90, 180) * life_mult), 'max_life': 180,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'embers', 'extra': 0.0
                })
            elif p_type == 'twinkle':
                self.particles.append({
                    'x': random.uniform(0, w), 'y': random.uniform(0, h),
                    'vx': random.uniform(-0.4, 0.4), 'vy': random.uniform(-0.4, 0.4),
                    'r': random.uniform(3.0, 6.5) * sz_mult,
                    'life': int(random.uniform(60, 140) * life_mult), 'max_life': 140,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'twinkle', 'extra': 0.0
                })
            elif p_type == 'bokeh':
                self.particles.append({
                    'x': random.uniform(0, w), 'y': init_y,
                    'vx': random.uniform(-0.2, 0.2), 'vy': -random.uniform(0.3, 0.8),
                    'r': random.uniform(14.0, 32.0) * sz_mult,
                    'life': int(random.uniform(120, 220) * life_mult), 'max_life': 220,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'bokeh', 'extra': 0.0
                })
            elif p_type == 'snow':
                self.particles.append({
                    'x': random.uniform(0, w), 'y': random.uniform(-20, h),
                    'vx': random.uniform(-0.5, 0.5), 'vy': random.uniform(0.6, 1.6),
                    'r': random.uniform(2.0, 4.5) * sz_mult,
                    'life': int(random.uniform(100, 200) * life_mult), 'max_life': 200,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'snow', 'extra': 0.0
                })
            elif p_type == 'petals':
                self.particles.append({
                    'x': random.uniform(0, w), 'y': random.uniform(-20, h),
                    'vx': random.uniform(-0.8, 0.8), 'vy': random.uniform(0.8, 2.0),
                    'r': random.uniform(4.0, 8.0) * sz_mult,
                    'life': int(random.uniform(100, 200) * life_mult), 'max_life': 200,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'petals', 'extra': random.uniform(0, math.pi)
                })
            elif p_type == 'bubbles':
                self.particles.append({
                    'x': random.uniform(w * 0.05, w * 0.95), 'y': init_y,
                    'vx': random.uniform(-0.4, 0.4), 'vy': -random.uniform(0.7, 1.8),
                    'r': random.uniform(4.0, 10.0) * sz_mult,
                    'life': int(random.uniform(80, 160) * life_mult), 'max_life': 160,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'bubbles', 'extra': 0.0
                })
            elif p_type == 'rain':
                self.particles.append({
                    'x': random.uniform(0, w), 'y': random.uniform(-30, h),
                    'vx': random.uniform(-1.2, -0.4), 'vy': random.uniform(7.0, 12.0),
                    'r': random.uniform(1.0, 2.5) * sz_mult,
                    'life': int(random.uniform(40, 80) * life_mult), 'max_life': 80,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'rain', 'extra': 0.0
                })
            elif p_type in ('drift_sparks', 'phonk', 'sparks'):
                self.particles.append({
                    'x': random.uniform(0, w), 'y': random.uniform(h * 0.25, h + 20),
                    'vx': random.choice([-1.0, 1.0]) * random.uniform(2.5, 6.5),
                    'vy': -random.uniform(1.2, 4.5),
                    'r': random.uniform(2.0, 4.5) * sz_mult,
                    'life': int(random.uniform(25, 55) * life_mult), 'max_life': 55,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'drift_sparks', 'extra': 0.0
                })
            elif p_type in ('golden_dust', 'afro', 'gold_dust'):
                self.particles.append({
                    'x': random.uniform(0, w), 'y': init_y,
                    'vx': random.uniform(-0.35, 0.35), 'vy': -random.uniform(0.4, 1.1),
                    'r': random.uniform(2.5, 5.5) * sz_mult,
                    'life': int(random.uniform(110, 210) * life_mult), 'max_life': 210,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'golden_dust', 'extra': 0.0
                })
            else:  # sparkle default
                self.particles.append({
                    'x': random.uniform(0, w), 'y': random.uniform(0, h),
                    'vx': random.uniform(-1.5, 1.5), 'vy': random.uniform(-1.5, 1.5),
                    'r': random.uniform(2.5, 6.0) * sz_mult,
                    'life': int(random.uniform(40, 90) * life_mult), 'max_life': 90,
                    'phase': random.uniform(0, math.pi * 2), 'type': 'sparkle', 'extra': 0.0
                })

        # ── 2. AUDIO REACTIVE BURST (Meletup dinamis saat beat keras) ──
        if is_hit and vol > 1.3:
            burst_count = max(2, min(8, int(p_amt * dens_mult * 0.6)))
            for _ in range(burst_count):
                if p_type == 'fireworks':
                    self.particles.append({
                        'x': random.uniform(w * 0.35, w * 0.65), 'y': random.uniform(h * 0.6, h * 0.85),
                        'vx': random.uniform(-5.5, 5.5), 'vy': random.uniform(-7.5, -2.5),
                        'r': random.uniform(3.0, 6.0) * sz_mult,
                        'life': int(random.uniform(40, 75) * life_mult), 'max_life': 75,
                        'phase': random.uniform(0, math.pi * 2), 'type': 'fireworks', 'extra': 0.0
                    })
                elif p_type == 'trail':
                    self.particles.append({
                        'x': random.uniform(0, w), 'y': random.uniform(0, h * 0.5),
                        'vx': random.uniform(-3.5, 3.5), 'vy': random.uniform(3.0, 6.5),
                        'r': random.uniform(3.0, 5.0) * sz_mult,
                        'life': int(random.uniform(40, 70) * life_mult), 'max_life': 70,
                        'phase': random.uniform(0, math.pi * 2), 'type': 'trail', 'extra': 0.0
                    })
                elif p_type == 'smoke':
                    self.particles.append({
                        'x': random.uniform(w * 0.15, w * 0.85), 'y': random.uniform(h * 0.7, h),
                        'vx': random.uniform(-0.5, 0.5), 'vy': random.uniform(-1.6, -0.4),
                        'r': random.uniform(3.0, 6.0) * sz_mult,
                        'life': int(random.uniform(50, 90) * life_mult), 'max_life': 90,
                        'phase': random.uniform(0, math.pi * 2), 'type': 'smoke', 'extra': random.uniform(0.04, 0.09)
                    })
                elif p_type in ('drift_sparks', 'phonk', 'sparks'):
                    burst_x = random.uniform(w * 0.2, w * 0.8)
                    burst_y = random.uniform(h * 0.55, h * 0.88)
                    angle = random.uniform(-math.pi * 0.88, -math.pi * 0.12)
                    speed = random.uniform(6.0, 15.0)
                    self.particles.append({
                        'x': burst_x, 'y': burst_y,
                        'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                        'r': random.uniform(2.5, 5.0) * sz_mult,
                        'life': int(random.uniform(20, 45) * life_mult), 'max_life': 45,
                        'phase': random.uniform(0, math.pi * 2), 'type': 'drift_sparks', 'extra': 1.0
                    })
                elif p_type in ('golden_dust', 'afro', 'gold_dust'):
                    self.particles.append({
                        'x': random.uniform(w * 0.1, w * 0.9), 'y': random.uniform(h * 0.65, h * 0.95),
                        'vx': random.uniform(-0.8, 0.8), 'vy': -random.uniform(1.0, 2.5),
                        'r': random.uniform(3.0, 6.0) * sz_mult,
                        'life': int(random.uniform(80, 150) * life_mult), 'max_life': 150,
                        'phase': random.uniform(0, math.pi * 2), 'type': 'golden_dust', 'extra': 0.5
                    })
                else:
                    self.particles.append({
                        'x': random.uniform(w * 0.2, w * 0.8), 'y': random.uniform(h * 0.5, h * 0.85),
                        'vx': random.uniform(-2.5, 2.5), 'vy': -random.uniform(1.5, 3.5),
                        'r': random.uniform(3.5, 7.5) * sz_mult,
                        'life': int(random.uniform(50, 100) * life_mult), 'max_life': 100,
                        'phase': random.uniform(0, math.pi * 2), 'type': p_type, 'extra': 0.0
                    })

        # ── 3. UPDATE FISIKA & DRAWING SOFT GLOW ──
        alive = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] <= 0:
                continue

            age = p['max_life'] - p['life']
            fade = min(1.0, min(age / 18.0, p['life'] / 22.0))
            ptype = p.get('type', 'embers')

            # ⚡ DRIFT SPARKS (PHONK): Percikan tajam cepat, velocity stretch line, meletup keras
            if ptype == 'drift_sparks':
                p['vx'] *= 0.97
                p['vy'] += 0.08 * spd
                p['x'] += p['vx'] * spd
                p['y'] += p['vy'] * spd
                flicker = 0.7 + 0.3 * math.sin(age * 0.4 + p['phase'])
                eff_a = part_alpha * fade * flicker * (1.0 + beat_boost * 0.6)
                eff_r = p['r'] * (1.0 + beat_boost * 0.4)
                sx, sy = int(p['x']), int(p['y'])
                tail_len = 1.8 if p.get('extra', 0) > 0 else 1.2
                ex = int(sx - p['vx'] * tail_len * spd)
                ey = int(sy - p['vy'] * tail_len * spd)
                self._draw_glow(frame, sx, sy, eff_r * 2.0, self.col_part, eff_a * 0.8)
                if 0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h:
                    cv2.line(frame, (sx, sy), (ex, ey), pcol(self.col_part, eff_a), max(1, int(eff_r * 0.6)), cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), max(1, int(eff_r * 0.4)), (255, 255, 255), -1)
                if -40 <= p['x'] <= w + 40 and -40 <= p['y'] <= h + 40:
                    alive.append(p)

            # ✨ GOLDEN DUST (AFRO): Butiran debu hangat berayun harmonis (dual sine) & pendaran lembut
            elif ptype == 'golden_dust':
                p['y'] += p['vy'] * spd
                sway = math.sin(age * 0.035 + p['phase']) * 1.5 + math.cos(age * 0.08 + p['phase']) * 0.7
                p['x'] += p['vx'] * spd + sway * spd
                twinkle = 0.5 + 0.5 * math.sin(age * 0.09 + p['phase'])
                eff_a = part_alpha * fade * twinkle * (1.0 + beat_boost * 0.5)
                eff_r = p['r'] * (1.0 + beat_boost * 0.3)
                self._draw_glow(frame, p['x'], p['y'], eff_r * 2.4, self.col_part, eff_a * 0.85)
                core_r = max(1, int(eff_r * 0.45))
                cv2.circle(frame, (int(p['x']), int(p['y'])), core_r, (255, 255, 255), -1)
                if p['y'] > -40 and -30 <= p['x'] <= w + 30:
                    alive.append(p)

            # 🌟 EMBERS / FIREFLIES: Melayang naik, berayun lembut gelombang sinus, soft glow + core
            elif ptype == 'embers':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.045 + p['phase']) * 1.5 * spd
                twinkle = 0.45 + 0.55 * math.sin(age * 0.12 + p['phase'])
                eff_a = part_alpha * fade * twinkle * (1.0 + beat_boost)
                eff_r = p['r'] * (1.0 + beat_boost * 0.4)
                self._draw_glow(frame, p['x'], p['y'], eff_r * 2.2, self.col_part, eff_a * 0.8)
                core_r = max(1, int(eff_r * 0.4))
                cv2.circle(frame, (int(p['x']), int(p['y'])), core_r, (255, 255, 255), -1)
                if p['y'] > -40 and -20 <= p['x'] <= w + 20:
                    alive.append(p)

            # ✨ TWINKLE: Bintang berkilau 4-sudut diamond cross + soft glow
            elif ptype == 'twinkle':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.03 + p['phase']) * 0.8
                twinkle = abs(math.sin(age * 0.10 + p['phase']))
                eff_a = part_alpha * fade * twinkle * (1.0 + beat_boost)
                eff_r = p['r'] * (1.0 + beat_boost * 0.5)
                self._draw_glow(frame, p['x'], p['y'], eff_r * 2.0, self.col_part, eff_a * 0.7)
                cx, cy = int(p['x']), int(p['y'])
                cr = int(eff_r * 1.8)
                if cr > 2 and 0 <= cx < w and 0 <= cy < h:
                    cv2.line(frame, (cx - cr, cy), (cx + cr, cy), pcol(self.col_part, eff_a), 1, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy - cr), (cx, cy + cr), pcol(self.col_part, eff_a), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), max(1, int(eff_r * 0.35)), (255, 255, 255), -1)
                if -20 <= p['x'] <= w + 20 and -20 <= p['y'] <= h + 20:
                    alive.append(p)

            # 🔮 BOKEH: Bulatan cahaya besar transparan & melayang lambat (kedalaman sinematik)
            elif ptype == 'bokeh':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.02 + p['phase']) * 0.6
                eff_a = part_alpha * fade * 0.45 * (1.0 + beat_boost * 0.3)
                self._draw_glow(frame, p['x'], p['y'], p['r'] * 1.8, self.col_part, eff_a)
                if p['y'] > -60 and -40 <= p['x'] <= w + 40:
                    alive.append(p)

            # ❄️ SNOW: Melayang turun perlahan dengan goyangan angin halus & pendaran lembut
            elif ptype == 'snow':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.07 + p['phase']) * 0.9
                eff_a = part_alpha * fade * 0.85
                self._draw_glow(frame, p['x'], p['y'], p['r'] * 1.8, (255, 255, 255), eff_a * 0.5)
                cv2.circle(frame, (int(p['x']), int(p['y'])), max(1, int(p['r'] * 0.7)), (255, 255, 255), -1)
                if p['y'] < h + 20 and -20 <= p['x'] <= w + 20:
                    alive.append(p)

            # ❅ PETALS: Kelopak bunga/bintang melayang anggun dengan rotasi
            elif ptype == 'petals':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.05 + p['phase']) * 1.3
                p['extra'] += 0.06
                eff_a = part_alpha * fade
                self._draw_star(frame, int(p['x']), int(p['y']), max(1, int(p['r'])), p['extra'], pcol(self.col_part, eff_a))
                if p['y'] < h + 30 and -30 <= p['x'] <= w + 30:
                    alive.append(p)

            # 🌧️ RAIN: Rintik hujan garis tipis miring & cepat
            elif ptype == 'rain':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd
                rx, ry = int(p['x']), int(p['y'])
                cv2.line(frame, (rx, ry), (int(rx - p['vx'] * 1.8), int(ry - p['vy'] * 1.2)), pcol(self.col_part, fade * 0.8), 1, cv2.LINE_AA)
                if p['y'] < h + 30 and -30 <= p['x'] <= w + 30:
                    alive.append(p)

            # 🫧 BUBBLES: Gelembung transparan naik & bergoyang dengan pantulan cahaya
            elif ptype == 'bubbles':
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.06 + p['phase']) * 0.7
                bx, by, br = int(p['x']), int(p['y']), max(2, int(p['r']))
                cv2.circle(frame, (bx, by), br, pcol(self.col_part, fade * 0.7), 1, cv2.LINE_AA)
                hs = max(1, int(br * 0.3))
                cv2.circle(frame, (bx - int(br * 0.3), by - int(br * 0.3)), hs, (255, 255, 255), -1)
                if p['y'] > -40 and -20 <= p['x'] <= w + 20:
                    alive.append(p)

            # 🎆 FIREWORKS: Letupan radial melengkung dengan gravitasi
            elif ptype == 'fireworks':
                p['vy'] += 0.22 * spd
                p['x'] += p['vx'] * spd
                p['y'] += p['vy'] * spd
                eff_a = part_alpha * fade
                self._draw_glow(frame, p['x'], p['y'], p['r'] * 1.6, self.col_part, eff_a * 0.7)
                cv2.circle(frame, (int(p['x']), int(p['y'])), max(1, int(p['r'] * 0.5)), (255, 255, 255), -1)
                if p['y'] < h + 30 and -30 <= p['x'] <= w + 30:
                    alive.append(p)

            # ☄ TRAIL: Komet meluncur dengan ekor cahaya
            elif ptype == 'trail':
                p['x'] += p['vx'] * spd
                p['y'] += p['vy'] * spd
                tx, ty = int(p['x']), int(p['y'])
                cv2.line(frame, (tx, ty), (int(tx - p['vx'] * 3.5), int(ty - p['vy'] * 3.5)), pcol(self.col_part, fade * 0.6), 1, cv2.LINE_AA)
                self._draw_glow(frame, tx, ty, p['r'] * 1.8, self.col_top, part_alpha * fade * 0.8)
                cv2.circle(frame, (tx, ty), max(1, int(p['r'] * 0.4)), (255, 255, 255), -1)
                if p['y'] < h + 40 and -40 <= p['x'] <= w + 40:
                    alive.append(p)

            # 🌫️ SMOKE: Gumpalan asap halus mengembang
            elif ptype == 'smoke':
                p['r'] += (p.get('extra', 0.05)) * spd * 0.5
                p['y'] += p['vy'] * spd
                p['x'] += p['vx'] * spd + math.sin(age * 0.04 + p['phase']) * 0.4
                eff_a = part_alpha * fade * 0.35
                self._draw_glow(frame, p['x'], p['y'], p['r'] * 2.0, (200, 200, 200), eff_a)
                if p['y'] > -50 and p['r'] < 90:
                    alive.append(p)

            # ✦ SPARKLE: Partikel berkilau klasik dengan soft glow
            else:
                p['x'] += p['vx'] * spd
                p['y'] += p['vy'] * spd
                twinkle = 0.5 + 0.5 * math.sin(age * 0.15 + p['phase'])
                eff_a = part_alpha * fade * twinkle
                self._draw_glow(frame, p['x'], p['y'], p['r'] * 1.8, self.col_part, eff_a * 0.8)
                cv2.circle(frame, (int(p['x']), int(p['y'])), max(1, int(p['r'] * 0.35)), (255, 255, 255), -1)
                if -20 <= p['x'] <= w + 20 and -20 <= p['y'] <= h + 20:
                    alive.append(p)

        self.particles = alive

    # ── helper: gambar bintang 5-kelopak untuk petals ──
    def _draw_star(self, frame, cx, cy, r, rotation, color):
        pts = []
        for i in range(10):
            angle = rotation + i * (math.pi / 5) - math.pi / 2
            rad = r if i % 2 == 0 else r * 0.45
            pts.append([int(cx + rad * math.cos(angle)), int(cy + rad * math.sin(angle))])
        cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], color)

def hex_to_rgb(h): return tuple(int(str(h).lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def render_video_core(task_id, audio_path, bg_paths, output_path, duration, cfg):
    # 🎨 Resolusi dinamis: baca dari cfg ('1080' atau '720'), fallback 720.
    _res = str(cfg.get('resolution', '720'))
    if _res == '1080':
        w, h = 1920, 1080
    else:
        w, h = 1280, 720
    fps = 30; total_f = int(duration * fps)
    c_bot = hex_to_rgb(cfg.get('color_bot', '#10b981'))
    c_top = hex_to_rgb(cfg.get('color_top', '#0ea5e9'))
    c_part = hex_to_rgb(cfg.get('color_part', '#ffffff'))
    bar_c = int(cfg.get('bar_count', 64))
    vis = VisualEngine(c_bot, c_top, c_part)
    bg = BackgroundManager(bg_paths, w, h)
    audio = AudioBrain(); audio.load(audio_path)
    _wm_state_ctx = {}
    # 🎨 Resolve font TTF sekali (cross-platform: Linux DejaVu / Windows Arial).
    # Dipakai untuk tracklist, watermark, timestamp agar mirip preview (font sistem).
    _ttf_bold = _resolve_ttf('M')
    _ttf_reg = _resolve_ttf('S')
    
    with db_lock:
        for d in active_tasks:
            if d['id'] == task_id: d['status'] = "Rendering Visual & Background... ⚡"
    save_tasks_db()

    render_preset = str(cfg.get('render_speed', 'ultrafast')).lower()
    if render_preset not in ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'):
        render_preset = 'ultrafast'

    ff_threads = str(max(1, min(4, os.cpu_count() or 2)))
    cmd = [
        get_ffmpeg_path(), '-y', '-threads', ff_threads, 
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps), 
        '-i', '-', 
        '-i', audio_path, 
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', render_preset, '-crf', '22', '-pix_fmt', 'yuv420p', output_path
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        for f in range(total_f):
            if stop_flags.get(task_id):
                raise Exception("Dibatalkan")
                
            v, is_hit, bars = audio.get_data(f/fps, bar_c)
            frame = vis.process(bg.get_frame(), v, is_hit, bars, cfg)

            # ── TRANSISI FADE IN / FADE OUT ──
            try:
                fade_dur = float(cfg.get('fade_duration', 0) or 0)
            except (ValueError, TypeError):
                fade_dur = 0.0
            if fade_dur > 0:
                fade_frames = int(fade_dur * fps)
                # fade in (awal video: dari hitam ke normal)
                if f < fade_frames:
                    alpha = f / fade_frames
                    frame = (frame * alpha).astype(np.uint8)
                # fade out (akhir video: dari normal ke hitam)
                if f >= total_f - fade_frames:
                    alpha = (total_f - f) / fade_frames
                    frame = (frame * alpha).astype(np.uint8)

            if cfg.get('use_floating_card', False) and 'track_schedule' in cfg:
                sec = f / fps
                current_track = None
                for track in cfg['track_schedule']:
                    if track['start'] <= sec < track['end']:
                        current_track = track
                        break
                
                if current_track:
                    t = sec - current_track['start'] 
                    if t < 10.0:
                        alpha = (t * 0.85) if t < 1.0 else ((10.0 - t) * 0.85 if t > 9.0 else 0.85)
                        if alpha > 0.05:
                            cw, ch = 500, 100
                            x, y = 40, h - ch - 40
                            roi = frame[y:y+ch, x:x+cw]
                            overlay = roi.copy()
                            cv2.rectangle(overlay, (0, 0), (cw, ch), (30, 20, 15), -1)
                            cv2.rectangle(overlay, (15, 15), (85, 85), (60, 200, 80), -1)
                            card_title = current_track['title']
                            ch_name = cfg.get('channel_name', 'KeiBot FM')
                            _draw_text(overlay, card_title[:35], (105, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, ttf_path=_ttf_bold, pil_size=22)
                            _draw_text(overlay, f"Now Playing . {ch_name}", (105, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA, ttf_path=_ttf_reg, pil_size=16)
                            _draw_text(overlay, "J", (36, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA, ttf_path=_ttf_bold, pil_size=48)
                            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

            # ── TRACK LIST (daftar lagu di dalam video) ──
            use_tl = cfg.get('use_tracklist', False)
            has_ts = 'track_schedule' in cfg
            if use_tl and has_ts:
                sec = f / fps
                cur_idx = -1
                for idx, track in enumerate(cfg['track_schedule']):
                    if track['start'] <= sec < track['end']:
                        cur_idx = idx
                        break
                if cur_idx < 0:
                    # cari track terakhir jika sudah lewat semua
                    if cfg['track_schedule'] and sec >= cfg['track_schedule'][-1]['end']:
                        cur_idx = len(cfg['track_schedule']) - 1

                tracks = cfg['track_schedule']
                tl_pos = str(cfg.get('tl_position', 'tr'))
                tl_size = str(cfg.get('tl_size', 'medium'))
                tl_bg = str(cfg.get('tl_bg', 'dark'))
                tl_font = str(cfg.get('tl_font', 'M'))
                tl_color = str(cfg.get('tl_color', '#ffffff'))

                # size config
                if tl_size == 'large':
                    item_h = 32; list_w = 320; font_s = 0.5; header_s = 0.5
                elif tl_size == 'small':
                    item_h = 22; list_w = 220; font_s = 0.35; header_s = 0.4
                else:  # medium
                    item_h = 28; list_w = 280; font_s = 0.4; header_s = 0.45

                max_show = min(len(tracks), 10)
                pad = 10
                list_h = max_show * item_h + pad * 2 + 20

                # position
                margin = 20
                if tl_pos == 'tl': list_x, list_y = margin, margin
                elif tl_pos == 'bl': list_x, list_y = margin, h - list_h - margin
                elif tl_pos == 'br': list_x, list_y = w - list_w - margin, h - list_h - margin
                elif tl_pos == 'cl': list_x, list_y = margin, (h - list_h) // 2
                elif tl_pos == 'cr': list_x, list_y = w - list_w - margin, (h - list_h) // 2
                else: list_x, list_y = w - list_w - margin, margin  # tr default

                # font mapping
                tl_font_map = {
                    'M': cv2.FONT_HERSHEY_DUPLEX, 'S': cv2.FONT_HERSHEY_SIMPLEX,
                    'I': cv2.FONT_HERSHEY_TRIPLEX, 'C': cv2.FONT_HERSHEY_PLAIN,
                }
                tfont = tl_font_map.get(tl_font, cv2.FONT_HERSHEY_SIMPLEX)

                # background style
                overlay_list = np.zeros((list_h, list_w, 3), dtype=np.uint8)
                if tl_bg == 'glass':
                    cv2.rectangle(overlay_list, (0, 0), (list_w, list_h), (20, 15, 15), -1)
                    cv2.rectangle(overlay_list, (0, 0), (list_w, list_h), (80, 70, 70), 1)
                    blend_alpha = 0.55
                elif tl_bg == 'minimal':
                    cv2.rectangle(overlay_list, (0, 0), (list_w, list_h), (0, 0, 0), -1)
                    cv2.rectangle(overlay_list, (0, 0), (list_w, list_h), (50, 50, 50), 1)
                    blend_alpha = 0.40
                elif tl_bg == 'transparent':
                    blend_alpha = 0.0  # langsung gambar ke frame
                else:  # dark
                    cv2.rectangle(overlay_list, (0, 0), (list_w, list_h), (15, 10, 10), -1)
                    cv2.rectangle(overlay_list, (0, 0), (list_w, list_h), (60, 50, 50), 1)
                    blend_alpha = 0.82

                # header
                _draw_text(overlay_list, str(cfg.get('tl_title', 'Playlist')), (pad, pad + 12), tfont, header_s, hex_to_rgb(tl_color)[::-1], 1, cv2.LINE_AA, ttf_path=_ttf_bold, pil_size=16)

                start_idx = max(0, cur_idx - 4)
                shown = 0
                for i in range(start_idx, min(len(tracks), start_idx + max_show)):
                    tr = tracks[i]
                    y0 = pad + 14 + shown * item_h
                    is_active = (i == cur_idx)

                    title_str = tr['title'][:35]
                    text_color = (255, 255, 255) if is_active else hex_to_rgb(tl_color)[::-1]
                    # 🐛 FIX: karakter Unicode ▶ (U+25B6) tidak bisa dirender OpenCV
                    # (Hershey fonts) -> muncul sebagai "??". Sebagai gantinya kita
                    # gambar segitiga play manual dengan cv2.fillPoly untuk track aktif.
                    num_x = pad + 4
                    num_y0 = y0
                    if is_active:
                        cv2.rectangle(overlay_list, (pad, y0), (list_w - pad, y0 + item_h - 2), (70, 140, 60), -1)
                        # segitiga play kecil di kiri nomor
                        tcx = pad + 6
                        tcy = y0 + item_h // 2 - 1
                        tri = np.array([[tcx, tcy - 5], [tcx, tcy + 5], [tcx + 8, tcy]], np.int32)
                        cv2.fillPoly(overlay_list, [tri], (255, 255, 255))
                        num_x = pad + 20
                    _draw_text(overlay_list, f"{i+1}.", (num_x, y0 + 14), tfont, font_s, text_color, 1, cv2.LINE_AA, ttf_path=_ttf_reg, pil_size=13)
                    # animasi active track
                    t_y = y0
                    t_x = pad + 46
                    tl_anim = str(cfg.get('tl_active', 'none'))
                    skip_draw = False
                    if is_active:
                        if tl_anim == 'blink' and int(f / fps * 2) % 2 == 0:
                            skip_draw = True
                        elif tl_anim == 'float':
                            t_y = y0 + int(math.sin(f / fps * 4) * 3)
                        elif tl_anim == 'scroll':
                            scroll_off = int((f / fps * 40) % (len(title_str) * 12))
                            t_x = pad + 46 - scroll_off
                    if not skip_draw:
                        _draw_text(overlay_list, title_str, (t_x, t_y + 14), tfont, font_s, text_color, 1, cv2.LINE_AA, ttf_path=_ttf_reg, pil_size=13)
                    shown += 1

                # blend ke frame
                roi_list = frame[list_y:list_y + list_h, list_x:list_x + list_w]
                if blend_alpha > 0:
                    cv2.addWeighted(overlay_list, blend_alpha, roi_list, 1.0 - blend_alpha, 0, roi_list)
                else:
                    # transparent: hanya salin pixel teks/border yang berwarna,
                    # biarkan background video tetap terlihat (overlay_list berisi nol
                    # kecuali area teks/aksen → gunakan mask "bukan hitam murni")
                    mask = cv2.cvtColor(overlay_list, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                    roi_list[mask > 0] = overlay_list[mask > 0]

            # ── WATERMARK TEKS ──
            if cfg.get('use_watermark', False):
                wm_text = str(cfg.get('wm_text', ''))
                if wm_text:
                    wm_color_hex = cfg.get('wm_color', '#ffffff')
                    wm_color = hex_to_rgb(wm_color_hex)
                    wm_color_bgr = (wm_color[2], wm_color[1], wm_color[0])  # BGR untuk cv2
                    wm_size = int(cfg.get('wm_size', 24))
                    wm_pos = cfg.get('wm_position', 'bl')
                    wm_move = cfg.get('wm_move', 'none')
                    wm_font = cfg.get('wm_font', 'M')
                    wm_speed = cfg.get('wm_speed', 'medium')

                    # font mapping
                    font_map = {
                        'M': cv2.FONT_HERSHEY_DUPLEX,
                        'S': cv2.FONT_HERSHEY_SIMPLEX,
                        'I': cv2.FONT_HERSHEY_TRIPLEX,
                        'C': cv2.FONT_HERSHEY_PLAIN,
                    }
                    font = font_map.get(wm_font, cv2.FONT_HERSHEY_SIMPLEX)
                    thickness = max(1, wm_size // 14)

                    # ukuran teks untuk posisi (pakai TTF jika PIL tersedia)
                    tw, th = _get_text_size(wm_text, _ttf_bold, int(wm_size), font, wm_size * 0.06, thickness)
                    margin = 30

                    # posisi dasar
                    base_positions = {
                        'tl': (margin, margin + th),
                        'tr': (w - tw - margin, margin + th),
                        'bl': (margin, h - margin),
                        'br': (w - tw - margin, h - margin),
                        'center': (w//2 - tw//2, h//2 + th//2),
                    }
                    bx, by = base_positions.get(wm_pos, (margin, h - margin))

                    # movement
                    frame_sec = f / fps
                    if wm_move == 'float':
                        offset = math.sin(frame_sec * 1.5) * 10
                        by += int(offset)
                    elif wm_move == 'scroll':
                        scroll_range = w + tw + margin * 2
                        offset = ((frame_sec * 40) % scroll_range) - tw - margin
                        bx = int(offset)
                    elif wm_move == 'pulse':
                        pulse = 0.7 + 0.3 * abs(math.sin(frame_sec * 2.5))
                    elif wm_move == 'random_walk':
                        # random walk kontinu
                        if not _wm_state_ctx:
                            _wm_state_ctx['x'] = float(bx)
                            _wm_state_ctx['y'] = float(by)
                            _wm_state_ctx['dx'] = 1.5
                            _wm_state_ctx['dy'] = 1.2
                        ws = _wm_state_ctx
                        wm_spd_mult = {'slow': 0.5, 'fast': 2.5}.get(wm_speed, 1.2)
                        # update arah gradual
                        if random.random() < 0.015:
                            a = random.random() * 2 * math.pi
                            ws['dx'] += (math.cos(a) * wm_spd_mult - ws['dx']) * 0.1
                            ws['dy'] += (math.sin(a) * wm_spd_mult - ws['dy']) * 0.1
                        ws['x'] += ws['dx']; ws['y'] += ws['dy']
                        # pantul di tepi
                        for side, limit, key in [(margin, w - tw - margin, 'x'), (margin + th, h - margin, 'y')]:
                            if ws[key] < side: ws[key] = side; (ws['dx'], ws['dy']) = (-ws['dx'], ws['dy']) if key == 'x' else (ws['dx'], -ws['dy'])
                            if ws[key] > limit: ws[key] = limit; (ws['dx'], ws['dy']) = (-ws['dx'], ws['dy']) if key == 'x' else (ws['dx'], -ws['dy'])
                        bx = int(ws['x']); by = int(ws['y'])
                        pulse = 1.0
                    else:
                        pulse = 1.0

                    # shadow
                    shadow_color = (0, 0, 0)
                    _draw_text(frame, wm_text, (bx+2, by+2), font, wm_size * 0.06, shadow_color, thickness, cv2.LINE_AA, ttf_path=_ttf_bold, pil_size=int(wm_size))
                    # draw text
                    if wm_move == 'pulse':
                        _draw_text(frame, wm_text, (bx, by), font, wm_size * 0.06 * pulse, wm_color_bgr, thickness, cv2.LINE_AA, ttf_path=_ttf_bold, pil_size=int(wm_size * pulse))
                    else:
                        _draw_text(frame, wm_text, (bx, by), font, wm_size * 0.06, wm_color_bgr, thickness, cv2.LINE_AA, ttf_path=_ttf_bold, pil_size=int(wm_size))

            # ── TIMESTAMP ──
            if cfg.get('use_timestamp', False):
                ts_size = int(cfg.get('ts_size', 20))
                ts_pos = cfg.get('ts_pos', 'bl')
                ts_font_name = cfg.get('ts_font', 'M')
                ts_color_hex = cfg.get('ts_color', '#ffffff')
                ts_offx = safe_float(cfg.get('ts_offx', 50), 50.0) / 100.0
                ts_offy = safe_float(cfg.get('ts_offy', 90), 90.0) / 100.0
                ts_color = hex_to_rgb(ts_color_hex)
                ts_color_bgr = (ts_color[2], ts_color[1], ts_color[0])
                ts_font_map = {'M': cv2.FONT_HERSHEY_DUPLEX, 'S': cv2.FONT_HERSHEY_SIMPLEX, 'I': cv2.FONT_HERSHEY_TRIPLEX, 
                               'C': cv2.FONT_HERSHEY_PLAIN, 'D': cv2.FONT_HERSHEY_DUPLEX, 'O': cv2.FONT_HERSHEY_PLAIN}
                ts_font = ts_font_map.get(ts_font_name, cv2.FONT_HERSHEY_SIMPLEX)
                ts_margin = 20
                total_sec_int = int(f / fps)
                ts_text = f"{total_sec_int//3600}:{total_sec_int%3600//60:02d}:{total_sec_int%60:02d}"
                tw, th = _get_text_size(ts_text, _ttf_reg, ts_size, ts_font, ts_size * 0.05, 1)
                # posisi
                base_tx = {'tl': ts_margin, 'bl': ts_margin, 'ct': int(w * ts_offx), 'cb': int(w * ts_offx)}
                base_ty = {'tl': ts_margin + th, 'ct': ts_margin + th, 'bl': h - ts_margin, 'cb': h - ts_margin}
                tx = base_tx.get(ts_pos, ts_margin)
                ty = base_ty.get(ts_pos, h - ts_margin)
                if ts_pos in ('tr', 'br'): tx = w - tw - ts_margin
                if ts_pos in ('br',): ty = h - ts_margin
                _draw_text(frame, ts_text, (tx+2, ty+2), ts_font, ts_size * 0.05, (0, 0, 0), 1, cv2.LINE_AA, ttf_path=_ttf_reg, pil_size=ts_size)
                _draw_text(frame, ts_text, (tx, ty), ts_font, ts_size * 0.05, ts_color_bgr, 1, cv2.LINE_AA, ttf_path=_ttf_reg, pil_size=ts_size)

            proc.stdin.write(frame.tobytes())
            
    except Exception as e:
        try: proc.stdin.close()
        except: pass
        try: proc.terminate()
        except: pass
        bg.close()
        try:
            del audio; del bg; del vis
        except: pass
        gc.collect()
        raise e
        
    try: proc.stdin.close()
    except: pass
    proc.wait()
    bg.close()
    try:
        del audio; del bg; del vis
    except: pass
    gc.collect()

# ==========================================
# 🚀 BACKGROUND WORKER: OTO-LOOP ULTIMATE
# ==========================================
def background_worker():
    global channel_cooldowns
    
    while True:
        task = render_queue.get()
        task_id = task['id']
        yt_id = task['yt_id']
        
        # Cek jika task sudah dibatalkan user sebelumnya saat masih di antrean
        if stop_flags.get(task_id):
            stop_flags.pop(task_id, None)
            render_queue.task_done()
            continue
        
        # 🔥 SMART COOLDOWN SYSTEM (PUTAR BALIK ANTREAN) 🔥
        if yt_id in channel_cooldowns:
            if time.time() < channel_cooldowns[yt_id]:
                sisa_menit = max(1, int((channel_cooldowns[yt_id] - time.time()) / 60))
                
                with db_lock:
                    for d in active_tasks:
                        if d['id'] == task_id:
                            d['status'] = f"Antrean Ditunda (Cooldown YT {sisa_menit} mnt) ⏳"
                save_tasks_db()
                
                render_queue.put(task)
                render_queue.task_done()
                time.sleep(5) 
                continue
            else:
                del channel_cooldowns[yt_id]
                
        temp_files = [
            os.path.join(BASE_UPLOAD, f"temp_a_{task_id}.mp3"),
            os.path.join(BASE_UPLOAD, f"temp_c_{task_id}.txt"),
            os.path.join(BASE_UPLOAD, f"temp_v_{task_id}.mp4"),
            os.path.join(BASE_UPLOAD, f"loop_{task_id}.txt"),
            os.path.join(BASE_DIR, f"static/final_{task_id}.mp4"),
        ]
        try:
            if not wait_for_resources(task_id): 
                raise Exception("Dibatalkan")
                
            task_start_ts = time.time()
            now_str = dt.datetime.now().strftime('%H:%M:%S')
            with db_lock:
                for d in active_tasks:
                    if d['id'] == task_id:
                        d['status'] = "Meracik Aset Gallery... ⚙️"
                        d['start_ts'] = task_start_ts
                        d['render_start'] = now_str
            save_tasks_db()

            audio_paths = get_all_audios(yt_id)
            if not audio_paths: raise Exception("Gallery Audio Kosong!")
            
            mp3_req = int(task.get('mp3_per_video', 5))
            
            # Filter audio yang sehat & valid (skip jika file 0-byte, corrupt, atau durasi 0)
            valid_audios = []
            for ap in audio_paths:
                if len(valid_audios) >= mp3_req:
                    break
                if not os.path.exists(ap) or os.path.getsize(ap) < 1024:
                    print(f"[KeiBot] Skip audio kosong/rusak (<1KB): {ap}")
                    continue
                probe = subprocess.run([get_ffprobe_path(), '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', ap], capture_output=True, text=True)
                try: dur = float(probe.stdout.strip())
                except: dur = 0.0
                if dur <= 0.0:
                    print(f"[KeiBot] Skip audio durasi tidak valid: {ap}")
                    continue
                title = os.path.splitext(os.path.basename(ap))[0]
                valid_audios.append((ap, dur, title))

            if not valid_audios:
                raise Exception("Tidak ada file audio yang valid di Gallery!")

            track_schedule = []
            current_sec = 0.0

            base_audio = os.path.join(BASE_UPLOAD, f"temp_a_{task_id}.mp3")
            c_txt = os.path.join(BASE_UPLOAD, f"temp_c_{task_id}.txt")
            with open(c_txt, 'w', encoding='utf-8') as f:
                for ap, dur, title in valid_audios:
                    safe_path = os.path.abspath(ap).replace('\\', '/')
                    # Escape karakter single quote agar sintaks concat FFmpeg tidak patah
                    safe_path_concat = safe_path.replace("'", "'\\''")
                    f.write(f"file '{safe_path_concat}'\n")
                    
                    track_schedule.append({
                        'title': title,
                        'path': safe_path, 
                        'start': current_sec,
                        'end': current_sec + dur,
                        'duration': dur
                    })
                    current_sec += dur

            try:
                subprocess.run([get_ffmpeg_path(), '-y', '-threads', '2', '-f', 'concat', '-safe', '0', '-i', c_txt, '-c:a', 'libmp3lame', '-q:a', '2', base_audio], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                err_text = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
                last_err = [l.strip() for l in err_text.splitlines() if l.strip()][-3:]
                raise Exception(f"Concat Audio Gagal: {' | '.join(last_err)}")

            probe = subprocess.run([
                get_ffprobe_path(), '-v', 'error', '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', base_audio
            ], capture_output=True, text=True, check=True)
            base_duration_sec = float(probe.stdout.strip())
            
            if base_duration_sec <= 0: raise Exception("Durasi audio tidak valid!")

            channel_data = next((c for c in database_channel if c['yt_id'] == yt_id), None)
            ch_name = channel_data['name'] if channel_data else "KeiBot FM"

            bg_count = int(task.get('bg_count', 1))
            bg_paths = get_multi_backgrounds(yt_id, count=bg_count)
            if not bg_paths: raise Exception("Gallery Background Kosong!")

            preset = task.get('vis_preset')
            allowed_presets = task.get('vis_presets_allowed', [])
            vis_mode = task.get('vis_mode')
            if vis_mode == 'random' or preset == 'random':
                preset = get_random_preset(allowed_presets)
            elif vis_mode == 'smart':
                smart_preset = get_smart_preset(base_audio)
                if smart_preset:
                    preset = smart_preset
            if not isinstance(preset, dict):
                preset = {"color_bot": "#00d4ff", "color_top": "#7c5cfc", "color_part": "#ffffff", "pos_x": 50, "pos_y": 85, "width_pct": 60, "max_height": 40, "idle_height": 5, "bar_count": 64, "reactivity": 0.66, "spacing": 3, "part_amount": 3, "part_speed": 1.0, "effect_type": "spectrum", "use_beat_pulse": False, "particle_type": "sparkle", "fade_duration": 0, "use_watermark": False, "wm_text": "", "wm_color": "#ffffff", "wm_font": "M", "wm_size": 24, "wm_position": "bl", "wm_move": "none", "use_tracklist": False, "tl_font": "M", "tl_size": "medium", "tl_position": "tr", "tl_bg": "dark", "tl_title": "PLAYLIST", "tl_color": "#ffffff", "tl_active": "none", "use_timestamp": False, "ts_pos": "bl", "ts_font": "M", "ts_size": 20, "ts_color": "#ffffff", "ts_offx": 50, "ts_offy": 90}

            user_cfg = task.get('vis_config') if isinstance(task.get('vis_config'), dict) else {}
            preset['yt_id'] = yt_id
            preset['use_floating_card'] = task.get('use_floating_card', False)
            preset['use_tracklist'] = task.get('use_tracklist', preset.get('use_tracklist', False))
            preset['use_watermark'] = task.get('use_watermark', preset.get('use_watermark', False))
            preset['use_timestamp'] = task.get('use_timestamp', preset.get('use_timestamp', False))
            preset['jj_mode'] = task.get('jj_mode', user_cfg.get('jj_mode', preset.get('jj_mode', 'off')))
            preset['flash_mode'] = task.get('flash_mode', user_cfg.get('flash_mode', preset.get('flash_mode', 'off')))

            # 🐛 FIX: di mode Random/Smart, preset dibangun ulang sehingga styling
            # tracklist/watermark/timestamp & pengaturan partikel yang dipilih user HILANG.
            # Karena itu kita selalu mengirim `vis_config` penuh dari frontend dan di sini
            # kita merge field styling-nya kembali ke preset agar hasil render sesuai preview.
            if vis_mode in ('random', 'smart'):
                # 🐛 FIX: di mode Random/Smart, preset dibangun ulang sehingga SEMUA
                # pengaturan spectrum user (tinggi/posisi/lebar/jumlah bar/warna/dll)
                # HILANG dan diganti nilai random. Frontend selalu mengirim `vis_config`
                # penuh, jadi kita merge kembali field spectrum inti + styling overlay
                # agar hasil render benar-benar sesuai preview live.
                for k in (
                    # ── spectrum dasar (tinggi, posisi, lebar, bar, reaktif, spasi) ──
                    'max_height', 'pos_x', 'pos_y', 'width_pct', 'idle_height',
                    'bar_count', 'reactivity', 'spacing', 'effect_type', 'bar_style',
                    'use_beat_pulse', 'part_amount', 'part_speed', 'smoothing',
                    'color_bot', 'color_top', 'color_part',
                    # ── tracklist ──
                    'use_tracklist', 'tl_font', 'tl_size', 'tl_position', 'tl_bg',
                    'tl_title', 'tl_color', 'tl_active',
                    # ── watermark ──
                    'use_watermark', 'wm_text', 'wm_color', 'wm_font', 'wm_size',
                    'wm_position', 'wm_move', 'wm_speed',
                    # ── timestamp ──
                    'use_timestamp', 'ts_pos', 'ts_font', 'ts_size', 'ts_color',
                    'ts_offx', 'ts_offy',
                    # ── partikel lanjutan ──
                    'particle_type', 'part_size', 'part_life', 'part_opacity',
                    'part_density', 'fade_duration', 'jj_mode', 'flash_mode'):
                    if k in user_cfg:
                        preset[k] = user_cfg[k]
            # ⏱️ TARGET DURATION & RENDER TIME OPTIMIZATION
            raw_target_duration = task.get('target_duration_hours', 1)
            is_original_duration = (str(raw_target_duration).strip().lower() in ('original', 'asli', 'auto'))
            if is_original_duration:
                target_hours = base_duration_sec / 3600.0
                target_sec = base_duration_sec
                render_duration = base_duration_sec
                loop_count = 1
            else:
                try:
                    target_hours = float(raw_target_duration)
                except (ValueError, TypeError):
                    target_hours = 1.0
                target_sec = target_hours * 3600

                # 🚀 OPTIMASI KRITIS: Jika target durasi lebih pendek dari gabungan MP3,
                # hanya render durasi yang dibutuhkan (jangan render 20 menit MP3 jika video hanya 6 menit!)
                render_duration = min(base_duration_sec, target_sec) if target_sec > 0 else base_duration_sec
                loop_count = math.ceil(target_sec / render_duration) if render_duration > 0 else 1

            # Filter tracklist schedule agar hanya memuat track yang masuk dalam durasi render
            preset['track_schedule'] = [tr for tr in track_schedule if tr['start'] < render_duration]
            preset['channel_name'] = ch_name
            preset['resolution'] = task.get('resolution', '720')
            preset['render_speed'] = task.get('render_speed', 'ultrafast')

            base_video = os.path.join(BASE_UPLOAD, f"temp_v_{task_id}.mp4")
            final_video = os.path.join(BASE_DIR, f"static/final_{task_id}.mp4")

            if stop_flags.get(task_id): raise Exception("Dibatalkan")
            with db_lock:
                for d in active_tasks:
                    if d['id'] == task_id:
                        if is_original_duration:
                            d['status'] = f"Rendering Visual Durasi Asli ({int(render_duration//60)}m {int(render_duration%60)}s)... 🎵"
                        else:
                            d['status'] = f"Rendering Visual ({int(render_duration//60)}m {int(render_duration%60)}s)... ⚡"
            save_tasks_db()

            render_video_core(task_id, base_audio, bg_paths, base_video, render_duration, preset)
            if stop_flags.get(task_id): raise Exception("Dibatalkan")

            # ── SMART CUT: potong & acak ulang chunk video ──
            smart_cut = task.get('smart_cut', False)
            cut_duration = safe_float(task.get('cut_duration', 5), 5.0)
            cut_remainder = task.get('cut_remainder', 'end')
            cut_use_remainder = task.get('cut_use_remainder', True)
            use_transition = task.get('use_transition', False)
            trans_dur = min(2.0, max(0.1, safe_float(task.get('transition_duration', 0.5), 0.5)))
            if smart_cut and cut_duration > 0 and render_duration > cut_duration:
                with db_lock:
                    for d in active_tasks:
                        if d['id'] == task_id: d['status'] = f"Smart Cut {cut_duration}s... ✂️"
                save_tasks_db()

                full_chunks = int(render_duration // cut_duration)
                remainder = render_duration - (full_chunks * cut_duration)

                # segmentasi menggunakan FFmpeg
                seg_dir = os.path.join(BASE_UPLOAD, f"seg_{task_id}")
                os.makedirs(seg_dir, exist_ok=True)
                seg_pattern = os.path.join(seg_dir, "chunk_%03d.mp4")
                subprocess.run([
                    get_ffmpeg_path(), '-y', '-i', base_video,
                    '-c', 'copy', '-map', '0',
                    '-f', 'segment', '-segment_time', str(cut_duration),
                    '-reset_timestamps', '1',
                    seg_pattern
                ], check=True, capture_output=True)

                # kumpulkan chunk files
                chunks = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.mp4')],
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))
                # potong sesuai full_chunks (abaikan chunk kelebihan)
                chunks = chunks[:full_chunks]

                if len(chunks) >= 2:
                    # 🐛 FIX: Jangan acak urutan chunk jika tracklist/floating card aktif.
                    # Shuffle akan mengacak timeline sehingga daftar lagu yang sudah
                    # di-overlay (sesuai urutan audio) menjadi kacau & tidak sinkron.
                    use_overlay_timeline = bool(task.get('use_tracklist', False)) or bool(task.get('use_floating_card', False))
                    if not use_overlay_timeline:
                        random.shuffle(chunks)

                    # sisipkan remainder (jika diaktifkan) → hanya relevan saat tidak pakai transisi xfade
                    if cut_use_remainder and remainder > 0.5 and not use_transition:
                        # ekstrak remainder dari base_video
                        rem_video = os.path.join(BASE_UPLOAD, f"rem_{task_id}.mp4")
                        subprocess.run([
                            get_ffmpeg_path(), '-y', '-i', base_video,
                            '-ss', str(full_chunks * cut_duration), '-t', str(remainder),
                            '-c', 'copy', rem_video
                        ], check=True, capture_output=True)

                        if cut_remainder == 'middle':
                            mid = len(chunks) // 2
                            chunks.insert(mid, rem_video)
                        elif cut_remainder == 'random':
                            pos = random.randint(0, len(chunks))
                            chunks.insert(pos, rem_video)
                        else:
                            chunks.append(rem_video)

                    smart_video = os.path.join(BASE_UPLOAD, f"smart_{task_id}.mp4")

                    if use_transition and trans_dur > 0:
                        # 🌊 Transisi halus antar chunk: xfade (video) + acrossfade (audio)
                        with db_lock:
                            for d in active_tasks:
                                if d['id'] == task_id: d['status'] = f"Transisi {trans_dur}s antar potongan... 🌊"
                        save_tasks_db()

                        # hitung durasi tiap chunk via ffprobe
                        chunk_durs = []
                        for ch in chunks:
                            p = subprocess.run([get_ffprobe_path(), '-v', 'error', '-show_entries',
                                'format=duration', '-of', 'csv=p=0', ch],
                                capture_output=True, text=True)
                            try: chunk_durs.append(float(p.stdout.strip()))
                            except: chunk_durs.append(cut_duration)

                        # sesuaikan trans_dur agar tidak melebihi durasi chunk terkecil
                        min_dur = min(chunk_durs)
                        if trans_dur >= min_dur:
                            trans_dur = max(0.1, min_dur * 0.5)

                        ff = get_ffmpeg_path()
                        cmd = [ff, '-y']
                        for ch in chunks:
                            cmd.extend(['-i', ch])

                        v_parts = []
                        a_parts = []
                        for i in range(len(chunks) - 1):
                            cum = sum(chunk_durs[:i+1])
                            off = max(0, cum - (i + 1) * trans_dur)
                            if i == 0:
                                v_parts.append(f"[{i}:v][{i+1}:v]xfade=transition=fade:duration={trans_dur}:offset={off}[v{i}]")
                                a_parts.append(f"[{i}:a][{i+1}:a]acrossfade=d={trans_dur}[a{i}]")
                            else:
                                v_parts.append(f"[v{i-1}][{i+1}:v]xfade=transition=fade:duration={trans_dur}:offset={off}[v{i}]")
                                a_parts.append(f"[a{i-1}][{i+1}:a]acrossfade=d={trans_dur}[a{i}]")

                        last_v = f"v{len(chunks)-2}"
                        last_a = f"a{len(chunks)-2}"
                        combined = ';'.join(v_parts + a_parts)

                        subprocess.run(cmd + [
                            '-filter_complex', combined,
                            '-map', f'[{last_v}]', '-map', f'[{last_a}]',
                            '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', smart_video
                        ], check=True, capture_output=True)

                        total_dur = sum(chunk_durs) - trans_dur * (len(chunks) - 1)
                        render_duration = total_dur
                    else:
                        # concat cepat tanpa transisi (copy stream)
                        smart_txt = os.path.join(BASE_UPLOAD, f"smart_{task_id}.txt")
                        with open(smart_txt, 'w', encoding='utf-8') as f:
                            for ch in chunks:
                                esc_ch = os.path.abspath(ch).replace(chr(92), '/').replace("'", "'\\''")
                                f.write(f"file '{esc_ch}'\n")
                        subprocess.run([
                            get_ffmpeg_path(), '-y', '-threads', '0', '-f', 'concat', '-safe', '0',
                            '-i', smart_txt, '-c', 'copy', smart_video
                        ], check=True, capture_output=True)
                        render_duration = full_chunks * cut_duration + (remainder if (cut_use_remainder and remainder > 0.5) else 0)

                    # ganti base_video dengan hasil smart cut
                    shutil.move(smart_video, base_video)

                # cleanup segment files
                shutil.rmtree(seg_dir, ignore_errors=True)

            if is_original_duration:
                loop_count = 1
                target_sec = render_duration
            else:
                loop_count = math.ceil(target_sec / render_duration) if (render_duration > 0 and target_sec > 0) else 1

            if loop_count > 1:
                with db_lock:
                    for d in active_tasks:
                        if d['id'] == task_id: d['status'] = f"Auto-Looping {loop_count}x ke {target_hours} Jam... 🚀"
                save_tasks_db()

                loop_txt = os.path.join(BASE_UPLOAD, f"loop_{task_id}.txt")
                with open(loop_txt, 'w', encoding='utf-8') as f:
                    for _ in range(loop_count):
                        safe_path_vid = os.path.abspath(base_video).replace('\\', '/').replace("'", "'\\''")
                        f.write(f"file '{safe_path_vid}'\n")

                if stop_flags.get(task_id): raise Exception("Dibatalkan")
                subprocess.run([
                    get_ffmpeg_path(), '-y', '-threads', '0', '-f', 'concat', '-safe', '0', '-i', loop_txt, 
                    '-c', 'copy', '-t', str(target_sec), final_video
                ], check=True)
            else:
                if stop_flags.get(task_id): raise Exception("Dibatalkan")
                subprocess.run([
                    get_ffmpeg_path(), '-y', '-i', base_video, '-c', 'copy', '-t', str(target_sec), final_video
                ], check=True)

            dest = task.get('output_dest', 'youtube')

            if dest == 'vps':
                # Simpan di VPS / Lokal — langsung selesai
                vps_folder = str(task.get('vps_folder', OUTPUT_DIR))
                if os.name == 'nt' and (vps_folder.startswith('/root') or not os.path.isabs(vps_folder)):
                    vps_folder = OUTPUT_DIR
                try:
                    os.makedirs(vps_folder, exist_ok=True)
                    dest_path = os.path.join(vps_folder, f"{task.get('title', 'video')}_{task_id}.mp4")
                    shutil.move(final_video, dest_path)
                except:
                    dest_path = final_video
                with db_lock:
                    for d in active_tasks:
                        if d['id'] == task_id: d['status'] = "Render Selesai ✅"
                save_tasks_db()
                dl_link = f"/download_vps/{os.path.basename(dest_path)}"
                move_to_history(task_id, f"Render Selesai ✅ <a href='{dl_link}' target='_blank'>[Download Video]</a>")
                return  # stop di sini, jangan lanjut upload YouTube
            elif channel_data:
                # YouTube upload (existing logic)
                creds_list = channel_data.get('creds_list', [channel_data.get('creds_json')])
                upload_berhasil = False
                pesan_error = "Token API Tidak Ditemukan/Kosong!" 
                
                for index_kunci, cred_str in enumerate(creds_list):
                    if not cred_str: continue
                    try:
                        creds = Credentials.from_authorized_user_info(json.loads(cred_str))
                        if creds.expired and creds.refresh_token: 
                            creds.refresh(Request())
                            
                        youtube = build('youtube', 'v3', credentials=creds)
                        try: sch_obj = datetime.strptime(task['publish_date'], "%Y-%m-%d %H:%M")
                        except: raise Exception("Format tanggal salah")
                        
                        raw_tags = task.get('tags', '')
                        clean_tags = raw_tags.replace('#', '').replace('<', '').replace('>', '').replace('"', '')
                        temp_tags = [t.strip() for t in clean_tags.split(',') if t.strip()]
                        
                        tags_list = []
                        char_count = 0
                        for t in temp_tags:
                            if char_count + len(t) <= 400:
                                tags_list.append(t)
                                char_count += len(t) + 1
                        
                        if not tags_list: tags_list = ['wavepush']
                        
                        body = {
                            'snippet': {'title': task['title'], 'description': task.get('description', ''), 'tags': tags_list, 'categoryId': '10'},
                            'status': {'privacyStatus': task.get('privacy', 'public')}
                        }
                        if sch_obj > datetime.now():
                            wib = ZoneInfo("Asia/Jakarta")
                            sch_aware = sch_obj.replace(tzinfo=wib)
                            sch_utc = sch_aware.astimezone(timezone.utc)
                            body['status']['publishAt'] = sch_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                            body['status']['privacyStatus'] = 'private'
                            
                        media = MediaFileUpload(final_video, chunksize=1024*1024*5, resumable=True)
                        req = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
                        resp = None
                        
                        max_retries = 5
                        retry_count = 0
                        
                        while resp is None:
                            if stop_flags.get(task_id): raise Exception("Dibatalkan")
                            try:
                                status, resp = req.next_chunk()
                                if status:
                                    with db_lock:
                                        for d in active_tasks:
                                            if d['id'] == task_id: d['status'] = f"Mengunggah (Key {index_kunci+1})... {int(status.progress()*100)}% 🚀"
                                    save_tasks_db()
                                retry_count = 0 
                            except HttpError as e:
                                if e.resp.status < 500:
                                    raise e
                                else:
                                    retry_count += 1
                                    if retry_count > max_retries: 
                                        raise Exception("Server YouTube Down/Timeout setelah 5x percobaan.")
                                    with db_lock:
                                        for d in active_tasks:
                                            if d['id'] == task_id: d['status'] = f"Koneksi Sinyal Lemah, Auto-Retry ({retry_count}/{max_retries})... 🔌"
                                    save_tasks_db()
                                    time.sleep(10)
                            except Exception as e:
                                retry_count += 1
                                if retry_count > max_retries: 
                                    raise Exception("Koneksi VPS Putus setelah dicoba 5x berturut-turut.")
                                with db_lock:
                                    for d in active_tasks:
                                        if d['id'] == task_id: d['status'] = f"Koneksi VPS Putus, Auto-Retry ({retry_count}/{max_retries})... 🔌"
                                save_tasks_db()
                                time.sleep(10)
                        
                        video_id = resp.get('id')
                        
                        thumb_path = get_and_consume_thumbnail(yt_id)
                        if thumb_path and os.path.exists(thumb_path):
                            try:
                                with db_lock:
                                    for d in active_tasks:
                                        if d['id'] == task_id: d['status'] = "Memasang Thumbnail... 🖼️"
                                save_tasks_db()
                                youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumb_path)).execute()
                                try:
                                    os.remove(thumb_path)
                                except Exception:
                                    pass
                            except: pass
                                
                        try:
                            if task.get('playlist_id'):
                                # 🐛 FIX: retry playlist insert (sebelumnya error ditelan diam-diam).
                                # Coba beberapa kali untuk menghindari kegagalan sesaat jaringan/kuota
                                # atau videoNotFound karena propagasi YT yang lambat.
                                playlist_inserted = False
                                for _attempt in range(3):
                                    try:
                                        youtube.playlistItems().insert(part='snippet', body={'snippet': {'playlistId': task['playlist_id'], 'resourceId': {'kind': 'youtube#video', 'videoId': video_id}}}).execute()
                                        playlist_inserted = True
                                        break
                                    except HttpError as ple:
                                        if ple.resp.status >= 500 or ple.resp.status == 404:
                                            time.sleep(5)
                                            continue
                                        raise
                                if not playlist_inserted:
                                    print(f"[KeiBot] Peringatan: video {video_id} GAGAL masuk playlist {task['playlist_id']} setelah retry.")
                        except Exception as ple_err:
                            print(f"[KeiBot] Playlist insert gagal untuk video {video_id}: {ple_err}")
                        move_to_history(task_id, f"Tayang! ✅ <a href='https://youtu.be/{video_id}' target='_blank'>[Lihat]</a>")
                        upload_berhasil = True
                        break
                        
                    except HttpError as e:
                        try:
                            err_info = json.loads(e.content.decode('utf-8'))
                            reason = err_info['error']['errors'][0]['reason']
                        except:
                            reason = str(e)
                            
                        if "quotaExceeded" in reason:
                            pesan_error = f"Limit Kuota Harian API Habis!"
                            continue 
                        elif "uploadLimitExceeded" in reason:
                            pesan_error = "Limit Upload Harian Channel Tercapai!"
                            channel_cooldowns[yt_id] = time.time() + (3600 * 24)
                            break
                        elif "rateLimitExceeded" in reason:
                            pesan_error = "Rate Limit (Terlalu Cepat) - Auto Cooldown 30 Menit"
                            channel_cooldowns[yt_id] = time.time() + 1800
                            break
                        else:
                            pesan_error = f"Ditolak YT: {reason}"
                            break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "invalid_grant" in err_str or "expired" in err_str or "revoked" in err_str:
                            pesan_error = "Sesi Kedaluwarsa (Tautkan Ulang!)"
                            channel_cooldowns[yt_id] = time.time() + (3600 * 24)
                        elif "timeout" in err_str or "connection" in err_str or "broken" in err_str:
                            pesan_error = "Koneksi VPS Putus/Timeout"
                        else:
                            pesan_error = f"Error: {str(e)[:40]}"
                        break
                        
                if not upload_berhasil:
                    if "API Habis" in pesan_error:
                        channel_cooldowns[yt_id] = time.time() + (3600 * 24)
                    raise Exception(pesan_error)
            else:
                move_to_history(task_id, f"Render Selesai ✅ <a href='/static/final_{task_id}.mp4' target='_blank'>[Download]</a>")
        
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                err_text = e.stderr.decode('utf-8', errors='ignore')
                last_lines = [l.strip() for l in err_text.splitlines() if l.strip()][-2:]
                err_msg = f"{e} | FFmpeg: {' '.join(last_lines)}"
            else:
                err_msg = str(e)
            if "Limit" in err_msg or "Cooldown" in err_msg or "Habis" in err_msg:
                with db_lock:
                    for d in active_tasks:
                        if d['id'] == task_id:
                            d['status'] = f"Gagal Upload, Antre Ulang ({err_msg}) ⏳"
                save_tasks_db()
                render_queue.put(task)
            else:
                move_to_history(task_id, f"Gagal ❌ ({err_msg})")
        finally:
            for path in temp_files:
                try: os.remove(path)
                except: pass
            # Bersihkan seg_dir dan file intermediate smart cut jika tersisa
            try:
                seg_dir = os.path.join(BASE_UPLOAD, f"seg_{task_id}")
                if os.path.exists(seg_dir): shutil.rmtree(seg_dir, ignore_errors=True)
                for extra in [f"rem_{task_id}.mp4", f"smart_{task_id}.mp4", f"smart_{task_id}.txt"]:
                    ep = os.path.join(BASE_UPLOAD, extra)
                    if os.path.exists(ep): os.remove(ep)
            except: pass
            stop_flags.pop(task_id, None)
            render_queue.task_done()
            gc.collect()

restore_and_resume_queue()
threading.Thread(target=background_worker, daemon=True).start()

# ==========================================
# 📊 API ENDPOINTS
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    return jsonify(system_notifications)

@app.route('/api/notifications/clear', methods=['POST'])
def clear_notifications():
    global system_notifications
    with db_lock:
        system_notifications.clear()
    return jsonify({"status": "success"})

@app.route('/api/get_dashboard_stats')
def get_dashboard_stats():
    sys = get_system_stats()
    return jsonify({
        "channels": len(database_channel), "active_tasks": len(active_tasks), "history_tasks": len(history_tasks),
        "sys_cpu": sys["cpu"], "sys_ram_pct": sys["ram_pct"], "sys_ram_text": f"{sys['ram_used']}GB / {sys['ram_total']}GB"
    })

@app.route('/api/get_youtube_analytics')
def get_youtube_analytics():
    data = []
    for c in database_channel:
        views, subs, videos = 0, 0, 0
        try:
            creds_list = c.get('creds_list', [c.get('creds_json')])
            if creds_list and creds_list[0]:
                creds = Credentials.from_authorized_user_info(json.loads(creds_list[0]))
                if creds.expired and creds.refresh_token: creds.refresh(Request())
                youtube = build('youtube', 'v3', credentials=creds)
                res = youtube.channels().list(part="statistics", id=c['yt_id']).execute()
                if res.get('items'):
                    stats = res['items'][0]['statistics']
                    views = int(stats.get('viewCount', 0))
                    subs = int(stats.get('subscriberCount', 0))
                    videos = int(stats.get('videoCount', 0))
        except Exception as e:
            pass
        data.append({"yt_id": c["yt_id"], "name": c["name"], "views": views, "subs": subs, "watch_hours": 0, "videos": videos})
    return jsonify(data)

@app.route('/api/get_schedule')
def get_schedule(): return jsonify({"active": active_tasks, "history": history_tasks})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    global history_tasks
    with db_lock: history_tasks.clear()
    save_tasks_db()
    return jsonify({"status": "success", "message": "Riwayat dibersihkan!"})

@app.route('/api/get_channels')
def get_channels():
    safe_c = [{"id": c["id"], "name": c["name"], "yt_id": c["yt_id"], "thumbnail": c["thumbnail"], "status": c["status"], "title_bank": c.get("title_bank", [])} for c in database_channel]
    return jsonify(safe_c)

@app.route('/api/delete_channel', methods=['POST'])
def delete_channel():
    yt_id = request.form.get('yt_id')
    global database_channel
    database_channel = [c for c in database_channel if c['yt_id'] != yt_id]
    save_channels(database_channel)
    return jsonify({"status": "success", "message": "Channel dihapus!"})

# --- PRESET API ---
@app.route('/api/save_preset', methods=['POST'])
def save_preset():
    data = request.json
    try:
        presets = {}
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, 'r') as f:
                try: presets = json.load(f)
                except: pass
        presets.update(data)
        with open(PRESETS_FILE, 'w') as f: json.dump(presets, f, indent=4)
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/get_presets', methods=['GET'])
def get_presets():
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, 'r') as f:
            try: 
                return jsonify(json.load(f))
            except: 
                pass
    return jsonify({})

@app.route('/api/delete_preset', methods=['POST'])
def delete_preset():
    data = request.json
    preset_name = data.get('name')
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, 'r') as f:
                presets = json.load(f)
                
            if preset_name in presets:
                del presets[preset_name]
                
                with open(PRESETS_FILE, 'w') as f: 
                    json.dump(presets, f, indent=4)
                    
                return jsonify({"status": "success"})
                
        return jsonify({"status": "error", "message": "Preset tidak ditemukan"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# --- METADATA PRESET API ---
@app.route('/api/save_metadata_preset', methods=['POST'])
def save_metadata_preset():
    data = request.json
    try:
        presets = {}
        if os.path.exists(METADATA_PRESETS_FILE):
            with open(METADATA_PRESETS_FILE, 'r', encoding='utf-8') as f:
                try: presets = json.load(f)
                except: pass
        presets.update(data)
        with open(METADATA_PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=4, ensure_ascii=False)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/get_metadata_presets', methods=['GET'])
def get_metadata_presets():
    if os.path.exists(METADATA_PRESETS_FILE):
        with open(METADATA_PRESETS_FILE, 'r', encoding='utf-8') as f:
            try:
                return jsonify(json.load(f))
            except:
                pass
    return jsonify({})

@app.route('/api/delete_metadata_preset', methods=['POST'])
def delete_metadata_preset():
    data = request.json
    preset_name = data.get('name')
    try:
        if os.path.exists(METADATA_PRESETS_FILE):
            with open(METADATA_PRESETS_FILE, 'r', encoding='utf-8') as f:
                presets = json.load(f)
            if preset_name in presets:
                del presets[preset_name]
                with open(METADATA_PRESETS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(presets, f, indent=4, ensure_ascii=False)
                return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Preset tidak ditemukan"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ============================================================
# 🖼️ GALLERY ENDPOINTS
# ============================================================
@app.route('/api/get_asset_counts')
def get_asset_counts():
    yt_id = request.args.get('yt_id')
    if not yt_id: return jsonify({"audios": 0, "backgrounds": 0, "thumbnails": 0})
    def count_files(sub):
        path = get_channel_folder(yt_id, sub)
        return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return jsonify({"audios": count_files("audios"), "backgrounds": count_files("backgrounds"), "thumbnails": count_files("thumbnails")})

@app.route('/api/get_gallery', methods=['GET'])
def get_gallery():
    yt_id = request.args.get('yt_id')
    if not yt_id: return jsonify({"audio": [], "background": [], "thumbnails": []})
    def get_files_data(sub):
        path = get_channel_folder(yt_id, sub)
        res = []
        if os.path.exists(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    size_mb = round(os.path.getsize(fp) / (1024*1024), 2)
                    res.append({"name": f, "size": f"{size_mb} MB"})
        return res
    return jsonify({
        "audio":      get_files_data("audios"),
        "background": get_files_data("backgrounds"),
        "thumbnails": get_files_data("thumbnails")
    })

@app.route('/api/get_audio_info', methods=['GET'])
def get_audio_info():
    yt_id = request.args.get('yt_id')
    if not yt_id: return jsonify([])
    # cek cache
    now = time.time()
    cached = _audio_info_cache.get(yt_id)
    if cached and (now - cached['ts']) < AUDIO_CACHE_TTL:
        return jsonify(cached['data'])
    # baca folder & hitung durasi via ffprobe
    path = get_channel_folder(yt_id, "audios")
    result = []
    if os.path.exists(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(('.mp3', '.wav')):
                fp = os.path.join(path, f)
                probe = subprocess.run([get_ffprobe_path(), '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', fp], capture_output=True, text=True)
                try: dur = float(probe.stdout.strip())
                except: dur = 0.0
                title = os.path.splitext(f)[0]
                result.append({"name": f, "title": title, "duration": dur})
    _audio_info_cache[yt_id] = {"data": result, "ts": now}
    return jsonify(result)

@app.route('/api/upload_gallery', methods=['POST'])
def upload_gallery():
    yt_id  = request.form.get('yt_id', '').strip()
    g_type = request.form.get('type',  '').strip()

    if not yt_id:
        return jsonify({"status": "error", "message": "yt_id tidak boleh kosong!"}), 400
    if not g_type:
        return jsonify({"status": "error", "message": "type tidak boleh kosong!"}), 400

    folder_name = resolve_folder(g_type)
    folder      = get_channel_folder(yt_id, folder_name)

    files = (request.files.getlist('files[]')
             or request.files.getlist('files')
             or request.files.getlist('file')
             or list(request.files.values()))

    if not files:
        return jsonify({"status": "error", "message": "Tidak ada file yang diterima!"}), 400

    saved, errors = 0, []
    for f in files:
        if not f or not f.filename:
            continue
        try:
            safe_name = os.path.basename(f.filename)
            dest = os.path.join(folder, safe_name)
            f.save(dest)
            if os.path.getsize(dest) == 0:
                try: os.remove(dest)
                except: pass
                errors.append(f"{f.filename}: file kosong (0 bytes)")
                continue
            saved += 1
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")

    if saved == 0:
        return jsonify({"status": "error", "message": "Tidak ada file yang berhasil disimpan. " + "; ".join(errors)}), 500

    msg = f"{saved} file berhasil diupload ke '{folder_name}'"
    if errors:
        msg += f" ({len(errors)} gagal: {'; '.join(errors[:3])})"
    return jsonify({"status": "success", "message": msg})

@app.route('/api/delete_gallery_file', methods=['POST'])
def delete_gallery_file():
    yt_id  = request.form.get('yt_id', '').strip()
    g_type = request.form.get('type',  '').strip()
    name   = request.form.get('name',  '').strip()

    folder_name = resolve_folder(g_type)
    path = os.path.join(get_channel_folder(yt_id, folder_name), os.path.basename(name))

    if os.path.exists(path):
        os.remove(path)
        return jsonify({"status": "success", "message": "File dihapus!"})
    return jsonify({"status": "error", "message": f"File tidak ditemukan: {path}"})

# ============================================================
# 📝 TITLE BANK ENDPOINT
# ============================================================
@app.route('/api/upload_title_bank', methods=['POST'])
def upload_title_bank():
    yt_id = (request.form.get('yt_id') or request.args.get('yt_id') or '').strip()
    txt_file = request.files.get('txt_file') or request.files.get('file')

    if not yt_id:
        return jsonify({"status": "error", "message": "yt_id tidak ditemukan. Pastikan channel sudah dipilih."}), 400
    if not txt_file:
        return jsonify({"status": "error", "message": "File .txt tidak ditemukan dalam request."}), 400

    try:
        raw_bytes = txt_file.read()
        try:   content = raw_bytes.decode('utf-8')
        except: content = raw_bytes.decode('latin-1', errors='ignore')

        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return jsonify({"status": "error", "message": "File .txt kosong atau tidak ada baris valid."}), 400

        global database_channel
        channel_found = False
        for c in database_channel:
            if c['yt_id'] == yt_id:
                existing = c.get('title_bank', [])
                merged   = list(dict.fromkeys(existing + lines))
                c['title_bank'] = merged
                channel_found = True
                save_channels(database_channel)
                return jsonify({
                    "status":  "success",
                    "message": f"{len(lines)} judul diimport! Total bank: {len(merged)} judul.",
                    "total":   len(merged),
                })

        if not channel_found:
            return jsonify({"status": "error", "message": f"Channel dengan yt_id '{yt_id}' tidak ditemukan di database."}), 404

    except Exception as e:
        return jsonify({"status": "error", "message": f"Gagal memproses file: {str(e)}"}), 500

@app.route('/api/get_playlists', methods=['GET'])
def get_playlists():
    yt_id = request.args.get('yt_id')
    if not yt_id: return jsonify([])
    channel = next((c for c in database_channel if c['yt_id'] == yt_id), None)
    if not channel: return jsonify([])
    try:
        creds = get_fresh_credentials(channel)
        youtube = build('youtube', 'v3', credentials=creds)
        res = youtube.playlists().list(part="snippet", mine=True, maxResults=50).execute()
        return jsonify([{"id": p['id'], "title": p['snippet']['title']} for p in res.get('items', [])])
    except: return jsonify([])

@app.route('/api/stop_task/<int:task_id>', methods=['POST'])
def stop_task(task_id):
    stop_flags[task_id] = True
    with db_lock:
        for t in active_tasks:
            if t['id'] == task_id and t.get('status') == "In Factory Queue ⚙️":
                t['status'] = "Dibatalkan Pengguna 🛑"
                t.pop('blueprint', None)
                history_tasks.insert(0, t)
                active_tasks.remove(t)
                if len(history_tasks) > 50: history_tasks.pop()
                break
    save_tasks_db()
    return jsonify({"status": "success", "message": "Dihentikan!"})

@app.route('/api/check_secret')
def check_secret():
    try: return jsonify({"exists": os.path.exists(CLIENT_SECRETS_FILE)})
    except: return jsonify({"exists": False})

@app.route('/api/upload_secret', methods=['POST'])
def upload_secret():
    try:
        file = request.files.get('secret_file')
        if file and file.filename.endswith('.json'):
            file.save(CLIENT_SECRETS_FILE)
            return jsonify({"status": "success", "message": "API Key diunggah!"})
        return jsonify({"status": "error", "message": "Harus .json!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Izin ditolak server: {str(e)}"})

@app.route('/api/generate_tv_link')
def generate_tv_link():
    if not os.path.exists(CLIENT_SECRETS_FILE): return jsonify({"auth_url": "", "error": "File client_secret.json belum ada!"})
    return jsonify({"auth_url": f"http://{request.host}/device_login"})

@app.route('/device_login')
def device_login():
    if not os.path.exists(CLIENT_SECRETS_FILE): return "File rahasia tidak ditemukan!"
    with open(CLIENT_SECRETS_FILE, 'r') as f:
        secret_data = json.load(f); client_config = secret_data.get('installed', secret_data.get('web', {})); client_id = client_config.get('client_id')
    res = requests.post('https://oauth2.googleapis.com/device/code', data={'client_id': client_id, 'scope': ' '.join(SCOPES)}).json()
    if 'error' in res: return f"Error Google: {res['error']}"
    html = f"""
    <html><head><title>Aktivasi YouTube</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial; text-align: center; background: #eef2f6; color: #1e293b; padding-top: 10vh; }}
        .box {{ background: #ffffff; width: 550px; margin: auto; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }}
        .step {{ text-align: left; margin-bottom: 25px; font-size: 14px; color: #64748b; font-weight:600; }}
        .input-group {{ display: flex; margin-top: 10px; }}
        .input-group input {{ flex: 1; padding: 15px; font-size: 16px; font-weight: bold; background: #f8fafc; color: #10b981; border: 1px solid #e2e8f0; border-radius: 8px 0 0 8px; text-align: center; outline:none; }}
        .input-group button {{ padding: 15px 25px; font-size: 14px; font-weight: bold; background: #10b981; color: white; border: none; border-radius: 0 8px 8px 0; cursor: pointer; transition: 0.3s; }}
    </style></head><body>
        <div class="box">
            <h2 style="margin-top:0;">🔗 Tautkan Channel Baru</h2>
            <div class="step"><b>Langkah 1:</b> Copy link ini dan Paste di browser target:
                <div class="input-group"><input type="text" id="glink" value="{res['verification_url']}" readonly><button onclick="document.getElementById('glink').select();document.execCommand('copy');">Copy Link</button></div>
            </div>
            <div class="step"><b>Langkah 2:</b> Masukkan Kode Rahasia ini:
                <div class="input-group"><input type="text" id="gcode" value="{res['user_code']}" readonly><button onclick="document.getElementById('gcode').select();document.execCommand('copy');">Copy Kode</button></div>
            </div>
            <div id="status" style="margin-top:30px; font-weight:bold;">⏳ Menunggu Anda memasukkan kode...</div>
        </div>
        <script>
            function poll() {{
                fetch('/api/poll_device_token', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{device_code: '{res['device_code']}'}})
                }})
                .then(r => r.json())
                .then(data => {{
                    if(data.status === 'success') {{
                        document.getElementById('status').innerHTML = "🎉 Berhasil! Mengalihkan...";
                        document.getElementById('status').style.color = "#10b981";
                        setTimeout(() => {{ window.location.href = '/'; }}, 2000);
                    }} else if(data.status === 'pending') {{
                        setTimeout(poll, data.interval || 5000);
                    }} else {{
                        document.getElementById('status').innerHTML = "❌ Gagal: " + (data.error || "Terjadi kesalahan saat menghubungkan channel.");
                        document.getElementById('status').style.color = "#ef4444";
                    }}
                }})
                .catch(err => {{
                    document.getElementById('status').innerHTML = "❌ Error respon server: " + err;
                    document.getElementById('status').style.color = "#ef4444";
                }});
            }}
            setTimeout(poll, 5000);
        </script>
    </body></html>
    """
    return html

@app.route('/api/poll_device_token', methods=['POST'])
def poll_device_token():
    try:
        device_code = request.json.get('device_code')
        with open(CLIENT_SECRETS_FILE, 'r') as f:
            s_data = json.load(f); conf = s_data.get('installed', s_data.get('web', {})); c_id = conf.get('client_id'); c_sec = conf.get('client_secret')
        res = requests.post('https://oauth2.googleapis.com/token', data={'client_id': c_id, 'client_secret': c_sec, 'device_code': device_code, 'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'}).json()
        if 'error' in res:
            err = res['error']
            if err == 'authorization_pending': return jsonify({"status": "pending", "interval": 5000})
            elif err == 'slow_down': return jsonify({"status": "pending", "interval": 10000})
            else: return jsonify({"status": "error", "error": err})
        creds = Credentials(token=res['access_token'], refresh_token=res.get('refresh_token'), token_uri='https://oauth2.googleapis.com/token', client_id=c_id, client_secret=c_sec, scopes=SCOPES)
        youtube = build('youtube', 'v3', credentials=creds)
        chan_res = youtube.channels().list(part="snippet", mine=True).execute()
        if chan_res.get('items'):
            item = chan_res['items'][0]; global database_channel
            c_idx = next((i for i, c in enumerate(database_channel) if c['yt_id'] == item['id']), None)
            if c_idx is None:
                new_c = {"id": len(database_channel)+1, "name": item['snippet']['title'], "yt_id": item['id'], "thumbnail": item['snippet']['thumbnails']['default']['url'], "status": "Connected 🟢 (1 Key)", "creds_list": [creds.to_json()]}
                database_channel.append(new_c)
            else:
                if 'creds_list' not in database_channel[c_idx]:
                    database_channel[c_idx]['creds_list'] = [database_channel[c_idx].get('creds_json', '')]
                if creds.to_json() not in database_channel[c_idx]['creds_list']:
                    database_channel[c_idx]['creds_list'].append(creds.to_json())
                database_channel[c_idx]['status'] = f"Connected 🟢 ({len(database_channel[c_idx]['creds_list'])} Keys)"
            save_channels(database_channel)
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "error": "Akun Google ini belum memiliki Channel YouTube. Silakan buat channel terlebih dahulu di youtube.com!"})
    except Exception as e:
        print(f"❌ Error poll_device_token: {e}")
        return jsonify({"status": "error", "error": str(e)})

# --- BATCH CREATOR ---
@app.route('/api/batch_create', methods=['POST'])
def batch_create():
    data = request.json
    yt_id = data.get('yt_id')
    count = data.get('count', 1)
    titles = data.get('generated_titles', [])
    durations_array = data.get('target_durations_array', []) 
    
    try:
        base_date = datetime.strptime(data['start_date'], '%Y-%m-%dT%H:%M')
    except:
        return jsonify({"status": "error", "message": "Format tanggal salah"}), 400
        
    # 🔥 FIX 1: Konversi interval_days ke float secara eksplisit untuk mencegah TypeError pada timedelta
    interval_days = float(data.get('interval_days', 1))
        
    for i in range(count):
        t_id = int(time.time()) + i
        v_date = base_date + timedelta(days=i * interval_days)
        
        if i < len(durations_array):
            vid_duration = durations_array[i]
        else:
            vid_duration = data.get('target_duration_hours', 1)
            
        blueprint = {
            "id": t_id, "yt_id": yt_id, "title": titles[i] if i < len(titles) else f"Auto Video #{i+1}",
            "publish_date": v_date.strftime('%Y-%m-%d %H:%M'),
            "mp3_per_video": data.get('mp3_per_video', 5), 
            "bg_count": data.get('bg_count', 1), 
            "target_duration_hours": vid_duration,
            "vis_mode": data.get('vis_mode'), "vis_preset": data.get('vis_preset'),
            "vis_config": data.get('vis_config', {}), "vis_presets_allowed": data.get('vis_presets_allowed', []), "description": data.get('description', ''),
            "tags": data.get('tags', ''), "privacy": data.get('privacy', 'public'), "playlist_id": data.get('playlist_id', ''),
            "use_floating_card": data.get('use_floating_card', False),
            "use_tracklist": data.get('use_tracklist', False),
            "use_watermark": data.get('use_watermark', False),
            "use_timestamp": data.get('use_timestamp', False),
            "resolution": str(data.get('resolution', '720')),
            "smart_cut": data.get('smart_cut', False),
            "cut_duration": data.get('cut_duration', 5),
            "cut_remainder": data.get('cut_remainder', 'end'),
            "cut_use_remainder": data.get('cut_use_remainder', True),
            "use_transition": data.get('use_transition', False),
            "transition_duration": data.get('transition_duration', 0.5),
            "render_speed": str(data.get('render_speed', 'ultrafast')),
            "output_dest": data.get('output_dest', 'youtube'),
            "vps_folder": data.get('vps_folder', OUTPUT_DIR)
        }
        task_entry = {
            "id": t_id,
            "title": blueprint['title'],
            "time": blueprint['publish_date'],
            "status": "In Factory Queue ⚙️",
            "type": "📺 VOD",
            "blueprint": blueprint
        }
        with db_lock:
            active_tasks.append(task_entry)
        
        # Masukkan blueprint ke antrean in-memory worker
        render_queue.put(blueprint)
        
    # 🔥 FIX 2: Pindahkan save_tasks_db() ke LUAR perulangan 'for' 
    # Menghindari penulisan beruntun ke disk I/O yang menyebabkan VPS freeze/lag
    save_tasks_db()
    
    return jsonify({"status": "success", "message": f"{count} Video diproses!"})
    
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(BASE_UPLOAD, filename)

@app.route('/download_vps/<path:filename>')
def serve_vps_download(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/api/get_output_videos')
def get_output_videos():
    # baca folder dari file konfigurasi atau default
    folder = OUTPUT_DIR
    if not os.path.exists(folder): return jsonify([])
    files = []
    for f in sorted(os.listdir(folder), reverse=True):
        if f.lower().endswith('.mp4'):
            fp = os.path.join(folder, f)
            size_mb = round(os.path.getsize(fp) / (1024*1024), 2)
            files.append({"name": f, "size": f"{size_mb} MB"})
    return jsonify(files)

@app.route('/api/delete_output_video', methods=['POST'])
def delete_output_video():
    name = request.json.get('name')
    if not name: return jsonify({"status": "error", "error": "Nama file diperlukan"})
    fp = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(fp):
        os.remove(fp)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "error": "File tidak ditemukan"})

@app.route('/api/get_logs')
def get_logs():
    try:
        lines = request.args.get('lines', '100')
        result = subprocess.run(['journalctl', '-u', 'keibot', '-n', str(lines), '--no-pager', '-q'], 
                               capture_output=True, text=True, timeout=10)
        return result.stdout[-50000:] or '(kosong)'
    except Exception as e:
        try:
            log_file = os.path.join(BASE_DIR, 'app.log')
            if os.path.exists(log_file):
                with open(log_file) as f: return f.read()[-50000:]
        except: pass
        return f"(Tidak bisa baca log: {str(e)[:50]})"

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    try:
        subprocess.run(['journalctl', '--rotate', '-u', 'keibot'], capture_output=True, timeout=5)
        subprocess.run(['journalctl', '--vacuum-time=1s', '-u', 'keibot'], capture_output=True, timeout=5)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)[:50]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
