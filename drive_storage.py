"""Google Drive storage untuk KeibotMod — galeri dari Drive saat disk VPS hampir penuh.

Tujuan: kalau disk VPS menipis, file galeri (audio/background/thumbnail)
diambil dari folder Google Drive Bos, bukan dari disk lokal — supaya render
tetap jalan dan VPS tidak terganggu.

Cara kerja:
- Konfigurasi tersimpan di config.json (key "drive"): 4 ID folder Drive
  (audios, backgrounds, thumbnails, renders), ambang batas disk, dan opsi
  arsip render. File ini sudah di-.gitignore.
- Token OAuth tersimpan di drive_token.json (di-.gitignore) — diperoleh
  lewat device-code flow yang sama dengan login YouTube (client_secret.json
  yang sama; proyek Google Cloud-nya harus mengaktifkan Drive API).
- Semua import Google bersifat lazy: modul ini tetap bisa diimpor meski
  library Google belum terpasang. Tanpa konfigurasi/token, semua fungsi
  gagal dengan pesan yang jelas — aplikasi tidak pernah crash dan token
  tidak pernah bocor ke log/respons.
"""

import json
import os
import shutil
import threading

# FIX: tulis config.json / drive_token.json tanpa kunci bisa saling timpa
# antar thread (baca-ubah-tulis: patch yang satu hilang, atau file JSON
# terpotong di tengah). Satu kunci untuk semua tulis file Drive di modul ini.
_drive_file_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_TOKEN_FILE = os.path.join(BASE_DIR, "drive_token.json")
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, "client_secret.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
TOKEN_URL = "https://oauth2.googleapis.com/token"

AUDIO_EXTS = {".mp3", ".wav"}
BG_EXTS = {".mp4", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mov"}
THUMB_EXTS = {".jpg", ".png", ".jpeg"}

# sub-folder galeri -> key konfigurasi folder Drive-nya
FOLDER_KEY_BY_SUB = {
    "audios": "folder_audios",
    "audio": "folder_audios",
    "backgrounds": "folder_backgrounds",
    "background": "folder_backgrounds",
    "thumbnails": "folder_thumbnails",
    "thumbnail": "folder_thumbnails",
}
EXTS_BY_SUB = {
    "audios": AUDIO_EXTS,
    "backgrounds": BG_EXTS,
    "thumbnails": THUMB_EXTS,
}

DEFAULT_CONFIG = {
    "folder_audios": "",
    "folder_backgrounds": "",
    "folder_thumbnails": "",
    "folder_renders": "",
    "free_gb_threshold": 5.0,
    "usage_pct_threshold": 85.0,
    "upload_renders": False,
    "delete_after_upload": False,
}


# ── Konfigurasi ──────────────────────────────────────────────────
def load_drive_config():
    """Baca konfigurasi Drive dari config.json (key "drive").
    Tidak pernah raise — kalau file tidak ada/rusak, kembalikan default."""
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        stored = data.get("drive") or {}
        if isinstance(stored, dict):
            for key in DEFAULT_CONFIG:
                if key in stored:
                    cfg[key] = stored[key]
    except Exception:
        pass
    # Normalisasi tipe supaya nilai aneh dari file tidak merusak logika.
    try:
        cfg["free_gb_threshold"] = float(cfg.get("free_gb_threshold", 5.0))
    except (TypeError, ValueError):
        cfg["free_gb_threshold"] = 5.0
    try:
        cfg["usage_pct_threshold"] = float(cfg.get("usage_pct_threshold", 85.0))
    except (TypeError, ValueError):
        cfg["usage_pct_threshold"] = 85.0
    for key in ("folder_audios", "folder_backgrounds",
                "folder_thumbnails", "folder_renders"):
        cfg[key] = str(cfg.get(key) or "").strip()
    cfg["upload_renders"] = bool(cfg.get("upload_renders"))
    cfg["delete_after_upload"] = bool(cfg.get("delete_after_upload"))
    return cfg


def save_drive_config(patch):
    """Simpan patch konfigurasi Drive ke config.json (key "drive").
    Key lain di config.json (PIN, secret_key, ...) dipertahankan.
    Mengembalikan (True, None) atau (False, pesan_error)."""
    if not isinstance(patch, dict):
        return False, "Data konfigurasi tidak valid."
    # Kunci selama SELURUH baca-ubah-tulis (termasuk load_drive_config di
    # awal) supaya dua penyimpanan bersamaan tidak saling menimpa patch.
    with _drive_file_lock:
        cfg = load_drive_config()
        for key, val in patch.items():
            if key in ("folder_audios", "folder_backgrounds",
                       "folder_thumbnails", "folder_renders"):
                if val is None:
                    val = ""
                val = str(val).strip()
                # ID folder Drive berupa huruf/angka/underscore/dash (atau kosong).
                if val and not all(c.isalnum() or c in "-_" for c in val):
                    return False, f"ID folder {key} tidak valid."
                cfg[key] = val
            elif key in ("free_gb_threshold", "usage_pct_threshold"):
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    return False, f"Nilai {key} harus angka."
                if key == "free_gb_threshold" and not (0.5 <= num <= 50):
                    return False, "Batas disk bebas harus 0,5–50 GB."
                if key == "usage_pct_threshold" and not (50 <= num <= 99):
                    return False, "Batas pemakaian disk harus 50–99%."
                cfg[key] = num
            elif key in ("upload_renders", "delete_after_upload"):
                cfg[key] = bool(val)
            # Key tidak dikenal diabaikan.
        try:
            data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
            if not isinstance(data, dict):
                data = {}
            data["drive"] = cfg
            with open(CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=4)
            return True, None
        except Exception as e:
            return False, f"Gagal menyimpan konfigurasi: {type(e).__name__}"


# ── Disk ─────────────────────────────────────────────────────────
def disk_usage_info(path=None):
    """Info pemakaian disk (default: folder aplikasi). Tidak pernah raise."""
    target = path or BASE_DIR
    try:
        total, used, free = shutil.disk_usage(target)
    except Exception:
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0,
                "pct_used": 0.0, "ok": False}
    total_gb = total / (1024 ** 3)
    return {
        "total_gb": round(total_gb, 2),
        "used_gb": round(used / (1024 ** 3), 2),
        "free_gb": round(free / (1024 ** 3), 2),
        "pct_used": round(used / total * 100, 1) if total else 0.0,
        "ok": True,
    }


def should_use_drive(config=None):
    """Apakah galeri sebaiknya diambil dari Drive?
    Syarat: minimal satu folder galeri Drive dikonfigurasi DAN
    disk bebas < ambang batas ATAU pemakaian >= ambang batas.
    Mengembalikan (bool, alasan)."""
    cfg = config or load_drive_config()
    if not any(cfg.get(k) for k in ("folder_audios", "folder_backgrounds",
                                    "folder_thumbnails")):
        return False, "Folder galeri Drive belum dikonfigurasi."
    disk = disk_usage_info()
    if not disk["ok"]:
        return False, "Info disk tidak bisa dibaca."
    if disk["free_gb"] < cfg["free_gb_threshold"]:
        return True, (f"Disk bebas tinggal {disk['free_gb']} GB "
                      f"(batas {cfg['free_gb_threshold']} GB) — galeri dari Drive.")
    if disk["pct_used"] >= cfg["usage_pct_threshold"]:
        return True, (f"Disk terpakai {disk['pct_used']}% "
                      f"(batas {cfg['usage_pct_threshold']}%) — galeri dari Drive.")
    return False, (f"Disk masih lega ({disk['free_gb']} GB bebas) — galeri lokal.")


def drive_mode(config=None):
    """Mode galeri saat ini: 'drive' / 'lokal' / 'belum-dikonfigurasi'."""
    cfg = config or load_drive_config()
    if not any(cfg.get(k) for k in ("folder_audios", "folder_backgrounds",
                                    "folder_thumbnails")):
        return "belum-dikonfigurasi"
    use_drive, _ = should_use_drive(cfg)
    return "drive" if use_drive else "lokal"


# ── Auth Google Drive (device-code flow, seperti login YouTube) ──
def _client_id_secret():
    """Baca client_id + client_secret dari client_secret.json.
    Mengembalikan (client_id, client_secret) atau (None, pesan_error)."""
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return None, "File client_secret.json belum ada — unggah dulu di Settings."
    try:
        with open(CLIENT_SECRETS_FILE, "r") as f:
            secret = json.load(f)
        conf = secret.get("installed", secret.get("web", {}))
        client_id = conf.get("client_id")
        client_secret = conf.get("client_secret")
        if not client_id or not client_secret:
            return None, "client_secret.json tidak berisi client_id/client_secret."
        return (client_id, client_secret), None
    except Exception:
        return None, "client_secret.json rusak / tidak bisa dibaca."


def start_device_flow():
    """Mulai aktivasi Drive: minta kode perangkat dari Google.
    Mengembalikan (data, None) atau (None, pesan_error).
    Data berisi verification_url, user_code, device_code, interval."""
    import requests

    creds, err = _client_id_secret()
    if err:
        return None, err
    client_id, _ = creds
    try:
        res = requests.post(
            DEVICE_CODE_URL,
            data={"client_id": client_id, "scope": " ".join(DRIVE_SCOPES)},
            timeout=20,
        ).json()
    except Exception:
        return None, "Tidak bisa menghubungi Google — periksa koneksi VPS."
    if "error" in res:
        return None, f"Google menolak: {res.get('error')}"
    if not res.get("device_code") or not res.get("user_code"):
        return None, "Respons Google tidak lengkap — coba lagi."
    return {
        "verification_url": res.get("verification_url", "https://www.google.com/device"),
        "user_code": res["user_code"],
        "device_code": res["device_code"],
        "interval": int(res.get("interval", 5)),
        "expires_in": int(res.get("expires_in", 1800)),
    }, None


def poll_device_flow(device_code):
    """Tukar device_code menjadi token Drive lalu simpan ke drive_token.json.
    Mengembalikan (True, None) saat berhasil, (False, "pending") saat Bos
    belum memasukkan kode, atau (False, pesan_error) saat gagal."""
    import requests

    if not device_code:
        return False, "Kode perangkat kosong."
    creds, err = _client_id_secret()
    if err:
        return False, err
    client_id, client_secret = creds
    try:
        res = requests.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=20,
        ).json()
    except Exception:
        return False, "Tidak bisa menghubungi Google — periksa koneksi VPS."
    if "error" in res:
        code = res.get("error", "")
        if code == "authorization_pending":
            return False, "pending"
        if code == "slow_down":
            return False, "pending-slow"
        # Hanya kode error-nya yang diteruskan — tidak ada isi token.
        return False, f"Google: {code}"
    if not res.get("access_token"):
        return False, "Respons Google tidak berisi access_token."
    try:
        from google.oauth2.credentials import Credentials

        creds_obj = Credentials(
            token=res["access_token"],
            refresh_token=res.get("refresh_token"),
            token_uri=TOKEN_URL,
            client_id=client_id,
            client_secret=client_secret,
            scopes=DRIVE_SCOPES,
        )
        # Tulis token di bawah kunci (lihat _drive_file_lock di atas).
        with _drive_file_lock:
            with open(DRIVE_TOKEN_FILE, "w") as f:
                f.write(creds_obj.to_json())
            try:
                os.chmod(DRIVE_TOKEN_FILE, 0o600)
            except Exception:
                pass
        return True, None
    except Exception:
        return False, "Gagal menyimpan token Drive."


def get_service():
    """Bangun service Google Drive dari drive_token.json (refresh bila perlu).
    Mengembalikan (service, None) atau (None, pesan_error).
    Pesan error tidak pernah berisi isi token."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
    except Exception:
        return None, "Library Google belum tersedia di VPS."
    if not os.path.exists(DRIVE_TOKEN_FILE):
        return None, "Token Drive belum ada — hubungkan Google Drive dulu dari panel."
    try:
        with open(DRIVE_TOKEN_FILE, "r") as f:
            info = json.load(f)
        creds = Credentials.from_authorized_user_info(info, DRIVE_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Refresh (jaringan) di luar kunci; hanya tulis file di bawah kunci.
            with _drive_file_lock:
                with open(DRIVE_TOKEN_FILE, "w") as f:
                    f.write(creds.to_json())
        service = build("drive", "v3", credentials=creds)
        return service, None
    except Exception:
        return None, "Token Drive tidak valid/kedaluwarsa — hubungkan ulang dari panel."


def disconnect():
    """Putuskan Drive: hapus drive_token.json. Mengembalikan True/False."""
    try:
        with _drive_file_lock:
            if os.path.exists(DRIVE_TOKEN_FILE):
                os.remove(DRIVE_TOKEN_FILE)
        return True
    except Exception:
        return False


# ── Operasi file Drive ───────────────────────────────────────────
def list_files(service, folder_id, exts=None, page_size=100):
    """Daftar file di folder Drive (bukan sampah). Difilter ekstensi bila ada.
    Mengembalikan (daftar_file, None) atau ([], pesan_error)."""
    if not folder_id:
        return [], "ID folder Drive kosong."
    try:
        query = f"'{folder_id}' in parents and trashed = false"
        # FIX: dulu hanya 1 halaman (pageSize=100) — folder berisi >100 file
        # kehilangan sisanya diam-diam. Sekarang loop semua halaman.
        files = []
        page_token = None
        while True:
            kwargs = dict(q=query, pageSize=page_size,
                          fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
                          orderBy="modifiedTime desc")
            if page_token:
                kwargs['pageToken'] = page_token
            resp = service.files().list(**kwargs).execute()
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        if exts:
            exts = {e.lower() for e in exts}
            files = [f for f in files
                     if os.path.splitext(f.get("name", ""))[1].lower() in exts]
        return files, None
    except Exception:
        return [], "Gagal membaca folder Drive — periksa ID folder & koneksi."


def download_file(service, file_id, dest_path):
    """Unduh satu file Drive ke dest_path. (True, None) / (False, error)."""
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except Exception:
        return False, "Library Google belum tersedia di VPS."
    try:
        request = service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return True, None
    except Exception:
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        return False, "Gagal mengunduh file dari Drive."


def sync_gallery_folder(folder_id, dest_dir, exts, config=None):
    """Unduh file galeri yang belum ada di disk lokal dari folder Drive.
    Hanya mengunduh yang belum ada (cocokkan nama + ukuran) — tidak pernah
    menghapus file lokal. Mengembalikan dict {downloaded, skipped, error}."""
    result = {"downloaded": [], "skipped": 0, "error": None}
    if not folder_id:
        result["error"] = "ID folder Drive kosong."
        return result
    service, err = get_service()
    if err:
        result["error"] = err
        return result
    files, err = list_files(service, folder_id, exts=exts)
    if err:
        result["error"] = err
        return result
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        name = f.get("name", "")
        if not name:
            continue
        dest_path = os.path.join(dest_dir, name)
        try:
            size = int(f.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        if os.path.exists(dest_path) and size and \
                os.path.getsize(dest_path) == size:
            result["skipped"] += 1
            continue
        ok, derr = download_file(service, f["id"], dest_path)
        if ok:
            result["downloaded"].append(name)
        else:
            result["error"] = derr
            return result
    return result


def upload_file(service, local_path, folder_id):
    """Unggah satu file lokal ke folder Drive.
    Mengembalikan (file_id, web_link) atau (None, pesan_error)."""
    try:
        from googleapiclient.http import MediaFileUpload
    except Exception:
        return None, "Library Google belum tersedia di VPS."
    if not os.path.exists(local_path):
        return None, "File lokal tidak ditemukan."
    if not folder_id:
        return None, "ID folder render Drive kosong."
    try:
        media = MediaFileUpload(local_path, resumable=True)
        body = {"name": os.path.basename(local_path), "parents": [folder_id]}
        created = service.files().create(
            body=body, media_body=media, fields="id,webViewLink").execute()
        return created.get("id"), created.get("webViewLink")
    except Exception:
        return None, "Gagal mengunggah file ke Drive."


# ── Ringkasan status (panel + Agent API) ─────────────────────────
def status_summary():
    """Ringkasan status Drive untuk panel/API. Tidak pernah raise dan
    tidak pernah menyertakan isi token/kunci apa pun."""
    cfg = load_drive_config()
    disk = disk_usage_info()
    use_drive, reason = should_use_drive(cfg)
    out = {
        "module_loaded": True,
        "configured": bool(any(cfg.get(k) for k in (
            "folder_audios", "folder_backgrounds",
            "folder_thumbnails", "folder_renders"))),
        "connected": False,
        "token_exists": os.path.exists(DRIVE_TOKEN_FILE),
        "mode": drive_mode(cfg),
        "use_drive": use_drive,
        "use_drive_reason": reason,
        "disk": disk,
        "folders_set": {
            "audios": bool(cfg.get("folder_audios")),
            "backgrounds": bool(cfg.get("folder_backgrounds")),
            "thumbnails": bool(cfg.get("folder_thumbnails")),
            "renders": bool(cfg.get("folder_renders")),
        },
        "config": cfg,
        "drive_user": None,
        "error": None,
    }
    if not out["token_exists"]:
        out["error"] = "Belum terhubung — hubungkan Google Drive dari panel."
        return out
    try:
        service, err = get_service()
        if err:
            out["error"] = err
            return out
        about = service.about().get(
            fields="user(displayName,emailAddress)").execute()
        user = about.get("user", {})
        out["connected"] = True
        out["drive_user"] = {
            "name": user.get("displayName"),
            "email": user.get("emailAddress"),
        }
    except Exception:
        out["error"] = "Tidak bisa memeriksa koneksi Drive."
    return out
