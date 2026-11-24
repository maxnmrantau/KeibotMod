# API Agen KeibotMod — Panduan Akses

Dokumen ini menjelaskan cara agen lain (misalnya Muse lain) mengakses dan
mengendalikan KeibotMod lewat API, tanpa perlu membuka panel web.

Spesifikasi mesin (OpenAPI 3.0) ada di [`openapi.yaml`](openapi.yaml) di
folder yang sama — agen bisa membacanya langsung untuk mengetahui semua
endpoint, parameter, dan bentuk respons.

## 1. Alamat dan autentikasi

- **Base URL:** `http://<IP-VPS>:5000` — ganti `<IP-VPS>` dengan alamat IP
  VPS KeibotMod. Alamat aslinya TIDAK ditulis di dokumen ini karena repo
  ini publik; Bos yang memberikannya bersama kunci API kepada agen yang
  dituju.
- **Autentikasi:** header `X-API-Key: <KUNCI_API>` pada SETIAP request.
  Kunci tersimpan di VPS pada `/root/keibotmod/.agent.env`
  (`KEIBOT_AGENT_KEY`). Tanpa kunci yang benar, API menjawab `401`.
  Jika kunci belum dikonfigurasi di VPS, API menjawab `503`.
- **Panel web** di alamat yang sama tetap memakai login PIN (terpisah
  dari API key). Aksi yang butuh PIN — sambung Google Drive, upload
  `client_secret.json`, login YouTube — harus dilakukan Bos lewat panel,
  tidak bisa lewat API ini.

## 2. Endpoint

Semua respons berupa JSON dengan field `status` (`"ok"`, `"success"`,
atau `"error"`).

| Method | Path | Fungsi |
|---|---|---|
| GET | `/api/agent/health` | Cek layanan hidup + ffmpeg + ukuran antrean |
| GET | `/api/agent/status` | Ringkasan: jumlah batch, channel, YouTube, Drive, 5 job terakhir |
| GET | `/api/agent/batches` | Daftar batch aktif (`?include_history=1` untuk + riwayat) |
| GET | `/api/agent/jobs` | Daftar job aktif + riwayat (`?limit=1-50`, default 20) |
| POST | `/api/agent/batch` | Buat batch video baru |
| POST | `/api/agent/upload` | Masukkan ulang satu task ke antrean render/upload |
| GET | `/api/agent/batch/<id>` | Pengaturan lengkap (blueprint) satu task aktif — untuk ditiru batch berikutnya |

Tidak ada endpoint hapus/batalkan lewat API ini.

### GET /api/agent/health

```bash
curl -s -H "X-API-Key: <KUNCI_API>" http://<IP-VPS>:5000/api/agent/health
```

Respons:

```json
{
  "status": "ok",
  "service": "keibot",
  "ffmpeg": true,
  "ffmpeg_path": "/usr/bin/ffmpeg",
  "queue_size": 0,
  "uptime_seconds": 12345
}
```

### GET /api/agent/status

```bash
curl -s -H "X-API-Key: <KUNCI_API>" http://<IP-VPS>:5000/api/agent/status
```

Respons (ringkas): `counts` berisi `batches_active`, `queued`,
`rendering_or_uploading`, `queue_size`, `uploaded`, `failed`,
`history_total`, `channels`; `youtube_connected` (bool);
`channels` berisi `{id, name, yt_id, status}`; `drive` berisi status
Drive; `recent_jobs` berisi 5 job terakhir (format Task, lihat §3).

### GET /api/agent/batches

```bash
curl -s -H "X-API-Key: <KUNCI_API>" "http://<IP-VPS>:5000/api/agent/batches?include_history=1"
```

Respons: `{status: "ok", count: N, batches: [Task, ...]}`.

### GET /api/agent/jobs

```bash
curl -s -H "X-API-Key: <KUNCI_API>" "http://<IP-VPS>:5000/api/agent/jobs?limit=10"
```

Respons: `{status: "ok", count: N, jobs: [Task, ...]}`. `limit`
dibatasi 1–50 (default 20).

### POST /api/agent/batch

Membuat batch video baru (jadwal render + upload otomatis).

```bash
curl -s -X POST -H "X-API-Key: <KUNCI_API>" -H "Content-Type: application/json" \
  -d '{"yt_id": "UCxxxx", "count": 3, "interval_days": 1, "start_date": "2026-11-25T19:00", "mp3_per_video": 5, "bg_count": 1}' \
  http://<IP-VPS>:5000/api/agent/batch
```

Body JSON:

| Field | Wajib | Aturan |
|---|---|---|
| `yt_id` | Ya | String, ID channel YouTube |
| `start_date` | Ya | Format `YYYY-MM-DDTHH:MM`, mis. `2026-11-25T19:00` |
| `count` | Tidak (default 1) | Bilangan bulat 1–100 |
| `interval_days` | Tidak (default 1) | Angka > 0 |
| `mp3_per_video` | Tidak (default 5) | Bilangan bulat 1–100 |
| `bg_count` | Tidak (default 1) | Bilangan bulat 1–50 |
| `generated_titles` | Tidak (default []) | Daftar judul |
| `target_durations_array` | Tidak (default []) | Daftar durasi target |

Sukses (200): `{"status": "success", "message": "3 Video diproses!",
"task_ids": [101, 102, 103]}`. Gagal validasi: 400 dengan `message`
penjelasan (mis. `yt_id wajib diisi`, `count harus bilangan bulat
1-100`, `Format tanggal salah`).

### POST /api/agent/upload

Memasukkan ulang satu task yang sudah ada ke antrean render/upload
(mis. setelah gagal upload).

```bash
curl -s -X POST -H "X-API-Key: <KUNCI_API>" -H "Content-Type: application/json" \
  -d '{"batch_id": 101}' \
  http://<IP-VPS>:5000/api/agent/upload
```

`batch_id` wajib berupa angka (id task). Sukses (200):
`{"status": "success", "message": "Task dimasukkan ulang ke antrean
render/upload", "task": {"id": 101, "title": "...", "status": "In
Factory Queue ⚙️"}}`. Gagal: 400 (batch_id kosong/bukan angka) atau
409 (task tidak punya blueprint / sudah ada di antrean).

### GET /api/agent/batch/<id>

Mengambil pengaturan lengkap (blueprint) satu task AKTIF, supaya
agen bisa meniru pengaturannya untuk batch berikutnya ("jadikan
template"): channel, jumlah lagu/background, durasi target,
pengaturan visualizer (`vis_config`), resolusi, Floating Card,
Playlist, Watermark, Timestamp, Smart Cut, dan lain-lain.

```bash
curl -s -H "X-API-Key: <KUNCI_API>" http://<IP-VPS>:5000/api/agent/batch/101
```

Sukses (200): `{"status": "ok", "task": {...}, "blueprint": {...}}`.
Gagal 404: task tidak ditemukan, atau task sudah selesai/tayang —
Keibot menghapus blueprint task yang sudah selesai untuk menghemat
tempat, jadi salin pengaturannya selagi task masih antre. Panel web
punya tombol yang sama: **📋 Template** di tiap baris task aktif
(Task Monitor), yang menerapkan blueprint ke form Batch Creator.

## 3. Format Task

Setiap item di `batches`, `jobs`, dan `recent_jobs`:

| Field | Arti |
|---|---|
| `id` | ID task (angka) |
| `name`, `title` | Judul video |
| `status` | Status bersih (tanpa HTML) |
| `status_raw` | Status apa adanya dari worker |
| `progress` | Perkiraan 0–100, atau `null` bila tidak bisa diperkirakan / sudah selesai-gagal |
| `type` | Tipe task |
| `publish_date` | Jadwal tayang |
| `yt_id` | ID channel YouTube |
| `duration` | Durasi video |
| `finish_time` | Waktu selesai |
| `video_url` | Link youtu.be, hanya bila sudah tayang |

## 4. Kode error

| Kode | Arti |
|---|---|
| 401 | API key salah atau tidak ada (header `X-API-Key`) |
| 400 | Input tidak valid (lihat `message`/`error` di respons) |
| 409 | Task tidak bisa diantre ulang (tidak ada blueprint / sudah antre) |
| 503 | API belum dikonfigurasi — `KEIBOT_AGENT_KEY` kosong di VPS |

## 5. Cara memberi akses ke agen lain

1. Bos memilih agen yang boleh mengakses KeibotMod.
2. Bos memberikan kepada agen itu: (a) alamat panel/VPS,
   (b) kunci API (`KEIBOT_AGENT_KEY`), (c) dokumen ini +
   `openapi.yaml`.
3. Agen itu memanggil endpoint di atas dengan `curl` atau HTTP
   biasa — tidak perlu SDK atau generator.

**Keamanan:** siapa pun yang memegang kunci API punya kendali penuh
(buat batch, antre ulang upload, baca status). Hanya berikan ke agen
yang Bos percaya. Untuk mencabut akses: ganti nilai
`KEIBOT_AGENT_KEY` di `/root/keibotmod/.agent.env` di VPS lalu
restart service `keibot` (`sudo systemctl restart keibot`).
Perhatian: install ulang KeibotMod menghapus file `.agent.env` —
kunci harus dibuat ulang setelah install ulang.

## 6. Batasan yang perlu diketahui

- KeibotMod TIDAK bisa jalan 24/7 dari GitHub Actions atau mesin
  yang tidur (mis. InstaBox) — butuh VPS yang selalu nyala, karena
  antrean, jadwal tayang, dan state tersimpan di disk VPS.
- API ini hanya membaca status dan membuat/mengantre-ulang batch.
  Login YouTube, sambung Drive, dan pengaturan panel tetap lewat
  panel web dengan PIN.
- Channel YouTube harus sudah terhubung di panel sebelum batch
  bisa diunggah (`youtube_connected: false` berarti belum).
