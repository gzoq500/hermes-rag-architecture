# Hermes Enterprise BIG RAG Architecture

Repository ini berisi checkpoint lengkap arsitektur Enterprise RAG (Zilliz Cloud + Google Gemini 3072D) untuk Hermes Agent. Sistem ini dirancang untuk menahan _context window_ hingga **500.000 Token** tanpa halusinasi, dan menekan biaya API secara drastis dengan men- _offload_ ingatan lawas ke Vector Database.

Dibangun oleh Kezem untuk Golem.

---

## 🏗️ Topologi Final

*   **Hermes Gateway:** Berjalan dengan `context_length: 500000`. Kompresi bawaan **dimatikan** (`compression: enabled: false`) agar kode 100+ file tidak pernah terpotong/diringkas oleh LLM pusat.
*   **RAG Proxy Worker (Port 20128):** Interceptor berbasis Python FastAPI. Bertugas mencegat API Chat ke 9Router:
    *   **Real-time Ingestion:** Menyedot chat (User/Assistant) dan alat (Tool) menjadi Vektor 3072 Dimensi. Menggunakan "Intelligent Chunking" (potong per baris baru, batas 8K char) agar _script_ utuh.
    *   **RAG Retrieval:** Jika pertanyaan User > 10 karakter, Proxy mencari 3 ingatan teratas di Zilliz, dan otomatis menyuntikkannya ke `System Prompt` dengan tag `[ARCHIVED MEMORY RECALLED FROM ZILLIZ CLOUD]`.
    *   **UI Bypass:** Jika URL mengandung kata `test` atau rute UI Dashboard Next.js, Proxy meneruskannya mentah-mentah ke port 20130.
*   **9Router Asli (Port 20130):** Digeser dari 20128 ke 20130. Berfungsi murni untuk me-_routing_ ke API LLM (cbai/codebuddy/dll).
*   **Zilliz Cloud (AWS ap-southeast-1):** _Serverless Vector Database_, menyimpan koleksi `hermes_gemini_memory`.
*   **Google Gemini API:** Menggunakan model `models/gemini-embedding-2` sebagai mesin _embedder_ (Gratis via Google AI Studio).

---

## 🚀 Panduan Migrasi / Instalasi di VPS Baru

Jika VPS mengalami migrasi atau _wipe_ data, ikuti langkah ini secara berurutan pada VPS yang akan menjadi rumah 9Router (misal VPS 133).

### 1. Persiapan Environment RAG
```bash
# Pastikan Python 3 dan venv terinstall
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv

# Buat Virtual Environment khusus RAG
python3 -m venv /root/rag-venv

# Install library utama (JANGAN pakai pip global agar tidak merusak paket OS)
/root/rag-venv/bin/pip install fastapi uvicorn httpx pymilvus
```

### 2. Konfigurasi Zilliz Cloud (Satu Kali Saja Jika Koleksi Hilang)
Jika Zilliz Cloud kamu juga baru / direset, kamu harus membuat Skema _Database_ nya.
1. Edit file `src/create_schema_gemini.py` dan masukkan **ZILLIZ TOKEN** aslimu.
2. Jalankan:
   ```bash
   /root/rag-venv/bin/python src/create_schema_gemini.py
   ```
*(Ini akan membuat koleksi `hermes_gemini_memory` berdimensi 3072).*

### 3. Deploy Worker RAG
1. Salin `src/main_gemini.py` ke `/root/rag-venv/main_gemini.py`.
2. Edit `/root/rag-venv/main_gemini.py`:
   *   Ganti `<REPLACE_WITH_ZILLIZ_TOKEN>` dengan rahasia Zilliz kamu.
   *   Ganti `<REPLACE_WITH_GOOGLE_GEMINI_KEY>` dengan kunci Google AI Studio.
3. Salin konfigurasi _Service_ Systemd ke OS:
   ```bash
   cp src/rag-worker.service /etc/systemd/system/rag-worker.service
   systemctl daemon-reload
   systemctl enable --now rag-worker.service
   ```

### 4. Menggeser 9Router Asli ke Port 20130
Karena RAG Worker kita mengambil alih singgasana port `20128`, 9Router asli **harus** dipindah.
1. Edit file Systemd milik 9Router (biasanya di `/etc/systemd/system/9router.service`).
2. Cari bagian `ExecStart` dan ubah portnya dari `-p 20128` menjadi `-p 20130`.
   *(Kamu bisa melihat contohnya di file `src/9router-shifted.service` repo ini).*
3. Restart 9Router:
   ```bash
   systemctl daemon-reload
   systemctl restart 9router.service
   ```

### 5. Konfigurasi Agen Hermes (Telegram)
Di `config.yaml` milik Hermes kamu, pastikan _Custom Provider_ menunjuk ke Port 20128 (Jalur RAG), bukan Port 20130:
```yaml
custom_providers:
  - name: Circuit03
    base_url: http://127.0.0.1:20128/v1
    key_env: HERMES_CUSTOM_KEY

agent:
  context_length: 500000

compression:
  enabled: false
```

### 6. Verifikasi Kesuksesan
1. Buka Dashboard 9Router via IP VPS (Port `20128`). Harus bisa login normal tanpa layar putih.
2. Tes koneksi LLM model di menu _Available Models_. Harus hijau.
3. Obrolkan sesuatu ke Telegram Hermes.
4. Cek log `journalctl -u rag-worker.service -f` dan pastikan muncul log `[RAG] Auto-upserted real-time memory`.

Selesai. Sistem Enterprise BIG RAG telah pulih.
