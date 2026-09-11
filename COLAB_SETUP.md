# 🏥 MediKiosk v2 — Google Colab Setup Guide

This guide explains how to run **MediKiosk v2** reliably in **Google Colab** on an NVIDIA T4 GPU using the [`run_colab.ipynb`](file:///d:/VS%20Code%20Projects/Medikiosk01/run_colab.ipynb) production runner.

---

## 🚀 1. Selecting the T4 GPU in Google Colab

MediKiosk runs its offline Speech-to-Text (`faster-whisper`) and Clinical LLM (`qwen2.5:7b`) with GPU acceleration:

1. Open your uploaded [`run_colab.ipynb`](file:///d:/VS%20Code%20Projects/Medikiosk01/run_colab.ipynb) in **Google Colab**.
2. In the top navigation menu, click **Runtime** → **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Set **GPU type** to **Standard** (free tier compatible).
5. Click **Save**.

---

## ▶️ 2. Running the Notebook

1. Once connected to a T4 GPU session, go to **Runtime** → **Run all** (or press `Ctrl+F9`).
2. Alternatively, run the cells in order from top to bottom.
3. The execution will take approximately **5 to 8 minutes** on the initial run (pulling the 4.7 GB `qwen2.5:7b` model and compiling the React frontend bundle).

---

## 📋 3. What Each Stage Does

| Stage | Name | Purpose |
|---|---|---|
| **Stage 0** | **Configuration** | Configures `REPO_URL` (`https://github.com/Shlok180207/Medikiosk-Personal.git`), workspace path (`/content/Medikiosk-Personal`), branch (`main`), and processor mode (`AUTO`, `FORCE_GPU`, `FORCE_CPU`). |
| **Stage 1** | **GPU / CUDA Verification** | Inspects PyTorch version, checks CUDA availability, VRAM, and prints `nvidia-smi`. Stops early if `FORCE_GPU` is requested without a GPU. |
| **Stage 2** | **Clone / Synchronize Repository** | Pulls or synchronizes the repository into `/content/Medikiosk-Personal` and prints commit hash and branch. |
| **Stage 3** | **System Dependencies** | Installs Linux packages: `tesseract-ocr`, language packs (`eng`, `hin`), `zstd`, `curl`, and `ffmpeg`. |
| **Stage 4** | **Python Dependencies** | Preserves Colab's existing CUDA PyTorch without blindly replacing it. Installs `PyMuPDF` (explicitly), `pytesseract`, `faster-whisper`, `torchxrayvision`, `rapidfuzz`, and `fastapi`. |
| **Stage 5** | **Verify OCR & PDF** | Confirms `import fitz` (PyMuPDF) and `pytesseract` work. Confirms working OCR fallback status. |
| **Stage 6** | **Hardware / Model Profile** | Evaluates final environment: allocates `qwen2.5:7b` + CUDA Whisper for GPU mode, or `qwen2.5:1.5b` + CPU Whisper for CPU mode. |
| **Stage 7** | **Install / Start Ollama** | Installs Ollama binary, launches the background daemon (`127.0.0.1:11434`), and verifies responsiveness. |
| **Stage 8** | **Pull & Test Selected LLM** | Downloads `qwen2.5:7b` (or CPU model) into Ollama and executes a warm-up prompt. |
| **Stage 9** | **Test Whisper on GPU** | Instantiates Faster-Whisper on `cuda` with `int8_float16`, verifies model loading, and immediately deloads to free VRAM for the server. |
| **Stage 10** | **Build Frontend** | Runs `npm install && npm run build` inside `frontend/` to generate production SPA bundle in `frontend/dist`. |
| **Stage 11** | **Automated Preflight Tests** | Validates `main.py`, `perception.router`, `perception.document_ocr`, `perception.prescription`, `perception.xray`, `perception.ecg`, and `perception.lab`. Runs synthetic PDF & OCR tests. |
| **Stage 12** | **Start FastAPI Backend** | Cleans up port 8000 and launches `uvicorn main:app` as a background process logging to `/content/fastapi.log` (does not block notebook execution). |
| **Stage 13** | **Local FastAPI Verification** | Queries `http://127.0.0.1:8000/api/health`, `/docs`, and `/api/abha-profiles` to ensure backend is answering requests. |
| **Stage 14** | **Start Cloudflare Tunnel** | Starts a Cloudflare quick tunnel pointing to port 8000 and extracts the live `https://*.trycloudflare.com` URL. |
| **Stage 15** | **Diagnostic Report & URLs** | Outputs a comprehensive diagnostic report and renders clickable buttons for the Kiosk, Doctor Dashboard, and API Docs. |

---

## ⚠️ 4. What to Do If CUDA Is Unavailable

If you see:
```text
❌ ERROR: CUDA is unavailable, but PROCESSOR_MODE is set to 'FORCE_GPU'!
```
or if Stage 1 reports `CUDA available: False`:

1. Check your runtime type:
   - Click **Runtime** → **Change runtime type**.
   - Ensure **T4 GPU** is selected, not **CPU** or **None**.
2. If Colab says "Cannot assign GPU" (due to free tier usage limits):
   - In **Stage 0**, change:
     ```python
     PROCESSOR_MODE = "FORCE_CPU"
     ```
   - Rerun the notebook. The runner will switch to the lightweight CPU profile:
     - LLM: `qwen2.5:1.5b`
     - Whisper: `base` (CPU, int8)
     - Perception: 100% CPU-bound (unchanged)
3. If a GPU was newly selected, restart the runtime:
   - **Runtime** → **Restart session** (or `Ctrl+M .`)
   - Run from **Stage 0**.

---

## 🌐 5. How to Find the Public MediKiosk URL

At the end of the notebook in **Stage 15**, a diagnostic box will display your live public URLs:

```text
============================================================
🔗 MEDIKIOSK ACCESS URLS
============================================================

Public Web Kiosk    : https://random-subdomain.trycloudflare.com
Doctor Dashboard    : https://random-subdomain.trycloudflare.com/doctor
API Swagger Docs    : https://random-subdomain.trycloudflare.com/docs
```

- **Public Web Kiosk**: The patient-facing intake interface (language selection, ABHA identification, conversational voice/text intake, document scanner).
- **Doctor Dashboard**: The clinician interface with synthesized bilingual doctor notes, ABHA history triage, and flagged lab parameters.
- **API Swagger Docs**: Interactive API documentation for testing endpoints directly.

> **Note**: Cloudflare quick tunnels generate a fresh URL on each run. Do not bookmark the previous run's URL.

---

## 🔍 6. How to Diagnose Document-Processing Failures

If document processing fails during patient intake or in Stage 11:

### 1. Check Module Preflight Output (Stage 11)
Stage 11 verifies that all perception pipelines can be imported:
- `[PASS] main`
- `[PASS] PDF/OCR` (PyMuPDF `fitz` and `document_ocr`)
- `[PASS] prescription OCR` (Sauvola + RapidFuzz drug lexicon)
- `[PASS] lab pipeline` (`perception.lab.analyze_lab_report`)
- `[PASS] X-ray pipeline` (`torchxrayvision` DenseNet-121)
- `[PASS] ECG pipeline` (OpenCV HSV grid stripping + SciPy peak detection)

### 2. Verify `PyMuPDF` (`fitz`)
If you encounter `No module named 'fitz'`:
- Verify `PyMuPDF` is installed:
  ```bash
  pip install -U PyMuPDF
  ```
- **Never** run `pip install fitz` (that installs an unrelated incompatible package).

### 3. Verify Tesseract OCR
If prescription or lab OCR fails to detect text:
- Check that the system binary is present:
  ```bash
  which tesseract
  tesseract --version
  ```
- Check language packs:
  ```bash
  tesseract --list-langs
  # Should list 'eng', 'hin', 'osd'
  ```

### 4. Inspect Live Server Logs
Use the utility cell at the bottom of the notebook:
```python
!tail -n 50 /content/fastapi.log
```
Or view the Ollama daemon logs:
```python
!tail -n 50 /content/ollama_daemon.log
```

---

## 🛑 7. Stopping Services

When finished with your Colab session, run the cleanup cell:
```python
!pkill -f cloudflared
!pkill -f uvicorn
!pkill -f "ollama serve"
```
Or disconnect the runtime via **Runtime** → **Disconnect and delete runtime**.
