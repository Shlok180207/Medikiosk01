# 🏥 MediKiosk v2 — AI Clinical Intake System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shlok180207/Medikiosk01/blob/main/run_medikiosk.ipynb)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![React](https://img.shields.io/badge/React-18-61dafb)

**MediKiosk v2** is a multimodal clinical intake system designed for hospital outpatient departments (OPD). It conducts multilingual voice consultations, performs clinical reasoning, detects emergency red flags, scans medical reports via vision models, and prepares structured medical summaries for attending physicians.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Google Colab Cloud GPU (T4 / A100)                   │
│                                                                        │
│  🎙️ Faster-Whisper (large-v3)    ── Multilingual Speech-to-Text (~0.5s)│
│  🧠 Ollama Qwen 2.5 (7B)         ── Clinical Triage, Red Flags, ICD-10 │
│  👁️ Moondream                    ── Vision OCR for Prescriptions/Labs  │
│  🔊 gTTS                         ── Pre-cached Multilingual Audio      │
│  ⚡ FastAPI Backend (Port 8000)   ── SQLite Patient Visit Records       │
│  🌐 Cloudflare HTTPS Tunnel      ── https://<subdomain>.trycloudflare.com│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Live HTTPS Connection
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Local Antigravity Workspace / Kiosk UI                 │
│                                                                        │
│  • frontend/.env -> VITE_API_BASE_URL=https://<tunnel-url>/api         │
│  • React + Vite Frontend (Port 5173)                                   │
│  • Microphones, Touchscreens, Scanner UI                               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart: Run on Google Colab GPU (Connected to Antigravity)

### Step 1: Open Notebook in Google Colab
Click the badge below to open the complete backend in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shlok180207/Medikiosk01/blob/main/run_medikiosk.ipynb)

### Step 2: Enable GPU Runtime
1. In Colab, go to **Runtime** > **Change runtime type**.
2. Select **T4 GPU** (or A100 / L4).
3. Click **Save**.

### Step 3: Run All Cells
Run all cells in the notebook (`Runtime` > `Run all` or press `Ctrl + F9`).
This will:
- Install Ollama and pull `qwen2.5:7b`, `qwen2.5:3b`, and `moondream`.
- Load `faster-whisper (large-v3)` directly into GPU VRAM.
- Launch the FastAPI backend on port 8000.
- Establish a secure Cloudflare tunnel and print:
  ```
  =================================================================
  🚀 MEDIKIOSK BACKEND IS LIVE ON GOOGLE COLAB GPU!
  =================================================================
  🌐 Public Base URL:     https://random-words.trycloudflare.com
  🩺 API Endpoint:        https://random-words.trycloudflare.com/api
  📖 Interactive Swagger: https://random-words.trycloudflare.com/docs
  =================================================================
  ```

### Step 4: Connect Your Antigravity Workspace
1. Copy the public API endpoint printed in Colab.
2. In your Antigravity project, open `frontend/.env` and update the URL:
   ```env
   VITE_API_BASE_URL=https://random-words.trycloudflare.com/api
   ```
3. Start the local frontend in the terminal:
   ```bash
   cd frontend
   npm run dev
   ```
4. Open `http://localhost:5173` in your browser. Your local kiosk UI is now streaming voice, image analysis, and clinical reasoning directly through Colab's GPU!

---

## 💻 Alternative: Run 100% Locally (Offline)

If you have a local machine with Python 3.10+ and Node.js:

```bash
# 1. Setup Python backend
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux / macOS
pip install -r requirements.txt

# 2. Install & Start Ollama
ollama serve
ollama pull qwen2.5:3b
ollama pull moondream

# 3. Pre-generate TTS audio & run server
python pre_generate_tts.py
uvicorn main:app --reload --port 8000

# 4. Start frontend
cd frontend
npm install
npm run dev
```

---

## 🌐 Supported Languages
- English (`en`)
- Hindi (`hi`)
- Tamil (`ta`)
- Telugu (`te`)
- Kannada (`kn`)
- Malayalam (`ml`)
- Marathi (`mr`)
- Bengali (`bn`)
- Gujarati (`gu`)
- Punjabi (`pa`)
- Urdu (`ur`)
