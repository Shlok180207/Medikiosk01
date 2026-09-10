# MediKiosk v2 — Project Memory & Architecture Guide

> **Primary Purpose**: Single-source-of-truth project memory to prevent high context bloat, slow tool searches, and token memory exhaustion when working with this repository.

---

## 1. Project Overview & Architecture

- **Project Name**: MediKiosk v2 (Smart India Hackathon 2026 - Edge AI Kiosk)
- **Concept**: 100% Offline AI-powered Clinical Intake Kiosk with multilingual speech intake, smart symptom investigation, vision document OCR, red-flag screening, ABHA history triage, and a Doctor Dashboard.
- **Tech Stack**:
  - **Backend**: Python FastAPI, SQLite (`medikiosk_v2.db`), SQLAlchemy, Uvicorn
  - **Frontend**: React 19, Vite 8, React Router 7, Vanilla CSS / Tailwind utilities, Oxlint
  - **AI Engines (Edge / Offline)**:
    - **Speech-to-Text (STT)**: Faster-Whisper (`medium`/`small` on CUDA GPU, int8, explicit lifecycle deloading)
    - **LLM Reasoning**: Qwen2.5-7B-Instruct (4k context via local llama.cpp / Ollama)
    - **Diagnostic Vision Perception (100% CPU-Bound, Zero VRAM)**:
      - Chest X-Ray: `torchxrayvision` DenseNet-121 on CPU (`perception/xray.py`)
      - 12-Lead ECG: OpenCV HSV grid stripping & SciPy peak detection on CPU (`perception/ecg.py`)
      - Prescription & Slip OCR: Sauvola adaptive thresholding, Dual OCR (TrOCR/PaddleOCR CPU), and RapidFuzz drug normalizer (`perception/prescription.py`)
    - **Drug Lexicon**: `data/indian_drug_lexicon.json` (25+ common Indian brands & generics, defensive unlisted drug flagging)
    - **Text-to-Speech (TTS)**: gTTS / pre-generated audio assets
- **Ports & Services**:
  - Backend API: `http://localhost:8000` (docs at `http://localhost:8000/docs`)
  - Frontend Dev Server: `http://localhost:5173`
  - Ollama Daemon: `http://127.0.0.1:11434`

---

## 2. Monolithic Backend Index (`main.py` ~2,836 lines)

**CRITICAL RULE FOR AGENTS**: Never dump or read all 2,836 lines of `main.py` at once! Always use targeted line slices.

| Line Range | Component / Feature | Details |
|---|---|---|
| **1 – 67** | Imports & Ollama Auto-Launcher | Startup socket check on 11434, auto-launch background Ollama |
| **68 – 118** | Faster-Whisper Pipeline | `get_whisper_pipeline()`, CUDA int8_float16, `unload_whisper()` to free VRAM |
| **119 – 230** | LLM Helpers & Prompt Wrappers | `call_llm()`, JSON extractor, cleanup helpers, 1m keep-alive |
| **231 – 568** | Hardcoded Multilingual Dictionaries | Hindi, English, Tamil, Telugu fallback questions & triage prompts |
| **569 – 606** | Pydantic Schemas | `PatientExtraction`, `FollowUpResponse`, `DocumentExtraction` |
| **607 – 676** | Database Models (SQLAlchemy) | `PatientRecord` (intake details, clinical impression, vitals), `VisitHistory` (ABHA records) |
| **677 – 700** | FastAPI Setup & CORS | `app = FastAPI(title="MediKiosk v2 API")`, CORS origins `*` |
| **696 – 1416** | Pre-generated TTS & Audio Handling | Dynamic TTS generator & cache (`/api/tts`) |
| **1417 – 1464** | ABHA Profile Endpoints | `GET /api/abha-profiles`, `GET /api/abha-profile/{abha_id}` |
| **1465 – 1529** | Voice Intake STT | `POST /api/process-audio` (multipart audio -> Whisper -> text) |
| **1530 – 1618** | Text Intake & Symptom Analysis | `POST /api/process-text` (initial triage, chief complaint extraction) |
| **1619 – 1646** | Follow-Up Dynamic Questions (Text) | `POST /api/follow-up-text` (conversational follow-up generation) |
| **1647 – 2386** | Follow-Up Dynamic Questions (Audio) | `POST /api/follow-up` (speech follow-up + TTS audio return) |
| **2220 – 2443** | Medical Document Scan & Perception | `POST /api/process-document` (CPU X-ray, ECG, and Prescription perception) |
| **2444 – 2464** | Finalize Intake | `POST /api/finalize-intake` (saves patient to SQLite database) |
| **2465 – 2485** | Resynthesize Clinical Notes | `POST /api/resynthesize` & `GET /api/resynthesize` |
| **2486 – 2529** | Red Flag Screener | `GET /api/red-flag-check` (critical triage flags like chest pain, stroke, sepsis) |
| **2530 – 2583** | Specialty Matching | `GET /api/specialty-match` (recommends Cardiology, Orthopedics, etc.) |
| **2584 – 2609** | Patient Management | `GET /api/patients`, `DELETE /api/patients/{patient_id}` |
| **2610 – 2682** | Doctor Clinical Summary | `GET /api/patient-summary` (structured summary for doctor review) |
| **2683 – 2743** | ABHA History & Triage | `GET /api/patient-history` (context-aware past records filtering) |
| **2744 – 2836** | Demo Data Seed & Main Server | `POST /api/demo-data`, `uvicorn.run(app, host="0.0.0.0", port=8000)` |

---

## 3. Frontend Map (`frontend/src`)

- **State Management**: `frontend/src/context/AppContext.jsx` (holds active patient state, language, audio state, ABHA profile).
- **Styling**: `frontend/src/index.css` (custom design system, glassmorphism, glowing badges, dark/light clinical UI).
- **Internationalization**: `frontend/src/utils/translations.js` (English, Hindi, Tamil, Telugu labels & UI copy).

### Routes (`frontend/src/App.jsx`)
1. `/` -> `pages/Landing.jsx` (Welcome screen & kiosk wake-up)
2. `/language` -> `pages/LanguageSelect.jsx` (English / Hindi / Tamil / Telugu selection)
3. `/patient-id` -> `pages/PatientID.jsx` (ABHA ID scan, manual entry, or Quick Demo selector)
4. `/consent` -> `pages/Consent.jsx` (Digital consent for data capture & ABHA sync)
5. `/clinical-mode` -> `pages/ClinicalMode.jsx` (Standard vs Express triage flow)
6. `/kiosk` -> `pages/Kiosk.jsx` (Voice & text conversational intake interface)
7. `/document-scan` -> `pages/DocumentScan.jsx` (Webcam/upload prescription/lab report OCR)
8. `/red-flag` -> `pages/RedFlag.jsx` (Urgent clinical alert page if emergency detected)
9. `/specialty` -> `pages/Specialty.jsx` (Recommended OPD specialty)
10. `/providers` -> `pages/Providers.jsx` (Matching doctor queue & token generation)
11. `/summary` -> `pages/Summary.jsx` (Patient-facing token & intake receipt)
12. `/patient-final` -> `pages/PatientFinal.jsx` (Final submission confirmation)
13. `/doctor` -> `pages/Dashboard.jsx` (Comprehensive Doctor Dashboard with ABHA history, clinical impression, lab flagging)

---

## 4. Key Files & Data Assets

- `dummy_abha_patients.txt`: 5 pre-configured ABHA profiles (Ramesh Sharma, Priya Patel, etc.) for testing context-aware triage.
- `medikiosk_v2.db`: SQLite database storing patients, intake records, and visit histories.
- `pre_generate_tts.py`: Script to pre-generate offline audio prompts for low-latency kiosk speech.
- `scripts/extract_doc_vl.py`: Helper script for vision document extraction experiments.

---

## 5. Token & Agent Performance Guidelines

To avoid connection timeouts (`wsarecv: An established connection was aborted`) and sluggish response times:
1. **Never read `main.py` in full**. Refer to the table in Section 2 and use `StartLine` and `EndLine` to read only the necessary ~50–100 lines.
2. **Never re-grep the entire repository**. Use specific target directories (`frontend/src/pages`, `frontend/src/components`, etc.).
3. **Keep edits atomic**. Use `replace_file_content` for surgical updates instead of rewriting large files.
4. **Preserve this file**. Keep `PROJECT_MEMORY.md` and `AGENTS.md` up-to-date whenever new routes or pages are added.
