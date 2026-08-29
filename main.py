"""
MediKiosk v2 — AI-Powered Clinical Intake Backend
100% Offline: Whisper (STT) + Ollama qwen2.5:3b (NLP) + llama3.2-vision (OCR)
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional
import os, uuid, json, re, io, tempfile, base64, hashlib
from gtts import gTTS
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

# ── Offline AI Models ──
import ollama
import torch
from transformers import pipeline

OLLAMA_MODEL = "qwen2.5:7b"  # Switched to qwen2.5:7b for faster performance and great native Hindi.
VISION_MODEL = "moondream"

# Load Whisper
device_id = 0 if torch.cuda.is_available() else -1
print(f"Loading Whisper on {'CUDA' if device_id == 0 else 'CPU'}...")
try:
    from faster_whisper import WhisperModel
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device_type == "cuda" else "int8"
    whisper_pipeline = WhisperModel("large-v3", device=device_type, compute_type=compute_type)
    print("✅ Whisper loaded.")
except Exception as e:
    print(f"❌ Whisper failed: {e}")
    whisper_pipeline = None


# ── LLM Helpers ──
def call_llm(prompt: str, image_bytes: Optional[bytes] = None) -> str:
    """Call Ollama. If image_bytes provided, uses llama3.2-vision."""
    messages = [{'role': 'user', 'content': prompt}]
    model = OLLAMA_MODEL

    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        messages[0]['images'] = [b64]
        model = VISION_MODEL

    print(f"→ Ollama ({model})...")
    response = ollama.chat(model=model, messages=messages, format='json')
    return response['message']['content']


def extract_json_string(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def unwrap_json(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    keys = list(data.keys())
    if len(keys) == 1 and isinstance(data[keys[0]], dict):
        return data[keys[0]]
    if "properties" in data and isinstance(data["properties"], dict):
        return data["properties"]
    return data


# ── Language Codes ──
LANGUAGE_CODES = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa", "Urdu": "ur"
}

# ── Pre-written Follow-up Questions (human-verified, no AI typos) ──
PHASE_QUESTIONS = {
    "Hindi": {
        "initial": "यह तकलीफ़ आपको कब से हो रही है?",
        0: "क्या यह दर्द शरीर के किसी और हिस्से में भी जाता है?",
        1: "इसके साथ और कोई तकलीफ़ है? जैसे बुखार, उल्टी, या कमज़ोरी?",
        2: "क्या आपने इसके लिए कोई दवा ली है? किसी चीज़ से आराम मिलता है या तकलीफ़ बढ़ती है?",
        3: "क्या आपको पहले कोई बीमारी रही है? परिवार में किसी को कोई बीमारी है? क्या आप धूम्रपान या शराब का सेवन करते हैं?",
        4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
        "default": "और कुछ बताना चाहेंगे?"
    },
    "English": {
        "initial": "How long have you been experiencing this problem?",
        0: "Does this pain spread or travel to any other part of your body?",
        1: "Are you experiencing any other symptoms like fever, nausea, or weakness?",
        2: "Have you taken any medicine for this? Does anything make it better or worse?",
        3: "Do you have any past medical conditions? Any diseases in your family? Do you smoke or drink alcohol?",
        4: "Are you allergic to any medicines or foods?",
        "default": "Is there anything else you would like to tell me?"
    },
    "Tamil": {
        "initial": "இந்த பிரச்சனை எவ்வளவு நாளாக இருக்கிறது?",
        0: "இந்த வலி உடலின் வேறு எந்த பகுதிக்கும் பரவுகிறதா?",
        1: "இதனுடன் காய்ச்சல், குமட்டல் அல்லது பலவீனம் போன்ற வேறு ஏதாவது தொந்தரவு இருக்கிறதா?",
        2: "இதற்கு ஏதாவது மருந்து எடுத்துக்கொண்டீர்களா? எதனால் சரியாகிறது அல்லது மோசமாகிறது?",
        3: "உங்களுக்கு முன்பு ஏதாவது நோய் இருந்ததா? குடும்பத்தில் யாருக்காவது நோய் இருக்கிறதா? புகைபிடிப்பீர்களா அல்லது மது அருந்துவீர்களா?",
        4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
        "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
    },
    "Telugu": {
        "initial": "ఈ సమస్య మీకు ఎంత కాలంగా ఉంది?",
        0: "ఈ నొప్పి శరీరంలో ఇతర భాగాలకు వ్యాపిస్తుందా?",
        1: "దీనితో పాటు జ్వరం, వాంతులు లేదా బలహీనత వంటి ఇతర సమస్యలు ఉన్నాయా?",
        2: "దీని కోసం ఏదైనా మందు వాడారా? దేనివల్ల తగ్గుతుంది లేదా పెరుగుతుంది?",
        3: "మీకు ఇంతకు ముందు ఏదైనా వ్యాధి ఉందా? కుటుంబంలో ఎవరికైనా వ్యాధి ఉందా? మీరు పొగ తాగుతారా లేదా మద్యం సేవిస్తారా?",
        4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
        "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
    },
    "Bengali": {
        "initial": "এই সমস্যা আপনার কতদিন ধরে হচ্ছে?",
        0: "এই ব্যথা কি শরীরের অন্য কোনো জায়গায় ছড়ায়?",
        1: "এর সাথে জ্বর, বমি বা দুর্বলতার মতো অন্য কোনো সমস্যা আছে?",
        2: "এর জন্য কি কোনো ওষুধ খেয়েছেন? কিসে আরাম হয় বা কষ্ট বাড়ে?",
        3: "আগে কি কোনো রোগ ছিল? পরিবারে কারো কি কোনো রোগ আছে? আপনি কি ধূমপান বা মদ্যপান করেন?",
        4: "আপনার কি কোনো ওষুধ বা খাবারে অ্যালার্জি আছে?",
        "default": "আর কিছু বলতে চান?"
    },
    "Marathi": {
        "initial": "ही तकलीफ तुम्हाला कधीपासून होत आहे?",
        0: "हा दुखणे शरीराच्या इतर कोणत्या भागात जातो का?",
        1: "याबरोबर ताप, उलटी किंवा अशक्तपणा असे काही त्रास आहे का?",
        2: "यासाठी काही औषध घेतले का? कशामुळे आराम पडतो किंवा त्रास वाढतो?",
        3: "तुम्हाला आधी काही आजार होता का? कुटुंबात कोणाला काही आजार आहे का? तुम्ही धूम्रपान किंवा दारू पिता का?",
        4: "तुम्हाला कोणत्या औषधाची किंवा खाण्याच्या पदार्थाची ऍलर्जी आहे का?",
        "default": "अजून काही सांगायचे आहे का?"
    },
}

def get_phase_question(language: str, phase) -> str:
    """Get a pre-written question template. Falls back to English if language not available."""
    lang_questions = PHASE_QUESTIONS.get(language, PHASE_QUESTIONS.get("English", {}))
    return lang_questions.get(phase, lang_questions.get("default", "Is there anything else you would like to tell me?"))

# ── Pydantic Models ──
PATIENT_JSON_TEMPLATE = """{
  "chief_complaint": "Main symptom (in English only)",
  "hpi": "History of present illness (in English only)",
  "is_emergency": false,
  "severity": "Low|Medium|High",
  "duration": "Duration of symptoms (in English)",
  "past_medical_history": "Any past conditions or 'None reported' (in English)",
  "family_history": "Family medical history or 'None reported' (in English)",
  "personal_history": "Smoking, alcohol, diet, lifestyle or 'None reported' (in English)",
  "allergies": "Known allergies or 'None reported' (in English)",
  "review_of_systems": "Other symptoms or 'None reported' (in English)",
  "prakriti": "Not assessed",
  "vikriti": "Not assessed",
  "agni": "Not assessed",
  "next_question": "Generated first question in patient's language"
}"""

FOLLOWUP_JSON_TEMPLATE = """{
  "updates": {
    "hpi": null,
    "past_medical_history": null,
    "family_history": null,
    "personal_history": null,
    "allergies": null,
    "review_of_systems": null,
    "prakriti": null,
    "vikriti": null,
    "agni": null
  },
  "is_complete": false,
  "next_question": "Generated question in patient's language"
}"""

DOCUMENT_JSON_TEMPLATE = """{
  "document_type": "Name or type of report (e.g. CBC, MRI, Prescription)",
  "diagnoses": ["list of diagnoses"],
  "medications": ["medicine with dose"],
  "flagged_values": ["abnormal lab values"],
  "document_date": "date or Unknown",
  "summary": "Brief summary"
}"""

class PatientExtraction(BaseModel):
    chief_complaint: str
    hpi: Optional[str] = "None reported"
    is_emergency: bool = False
    severity: str = "Low"
    duration: str = "Unknown"
    past_medical_history: Optional[str] = "None reported"
    family_history: Optional[str] = "None reported"
    personal_history: Optional[str] = "None reported"
    allergies: Optional[str] = "None reported"
    review_of_systems: Optional[str] = "None reported"
    prakriti: Optional[str] = "Not assessed"
    vikriti: Optional[str] = "Not assessed"
    agni: Optional[str] = "Not assessed"
    next_question: Optional[str] = "Could you tell me more about this issue?"

class FollowUpResponse(BaseModel):
    updates: dict = {}
    is_complete: bool = False
    next_question: Optional[str] = "Is there anything else?"

class DocumentExtraction(BaseModel):
    document_type: str = "Medical Document"
    diagnoses: List[str] = []
    medications: List[str] = []
    flagged_values: List[str] = []
    document_date: str = "Unknown"
    summary: str = ""


# ── Database ──
SQLALCHEMY_DATABASE_URL = "sqlite:///./medikiosk_v2.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    chief_complaint = Column(Text)
    hpi = Column(Text)
    is_emergency = Column(Boolean, default=False)
    severity = Column(String)
    duration = Column(String)
    past_medical_history = Column(String)
    family_history = Column(String)
    personal_history = Column(String)
    allergies = Column(String)
    review_of_systems = Column(Text)
    prakriti = Column(String)
    vikriti = Column(String)
    agni = Column(String)
    flagged_lab_values = Column(Text, default="[]")
    created_at = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── FastAPI App ──
app = FastAPI(title="MediKiosk v2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ── TTS (Text-to-Speech) Endpoint with Local Cache ──
TTS_CACHE_DIR = os.path.join("uploads", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

TTS_LANG_MAP = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur"
}

@app.get("/api/tts")
async def get_tts(text: str, lang: str = "hi"):
    """Text-to-speech with local audio caching for 100% offline playback."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    lang_code = TTS_LANG_MAP.get(lang, lang.lower())
    if len(lang_code) > 2 and '-' in lang_code:
        lang_code = lang_code.split('-')[0]

    cache_key = hashlib.md5(f"{lang_code}_{text.strip()}".encode("utf-8")).hexdigest()
    cache_file = os.path.join(TTS_CACHE_DIR, f"{cache_key}.mp3")

    if not os.path.exists(cache_file):
        try:
            tts = gTTS(text=text.strip(), lang=lang_code)
            tts.save(cache_file)
        except Exception as e:
            print(f"TTS generation error: {e}")
            try:
                tts = gTTS(text=text.strip(), lang="en")
                tts.save(cache_file)
            except Exception as e2:
                print(f"TTS fallback error: {e2}")
                raise HTTPException(status_code=500, detail="TTS generation failed")

    return FileResponse(cache_file, media_type="audio/mpeg")



# ── Core: Build patient from transcript ──
def build_patient_from_transcript(transcript, language, is_ayush, pt_id, db):
    ayush_inst = (
        "The setting is Ayurvedic OPD. Assess Prakriti, Vikriti, Agni if evident."
        if is_ayush else
        "The setting is standard Allopathic. Set prakriti, vikriti, agni to 'Not assessed'."
    )

    prompt = f"""You are a medical data extraction AI. The patient is speaking {language}.
{ayush_inst}

Patient's transcript: "{transcript}"

TASK: Extract ALL clinical information from the transcript into the structured JSON fields.
CRITICAL RULES:
1. ALL extracted values MUST be translated to standard medical English.
2. DO NOT include any {language} text in the extracted values. The JSON values MUST be strictly English.
3. DO NOT invent details. If something is not mentioned, use 'None reported' or 'Unknown'.
4. PHONETIC ERROR CORRECTION: The transcript is generated by an AI speech-to-text tool and may contain phonetic errors (e.g. "bete" instead of "pait/stomach"). You MUST interpret the transcript in a strictly MEDICAL context and auto-correct these errors before extracting symptoms.
5. Set `next_question` to "next" (the system will handle question generation).
6. Ensure the output is strictly valid JSON.

Output ONLY valid JSON:
{PATIENT_JSON_TEMPLATE}"""

    response_text = call_llm(prompt)
    result_json = json.loads(extract_json_string(response_text))
    result_json = unwrap_json(result_json)
    extraction = PatientExtraction(**result_json)

    patient = PatientRecord(
        patient_id=pt_id,
        chief_complaint=extraction.chief_complaint,
        hpi=extraction.hpi,
        is_emergency=extraction.is_emergency,
        severity=extraction.severity,
        duration=extraction.duration,
        past_medical_history=extraction.past_medical_history,
        family_history=extraction.family_history,
        personal_history=extraction.personal_history,
        allergies=extraction.allergies,
        review_of_systems=extraction.review_of_systems,
        prakriti=extraction.prakriti,
        vikriti=extraction.vikriti,
        agni=extraction.agni,
        created_at=datetime.now().strftime("%I:%M %p")
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    # Use SOCRATES-framework question template
    initial_question = get_phase_question(language, "initial")
    return patient, initial_question


# ═══════════════ ENDPOINTS ═══════════════

@app.get("/")
async def root():
    return {"message": "MediKiosk v2 Backend running (Offline Mode)"}


# ── Initial Complaint (Audio) ──
@app.post("/api/process-audio")
async def process_audio(
    audio: UploadFile = File(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    db: Session = Depends(get_db)
):
    audio_bytes = await audio.read()
    pt_id = f"PT-{str(uuid.uuid4())[:4].upper()}"

    # 1. Whisper STT
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    lang_code = LANGUAGE_CODES.get(language, "en")
    segments, info = whisper_pipeline.transcribe(
        tmp_path, 
        language=lang_code, 
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False
    )
    transcript = " ".join([segment.text for segment in segments]).strip()
    os.remove(tmp_path)
    print(f"Transcript: {transcript}")

    # 2. LLM extraction
    patient, next_q = build_patient_from_transcript(transcript, language, is_ayush, pt_id, db)

    return {
        "status": "success",
        "extracted_complaint": patient.chief_complaint,
        "is_emergency": patient.is_emergency,
        "patient_id": patient.patient_id,
        "transcript": transcript,
        "next_question": next_q
    }


# ── Initial Complaint (Text) ──
@app.post("/api/process-text")
async def process_text(
    transcript: str = Form(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    db: Session = Depends(get_db)
):
    pt_id = f"PT-{str(uuid.uuid4())[:4].upper()}"
    patient, next_q = build_patient_from_transcript(transcript, language, is_ayush, pt_id, db)

    return {
        "status": "success",
        "extracted_complaint": patient.chief_complaint,
        "is_emergency": patient.is_emergency,
        "patient_id": patient.patient_id,
        "transcript": transcript,
        "next_question": next_q
    }


def handle_followup_extraction(
    patient: PatientRecord,
    transcript: str,
    language: str,
    conversation_context: str,
    is_ayush: bool,
    follow_up_count: int,
    db: Session
):
    ayush_extra = " Also probe for Ayurvedic parameters if relevant." if is_ayush else ""

    answering_phase_desc = {
        0: "The patient is answering about ONSET and DURATION of symptoms. Extract into 'duration' and 'hpi'.",
        1: "The patient is answering whether pain RADIATES or spreads to other parts of the body. Extract into 'hpi' (e.g. 'No radiation to other body parts' if denied, or specify where it radiates).",
        2: "The patient is answering REVIEW OF SYSTEMS (associated symptoms: fever, nausea, vomiting, weakness, etc.). Extract into 'review_of_systems'. If patient denies or says no/none, write: 'Patient denies fever, nausea, vomiting, weakness, or other associated symptoms'.",
        3: "The patient is answering about MEDICATIONS taken and AGGRAVATING/RELIEVING factors. Extract into 'hpi'. If patient denies/none, write: 'No medications taken; no aggravating or relieving factors reported'.",
        4: "The patient is answering about PAST MEDICAL HISTORY, FAMILY HISTORY, and LIFESTYLE (smoking, alcohol, diet). Extract into 'past_medical_history', 'family_history', and 'personal_history'. If no past medical history, write: 'No significant past medical history reported'.",
        5: "The patient is answering about ALLERGIES (medicines, food). Extract into 'allergies'. If patient denies or says no/none, write: 'No known drug or food allergies (NKDA)'."
    }.get(follow_up_count, "Extract any relevant clinical information.")

    prompt = f"""You are a medical data extraction AI. The patient is speaking {language}.
You are reviewing follow-up answers for a patient whose chief complaint is: "{patient.chief_complaint}".

CURRENT CLINICAL FOCUS:
{answering_phase_desc}

Previous conversation context:
{conversation_context}

Patient's latest answer: "{transcript}"

TASK:
1. Extract any NEW clinical information from the patient's latest answer into the "updates" dictionary.
2. CRITICAL: The extracted information inside the "updates" dictionary MUST BE TRANSLATED TO ENGLISH.
3. PHONETIC ERROR CORRECTION: Interpret speech-to-text errors in a strictly MEDICAL context.
4. NEGATIVE RESPONSES: If the patient denies symptoms or says "no" / "none" / "नहीं":
   - For Review of Systems (step 2): write "Patient denies fever, nausea, vomiting, weakness, or other associated symptoms"
   - For Allergies (step 5): write "No known drug or food allergies (NKDA)"
   - For Radiation (step 1): write "No radiation to other body parts"
   - For Past History (step 4): write "No significant past medical history reported"
5. Do NOT leave a category as null if the patient answered it (even if negative/denial).
6. Set `next_question` to "next".
7. Set `is_complete` to false.

Output ONLY valid JSON:
{FOLLOWUP_JSON_TEMPLATE}"""

    try:
        response_text = call_llm(prompt)
        result_json = json.loads(extract_json_string(response_text))
        result_json = unwrap_json(result_json)
        fu_result = FollowUpResponse(**result_json)
    except Exception as e:
        print(f"Follow-up error: {e}")
        fu_result = FollowUpResponse(updates={}, is_complete=False)

    updates = fu_result.updates if isinstance(fu_result.updates, dict) else {}

    # Deterministic safeguard: if patient answered No/negative to a specific phase
    t_clean = transcript.strip().lower()
    is_neg = t_clean in ["no", "nahi", "nahin", "नहीं", "no allergies", "none", "nothing", "kuch nahi", "kuch nahi hai", "na", "not sure", "न", "ना", "n"]

    if follow_up_count == 2:
        ros_val = updates.get("review_of_systems")
        if not ros_val or ros_val.strip().lower() in ["null", "none", "none reported", "unknown", ""]:
            if is_neg:
                updates["review_of_systems"] = "Patient denies fever, nausea, vomiting, weakness, or other associated symptoms"
            elif len(transcript.strip()) > 0:
                updates["review_of_systems"] = f"Reported: {transcript.strip()}"

    elif follow_up_count == 5:
        allg_val = updates.get("allergies")
        if not allg_val or allg_val.strip().lower() in ["null", "none", "none reported", "unknown", ""]:
            if is_neg:
                updates["allergies"] = "No known drug or food allergies (NKDA)"
            elif len(transcript.strip()) > 0:
                updates["allergies"] = f"Reported: {transcript.strip()}"

    elif follow_up_count == 1:
        if is_neg and not updates.get("hpi"):
            updates["hpi"] = "No radiation to other body parts"

    elif follow_up_count == 4:
        pmh_val = updates.get("past_medical_history")
        if not pmh_val or pmh_val.strip().lower() in ["null", "none", "none reported", "unknown", ""]:
            updates["past_medical_history"] = "No significant past medical history reported"

    # Apply updates to patient record in DB
    for field, info in updates.items():
        if info and isinstance(info, str) and info.strip().lower() not in ["null", "none", ""]:
            if "updated info in english" in info.lower() or "updated hpi" in info.lower():
                continue
            if hasattr(patient, field):
                current = getattr(patient, field) or ""
                info_clean = info.lstrip('• ').strip()
                if current and current not in ("None reported", "Not assessed", "Unknown", ""):
                    setattr(patient, field, current.strip() + "\n• " + info_clean)
                else:
                    setattr(patient, field, "• " + info_clean)

    db.commit()
    db.refresh(patient)

    is_complete = True if follow_up_count >= 5 else False
    next_question = get_phase_question(language, follow_up_count) if follow_up_count < 5 else "Thank you. Let us proceed to document scanning."

    return {
        "status": "success",
        "extracted_info": " | ".join(f"{k}: {v}" for k, v in updates.items() if v and str(v).lower() not in ["null", ""]),
        "is_complete": is_complete,
        "next_question": next_question,
        "transcript": transcript
    }


# ── Follow-up (Text) ──
@app.post("/api/follow-up-text")
async def follow_up_text(
    transcript: str = Form(...),
    language: str = Form("English"),
    patient_id: str = Form(...),
    conversation_context: str = Form(""),
    is_ayush: bool = Form(False),
    follow_up_count: int = Form(0),
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return handle_followup_extraction(
        patient=patient,
        transcript=transcript,
        language=language,
        conversation_context=conversation_context,
        is_ayush=is_ayush,
        follow_up_count=follow_up_count,
        db=db
    )


# ── Follow-up (Audio) ──
@app.post("/api/follow-up")
async def follow_up_audio(
    audio: UploadFile = File(...),
    language: str = Form("English"),
    patient_id: str = Form(...),
    conversation_context: str = Form(""),
    is_ayush: bool = Form(False),
    follow_up_count: int = Form(0),
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    lang_code = LANGUAGE_CODES.get(language, "en")
    segments, info = whisper_pipeline.transcribe(
        tmp_path, 
        language=lang_code, 
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False
    )
    transcript = " ".join([segment.text for segment in segments]).strip()
    os.remove(tmp_path)
    print(f"Follow-up transcript: {transcript}")

    return handle_followup_extraction(
        patient=patient,
        transcript=transcript,
        language=language,
        conversation_context=conversation_context,
        is_ayush=is_ayush,
        follow_up_count=follow_up_count,
        db=db
    )


def process_document_background(file_bytes: bytes, filename: str, content_type: str, file_url: str, patient_id_db: int):
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        if not patient:
            return
            
        structured_data = None
        extracted_text = ""
        try:
            prompt_base = """Analyze this medical document carefully.

Extract into JSON:
- document_type: Name or type of report (e.g. CBC, MRI, Prescription)
- diagnoses: list of clinical diagnoses (English)
- medications: list of medicines with dosages (English)
- flagged_values: list of abnormal lab values or critical findings
- document_date: date on document, or 'Unknown'
- summary: concise summary of key findings

Output ONLY valid JSON:
""" + DOCUMENT_JSON_TEMPLATE

            if filename.lower().endswith('.pdf') or content_type == 'application/pdf':
                import fitz
                pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in pdf_doc:
                    extracted_text += page.get_text() + "\n"
                prompt = f"Extracted Text:\n{extracted_text}\n\n{prompt_base}"
                response_text = call_llm(prompt)
            else:
                print(f"Using {VISION_MODEL} for image in background...")
                response_text = call_llm(prompt_base, image_bytes=file_bytes)

            result_json = json.loads(extract_json_string(response_text))
            result_json = unwrap_json(result_json)
            extraction = DocumentExtraction(**result_json)
            structured_data = {
                "document_type": extraction.document_type,
                "diagnoses": extraction.diagnoses,
                "medications": extraction.medications,
                "flagged_values": extraction.flagged_values,
                "document_date": extraction.document_date,
                "summary": extraction.summary,
                "file_url": file_url,
                "raw_text": extracted_text
            }
            print(f"Background Extracted: {structured_data}")
        except Exception as e:
            print(f"Background Document processing failed: {e}")

        if not structured_data:
            structured_data = {
                "document_type": "Unknown Document",
                "diagnoses": ["Extraction failed — please try again"],
                "medications": [],
                "flagged_values": [],
                "document_date": "Unknown",
                "summary": "Could not process this document. Please try a clearer image.",
                "file_url": file_url,
                "raw_text": extracted_text
            }

        existing = []
        if patient.flagged_lab_values and patient.flagged_lab_values != "[]":
            try:
                parsed = json.loads(patient.flagged_lab_values)
                if isinstance(parsed, list):
                    existing = [i for i in parsed if isinstance(i, dict)]
            except:
                pass
        existing.append(structured_data)
        patient.flagged_lab_values = json.dumps(existing)
        db.commit()
    finally:
        db.close()


# ── Document Processing ──
@app.post("/api/process-document")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    patient = None
    if patient_id:
        patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        patient = db.query(PatientRecord).order_by(PatientRecord.id.desc()).first()
    if not patient:
        raise HTTPException(status_code=404, detail="No patient found")

    file_bytes = await file.read()
    print(f"Document received: {file.filename}, {len(file_bytes)} bytes. Dispatching background task.")

    # Save file for viewing later
    os.makedirs("uploads", exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or '.png'
    saved_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join("uploads", saved_filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    # We will pass the full url, assuming frontend is on same host or API is absolute
    base_url = "http://localhost:8000" 
    file_url = f"{base_url}/uploads/{saved_filename}"

    # Dispatch to background task
    background_tasks.add_task(
        process_document_background,
        file_bytes,
        file.filename,
        file.content_type,
        file_url,
        patient.id
    )

    return {
        "status": "success", 
        "message": "Document is being processed asynchronously.",
        "extracted_document": {
            "document_type": "Processing...",
            "diagnoses": ["Analyzing document in background..."],
            "medications": [],
            "flagged_values": [],
            "document_date": "Pending",
            "summary": "Document securely uploaded and queued for processing.",
            "file_url": file_url,
            "raw_text": ""
        }
    }


# ── Red Flag Check ──
@app.get("/api/red-flag-check")
async def red_flag_check(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        return {"has_red_flags": False, "flags": [], "message": "Patient not found"}

    prompt = f"""You are a medical triage safety system. Based on the patient's reported symptoms, identify any RED FLAG symptoms that may require urgent clinical assessment.

Chief Complaint: {patient.chief_complaint}
HPI: {patient.hpi}
Severity: {patient.severity}

Respond with JSON:
{{"has_red_flags": true/false, "flags": ["list of concerning symptoms"], "message": "brief explanation"}}

IMPORTANT: Do NOT diagnose. Only flag potentially urgent symptoms. Be conservative — flag if uncertain."""

    try:
        response_text = call_llm(prompt)
        result = json.loads(extract_json_string(response_text))
        result = unwrap_json(result)
        return result
    except:
        return {"has_red_flags": False, "flags": [], "message": "Safety check completed — no urgent flags detected."}


# ── Specialty Matching ──
@app.get("/api/specialty-match")
async def specialty_match(patient_id: str, language: str = "English", db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        return {"specialty": "General Medicine", "reason": "Default", "confidence": "Low"}

    prompt = f"""Based on the patient's reported symptoms, suggest the most appropriate medical specialty.

Chief Complaint: {patient.chief_complaint}
HPI: {patient.hpi}

Respond with JSON:
{{"specialty": "Specialty name in {language}", "reason": "Brief explanation in {language}", "confidence": "Low|Medium|High"}}

Common specialties: General Medicine, Cardiology, Pulmonology, Gastroenterology, Neurology, Orthopedics, Dermatology, ENT, Ophthalmology, Psychiatry, Obstetrics & Gynecology, Pediatrics, Urology, Surgery.

IMPORTANT: The JSON keys must remain in English, but the VALUES for 'specialty' and 'reason' MUST be accurately translated into {language}. Do NOT diagnose. Only suggest which specialty is most appropriate for the described symptoms."""

    try:
        response_text = call_llm(prompt)
        result = json.loads(extract_json_string(response_text))
        result = unwrap_json(result)
        return result
    except:
        return {"specialty": "General Medicine", "reason": "Default recommendation", "confidence": "Medium"}


# ── Patient Queue ──
@app.get("/api/patients")
async def get_patients(db: Session = Depends(get_db)):
    patients = db.query(PatientRecord).order_by(PatientRecord.id.desc()).all()
    return [{
        "patient_id": p.patient_id,
        "created_at": p.created_at,
        "is_emergency": p.is_emergency
    } for p in patients]

@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    db.delete(patient)
    db.commit()
    return {"status": "success", "message": f"Patient {patient_id} deleted"}


# ── Patient Summary ──
@app.get("/api/patient-summary")
async def get_patient_summary(patient_id: Optional[str] = None, db: Session = Depends(get_db)):
    if patient_id:
        patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    else:
        patient = db.query(PatientRecord).order_by(PatientRecord.id.desc()).first()

    if not patient:
        return {"status": "No patients yet"}

    return {
        "patient_id": patient.patient_id,
        "chief_complaint": patient.chief_complaint or "Not recorded",
        "hpi": patient.hpi or "None reported",
        "is_emergency": patient.is_emergency,
        "severity": patient.severity or "Unknown",
        "duration": patient.duration or "Unknown",
        "past_medical_history": patient.past_medical_history or "None reported",
        "family_history": patient.family_history or "None reported",
        "personal_history": patient.personal_history or "None reported",
        "allergies": patient.allergies or "None reported",
        "review_of_systems": patient.review_of_systems or "None reported",
        "prakriti": patient.prakriti or "Not assessed",
        "vikriti": patient.vikriti or "Not assessed",
        "agni": patient.agni or "Not assessed",
        "flagged_lab_values": patient.flagged_lab_values or "[]",
        "created_at": patient.created_at,
    }


@app.post("/api/demo-data")
async def demo_data(db: Session = Depends(get_db)):
    pt_id = f"PT-DEMO-{str(uuid.uuid4())[:4].upper()}"
    
    demo_doc = {
        "document_type": "Complete Blood Count (CBC)",
        "diagnoses": ["Anemia"],
        "medications": ["Iron supplements (prescribed)"],
        "flagged_values": ["Hemoglobin 9.2 g/dL (Low)", "RBC count low"],
        "document_date": datetime.now().strftime("%Y-%m-%d"),
        "summary": "Blood test indicates moderate anemia with low hemoglobin levels.",
        "file_url": "",
        "raw_text": "Demo document text"
    }

    patient = PatientRecord(
        patient_id=pt_id,
        chief_complaint="Severe headache and persistent fatigue",
        hpi="• Started 3 days ago\n• Pain is throbbing and located in the frontal region\n• Accompanied by mild nausea",
        is_emergency=False,
        severity="Medium",
        duration="3 days",
        past_medical_history="• Hypertension (controlled)",
        family_history="• Mother had diabetes",
        personal_history="• Non-smoker, occasional alcohol",
        allergies="• Penicillin (causes rash)",
        review_of_systems="• No chest pain\n• No shortness of breath",
        prakriti="Not assessed",
        vikriti="Not assessed",
        agni="Not assessed",
        flagged_lab_values=json.dumps([demo_doc]),
        created_at=datetime.now().strftime("%I:%M %p")
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return {"status": "success", "patient_id": pt_id}


if __name__ == "__main__":
    print("🏥 Starting MediKiosk v2 Backend (Offline Mode)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
