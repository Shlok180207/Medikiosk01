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
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

# ── Offline AI Models ──
import ollama

OLLAMA_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "qwen2.5:3b"
DOC_EXTRACTION_MODEL = "qwen2.5:7b"

# ── Perception & Synthesis Engines (100% CPU Perception + GPU LLM) ──
from perception.xray import analyze_xray
from perception.ecg import analyze_ecg
from perception.prescription import analyze_prescription, normalize_drugs
from audio.transcriber import AudioTranscriber, get_vram_mb
from synthesis.llm import synthesize_clinical_case

def ensure_ollama_running() -> bool:
    """Checks if local Ollama daemon is active on 127.0.0.1:11434. If not, auto-starts 'ollama serve'."""
    import socket, subprocess, time
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.6)
    try:
        sock.connect(("127.0.0.1", 11434))
        sock.close()
        return True
    except Exception:
        pass
    
    print("⚙️ Ollama server not detected on 127.0.0.1:11434. Auto-launching 'ollama serve' in background...")
    try:
        if sys.platform == "win32":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS, shell=True)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        return True
    except Exception as e:
        print(f"⚠️ Could not auto-launch 'ollama serve': {e}")
        return False

# Ensure Ollama is running on module import
try:
    ensure_ollama_running()
except Exception as _oe:
    print(f"Initial Ollama check note: {_oe}")

# ── Dynamic Whisper STT (CUDA GPU with Lifecycle Management) ──
import threading
_whisper_lock = threading.Lock()
whisper_pipeline = None

def get_whisper_pipeline():
    """Lazily load Faster-Whisper on CUDA GPU (int8_float16) for blazing-fast 0.5s speech transcription."""
    global whisper_pipeline
    with _whisper_lock:
        if whisper_pipeline is None:
            print("⚡ Loading Faster-Whisper on CUDA GPU (int8_float16)...")
            try:
                from faster_whisper import WhisperModel
                whisper_pipeline = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
                print("✅ Faster-Whisper loaded on CUDA GPU (transcription latency: ~0.5s).")
            except Exception as e:
                print(f"❌ Faster-Whisper failed to load: {e}")
                whisper_pipeline = None
        return whisper_pipeline

def unload_whisper():
    """
    Deload Faster-Whisper after conversation finishes to return ~2.5GB VRAM to GPU.
    Gives 100% VRAM headroom to GPU LLM clinical synthesis (Qwen2.5-7B).
    """
    global whisper_pipeline
    with _whisper_lock:
        if whisper_pipeline is not None:
            print("🧹 Deloading Faster-Whisper from GPU to maximize VRAM for LLM...")
            try:
                del whisper_pipeline
            except Exception:
                pass
            whisper_pipeline = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("✅ Faster-Whisper deloaded. Full GPU VRAM returned to system.")

# Faster-Whisper is lazily loaded on demand inside speech endpoints (/api/process-audio)


# ── LLM Helpers ──
OLLAMA_KEEP_ALIVE = "1m"  # Auto-unload models after 1 min idle to reclaim ~4.7 GB VRAM

def call_llm(prompt: str, image_bytes: Optional[bytes] = None, model: Optional[str] = None) -> str:
    """Call Ollama / local LLM for clinical text reasoning (Zero VRAM allocated to vision)."""

    messages = [{'role': 'user', 'content': prompt}]
    target_model = model or OLLAMA_MODEL

    print(f"→ Ollama ({target_model})...")
    try:
        response = ollama.chat(model=target_model, messages=messages, format='json', options={
            'num_ctx': 4096,       # 4k context takes only 250MB KV cache (prevents VRAM overflow)
            'temperature': 0.1     
        }, keep_alive=OLLAMA_KEEP_ALIVE)
        return response['message']['content']
    except Exception as e:
        print(f"⚠️ {target_model} failed or Ollama connection lost ({e}). Verifying Ollama status...")
        ensure_ollama_running()
        if target_model != FALLBACK_MODEL:
            print(f"🔄 Falling back to lightweight local model {FALLBACK_MODEL}...")
            try:
                response = ollama.chat(model=FALLBACK_MODEL, messages=messages, format='json', options={
                    'num_ctx': 4096,
                    'temperature': 0.1
                }, keep_alive=OLLAMA_KEEP_ALIVE)
                return response['message']['content']
            except Exception as fb_err:
                print(f"⚠️ Fallback {FALLBACK_MODEL} also failed: {fb_err}")
                raise fb_err
        raise e


def extract_json_string(text: str) -> str:
    if not text:
        return "{}"
    clean = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        return match.group(1)
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return clean[first_brace:last_brace + 1]
    return clean


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

# ── Pre-written Symptom-Specific Question Sets (human-verified, no AI typos) ──
CATEGORY_QUESTIONS = {
    "chest_pain": {
        "Hindi": {
            "initial": "यह सीने में दर्द या सांस लेने में तकलीफ़ आपको कब से हो रही है?",
            0: "क्या यह दर्द आपके बाएं हाथ, कंधे, जबड़े या पीठ की तरफ फैलता है?",
            1: "क्या इसके साथ सांस फूलना, ठंडा पसीना, चक्कर या घबराहट हो रही है?",
            2: "क्या चलने या मेहनत करने से दर्द बढ़ता है और आराम करने से घटता है?",
            3: "क्या आपको पहले बीपी, शुगर, दिल की बीमारी रही है या आप धूम्रपान करते हैं?",
            4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you been having this chest discomfort or breathing trouble?",
            0: "Does this pain spread to your left arm, shoulder, jaw, or back?",
            1: "Are you experiencing shortness of breath, cold sweats, dizziness, or nausea?",
            2: "Does walking or exertion make it worse, and does resting relieve it?",
            3: "Do you have a history of high BP, diabetes, heart disease, or smoking?",
            4: "Are you allergic to any medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த நெஞ்சு வலி அல்லது மூச்சுத்திணறல் எவ்வளவு நாளாக இருக்கிறது?",
            0: "இந்த வலி இடது கை, தோள்பட்டை, தாடை அல்லது முதுகுக்கு பரவுகிறதா?",
            1: "இதனுடன் மூச்சுத்திணறல், குளிர்ந்த வியர்வை, மயக்கம் அல்லது குமட்டல் உள்ளதா?",
            2: "நடக்கும்போது வலி அதிகமாகி ஓய்வெடுக்கும்போது குறைகிறதா?",
            3: "உங்களுக்கு ரத்த அழுத்தம், சர்க்கரை நோய், இதய நோய் உள்ளதா அல்லது புகைபிடிப்பீர்களா?",
            4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ గుండె నొప్పి లేదా శ్వాస తీసుకోవడంలో ఇబ్బంది ఎంత కాలంగా ఉంది?",
            0: "ఈ నొప్పి ఎడమ చేయి, భుజం, దవడ లేదా వెనుక భాగానికి వ్యాపిస్తుందా?",
            1: "దీనితో పాటు శ్వాస ఆడకపోవడం, చల్లని చెమటలు, తలతిరగడం లేదా వికారం ఉన్నాయా?",
            2: "నడిచినప్పుడు నొప్పి పెరిగి, విశ్రాంతి తీసుకున్నప్పుడు తగ్గుతుందా?",
            3: "మీకు బీపీ, షుగర్, గుండె జబ్బుల చరిత్ర ఉందా లేదా పొగ తాగుతారా?",
            4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "stomach_pain": {
        "Hindi": {
            "initial": "यह पेट दर्द आपको कब से हो रहा है?",
            0: "क्या यह दर्द पेट के ऊपरी हिस्से में है, नीचे की तरफ या पीठ में जाता है?",
            1: "क्या उल्टी, दस्त, खट्टी डकार, जलन या बुखार जैसा लग रहा है?",
            2: "क्या कुछ खाने-पीने से दर्द बढ़ता है या खाली पेट रहने से?",
            3: "क्या आपको पहले अल्सर, गैस, पथरी की शिकायत रही है या बाहर का खाना खाया था?",
            4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you had this stomach or abdominal pain?",
            0: "Is the pain in the upper abdomen, lower belly, or radiating to the back?",
            1: "Are you experiencing vomiting, loose motions, acidity, burning, or fever?",
            2: "Does eating food or drinking water make the pain worse or better?",
            3: "Do you have a history of ulcers, acidity, gallstones, or recent outside food?",
            4: "Are you allergic to any medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த வயிற்று வலி உங்களுக்கு எவ்வளவு நாளாக இருக்கிறது?",
            0: "இந்த வலி வயிற்றின் மேல் பகுதியிலா, கீழ் பகுதியிலா அல்லது முதுகில் பரவுகிறதா?",
            1: "இதனுடன் வாந்தி, வயிற்றுப்போக்கு, நெஞ்செரிச்சல் அல்லது காய்ச்சல் உள்ளதா?",
            2: "சாப்பிட்ட பிறகு வலி அதிகமாகிறதா அல்லது குறைகிறதா?",
            3: "உங்களுக்கு குடல் புண், பித்தப்பை கல் அல்லது வெளி உணவு சாப்பிட்ட வரலாறு உள்ளதா?",
            4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ కడుపు నొప్పి మీకు ఎంత కాలంగా ఉంది?",
            0: "నొప్పి కడుపు పైభాగంలో ఉందా, కింద ఉందా లేదా వీపులోకి వ్యాపిస్తుందా?",
            1: "వాంతులు, విరేచనాలు, ఎసిడిటీ, మంట లేదా జ్వరం ఉన్నాయా?",
            2: "ఆహారం తిన్న తర్వాత నొప్పి పెరుగుతుందా లేదా తగ్గుతుందా?",
            3: "మీకు అల్సర్, గ్యాస్ట్రిక్, పిత్తాశయ రాళ్ల సమస్య ఉందా లేదా బయటి ఆహారం తిన్నారా?",
            4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "headache": {
        "Hindi": {
            "initial": "यह सिरदर्द या चक्कर आपको कब से आ रहे हैं?",
            0: "क्या यह सिरदर्द आधे सिर में है, माथे पर या गर्दन के पीछे की तरफ?",
            1: "क्या इसके साथ उल्टी का मन, आंखों के आगे अंधेरा, तेज रोशनी से चिढ़ या कमज़ोरी है?",
            2: "क्या तनाव, नींद की कमी या स्क्रीन देखने से दर्द बढ़ता है?",
            3: "क्या आपको हाई बीपी, चश्मे का नंबर, साइनस या परिवार में माइग्रेन की शिकायत है?",
            4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you been experiencing this headache or dizziness?",
            0: "Is the headache throbbing on one side, frontal, or radiating down the neck?",
            1: "Do you have nausea, sensitivity to bright light, blurred vision, or weakness?",
            2: "Does stress, lack of sleep, or screen time trigger or worsen the headache?",
            3: "Do you have a history of high blood pressure, sinus issues, or family migraine?",
            4: "Are you allergic to any medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த தலைவலி அல்லது மயக்கம் எவ்வளவு நாளாக இருக்கிறது?",
            0: "தலைவலி ஒரு பக்கத்திலா, நெற்றியிலா அல்லது கழுத்தின் பின்புறத்திலா?",
            1: "இதனுடன் குமட்டல், வெளிச்சத்தை பார்க்க முடியாத நிலை அல்லது பார்வை மங்கலாகுதல் உள்ளதா?",
            2: "மன அழுத்தம் அல்லது தூக்கமின்மையால் தலைவலி அதிகரிக்கிறதா?",
            3: "உங்களுக்கு ரத்த அழுத்தம், சைனஸ் அல்லது குடும்பத்தில் மைக்ரேன் வரலாறு உள்ளதா?",
            4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ తలనొప్పి లేదా తలతిరగడం మీకు ఎంత కాలంగా ఉంది?",
            0: "తలనొప్పి ఒక వైపున, నుదిటిపై లేదా మెడ వెనుక భాగంలో ఉందా?",
            1: "వికారం, కాంతిని చూడలేకపోవడం, మసకబారిన చూపు లేదా బలహీనత ఉన్నాయా?",
            2: "ఒత్తిడి, నిద్రలేమి లేదా స్క్రీన్ చూడటం వల్ల తలనొప్పి పెరుగుతుందా?",
            3: "మీకు హై బీపీ, సైనస్ లేదా కుటుంబంలో మైగ్రేన్ సమస్యలు ఉన్నాయా?",
            4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "fever": {
        "Hindi": {
            "initial": "यह बुखार आपको कितने दिनों से आ रहा है?",
            0: "क्या बुखार ठंड और कंपकंपी के साथ आता है? क्या यह किसी खास समय तेज होता है?",
            1: "क्या इसके साथ खांसी, गले में खराश, बदन दर्द, दाने या पेशाब में जलन है?",
            2: "क्या आपने पैरासिटामोल ली है? क्या दवा लेने पर बुखार उतरता है?",
            3: "क्या घर या पड़ोस में किसी को डेंगू, मलेरिया, टाइफाइड या वायरल बुखार हुआ है?",
            4: "क्या आपको किसी एंटीबायोटिक या दवा से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How many days have you had this fever?",
            0: "Does the fever come with chills and shivering, and does it spike at a specific time?",
            1: "Do you have cough, sore throat, severe body aches, rashes, or burning urination?",
            2: "Have you taken paracetamol? Does the temperature come down after medicine?",
            3: "Has anyone in your home or area had dengue, malaria, typhoid, or viral fever recently?",
            4: "Are you allergic to any antibiotics or medicines?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த காய்ச்சல் எத்தனை நாட்களாக இருக்கிறது?",
            0: "காய்ச்சல் குளிர் மற்றும் நடுக்கத்துடன் வருகிறதா? குறிப்பிட்ட நேரத்தில் அதிகமாகிறதா?",
            1: "இதனுடன் இருமல், தொண்டை வலி, உடல் வலி அல்லது சிறுநீரில் எரிச்சல் உள்ளதா?",
            2: "பாராசிட்டமால் மாத்திரை சாப்பிட்டீர்களா? மருந்து எடுத்தவுடன் காய்ச்சல் குறைகிறதா?",
            3: "அருகில் யாருக்காவது டெங்கு, மலேரியா அல்லது டைபாய்டு காய்ச்சல் உள்ளதா?",
            4: "உங்களுக்கு ஏதேனும் ஆண்டிபயாடிக் அல்லது மருந்துக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ జ్వరం మీకు ఎన్ని రోజులుగా వస్తోంది?",
            0: "జ్వరం చలి మరియు వణుకుతో వస్తుందా? ఏదైనా నిర్దిష్ట సమయంలో పెరుగుతుందా?",
            1: "దగ్గు, గొంతు నొప్పి, తీవ్రమైన ఒళ్లు నొప్పులు లేదా మూత్రంలో మంట ఉన్నాయా?",
            2: "పారాసిటమాల్ వేసుకున్నారా? మందు వేసుకున్న తర్వాత జ్వరం తగ్గుతుందా?",
            3: "ఇంట్లో లేదా చుట్టుపక్కల ఎవరికైనా డెంగ్యూ, మలేరియా లేదా టైఫాయిడ్ వచ్చిందా?",
            4: "మీకు ఏదైనా యాంటీబయాటిక్ లేదా మందుకు అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "joint_pain": {
        "Hindi": {
            "initial": "यह जोड़ों या कमर का दर्द आपको कब से हो रहा है?",
            0: "क्या जोड़ पर सूजन, लालिमा या सुबह उठने पर जकड़न महसूस होती है?",
            1: "क्या कोई चोट लगी थी? क्या चलने-फिरने या सीढ़ियां चढ़ने में तकलीफ़ होती है?",
            2: "क्या आराम करने या गर्म सिकाई करने से दर्द में राहत मिलती है?",
            3: "क्या आपको पहले गठिया, यूरिक एसिड, साइटिका या हड्डियों की कमज़ोरी रही है?",
            4: "क्या आपको किसी दर्द निवारक (पेनकिलर) दवा से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you had this joint or back pain?",
            0: "Is there visible swelling, redness, or morning stiffness in the joint?",
            1: "Did you have a fall or injury? Is it difficult to walk or climb stairs?",
            2: "Does rest or heat application provide relief from the pain?",
            3: "Do you have a history of arthritis, high uric acid, sciatica, or osteoporosis?",
            4: "Are you allergic to any painkiller medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த மூட்டு அல்லது முதுகு வலி எவ்வளவு நாளாக இருக்கிறது?",
            0: "மூட்டில் வீக்கம், சிவத்தல் அல்லது காலையில் விறைப்பு தன்மை உள்ளதா?",
            1: "ஏதாவது காயம் ஏற்பட்டதா? நடப்பதற்கோ அல்லது படிக்கட்டுகள் ஏறுவதற்கோ சிரமமாக உள்ளதா?",
            2: "ஓய்வெடுப்பதாலோ அல்லது ஒத்தடம் கொடுப்பதாலோ வலி குறைகிறதா?",
            3: "உங்களுக்கு மூட்டுவாதம், யூரிக் அமிலம் அல்லது எலும்பு தேய்மானம் உள்ளதா?",
            4: "உங்களுக்கு வலி நிவாரணி மருந்துகளுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ కీళ్ల లేదా వెన్ను నొప్పి మీకు ఎంత కాలంగా ఉంది?",
            0: "కీళ్లపై వాపు, ఎరుపుదనం లేదా ఉదయం పూట బిగుతుగా ఉండటం ఉందా?",
            1: "ఏదైనా గాయం అయిందా? నడవడానికి లేదా మెట్లు ఎక్కడానికి కష్టంగా ఉందా?",
            2: "విశ్రాంతి లేదా వేడి కాపడం వల్ల నొప్పి తగ్గుతుందా?",
            3: "మీకు ఆర్థరైటిస్, యూరిక్ యాసిడ్ లేదా ఎముకల బలహీనత సమస్యలు ఉన్నాయా?",
            4: "మీకు పెయిన్‌కిల్లర్ మందులకు అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "general": {
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
            1: "এর সাথে জ্বর, বমি বা দুর্বলতার মতো অন্য কোনো समस्या আছে?",
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
        }
    }
}

# Pre-compiled symptom category regexes (compiled once at module load, not per-call)
_SYMPTOM_REGEXES = [
    ("chest_pain", re.compile(
        r"chest|सीने|सीना|छाती|दिल|सांस|breath|palpitation|घबराहट|धड़कन|angina|cardiac|"
        r"seene|seena|chhati|chati|dil|saans|sans|ghabrahat|dhadkan|நெஞ்சு|மார்பு|గుండె|छातीत|বুক",
        re.IGNORECASE
    )),
    ("stomach_pain", re.compile(
        r"stomach|abdomen|abdominal|belly|पेट|pet|pait|vomit|उल्टी|ulti|loose motion|"
        r"दस्त|dast|acidity|gas|गैस|जलन|jalan|constipation|कब्ज|kabz|ulcer|appetite|भूख|bhook|வயிறு|కడుపు|पोट|পেট",
        re.IGNORECASE
    )),
    ("headache", re.compile(
        r"headache|head pain|head|सिर|सर|sir|sar|migraine|माइग्रेन|dizziness|चक्कर|chakkar|"
        r"faint|बेहोश|behosh|vision|धुंधला|stroke|தலைவலி|తలనొప్పి|डोकेदुखी|মাথাব্যথা",
        re.IGNORECASE
    )),
    ("fever", re.compile(
        r"fever|बुखार|ताप|bukhar|taap|chills|ठंड|thand|shivering|कंपकंपी|kampkampi|"
        r"dengue|डेंगू|malaria|मलेरिया|typhoid|टाइफाइड|viral|காய்ச்சல்|జ్వరం|জ্বর",
        re.IGNORECASE
    )),
    ("joint_pain", re.compile(
        r"joint|जोड़|jod|jodon|knee|घुटने|ghutne|ghutna|back pain|कमर|kamar|spine|रीढ़|"
        r"bone|हड्डी|haddi|swelling|सूजन|sujan|arthritis|गठिया|gathiya|fracture|stiffness|जकड़न|jakdan|மூட்டு|కీళ్ల|సాంధేదుఖీ|গাঁটের ব্যথা",
        re.IGNORECASE
    )),
]

def detect_symptom_category(transcript: str) -> str:
    """Classifies patient transcript into targeted symptom tracks using pre-compiled multilingual regexes."""
    if not transcript:
        return "general"
    for category, pattern in _SYMPTOM_REGEXES:
        if pattern.search(transcript):
            return category
    return "general"

def get_phase_question(language: str, phase, category: str = "general") -> str:
    """Get a pre-written question template tailored to language and symptom category."""
    cat_dict = CATEGORY_QUESTIONS.get(category, CATEGORY_QUESTIONS.get("general", {}))
    lang_questions = cat_dict.get(language, cat_dict.get("English", {}))
    if not lang_questions:
        gen_cat = CATEGORY_QUESTIONS.get("general", {})
        lang_questions = gen_cat.get(language, gen_cat.get("English", {}))
    return lang_questions.get(phase, lang_questions.get("default", "Is there anything else you would like to tell me?"))

# ── Pydantic Models ──
PATIENT_JSON_TEMPLATE = """{
  "chief_complaint": "Main presenting symptom (translated to clinical English)",
  "hpi": "Chronological narrative of the CURRENT presenting complaint ONLY: onset, duration, character, radiation, severity, aggravating/relieving factors. STRICTLY EXCLUDE past history, family history, lifestyle/habits, allergies, and general systemic symptom denials.",
  "is_emergency": false,
  "severity": "Low|Medium|High",
  "duration": "Symptom duration (e.g. '2 days')",
  "past_medical_history": "Past medical conditions (or 'Uncertain / unconfirmed (patient does not recall)' if unsure, or 'Patient denies past chronic medical conditions / No significant past medical history' if denied)",
  "family_history": "Family history (or 'Patient denies family history of similar complaints / No significant family history' if denied, or 'Uncertain / unconfirmed' if unsure)",
  "personal_history": "Smoking, alcohol, diet, habits (or 'No significant lifestyle or habit risks reported')",
  "allergies": "Drug/food allergies (or 'No known drug or food allergies (NKDA)' if denied)",
  "review_of_systems": "Summary of systemic positive and negative symptoms (e.g. 'Patient reports diaphoresis; denies fever or vomiting')",
  "clinical_impression": {
    "clinical_synthesis": [
      "Key acute symptoms, duration, and anatomical localization reported today",
      "Corroborating objective findings from today's uploaded reports/labs (or 'No acute lab flags reported')",
      "Historical ABHA risk context and underlying clinical etiology rationale"
    ],
    "probable_diagnoses": [
      {
        "condition": "Primary Suspected Condition Name",
        "likelihood": "High|Medium|Low",
        "supporting_evidence": "Clinical rationale tying together symptoms, lab values, and past history."
      }
    ],
    "suggested_investigations": [
      "Key diagnostic test or scan 1",
      "Key diagnostic test or scan 2"
    ],
    "critical_rule_outs": [
      "Critical high-risk condition to actively rule out"
    ]
  },
  "prakriti": "Not assessed",
  "vikriti": "Not assessed",
  "agni": "Not assessed",
  "next_question": "complete"
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
  "document_type": "Specific name of scan or report (e.g. Head X-Ray, 12-Lead ECG, Knee Radiograph, Biopsy, CBC, Prescription)",
  "modality": "radiology|ecg|pathology|endoscopy|document",
  "diagnoses": ["list of clinical diagnoses/radiological findings"],
  "medications": ["medicine with dose if prescription, else empty list"],
  "flagged_values": [
    {
      "test_name": "Parameter name (e.g. FEV1, Hemoglobin, RBC)",
      "measured_value": 0.0,
      "unit": "unit of measurement (e.g. L, L/s, g/dL, mg/dL, /cumm)",
      "reference_low": 0.0,
      "reference_high": 0.0,
      "status": "Low or High or Normal"
    }
  ],
  "document_date": "date on document or 'Visual Scan'",
  "summary": "Brief concise clinical summary"
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
    clinical_impression: Optional[dict] = None
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
    modality: str = "document"
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
    abha_id = Column(String, index=True, nullable=True)  # Links visits by ABHA identity
    is_ayush = Column(Boolean, default=False)
    patient_name = Column(String, default="Patient")
    age = Column(String, default="")
    gender = Column(String, default="")
    phone = Column(String, default="")
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
    symptom_category = Column(String, default="general")
    clinical_impression_json = Column(Text, default="{}")
    prakriti = Column(String)
    vikriti = Column(String)
    agni = Column(String)
    flagged_lab_values = Column(Text, default="[]")
    raw_dialogue = Column(Text, default="")
    is_synthesized = Column(Boolean, default=False)
    abha_relevance_json = Column(Text, default="{}")
    created_at = Column(String)


class VisitHistory(Base):
    """Stores past visit records linked by ABHA ID for history continuity."""
    __tablename__ = "visit_history"
    id = Column(Integer, primary_key=True, index=True)
    abha_id = Column(String, index=True)
    visit_date = Column(String)
    chief_complaint = Column(Text)
    diagnoses = Column(Text, default="[]")        # JSON list
    medications = Column(Text, default="[]")       # JSON list
    flagged_values = Column(Text, default="[]")    # JSON list
    summary = Column(Text)
    specialty = Column(String)
    is_relevant = Column(Boolean, default=False)   # Set by AI filter
    relevance_reason = Column(Text)                # Why AI thinks it's relevant


Base.metadata.create_all(bind=engine)

# Auto-migrate SQLite schema if columns don't exist
try:
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(patients)")).fetchall()]
        for col_name, col_type in [("abha_id", "VARCHAR"), ("patient_name", "VARCHAR"), ("age", "VARCHAR"), ("gender", "VARCHAR"), ("phone", "VARCHAR"), ("symptom_category", "VARCHAR"), ("raw_dialogue", "TEXT"), ("is_synthesized", "BOOLEAN"), ("abha_relevance_json", "TEXT"), ("clinical_impression_json", "TEXT")]:
            if col_name not in cols:
                conn.execute(text(f"ALTER TABLE patients ADD COLUMN {col_name} {col_type}"))
        conn.commit()
except Exception as e:
    print(f"Auto-migration note: {e}")

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



# ── 5 Diverse ABHA Patient Profiles (Mock ABDM Gateway) ──
ABHA_PROFILES = {
    "12-3456-7890-1234": {
        "name": "Ramesh Sharma",
        "age": 58,
        "gender": "Male",
        "phone": "9876543210",
        "abha_id": "12-3456-7890-1234",
        "avatar": "👨‍🦳",
        "badge": "Cardiology & Diabetes",
        "history": [
            {
                "visit_date": "2024-01-15",
                "chief_complaint": "Acute chest pain radiating to left arm with cold sweat",
                "diagnoses": json.dumps(["Myocardial Infarction (STEMI)", "Coronary Artery Disease"]),
                "medications": json.dumps(["Aspirin 75mg OD", "Clopidogrel 75mg OD", "Atorvastatin 40mg OD", "Metoprolol 25mg BD"]),
                "flagged_values": json.dumps(["Troponin I: 8.5 ng/mL (Critical High)", "CK-MB: 45 U/L (High)"]),
                "summary": "Admitted with acute STEMI. Underwent primary PCI with drug-eluting stent to LAD. Discharged on dual antiplatelet therapy.",
                "specialty": "Cardiology"
            },
            {
                "visit_date": "2024-06-20",
                "chief_complaint": "Follow-up cardiac & lipid evaluation",
                "diagnoses": json.dumps(["Hyperlipidemia", "Post-MI follow-up"]),
                "medications": json.dumps(["Atorvastatin 40mg OD", "Ramipril 5mg OD", "Aspirin 75mg OD"]),
                "flagged_values": json.dumps(["LDL Cholesterol: 145 mg/dL (High)", "Total Cholesterol: 230 mg/dL (High)"]),
                "summary": "6-month post-MI follow-up. Echocardiogram shows EF 50%. High LDL, statin dose maintained.",
                "specialty": "Cardiology"
            },
            {
                "visit_date": "2023-04-12",
                "chief_complaint": "Left ankle sprain after minor trip",
                "diagnoses": json.dumps(["Grade 1 Ankle Sprain"]),
                "medications": json.dumps(["Paracetamol 650mg SOS", "Diclofenac gel"]),
                "flagged_values": json.dumps([]),
                "summary": "Minor ligament strain. Fully resolved in 2 weeks.",
                "specialty": "Orthopedics"
            },
            {
                "visit_date": "2025-02-10",
                "chief_complaint": "Fatigue, increased thirst, and frequent urination",
                "diagnoses": json.dumps(["Type 2 Diabetes Mellitus (Uncontrolled)", "Hypertension Stage 2"]),
                "medications": json.dumps(["Metformin 500mg BD", "Glimepiride 1mg OD", "Telmisartan 40mg OD"]),
                "flagged_values": json.dumps(["HbA1c: 8.2% (High)", "Fasting Blood Glucose: 168 mg/dL (High)", "BP: 152/94 mmHg"]),
                "summary": "Uncontrolled Type 2 Diabetes with Stage 2 HTN. Oral hypoglycemics adjusted.",
                "specialty": "General Medicine"
            }
        ]
    },
    "23-4567-8901-2345": {
        "name": "Priya Patel",
        "age": 32,
        "gender": "Female",
        "phone": "9812345678",
        "abha_id": "23-4567-8901-2345",
        "avatar": "👩",
        "badge": "Pulmonology & Asthma",
        "history": [
            {
                "visit_date": "2024-10-05",
                "chief_complaint": "Severe acute breathlessness, dry cough, and wheezing",
                "diagnoses": json.dumps(["Acute Exacerbation of Bronchial Asthma", "Bronchospasm"]),
                "medications": json.dumps(["Salbutamol Nebulization SOS", "Budecort Inhaler 200mcg BD", "Montelukast 10mg OD"]),
                "flagged_values": json.dumps(["Serum IgE: 650 IU/mL (Markedly Elevated)", "Peak Expiratory Flow: 220 L/min (Low)"]),
                "summary": "Severe asthma attack triggered by dust and cold weather. Responsive to bronchodilators.",
                "specialty": "Pulmonology"
            },
            {
                "visit_date": "2023-02-18",
                "chief_complaint": "Facial skin acne breakouts",
                "diagnoses": json.dumps(["Acne Vulgaris"]),
                "medications": json.dumps(["Clindamycin 1% gel", "Benzoyl Peroxide 2.5%"]),
                "flagged_values": json.dumps([]),
                "summary": "Mild papular acne treated with topical antibiotics.",
                "specialty": "Dermatology"
            },
            {
                "visit_date": "2024-03-14",
                "chief_complaint": "Persistent nighttime coughing fits",
                "diagnoses": json.dumps(["Cough-Variant Asthma", "Allergic Bronchitis"]),
                "medications": json.dumps(["Levocetirizine 5mg OD", "Formoterol + Budesonide Inhaler"]),
                "flagged_values": json.dumps([]),
                "summary": "Nocturnal asthma symptoms controlled with combination inhaler therapy.",
                "specialty": "Pulmonology"
            }
        ]
    },
    "34-5678-9012-3456": {
        "name": "Sunita Devi",
        "age": 64,
        "gender": "Female",
        "phone": "9765432109",
        "abha_id": "34-5678-9012-3456",
        "avatar": "👵",
        "badge": "Orthopedics & Arthritis",
        "history": [
            {
                "visit_date": "2024-08-11",
                "chief_complaint": "Severe right knee joint pain, crepitus, and inability to climb stairs",
                "diagnoses": json.dumps(["Primary Osteoarthritis of Right Knee (Grade 3)", "Synovial Effusion"]),
                "medications": json.dumps(["Aceclofenac 100mg + Paracetamol 325mg BD", "Glucosamine 1500mg OD"]),
                "flagged_values": json.dumps(["Knee X-Ray: Medial compartment joint space narrowing with subchondral sclerosis"]),
                "summary": "Advanced knee osteoarthritis. Advised physiotherapy, quadriceps strengthening, and knee brace.",
                "specialty": "Orthopedics"
            },
            {
                "visit_date": "2023-11-20",
                "chief_complaint": "Diffuse lower back pain and bone aches",
                "diagnoses": json.dumps(["Osteopenia", "Severe Vitamin D3 Deficiency"]),
                "medications": json.dumps(["Cholecalciferol (Vit D3) 60,000 IU weekly", "Calcium Carbonate 500mg BD"]),
                "flagged_values": json.dumps(["Serum Vitamin D: 9.8 ng/mL (Deficient)", "DEXA T-Score: -2.1 (Osteopenia)"]),
                "summary": "Osteopenia identified on DEXA scan. Intensive Vitamin D and Calcium supplementation started.",
                "specialty": "Rheumatology"
            },
            {
                "visit_date": "2022-05-04",
                "chief_complaint": "Difficulty reading small text and driving at night",
                "diagnoses": json.dumps(["Early Immature Senile Cataract", "Presbyopia"]),
                "medications": json.dumps(["Carboxymethylcellulose eye drops"]),
                "flagged_values": json.dumps([]),
                "summary": "Prescription glasses updated. Annual eye review advised.",
                "specialty": "Ophthalmology"
            }
        ]
    },
    "45-6789-0123-4567": {
        "name": "Mohammed Ali",
        "age": 45,
        "gender": "Male",
        "phone": "9654321098",
        "abha_id": "45-6789-0123-4567",
        "avatar": "👨",
        "badge": "Gastroenterology & Liver",
        "history": [
            {
                "visit_date": "2024-04-22",
                "chief_complaint": "Burning epigastric pain, acid regurgitation, and post-meal fullness",
                "diagnoses": json.dumps(["Erosive Reflux Esophagitis (Grade B)", "Antral Gastritis"]),
                "medications": json.dumps(["Pantoprazole 40mg OD", "Sucralfate suspension 10ml TDS"]),
                "flagged_values": json.dumps(["Endoscopy: Multiple superficial linear mucosal erosions in lower third of esophagus"]),
                "summary": "Endoscopy confirmed erosive GERD. 8-week course of PPI and mucosal protectant prescribed.",
                "specialty": "Gastroenterology"
            },
            {
                "visit_date": "2024-11-19",
                "chief_complaint": "Dull ache in right upper quadrant of abdomen and mild nausea",
                "diagnoses": json.dumps(["Non-Alcoholic Fatty Liver Disease (Grade 1 NAFLD)", "Elevated Liver Enzymes"]),
                "medications": json.dumps(["Ursodeoxycholic Acid (UDCA) 300mg BD", "Vitamin E 400mg OD"]),
                "flagged_values": json.dumps(["SGPT/ALT: 68 U/L (High)", "SGOT/AST: 54 U/L (High)", "Serum Bilirubin: 1.1 mg/dL"]),
                "summary": "Abdominal ultrasound showed Grade 1 hepatic steatosis with elevated transaminases.",
                "specialty": "Gastroenterology"
            },
            {
                "visit_date": "2023-07-08",
                "chief_complaint": "Right ear itching and discomfort after swimming",
                "diagnoses": json.dumps(["Acute Otitis Externa"]),
                "medications": json.dumps(["Ciprofloxacin ear drops", "Ibuprofen 400mg"]),
                "flagged_values": json.dumps([]),
                "summary": "Swimmer's ear cleared completely after 5 days of topical antibiotic drops.",
                "specialty": "ENT"
            }
        ]
    },
    "56-7890-1234-5678": {
        "name": "Anita Verma",
        "age": 26,
        "gender": "Female",
        "phone": "9543210987",
        "abha_id": "56-7890-1234-5678",
        "avatar": "👩‍🦰",
        "badge": "ENT & Allergy",
        "history": [
            {
                "visit_date": "2024-12-01",
                "chief_complaint": "Bilateral throbbing facial pressure, thick nasal discharge, and frontal headache",
                "diagnoses": json.dumps(["Acute Exacerbation of Chronic Maxillary Sinusitis"]),
                "medications": json.dumps(["Amoxicillin-Clavulanate 625mg BD x 7d", "Fluticasone Furoate Nasal Spray 1 puff BD", "Saline rinse"]),
                "flagged_values": json.dumps(["PNS X-Ray: Bilateral maxillary sinus haziness with mucosal thickening"]),
                "summary": "Bacterial sinusitis flare-up treated with oral antibiotics and steroid nasal spray.",
                "specialty": "ENT"
            },
            {
                "visit_date": "2023-09-15",
                "chief_complaint": "Excessive morning sneezing bouts (15-20 sneezes) and watery itchy eyes",
                "diagnoses": json.dumps(["Perennial Allergic Rhinitis (Dust Mite Allergy)"]),
                "medications": json.dumps(["Bilastine 20mg OD", "Montelukast 10mg OD"]),
                "flagged_values": json.dumps(["Skin Prick Test: Positive for Dermatophagoides pteronyssinus (House Dust Mite)"]),
                "summary": "Allergen sensitization confirmed. Prescribed non-sedating antihistamines and dust avoidance measures.",
                "specialty": "Allergy / Immunology"
            },
            {
                "visit_date": "2022-01-10",
                "chief_complaint": "Right wrist tenderness from repetitive typing",
                "diagnoses": json.dumps(["De Quervain's Tenosynovitis (Right Wrist)"]),
                "medications": json.dumps(["Thumb spica splint", "Diclofenac gel"]),
                "flagged_values": json.dumps([]),
                "summary": "Repetitive strain injury. Advised ergonomic workstation setup and rest.",
                "specialty": "Orthopedics"
            }
        ]
    }
}


def seed_abha_history(abha_id: str, db: Session):
    """Seed mock ABHA history into VisitHistory table if not already present."""
    if not abha_id:
        return
    # Check if already seeded for this specific abha_id
    existing = db.query(VisitHistory).filter(VisitHistory.abha_id == abha_id).first()
    if existing:
        return
    
    profile = ABHA_PROFILES.get(abha_id)
    if not profile or "history" not in profile:
        return
    
    for visit in profile["history"]:
        record = VisitHistory(
            abha_id=abha_id,
            visit_date=visit["visit_date"],
            chief_complaint=visit["chief_complaint"],
            diagnoses=visit["diagnoses"],
            medications=visit["medications"],
            flagged_values=visit["flagged_values"],
            summary=visit["summary"],
            specialty=visit["specialty"],
            is_relevant=False,
            relevance_reason=None
        )
        db.add(record)
    db.commit()
    print(f"✅ Seeded {len(profile['history'])} mock ABHA history records for {profile['name']} ({abha_id})")


# ── Context-Aware History Filter (Background Task) ──
HISTORY_FILTER_TEMPLATE = """{
  "results": [
    {"visit_id": 1, "is_relevant": true, "reason": "Detailed clinical correlation explaining why this past record is directly relevant to today's complaint"},
    {"visit_id": 2, "is_relevant": false, "reason": "Clinical explanation why this past visit is unrelated"}
  ]
}"""


def filter_history_background(patient_id_db: int, abha_id: str, chief_complaint: str):
    """Background task: Uses Qwen to filter past ABHA history strictly by relevance to THIS patient's complaint."""
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        past_visits = db.query(VisitHistory).filter(VisitHistory.abha_id == abha_id).all()
        if not past_visits or not patient:
            return
        
        # Build concise history summary for LLM
        visits_for_llm = []
        for v in past_visits:
            visits_for_llm.append({
                "visit_id": v.id,
                "date": v.visit_date,
                "complaint": v.chief_complaint,
                "diagnoses": json.loads(v.diagnoses) if v.diagnoses else [],
                "medications": json.loads(v.medications) if v.medications else [],
                "flagged_values": json.loads(v.flagged_values) if v.flagged_values else [],
                "specialty": v.specialty
            })
        
        prompt = f"""You are a clinical decision support AI acting on behalf of a doctor reviewing a patient's historical medical records.
The patient is presenting TODAY at the triage kiosk with the following Chief Complaint:
"{chief_complaint}"

Here is the patient's verified past medical history from ABHA:
{json.dumps(visits_for_llm, indent=2)}

TASK: For EACH past visit, determine if it is MEDICALLY RELEVANT to today's chief complaint ("{chief_complaint}").

CLINICAL RELEVANCE RULES:
1. RELEVANT (is_relevant: true):
   - Involves the SAME organ system, anatomical region, or related etiology (e.g. past STEMI/Heart Attack or HTN is RELEVANT when patient has chest pain or breathlessness; Gastroenterology/Gastritis is RELEVANT when presenting for abdominal pain).
   - Involves active medications that could interact or explain current symptoms.
   - Contains lab flags or chronic diagnoses directly tied to the current complaint.
2. NOT RELEVANT (is_relevant: false):
   - Belongs to a completely unrelated specialty/organ system (e.g., Cardiology/Heart Attack or Knee Osteoarthritis when presenting for Abdominal pain).
   - A minor, completely resolved past issue with no clinical bearing on today's presentation.

IMPORTANT: Set `is_relevant` to true or false. Provide a concise, professional clinical reason in `reason`.

Output ONLY valid JSON:
{HISTORY_FILTER_TEMPLATE}"""

        print(f"→ Running ABHA clinical relevance filter for '{chief_complaint}' on patient {patient.patient_id}...")
        response_text = call_llm(prompt)
        result_json = json.loads(extract_json_string(response_text))
        result_json = unwrap_json(result_json)
        
        results = result_json.get("results", [])
        if not isinstance(results, list):
            results = [result_json] if isinstance(result_json, dict) else []
        
        relevance_map = {}
        for item in results:
            visit_id = item.get("visit_id")
            raw_rel = item.get("is_relevant")
            if isinstance(raw_rel, bool):
                is_relevant = raw_rel
            else:
                is_relevant = str(raw_rel).strip().lower() in ["true", "1", "yes"]
            reason = str(item.get("reason", "")).strip()
            if visit_id is not None:
                relevance_map[str(visit_id)] = {
                    "is_relevant": is_relevant,
                    "reason": reason
                }
        
        patient.abha_relevance_json = json.dumps(relevance_map)
        db.commit()
        print(f"✅ ABHA clinical relevance filter complete for patient {patient.patient_id}. {sum(1 for r in relevance_map.values() if r.get('is_relevant'))} relevant visits attached.")
    except Exception as e:
        print(f"❌ History filter error: {e}")
    finally:
        db.close()


# ── Unsure vs Denial Helpers ──
UNSURE_PATTERNS = [
    "not sure", "unsure", "dont know", "don't know", "not certain", "uncertain", 
    "no idea", "pata nahi", "pata nahin", "malum nahi", "maloom nahi", "nahi pata", 
    "yaad nahi", "yaad nhi", "yaad nahin", "याद नहीं", "याद नाही", "confirm nahi", 
    "confirm nhi", "not confirmed", "unconfirmed", "bhul gaya", "bhool gaya", 
    "mai confirm nahi", "main confirm nahi", "mujhe yaad nahi", "mujhe yaad nhi", 
    "mujhe confirm nahi", "mujhe confirm nhi", "shayad", "maybe", "not remembered",
    "पता नहीं", "मालूम नहीं", "माहित नाही", "తెలియదు", "தெரியாது", "জানা নেই", 
    "ખબર નથી", "ಗೊತ್ತಿಲ್ಲ", "ಅറിയിಲ್ಲ", "ਨਹੀਂ ਪਤਾ", "🤷 not sure", "🤷"
]

DENIAL_PATTERNS = [
    "no", "nahi", "nahin", "नहीं", "no allergies", "none", "nothing", 
    "kuch nahi", "kuch nahi hai", "na", "न", "ना", "n", "nope", "never", "nil", "✓ no", "✗ no"
]

def is_unsure_response(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(p in t for p in UNSURE_PATTERNS)

def is_denial_response(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if is_unsure_response(t):
        return False
    if t in DENIAL_PATTERNS:
        return True
    return any(t == p or t.startswith(p + " ") or t.endswith(" " + p) for p in DENIAL_PATTERNS)


# ── Clinical HPI Cleaner (Eliminates Cross-Section Redundancy) ──
# Pre-compiled at module level — avoids re-compiling 14 regex patterns on every /api/patient-summary poll
_HPI_FILTER_RE = re.compile('|'.join([
    # Past medical history mentions / denials
    r'\b(?:past\s+(?:chronic\s+)?medical\s+(?:history|conditions?|illness|issues?)|past\s+medical|past\s+surgical|past\s+illness)\b',
    r'\b(?:denies\s+(?:any\s+)?past\s+(?:chronic\s+)?(?:medical|illness|conditions?))\b',
    r'\b(?:no\s+significant\s+past\s+medical)\b',
    r'\b(?:reports?\s+no\s+past\s+(?:medical|chronic))\b',
    # Family history mentions / denials
    r'\b(?:family\s+history|hereditary\s+conditions?|family\s+members?\s+(?:have|had))\b',
    r'\b(?:denies\s+(?:any\s+)?(?:significant\s+)?family\s+history)\b',
    r'\b(?:no\s+significant\s+family\s+history)\b',
    # Allergies mentions / denials
    r'\b(?:drug\s+or\s+food\s+allergies|known\s+drug|food\s+allergies|allergic\s+to\s+(?:any\s+)?(?:medication|food|drugs?)|allergies\b|nkda\b)',
    r'\b(?:denies\s+(?:any\s+)?(?:known\s+)?(?:drug|food\s+)?allergies)\b',
    r'\b(?:no\s+known\s+(?:drug|food\s+)?allergies)\b',
    # Personal / Lifestyle mentions / denials
    r'\b(?:lifestyle\s+or\s+habit|lifestyle\s+risks?|habit\s+risks?|smoking\s+(?:or|and)\s+alcohol|tobacco|substance\s+use)\b',
    r'\b(?:denies\s+(?:any\s+)?significant\s+lifestyle)\b',
    r'\b(?:no\s+significant\s+lifestyle)\b',
    # Systemic ROS checklist denials
    r'\b(?:associated\s+systemic\s+symptoms|denies\s+(?:any\s+)?associated\s+systemic|review\s+of\s+systems)\b',
]), re.IGNORECASE)
_HPI_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def clean_hpi_text(hpi: str) -> str:
    """
    Cleans History of Present Illness (HPI) text by removing sentences that mistakenly 
    duplicate or bundle Past Medical History, Family History, Allergies, Personal/Lifestyle History, 
    or general Review of Systems denials.
    """
    if not hpi or not isinstance(hpi, str) or not hpi.strip():
        return hpi

    cleaned_lines = []
    for line in hpi.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        
        if _HPI_FILTER_RE.search(line_str):
            sentences = _HPI_SENTENCE_SPLIT_RE.split(line_str)
            valid_sentences = [s.strip() for s in sentences if s.strip() and not _HPI_FILTER_RE.search(s)]
            if valid_sentences:
                cleaned_lines.append(" ".join(valid_sentences))
        else:
            cleaned_lines.append(line_str)

    cleaned_result = "\n".join(cleaned_lines).strip()
    return cleaned_result if cleaned_result else hpi


# ── Instant Patient Builder (0ms LLM Overhead) ──
def build_patient_from_transcript(transcript, language, is_ayush, pt_id, db, abha_id=None, patient_name="Patient", age="", gender="", phone=""):
    t_lower = transcript.lower()
    is_emerg = any(w in t_lower for w in [
        "chest pain", "heart", "attack", "breath", "stroke", "paralysis", "unconscious", "bleeding", "accident",
        "सीने में दर्द", "हार्ट", "अटैक", "सांस", "बेहोश", "खून", "छातीत दुखणे", "గుండె", "நெஞ்சு வலி", "বুকের ব্যথা"
    ])

    symptom_cat = detect_symptom_category(transcript)
    print(f"🎯 Detected primary symptom track: '{symptom_cat}' for complaint: '{transcript}'")

    dialogue_entry = f"Patient (Chief Complaint - {language}): {transcript}\n"

    if is_ayush:
        cat_map = {
            "chest_pain": ("Vata-Pitta dominant", "Pranavaha Srotorodha (Vata-Kaphaja Hridshoola)", "Vishamagni (Irregular digestive fire)"),
            "stomach_pain": ("Pitta-Vata dominant", "Annavaha Srotas Dushti (Pitta-Vataja Amlapitta / Parinama Shoola)", "Mandagni with Ama accumulation"),
            "joint_pain": ("Vata-Kapha dominant", "Asthivaha Srotas Dushti (Vataja Sandhivata / Amavata)", "Mandagni (Sluggish metabolic fire)"),
            "headache": ("Vata-Pitta dominant", "Majjavaha Srotas Dushti (Shirashoola / Suryavarta)", "Vishamagni (Irregular fire)"),
            "fever": ("Pitta-Kapha dominant", "Rasavaha Srotas Dushti (Vata-Pitta Jwara)", "Mandagni (Jwaragni state)"),
        }
        init_prak, init_vik, init_agni = cat_map.get(symptom_cat, ("Vata-Pitta dominant", "Doshic Vaishamya (Vata-Pitta Dushti)", "Vishamagni"))
    else:
        init_prak, init_vik, init_agni = "Not assessed", "Not assessed", "Not assessed"

    patient = PatientRecord(
        patient_id=pt_id,
        abha_id=abha_id,
        is_ayush=bool(is_ayush),
        patient_name=patient_name or "Patient",
        age=str(age) if age else "",
        gender=str(gender) if gender else "",
        phone=str(phone) if phone else "",
        symptom_category=symptom_cat,
        chief_complaint=transcript,
        hpi=f"Patient reports: {transcript}",
        is_emergency=is_emerg,
        severity="High" if is_emerg else "Medium",
        duration="Recording in progress",
        past_medical_history="Awaiting synthesis",
        family_history="Awaiting synthesis",
        personal_history="Awaiting synthesis",
        allergies="Awaiting synthesis",
        review_of_systems="Awaiting synthesis",
        prakriti=init_prak,
        vikriti=init_vik,
        agni=init_agni,
        raw_dialogue=dialogue_entry,
        is_synthesized=False,
        created_at=datetime.now().strftime("%I:%M %p")
    )
    if db is not None:
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # Seed mock ABHA history if available
    if abha_id and db is not None:
        seed_abha_history(abha_id, db)

    # Use targeted SOCRATES-framework question template for the detected symptom track
    initial_question = get_phase_question(language, "initial", category=symptom_cat)
    return patient, initial_question


# ── Stage 2: Post-Interview Holistic Clinical Synthesis (Background Task) ──
def synthesize_and_filter_patient_background(patient_id_db: int, abha_id: Optional[str], language: str, is_ayush: bool):
    """Runs after intake/document scan finishes: synthesizes full consultation + uploaded documents and filters ABHA records."""
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        if not patient:
            return
        is_ayush_mode = bool(is_ayush or patient.is_ayush)
        patient.is_ayush = is_ayush_mode
        
        full_transcript = patient.raw_dialogue or patient.chief_complaint
        if is_ayush_mode:
            ayush_inst = """
CRITICAL AYUSH (AYURVEDIC / INTEGRATIVE MEDICINE) OPD DIRECTIVE:
The patient has registered at the AYUSH OPD. You MUST provide an authentic, high-caliber Ayurvedic & Integrative CDSS assessment:
1. `prakriti`: Identify likely baseline Ayurvedic constitutional Prakriti (e.g. "Vata-Pitta dominant", "Pitta-Kaphaja", "Vata-Kaphaja", "Kapha-Pradhana").
2. `vikriti`: Identify the active doshic imbalance & affected body channels / srotas (e.g. "Pranavaha Srotas Dushti with Vata-Kaphaja Hridshoola / Hridroga", "Annavaha & Purishavaha Srotas Dushti with Pitta-Vataja Amlapitta", "Asthivaha & Sandhi Srotas Dushti with Vataja Sandhivata").
3. `agni`: Assess metabolic and digestive state (e.g. "Vishamagni (Irregular digestive fire)", "Mandagni (Sluggish metabolic fire with Ama accumulation)", "Tikshnagni (Hyperactive Pitta fire)", "Samagni (Balanced)").
4. In `clinical_impression`:
   - Under `probable_diagnoses`, provide BOTH the classical Ayurvedic disease entity (e.g. "Hridshoola / Vata-Kaphaja Hridroga (correlating with Acute Coronary Syndrome)", "Amlapitta / Parinama Shoola (correlating with Peptic/Gastric disorder)", "Sandhivata / Amavata (correlating with Osteoarthritis/Inflammatory Arthritis)", "Shirashoola / Suryavarta (correlating with Cephalea/Migraine)") AND its modern clinical correlate.
   - Under `suggested_investigations`, provide essential modern safety diagnostics (e.g., ECG, Troponin, CBC) PLUS key Ayurvedic & lifestyle recommendations (e.g., "Hridaya Basti & Snehana therapy", "Arjuna Ksheerapaka & Prabhakar Vati evaluation", "Pathya-Apathya: Vata-shamaka light warm diet").
   - Under `critical_rule_outs`, list urgent red flags requiring emergency allopathic stabilization (e.g., "Acute Myocardial Infarction / STEMI").
"""
        else:
            ayush_inst = "The setting is standard Allopathic. Set prakriti, vikriti, agni to 'Not assessed'."

        # Tier 1: Patient's Today's Spoken Input (Primary Clinical Anchor)
        tier1_str = f"=== TIER 1: PATIENT'S TODAY'S SPOKEN INTAKE (PRIMARY CLINICAL ANCHOR) ===\n{full_transcript}"

        # Tier 2: Currently Uploaded Documents & Lab Reports (Immediate Objective Corroboration)
        tier2_str = "=== TIER 2: CURRENTLY UPLOADED MEDICAL DOCUMENTS & LAB REPORTS ===\n(No documents uploaded today)"
        if patient.flagged_lab_values and patient.flagged_lab_values != "[]":
            try:
                docs_list = json.loads(patient.flagged_lab_values)
                if isinstance(docs_list, list) and len(docs_list) > 0:
                    tier2_str = "=== TIER 2: CURRENTLY UPLOADED MEDICAL DOCUMENTS & LAB REPORTS (IMMEDIATE CORROBORATION) ===\n"
                    for idx, d in enumerate(docs_list, 1):
                        if isinstance(d, dict):
                            mod_tag = d.get('modality', 'document').upper()
                            tier2_str += f"- Document #{idx} [{mod_tag}] ({d.get('document_type', 'Report')} - Date: {d.get('document_date', 'Unknown')}): Summary: {d.get('summary', '')}, Diagnoses/Findings: {d.get('diagnoses', [])}, Meds: {d.get('medications', [])}, Flagged Abnormalities: {d.get('flagged_values', [])}\n"
            except Exception as e:
                print(f"Error parsing docs for synthesis: {e}")

        # Tier 3: Verified Past ABHA Historical Records (Background Context & Risk Filter)
        tier3_str = "=== TIER 3: VERIFIED PAST ABHA MEDICAL HISTORY (BACKGROUND CONTEXT ONLY) ===\n(No past ABHA records on file)"
        if abha_id:
            try:
                past_visits = db.query(VisitHistory).filter(VisitHistory.abha_id == abha_id).all()
                if past_visits:
                    tier3_str = "=== TIER 3: VERIFIED PAST ABHA MEDICAL HISTORY (BACKGROUND CONTEXT ONLY) ===\n"
                    for idx, v in enumerate(past_visits, 1):
                        tier3_str += f"- Historical Visit #{idx} ({v.visit_date} - {v.specialty}): Chief Complaint: {v.chief_complaint}, Diagnoses: {v.diagnoses}, Meds: {v.medications}, Flags: {v.flagged_values}, Summary: {v.summary}\n"
            except Exception as e:
                print(f"Error loading ABHA history for synthesis: {e}")

        prompt = f"""You are an expert Chief Medical Officer and AI Clinical Decision Support Specialist.
Review the patient's data below following a strict 3-tier clinical diagnostic reasoning hierarchy:

Language spoken: {language}
{ayush_inst}

{tier1_str}

{tier2_str}

{tier3_str}

TASK: Perform high-precision clinical synthesis and generate Clinical Decision Support (CDSS) insights into a standard medical EHR record.

CRITICAL CLINICAL REASONING ORDER FOR PROBABLE DIAGNOSES (CDSS):
You MUST follow this exact sequential diagnostic reasoning flow:
1. STEP 1 — ANCHOR ON PATIENT'S CURRENT PRESENTATION (TIER 1):
   - The primary suspected condition MUST be anchored strictly to the patient's active complaints, onset, location, character, and systemic symptoms reported TODAY.
2. STEP 2 — CORROBORATE WITH CURRENTLY UPLOADED DOCUMENTS (TIER 2):
   - Cross-examine Tier 1 symptoms against today's scanned blood tests, ECGs, or imaging flags to confirm or refine the acute diagnosis.
3. STEP 3 — FILTER BACKGROUND CONTEXT FROM PAST ABHA HISTORY (TIER 3):
   - Check historical ABHA visits ONLY to identify relevant risk factors, past recurrent conditions, or chronic co-morbidities (e.g. past CAD stenting when presenting with chest pain).
   - STRICT WARNING: NEVER allow unrelated past history (e.g. past ankle sprain or cataract) to override or misguide today's acute diagnosis when today's symptoms represent a different organ system!
4. STEP 4 — FORMULATE DIFFERENTIAL DIAGNOSES:
   - Generate top 2-3 differential diagnoses reflecting this exact priority order. Each diagnosis must clearly state its likelihood ("High"|"Medium"|"Low") and supporting evidence linking Tier 1 -> Tier 2 -> Tier 3.

SECTION SPECIFIC RULES:
1. HISTORY OF PRESENT ILLNESS (HPI) — STRICT BOUNDARIES:
   - `hpi` MUST ONLY describe the chronology of the CURRENT presenting complaint (onset, duration, anatomical site, character, severity, progression, aggravating/relieving factors).
   - STRICT PROHIBITION: DO NOT mention past medical history, family history, lifestyle/personal habits, allergies, or general review-of-systems denials in the `hpi` field. Each belongs ONLY in its dedicated section.

2. ACCURATELY DISTINGUISH DENIAL ("NO") VS UNCERTAINTY ("NOT SURE / DON'T REMEMBER") VS NOT ASKED:
   - When patient clearly DENIES: Write "Patient denies..."
   - When patient expresses UNCERTAINTY / LACK OF MEMORY: Record as "Uncertain / unconfirmed (patient does not recall / unsure)". DO NOT write "No" or "Denies"!
   - When NOT asked: Record as "Not assessed".

3. REVIEW OF SYSTEMS (ROS):
   - Actively summarize associated systemic symptoms asked or reported during the interview (e.g. "Patient denies fever, vomiting, or dyspnea; reports diaphoresis").

4. CLINICAL DECISION SUPPORT (CDSS) & DIAGNOSTIC IMPRESSION:
   - `clinical_impression`:
     * `clinical_synthesis`: Array of 2-3 concise bullet points: (1) Current acute presentation/timeline, (2) Corroborating lab/imaging findings, (3) Relevant historical ABHA context & primary clinical etiology rationale.
     * `probable_diagnoses`: Top 2-3 differentials with `condition`, `likelihood` ("High"|"Medium"|"Low"), and `supporting_evidence`.
     * `suggested_investigations`: 2-4 recommended next diagnostic tests/scans.
     * `critical_rule_outs`: 1-3 high-risk life-threatening conditions to actively exclude.

5. EMERGENCY TRIAGE & SEVERITY:
   - Set `is_emergency`: true if red flags (acute coronary syndrome, stroke signs, severe trauma, acute respiratory distress), else false.
   - Set `severity`: "High" | "Medium" | "Low".

6. Set `next_question` to 'complete'.

Output ONLY valid JSON:
{PATIENT_JSON_TEMPLATE}"""

        print(f"→ Synthesizing tiered clinical record + CDSS (Tier 1 Input -> Tier 2 Docs -> Tier 3 ABHA) for patient {patient.patient_id} in background...")
        response_text = call_llm(prompt)
        result_json = json.loads(extract_json_string(response_text))
        result_json = unwrap_json(result_json)
        ext = PatientExtraction(**result_json)

        # Update patient record with synthesized clinical data
        patient.chief_complaint = ext.chief_complaint or patient.chief_complaint
        patient.hpi = clean_hpi_text(ext.hpi or patient.hpi)
        patient.is_emergency = ext.is_emergency
        patient.severity = ext.severity or patient.severity
        patient.duration = ext.duration or "Unknown"
        patient.past_medical_history = ext.past_medical_history or "No significant past medical history"
        patient.family_history = ext.family_history or "No significant family history"
        patient.personal_history = ext.personal_history or "No significant lifestyle risks"
        patient.allergies = ext.allergies or "No known drug allergies (NKDA)"
        patient.review_of_systems = ext.review_of_systems or "Patient denies associated systemic symptoms"
        if ext.clinical_impression and isinstance(ext.clinical_impression, dict):
            patient.clinical_impression_json = json.dumps(ext.clinical_impression)
        
        if is_ayush_mode:
            cat_map = {
                "chest_pain": ("Vata-Pitta dominant", "Pranavaha Srotorodha (Vata-Kaphaja Hridroga / Hridshoola)", "Vishamagni (Irregular digestive fire)"),
                "stomach_pain": ("Pitta-Vata dominant", "Annavaha Srotas Dushti (Pitta-Vataja Amlapitta / Parinama Shoola)", "Mandagni with Ama accumulation"),
                "joint_pain": ("Vata-Kapha dominant", "Asthivaha Srotas Dushti (Vataja Sandhivata / Amavata)", "Mandagni (Sluggish metabolic fire)"),
                "headache": ("Vata-Pitta dominant", "Majjavaha Srotas Dushti (Shirashoola / Suryavarta)", "Vishamagni (Irregular fire)"),
                "fever": ("Pitta-Kapha dominant", "Rasavaha Srotas Dushti (Vata-Pitta Jwara)", "Mandagni (Jwaragni state)"),
            }
            def_prak, def_vik, def_agni = cat_map.get(patient.symptom_category or "general", ("Vata-Pitta dominant", "Doshic Vaishamya (Vata-Pitta Dushti)", "Vishamagni"))
            patient.prakriti = ext.prakriti if (ext.prakriti and ext.prakriti != "Not assessed") else def_prak
            patient.vikriti = ext.vikriti if (ext.vikriti and ext.vikriti != "Not assessed") else def_vik
            patient.agni = ext.agni if (ext.agni and ext.agni != "Not assessed") else def_agni
        else:
            patient.prakriti = "Not assessed"
            patient.vikriti = "Not assessed"
            patient.agni = "Not assessed"

        patient.is_synthesized = True

        db.commit()
        print(f"✅ Tiered Clinical record & CDSS synthesis complete for {patient.patient_id} ({patient.chief_complaint})")

        # Now correlate and filter past ABHA visit records against the full synthesized clinical profile
        if abha_id:
            filter_history_background(patient_id_db, abha_id, patient.chief_complaint)

    except Exception as e:
        print(f"❌ Synthesis error: {e}. Falling back to dialogue-based clinical extraction...")
        try:
            raw_text = patient.raw_dialogue or ""
            dialogue_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            patient_answers = []
            for line in dialogue_lines:
                if line.startswith("Patient") and ":" in line:
                    ans = line.split(":", 1)[1].strip()
                    if ans:
                        patient_answers.append(ans)
            
            cc = patient.chief_complaint or (patient_answers[0] if patient_answers else "Acute medical evaluation requested")
            patient.chief_complaint = cc
            if not patient.hpi or "Awaiting" in patient.hpi or patient.hpi == f"Patient reports: {cc}":
                if patient_answers:
                    patient.hpi = f"Patient presents with {cc}. Interview notes: " + "; ".join(patient_answers[:4])
                else:
                    patient.hpi = f"Patient presents with: {cc}."
            
            if not patient.past_medical_history or "Awaiting" in patient.past_medical_history:
                patient.past_medical_history = "No significant chronic medical history reported during intake"
            if not patient.family_history or "Awaiting" in patient.family_history:
                patient.family_history = "No significant hereditary or family illness reported"
            if not patient.personal_history or "Awaiting" in patient.personal_history:
                patient.personal_history = "No adverse lifestyle or substance risks reported"
            if not patient.allergies or "Awaiting" in patient.allergies:
                patient.allergies = "No known drug or food allergies (NKDA)"
            if not patient.review_of_systems or "Awaiting" in patient.review_of_systems:
                patient.review_of_systems = "Systemic symptoms reviewed during intake"
            
            if not patient.clinical_impression_json or patient.clinical_impression_json == "{}":
                fallback_impression = {
                    "clinical_synthesis": [
                        f"Patient presented with: {cc}",
                        f"Track: {patient.symptom_category or 'General OPD'} with acuity status: {patient.severity or 'Medium'}.",
                        "Synthesized using structured intake dialogue."
                    ],
                    "probable_diagnoses": [
                        {
                            "condition": f"Symptom complex: {patient.symptom_category or cc}",
                            "likelihood": "Medium",
                            "supporting_evidence": f"Patient reports: {cc}"
                        }
                    ],
                    "suggested_investigations": ["Clinical evaluation by physician", "Vital signs recording", "Basic diagnostic panel"],
                    "critical_rule_outs": ["Acute cardiopulmonary / surgical red flags"]
                }
                patient.clinical_impression_json = json.dumps(fallback_impression)
            
            patient.is_synthesized = True
            db.commit()
            print(f"✅ Fallback structured clinical record saved for {patient.patient_id}")
        except Exception as fb_err:
            print(f"❌ Fallback extraction error: {fb_err}")
    finally:
        db.close()


# ═══════════════ ENDPOINTS ═══════════════

@app.get("/")
async def root():
    return {"message": "MediKiosk v2 Backend running (Offline Mode)"}


# ── ABHA Master Profiles Endpoint ──
@app.get("/api/abha-profiles")
async def get_abha_profiles():
    """Returns the 5 pre-configured ABHA patient profiles for quick lookup."""
    return [
        {
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "phone": p["phone"],
            "abha_id": p["abha_id"],
            "avatar": p["avatar"],
            "badge": p["badge"],
            "history_count": len(p["history"])
        }
        for p in ABHA_PROFILES.values()
    ]

@app.get("/api/abha-profile/{abha_id}")
async def get_abha_profile_by_id(abha_id: str):
    """Lookup demographic details for a given ABHA ID with hyphen/space normalization."""
    raw_clean = re.sub(r'[\s\-]', '', abha_id.strip())
    
    for key, p in ABHA_PROFILES.items():
        key_clean = re.sub(r'[\s\-]', '', key)
        if raw_clean == key_clean or abha_id.strip().lower() == key.lower():
            return {
                "found": True,
                "name": p["name"],
                "age": p["age"],
                "gender": p["gender"],
                "phone": p["phone"],
                "abha_id": p["abha_id"],
                "badge": p["badge"],
                "summary": p["summary"] if "summary" in p else ""
            }
    return {
        "found": False,
        "message": f"No ABHA account found with number '{abha_id}'. Please check the 14-digit number and try again."
    }


# ── Initial Complaint (Audio) ──
@app.post("/api/process-audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    abha_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form("Patient"),
    age: Optional[str] = Form(""),
    gender: Optional[str] = Form(""),
    phone: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    audio_bytes = await audio.read()
    pt_id = f"PT-{str(uuid.uuid4())[:4].upper()}"

    # 1. Whisper STT (In-Memory Streaming — Zero Disk I/O)
    audio_stream = io.BytesIO(audio_bytes)
    lang_code = LANGUAGE_CODES.get(language, "en")
    wp = get_whisper_pipeline()
    if not wp:
        raise HTTPException(status_code=503, detail="Whisper speech recognition model unavailable")
    segments, info = wp.transcribe(
        audio_stream, 
        language=lang_code, 
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400,
            threshold=0.5
        ),
        initial_prompt="A clinical consultation in a hospital OPD. Symptoms, pain, fever, duration, past history.",
        condition_on_previous_text=False
    )
    transcript = " ".join([segment.text for segment in segments]).strip()
    print(f"Transcript: {transcript}")

    # 2. Instant patient initialization (0ms LLM calls)
    patient, next_q = build_patient_from_transcript(
        transcript=transcript,
        language=language,
        is_ayush=is_ayush,
        pt_id=pt_id,
        db=db,
        abha_id=abha_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        phone=phone
    )

    return {
        "status": "success",
        "extracted_complaint": patient.chief_complaint,
        "is_emergency": patient.is_emergency,
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name,
        "abha_id": patient.abha_id,
        "transcript": transcript,
        "next_question": next_q
    }


# ── Initial Complaint (Text) ──
@app.post("/api/process-text")
async def process_text(
    background_tasks: BackgroundTasks,
    transcript: str = Form(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    abha_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form("Patient"),
    age: Optional[str] = Form(""),
    gender: Optional[str] = Form(""),
    phone: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    pt_id = f"PT-{str(uuid.uuid4())[:4].upper()}"
    patient, next_q = build_patient_from_transcript(
        transcript=transcript,
        language=language,
        is_ayush=is_ayush,
        pt_id=pt_id,
        db=db,
        abha_id=abha_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        phone=phone
    )

    return {
        "status": "success",
        "extracted_complaint": patient.chief_complaint,
        "is_emergency": patient.is_emergency,
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name,
        "abha_id": patient.abha_id,
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
    db: Session,
    background_tasks: Optional[BackgroundTasks] = None
):
    phase_names = {
        0: "Duration / Onset",
        1: "Radiation of Pain",
        2: "Associated Symptoms (ROS)",
        3: "Medications & Relieving Factors",
        4: "Past Medical / Family / Lifestyle",
        5: "Allergies"
    }
    phase_label = phase_names.get(follow_up_count, f"Phase {follow_up_count}")
    symptom_cat = patient.symptom_category or "general"
    prev_q = get_phase_question(language, follow_up_count - 1 if follow_up_count > 0 else "initial", category=symptom_cat)
    dialogue_entry = f"Doctor ({phase_label}): {prev_q}\nPatient ({language}): {transcript}\n"

    current_dialogue = patient.raw_dialogue or ""
    patient.raw_dialogue = current_dialogue + dialogue_entry

    is_complete = True if follow_up_count >= 5 else False
    next_question = get_phase_question(language, follow_up_count, category=symptom_cat) if follow_up_count < 5 else "Thank you. Let us proceed to document scanning."

    # When the 5-question interview finishes, deload Whisper to free 2.5GB VRAM, then trigger AI synthesis
    if is_complete:
        unload_whisper()
        if background_tasks is not None:
            print(f"🚀 Patient intake complete! Dispatching holistic AI synthesis & ABHA history filter for {patient.patient_id}...")
            background_tasks.add_task(synthesize_and_filter_patient_background, patient.id, patient.abha_id, language, is_ayush)

    if db is not None:
        db.commit()
        db.refresh(patient)

    return {
        "status": "success",
        "extracted_info": f"Recorded: {transcript}",
        "is_complete": is_complete,
        "next_question": next_question,
        "transcript": transcript
    }


# ── Follow-up (Text) ──
@app.post("/api/follow-up-text")
async def follow_up_text(
    background_tasks: BackgroundTasks,
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
        db=db,
        background_tasks=background_tasks
    )


# ── Follow-up (Audio) ──
@app.post("/api/follow-up")
async def follow_up_audio(
    background_tasks: BackgroundTasks,
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
    # In-Memory Streaming — Zero Disk I/O
    audio_stream = io.BytesIO(audio_bytes)
    lang_code = LANGUAGE_CODES.get(language, "en")
    wp = get_whisper_pipeline()
    if not wp:
        raise HTTPException(status_code=503, detail="Whisper speech recognition model unavailable")
    segments, info = wp.transcribe(
        audio_stream, 
        language=lang_code, 
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400,
            threshold=0.5
        ),
        initial_prompt="A clinical consultation in a hospital OPD. Symptoms, pain, fever, duration, past medical history.",
        condition_on_previous_text=False
    )
    transcript = " ".join([segment.text for segment in segments]).strip()
    print(f"Follow-up transcript: {transcript}")

    return handle_followup_extraction(
        patient=patient,
        transcript=transcript,
        language=language,
        conversation_context=conversation_context,
        is_ayush=is_ayush,
        follow_up_count=follow_up_count,
        db=db,
        background_tasks=background_tasks
    )


def detect_visual_modality(image_bytes: bytes, filename: str = "") -> tuple[str, str]:
    """
    Ultra-fast, zero-VRAM mathematical pixel-space classifier using PIL and numpy (<15MB RAM, <5ms).
    Accurately classifies medical documents into 5 clinical imaging domains:
      1. 'ecg': Periodic millimeter pink/salmon grid lines with continuous 12-lead signal waveforms.
      2. 'pathology': H&E dye spectrum (high violet/magenta cellular clusters).
      3. 'endoscopy': Intraluminal mucosal warm tones, scope lighting, circular cavity frames, or dermoscopy.
      4. 'radiology': Grayscale radiograph or blue-tinted radiograph with dark background polarity (X-Ray, CT, MRI, Ultrasound).
      5. 'document': High brightness white/cream paper background with dark text strokes (Prescription, Lab Report).
    Returns (modality_key, domain_conditioned_prompt).
    """
    radiology_prompt = (
        "You are an expert Radiologist & Diagnostic Imaging Specialist. "
        "Analyze this radiological scan (X-Ray, CT Scan, MRI, Ultrasound, or Mammogram). "
        "Identify the exact anatomical region (e.g. Skull/Head, Chest/Lungs, Spine, Knee, Pelvis, Abdomen). "
        "Carefully evaluate bone cortical continuity for fracture lines, joint spaces, soft tissue swelling, lung opacities, cardiomegaly, or focal lesions, "
        "and transcribe any printed radiologic annotations, patient labels, or orientation markers (L/R)."
    )
    ecg_prompt = (
        "You are an expert Clinical Cardiologist & ECG Specialist. "
        "Analyze this 12-Lead Electrocardiogram (ECG / EKG) waveform report. "
        "Examine rhythm regularity, heart rate, P-wave morphology, PR-interval, QRS complex width, "
        "and ST-segments across leads (I, II, III, aVR, aVL, aVF, V1-V6). "
        "Report any ST-segment elevation (STEMI), depression, T-wave inversion, bundle branch block, or arrhythmia, "
        "and transcribe all visible patient or machine calibration text."
    )
    pathology_prompt = (
        "You are an expert Histopathologist & Biopsy Specialist. "
        "Analyze this pathology, cytology, or biopsy microscopic image or report. "
        "Describe cellular architecture, nuclear pleomorphism, tissue margins, cytoplasmic features, mitotic activity, "
        "and transcribe the anatomical specimen site, histological grade, or diagnostic conclusion."
    )
    endoscopy_prompt = (
        "You are an expert Endoscopy & Cavity Diagnostic Specialist. "
        "Analyze this internal cavity image (Endoscopy, Colonoscopy, Laparoscopy, Bronchoscopy, Dermoscopy, or Retinal Fundus). "
        "Carefully evaluate the mucosal lining, vascular pattern, and check for ulcers, erosions, polyps, bleeding, erythema, or lesions. "
        "Transcribe any visible procedure annotations, anatomical landmarks, or instrument markings."
    )
    doc_prompt = (
        "You are an expert Medical OCR & Clinical Documentation Specialist. "
        "Analyze this clinical prescription, doctor consultation note, or laboratory diagnostic report. "
        "Transcribe the clinic/hospital header, doctor name, patient details, presenting symptoms, "
        "clinical diagnoses, prescribed medications (with dosages, route, and frequency), advised diagnostic tests, and doctor instructions."
    )

    # 1. Immediate high-confidence filename heuristics
    fn = (filename or "").lower()
    if any(k in fn for k in ["xray", "x-ray", "cxr", "chest", "radiology", "radiograph", "ct_scan", "ct-scan", "mri", "ultrasound"]):
        return "radiology", radiology_prompt
    if any(k in fn for k in ["ecg", "ekg", "cardio", "rhythm", "holter"]):
        return "ecg", ecg_prompt
    if any(k in fn for k in ["biopsy", "patholog", "histolog", "cytolog", "specimen"]):
        return "pathology", pathology_prompt
    if any(k in fn for k in ["prescription", "doctor_slip", "rx_slip", "consultation_slip"]):
        return "document", doc_prompt

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Fast thumbnail resize (<2ms, <1MB RAM)
        img_thumb = img.resize((150, 150))
        hsv = np.array(img_thumb.convert('HSV'))

        h = hsv[:, :, 0]  # 0..255 (Hue)
        s = hsv[:, :, 1]  # 0..255 (Saturation)
        v = hsv[:, :, 2]  # 0..255 (Value / Brightness)

        mean_s = float(np.mean(s))
        mean_v = float(np.mean(v))
        dark_pixel_ratio = float(np.mean(v < 75))
        bright_pixel_ratio = float(np.mean(v > 180))

        # Inspect central 60% of image where clinical document / strip body resides
        h_c = h[30:120, 30:120]
        s_c = s[30:120, 30:120]
        v_c = v[30:120, 30:120]

        center_white_paper_ratio = float(np.mean((s_c < 45) & (v_c > 140)))

        # 1. ECG Pink/Salmon grid: Real ECG paper has calibrated millimetric pink grid lines
        # across the paper body itself (not on background wooden desk borders).
        # Specifically: Hue in [230..255] (pink/magenta) or [0..12] (bright red/salmon),
        # Saturation >= 45, Brightness >= 130 (NOT dark wood desks which have V < 130 and S in 30..90)
        ecg_center_mask = ((h_c >= 230) | (h_c <= 12)) & (s_c >= 45) & (s_c <= 200) & (v_c >= 130)
        ecg_center_ratio = float(np.mean(ecg_center_mask))

        # 2. Histopathology H&E violet/purple/magenta stain: Hue in [175..235], Saturation > 35, Brightness > 60
        pathology_mask = (h_c >= 175) & (h_c <= 235) & (s_c >= 35) & (v_c >= 60)
        pathology_ratio = float(np.mean(pathology_mask))

        # 3. Endoscopy / Mucosal / Dermoscopy warm tones: Hue in [0..30] or [240..255], Saturation > 45
        endoscopy_mask = ((h_c <= 30) | (h_c >= 240)) & (s_c >= 45)
        endoscopy_ratio = float(np.mean(endoscopy_mask))

        # 4. Radiograph Blue/Cyan tint (common Kodak/digital monitor tint): Hue in [130..180], Saturation > 20
        blue_cyan_mask = (h >= 130) & (h <= 180) & (s >= 20)
        blue_cyan_ratio = float(np.mean(blue_cyan_mask))

        # White paper document suppression: if center is predominantly white paper (>35%), it is ALWAYS a document/prescription
        img_aspect = float(img.width) / max(float(img.height), 1.0)
        if center_white_paper_ratio > 0.35:
            modality = "document"
            prompt = doc_prompt
        elif ecg_center_ratio > 0.15 and center_white_paper_ratio < 0.30 and img_aspect >= 0.9:
            modality = "ecg"
            prompt = ecg_prompt
        elif pathology_ratio > 0.20:
            modality = "pathology"
            prompt = pathology_prompt
        elif endoscopy_ratio > 0.30 and mean_s > 40:
            modality = "endoscopy"
            prompt = endoscopy_prompt
        elif ((mean_s < 30) or (blue_cyan_ratio > 0.30)) and (mean_v < 175 or dark_pixel_ratio > 0.15) and (bright_pixel_ratio < 0.55):
            modality = "radiology"
            prompt = radiology_prompt
        else:
            modality = "document"
            prompt = doc_prompt

        return modality, prompt
    except Exception as e:
        print(f"Modality detection error: {e}")
        return "document", (
            "You are an expert Medical Imaging & Clinical OCR specialist. "
            "Analyze this medical document or scan. Describe the visual findings, anatomical structures shown, "
            "and transcribe all visible text, diagnoses, medications, or advised tests."
        )


# ── Wide Physiological Plausibility Bounds ──
# These are NOT reference ranges. They define the widest possible human-physiological
# bounds used ONLY for: (a) sanity-checking extracted values, (b) decimal restoration.
# The actual patient-specific reference range always comes from the report itself.
PHYSIOLOGICAL_BOUNDS = {
    # Lung Volumes & Spirometry — widest plausible human range
    "FEV1/FVC": (20.0, 100.0, "%"),
    "FEF25-75":  (0.3, 15.0, "L/s"),
    "FEF25":     (0.3, 15.0, "L/s"),
    "FEF50":     (0.3, 15.0, "L/s"),
    "FEF75":     (0.2, 10.0, "L/s"),
    "FVC":       (0.5, 8.0,  "L"),
    "FEV1":      (0.3, 7.0,  "L"),
    "PEF":       (1.0, 16.0, "L/s"),
    "TLC":       (2.0, 10.0, "L"),
    "VC":        (0.5, 8.0,  "L"),
    "IC":        (0.5, 6.0,  "L"),
    "FRC":       (1.0, 6.0,  "L"),
    "ERV":       (0.3, 4.0,  "L"),
    "RV":        (0.3, 4.0,  "L"),
    "RV/TLC":    (10.0, 60.0, "%"),
    "DLCO/VA":   (1.0, 10.0, "mL/min/mmHg/L"),
    "DLCO":      (5.0, 50.0, "mL/min/mmHg"),
    "VA":        (2.0, 12.0, "L"),

    # Complete Blood Count (CBC)
    "HEMOGLOBIN": (3.0, 22.0, "g/dL"),
    "HB":         (3.0, 22.0, "g/dL"),
    "RBC":        (1.0, 8.0,  "mill/cumm"),
    "WBC":        (1000, 50000, "/cumm"),
    "LEUCOCYTE":  (1000, 50000, "/cumm"),
    "PLATELET":   (10000, 900000, "/cumm"),
    "PCV":        (15.0, 65.0, "%"),
    "MCV":        (50.0, 130.0, "fL"),
    "MCH":        (15.0, 45.0, "pg"),
    "MCHC":       (25.0, 40.0, "g/dL"),

    # Biochemistry / Renal / Hepatic
    "CREATININE": (0.1, 15.0, "mg/dL"),
    "BILIRUBIN":  (0.1, 30.0, "mg/dL"),
    "SGOT":       (0, 500, "U/L"),
    "AST":        (0, 500, "U/L"),
    "SGPT":       (0, 500, "U/L"),
    "ALT":        (0, 500, "U/L"),
    "ALKALINE PHOSPHATASE": (10, 500, "U/L"),
    "ALP":        (10, 500, "U/L"),
    "GLUCOSE":    (20, 600, "mg/dL"),
}


def _try_restore_decimal(val: float, bounds_low: float, bounds_high: float) -> float:
    """
    If `val` is outside physiological bounds, try dividing by 10 or 100
    to restore a dropped decimal point. Returns the best candidate.
    """
    if bounds_low <= val <= bounds_high:
        return val  # already plausible
    # Try /10
    v10 = val / 10.0
    if bounds_low <= v10 <= bounds_high:
        return v10
    # Try /100
    v100 = val / 100.0
    if bounds_low <= v100 <= bounds_high:
        return v100
    return val  # can't fix, return original


def _find_physiological_key(param_name: str) -> Optional[str]:
    """
    Match a test parameter name to its PHYSIOLOGICAL_BOUNDS key.
    Handles names like 'Erythrocytes Count (RBC)', 'Mean Corpuscular Volume (MCV)',
    'Total RBC', 'HB', etc.
    """
    if not param_name:
        return None
    # Clean the parameter name: strip parenthetical abbreviations, special chars
    # e.g. "Erythrocytes Count (RBC)" -> try both "ERYTHROCYTESCOUNT" and "RBC"
    param_upper = param_name.upper().strip()
    # Extract abbreviation from parentheses if present, e.g. "(RBC)" -> "RBC"
    abbrev_match = re.search(r'\(([A-Z0-9/\-]+)\)', param_upper)
    abbrev = abbrev_match.group(1) if abbrev_match else None
    # Clean full name for matching (remove all non-alphanumeric)
    param_clean = re.sub(r'[^A-Z0-9]', '', param_upper)
    
    # Try matching abbreviation first (most specific), then full name
    for k in sorted(PHYSIOLOGICAL_BOUNDS.keys(), key=lambda x: len(x), reverse=True):
        clean_k = re.sub(r'[^A-Z0-9]', '', k.upper())
        # Match by abbreviation
        if abbrev and (clean_k == abbrev or abbrev == clean_k):
            return k
        # Match by full cleaned name
        if clean_k == param_clean or param_clean.startswith(clean_k) or param_clean.endswith(clean_k):
            return k
    return None


def _validate_and_format_flagged(
    test_name: str,
    measured_val: float,
    unit: str,
    ref_low: Optional[float],
    ref_high: Optional[float],
    original_text: str = ""
) -> Optional[str]:
    """
    Core validation logic that works directly on numeric values.
    Returns formatted string if abnormal, None if normal (to suppress).
    """
    matched_key = _find_physiological_key(test_name)

    if matched_key:
        bounds_low, bounds_high, bounds_unit = PHYSIOLOGICAL_BOUNDS[matched_key]
        if not unit:
            unit = bounds_unit

        # Reject physiologically impossible values (e.g. negative percentages)
        if measured_val < 0 and bounds_low >= 0:
            return None  # impossible value, suppress

        # Decimal restoration for MEASURED VALUE
        measured_val = _try_restore_decimal(measured_val, bounds_low, bounds_high)

        # Decimal restoration for REFERENCE RANGE bounds
        if ref_low is not None and ref_high is not None:
            ref_low = _try_restore_decimal(ref_low, bounds_low, bounds_high)
            ref_high = _try_restore_decimal(ref_high, bounds_low, bounds_high)
            if ref_low > ref_high:
                ref_low, ref_high = ref_high, ref_low
            if abs(ref_high - ref_low) < 0.01:
                ref_low, ref_high = None, None

    if ref_low is not None and ref_high is not None:
        if measured_val < ref_low:
            status = "Low"
        elif measured_val > ref_high:
            status = "High"
        else:
            # Value is WITHIN normal range — suppress it
            context = (original_text or "").lower()
            if any(w in context for w in ['reversib', '+', 'chg', 'post', 'asthma', 'airway']):
                status = "Reversible Airway Response"
            else:
                return None

        # Use a clean short name (abbreviation if available, otherwise original)
        display_name = test_name
        unit_str = f" {unit}" if unit else ""
        return f"{display_name}: {measured_val:.2f}{unit_str} ({status}) [Ref: {ref_low} - {ref_high}{unit_str}]"

    # No valid range — pass through the original text as-is (for non-lab findings like fracture descriptions)
    return original_text if original_text else None


def sanitize_clinical_flagged_value(item_str: str) -> Optional[str]:
    """
    String-based fallback validator for legacy stored flagged values.
    Parses structured text like 'TestName: 5.5 g/dL (High) [Ref: 4.0 - 6.0]'
    and re-validates mathematically.
    """
    if not isinstance(item_str, str):
        return None

    # Clean double dots e.g. 1.2.4 -> 1.24
    item_str = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2\3', item_str)

    # Updated regex: allows parentheses in test names, negative values, complex units
    m = re.search(
        r'^\s*([A-Za-z0-9\-/%\s\(\)]+?):\s*(-?[\d\.]+)\s*([A-Za-z0-9/%\s]*?)?'
        r'(?:\s*\([^\)]*\))?\s*'
        r'(?:\[(?:Ref:)?\s*(-?[\d\.]+)?\s*[-\u2013~]?\s*(-?[\d\.]+)?\s*(?:[A-Za-z/%\s]*)?\])?$',
        item_str
    )
    if not m:
        return item_str  # Non-lab findings (fracture descriptions etc.) pass through

    param = m.group(1).strip()

    try:
        val = float(m.group(2))
    except (ValueError, TypeError):
        return item_str

    unit = (m.group(3) or "").strip()

    low, high = None, None
    if m.group(4) and m.group(5):
        try:
            low = float(m.group(4))
            high = float(m.group(5))
            if low > high:
                low, high = high, low
        except ValueError:
            pass

    return _validate_and_format_flagged(param, val, unit, low, high, item_str)


def normalize_extracted_flagged_values(raw_list) -> List[str]:
    """
    Universally normalizes flagged values into clean medical observation strings.
    Uses DIRECT numeric validation on structured dict objects (bypasses fragile regex).
    Falls back to string-based regex parsing for legacy string data.
    """
    if not isinstance(raw_list, list):
        return []
    res = []
    for item in raw_list:
        verified = None

        if isinstance(item, dict):
            # ── DIRECT NUMERIC VALIDATION (no string serialization) ──
            test = str(item.get("test_name") or item.get("test") or item.get("parameter") or item.get("name") or item.get("description") or "")
            unit = str(item.get("units") or item.get("unit") or "")

            # Extract measured value
            raw_val = item.get("measured_value") or item.get("observed_value") or item.get("value")
            try:
                measured_val = float(raw_val) if raw_val is not None else None
            except (ValueError, TypeError):
                measured_val = None

            # Extract reference range (structured fields preferred)
            ref_low, ref_high = None, None
            raw_ref_low = item.get("reference_low")
            raw_ref_high = item.get("reference_high")
            if raw_ref_low is not None and raw_ref_high is not None:
                try:
                    ref_low = float(raw_ref_low)
                    ref_high = float(raw_ref_high)
                except (ValueError, TypeError):
                    pass
            # Fallback to string-based range
            if ref_low is None or ref_high is None:
                range_str = str(item.get("reference_range") or item.get("normal_range") or item.get("reference_interval") or "")
                rm = re.search(r'(-?[\d\.]+)\s*[-\u2013~]\s*(-?[\d\.]+)', range_str)
                if rm:
                    try:
                        ref_low = float(rm.group(1))
                        ref_high = float(rm.group(2))
                    except ValueError:
                        pass

            if test and measured_val is not None:
                verified = _validate_and_format_flagged(test, measured_val, unit, ref_low, ref_high)
            else:
                # Unstructured dict — format as string and use legacy validator
                desc = str(item.get("description") or item.get("finding") or item)
                verified = desc if desc else None

        elif isinstance(item, str):
            # ── STRING FALLBACK (legacy stored data) ──
            verified = sanitize_clinical_flagged_value(item)

        if verified:
            res.append(verified)
    return res


def enforce_clinical_safety_guardrail(data: dict, raw_text: str) -> dict:
    """
    Deterministic clinical safety guardrail to eliminate AI hallucinations
    from garbled OCR on handwritten prescriptions or consultation slips.
    """
    if not isinstance(data, dict):
        return data

    raw_lower = (raw_text or "").lower()
    doc_type = str(data.get("document_type", "")).lower()
    
    # Check if document is a doctor prescription / consultation slip
    rx_indicators = ["dr.", "clinic", "rx", "hospital", "prescription", "consultant", "timing:", "patient name", "for medicine", "consulting physician", "mbbs", "md", "reg. no"]
    lab_indicators = ["reference range", "normal range", "lipid profile", "complete blood count", "cbc report", "spirometry", "pulmonary function", "liver function", "renal function", "kft", "lft", "urine routine"]
    
    has_rx_sign = any(k in raw_lower for k in rx_indicators) or "prescription" in doc_type or "clinic" in doc_type
    has_lab_sign = any(k in raw_lower for k in lab_indicators)

    is_prescription = has_rx_sign and not has_lab_sign

    if is_prescription:
        data["document_type"] = "Prescription / Doctor Consultation Slip"
        data["modality"] = "document"

        # Rule 1: Prescriptions NEVER have numerical lab flagged values (e.g. FEV1, Hemoglobin, Creatinine)
        data["flagged_values"] = []

        # Rule 2: Anti-Hallucination filter for severe medical diagnoses
        # Do not allow the model to invent Tuberculosis, Cancer, Fracture, Asthma from garbled OCR noise
        high_stakes_conditions = ["tuberculosis", "cancer", "carcinoma", "fracture", "fev1", "asthma", "heart failure"]
        cleaned_diagnoses = []
        for d in data.get("diagnoses", []):
            if isinstance(d, dict):
                d_str = str(d.get("name") or d.get("diagnosis") or d.get("symptom") or d.get("complaint", "")).strip()
            else:
                d_str = str(d).strip()
            d_lower = d_str.lower()
            hallucinated = False
            for cond in high_stakes_conditions:
                if cond in d_lower and cond not in raw_lower:
                    hallucinated = True
                    break
            if not hallucinated and d_str:
                cleaned_diagnoses.append(d_str)

        if not cleaned_diagnoses:
            cleaned_diagnoses = ["Clinical consultation recorded (Symptoms/Rx noted)"]
        data["diagnoses"] = cleaned_diagnoses

        # Normalize medications (flatten dicts if model returned structured objects)
        cleaned_meds = []
        for m in data.get("medications", []):
            if isinstance(m, dict):
                parts = [
                    str(m.get("name", "")).strip(),
                    str(m.get("strength") or m.get("dose", "")).strip(),
                    str(m.get("frequency", "")).strip(),
                    str(m.get("duration", "")).strip(),
                    str(m.get("instructions", "")).strip()
                ]
                med_str = " ".join(p for p in parts if p).strip()
                if med_str:
                    cleaned_meds.append(med_str)
            elif isinstance(m, str) and m.strip():
                cleaned_meds.append(m.strip())
        data["medications"] = cleaned_meds

        # Rule 3: Clinical safety pharmacy advisory on medications and summary
        safety_notice = "⚠️ Clinical Safety Notice: Cursive doctor handwriting requires pharmacist verification before dispensing medications."
        existing_summary = str(data.get("summary", "")).strip()

        # Clean generic prompt instruction echos
        if not existing_summary or any(k in existing_summary.lower() for k in ["full clinical summary", "concise summary", "brief summary", "output only"]):
            diag_str = ", ".join(cleaned_diagnoses) if cleaned_diagnoses else "Not specified"
            med_str = ", ".join(cleaned_meds) if cleaned_meds else "None noted"
            existing_summary = f"Prescription / Doctor Consultation Slip. Symptoms/Complaints: {diag_str}. Prescribed: {med_str}."

        # Remove any hallucinated mentions of high-stakes conditions from the summary if not in raw text
        for cond in high_stakes_conditions:
            if cond in existing_summary.lower() and cond not in raw_lower:
                existing_summary = re.sub(rf"(?i)\b{cond}\b[^.]*\.?", "", existing_summary).strip()

        if "pharmacist" not in existing_summary.lower():
            data["summary"] = f"{existing_summary} | {safety_notice}" if existing_summary else safety_notice
        else:
            data["summary"] = existing_summary

    return data


def process_document_background(file_bytes: bytes, filename: str, content_type: str, file_url: str, patient_id_db: int):
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        if not patient:
            return
            
        structured_data = None
        extracted_text = ""
        detected_modality = "document"
        try:
            if filename.lower().endswith('.pdf') or content_type == 'application/pdf':
                import fitz
                pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in pdf_doc:
                    extracted_text += page.get_text() + "\n"
                
                extracted_text_l = extracted_text.lower()
                is_pathology_pdf = any(k in extracted_text_l for k in [
                    "histopathology", "biopsy", "microscopic examination", "gross examination",
                    "cytology", "carcinoma", "neoplasm", "tissue pieces", "specimen received"
                ])
                is_lab_pdf = any(k in extracted_text_l for k in [
                    "reference range", "normal range", "lipid profile", "complete blood count",
                    "cbc report", "kft", "lft", "urine routine"
                ])

                if is_pathology_pdf:
                    detected_modality = "pathology"
                    diagnoses = []
                    flagged_pathology = []

                    imp_match = re.search(r'(?i)(?:IMPRESSION|FINAL DIAGNOSIS|DIAGNOSIS)\s*:\s*([^\n\r]+(?:\n[^\n\r]+)?)', extracted_text)
                    if imp_match:
                        imp_text = imp_match.group(1).strip().replace('\n', ' ')
                        diagnoses.append(imp_text)
                        flagged_pathology.append(f"🔬 Impression: {imp_text[:80]}")

                    clin_match = re.search(r'(?i)(?:Clinical Details|Clinical History)\s*[:\s]*([^\n\r]+)', extracted_text)
                    if clin_match:
                        clin_text = clin_match.group(1).strip()
                        if clin_text and clin_text != ':':
                            diagnoses.append(f"Clinical Indication: {clin_text}")
                            flagged_pathology.append(f"🔬 Indication: {clin_text[:60]}")

                    spec_match = re.search(r'(?i)(?:Nature of Material Received|Specimen)\s*[:\s]*([^\n\r]+)', extracted_text)
                    spec_text = spec_match.group(1).strip() if spec_match else "Biopsy Specimen"

                    summary_text = f"Histopathology Report ({spec_text}): " + ("; ".join(diagnoses) if diagnoses else "Biopsy examined.")

                    structured_data = {
                        "document_type": "Histopathology / Biopsy Report",
                        "modality": "pathology",
                        "diagnoses": diagnoses if diagnoses else ["Histopathological Examination"],
                        "medications": [],
                        "flagged_values": flagged_pathology,
                        "document_date": "Pathology Report",
                        "summary": summary_text,
                        "file_url": file_url,
                        "raw_text": extracted_text
                    }
                elif is_lab_pdf:
                    detected_modality = "document"
                    structured_data = {
                        "document_type": "Laboratory Diagnostic Report",
                        "modality": "document",
                        "diagnoses": ["Clinical Laboratory Analysis"],
                        "medications": [],
                        "flagged_values": [],
                        "document_date": "Lab Report",
                        "summary": "Laboratory diagnostic report processed.",
                        "file_url": file_url,
                        "raw_text": extracted_text
                    }
                else:
                    detected_modality = "document"
                    # Run CPU drug normalizer on the extracted digital text
                    normalized_drugs_list = normalize_drugs(extracted_text)
                    verified_meds = [f"{d['drug']} ({d.get('strength', '')}) [{d['status']}]" for d in normalized_drugs_list]
                    flagged_rx = [f"⚠️ Unverified Rx Token: '{d['raw_token']}' (Score: {d['score']}%)" for d in normalized_drugs_list if d["status"] == "FLAGGED_FOR_DOCTOR"]

                    structured_data = {
                        "document_type": "Prescription / Clinical Report" if verified_meds else "Clinical Document",
                        "modality": "document",
                        "diagnoses": ["Clinical Consultation Record"],
                        "medications": verified_meds,
                        "flagged_values": flagged_rx,
                        "document_date": "Digital Upload",
                        "summary": f"Digital document processed: {len(normalized_drugs_list)} medications identified.",
                        "file_url": file_url,
                        "raw_text": extracted_text
                    }
            else:
                # 1. Zero-VRAM mathematical pixel classifier (<5ms, CPU memory)
                detected_modality, domain_prompt = detect_visual_modality(file_bytes, filename=filename)
                print(f"📸 Detected Visual Modality: {detected_modality.upper()} | Routing to CPU Perception Engine...")

                if detected_modality == "radiology":
                    print("🩻 Executing Radiology Perception Engine on CPU (Zero VRAM)...")
                    xray_res = analyze_xray(file_bytes, file_url=file_url, filename=filename)
                    structured_data = xray_res.get("dashboard_payload", {})
                elif detected_modality == "ecg":
                    print("📈 Executing OpenCV ECG Waveform Analysis on CPU (Zero VRAM)...")
                    ecg_res = analyze_ecg(file_bytes, file_url=file_url)
                    structured_data = ecg_res.get("dashboard_payload", {})
                else:
                    # Prescriptions, doctor consultation slips, and general documents
                    print("📄 Executing Sauvola Preprocessing + Dual OCR + RapidFuzz Drug Matcher on CPU...")
                    presc_res = analyze_prescription(file_bytes, file_url=file_url)
                    structured_data = presc_res.get("dashboard_payload", {})

            print(f"✅ Background CPU Perception Complete [{structured_data.get('modality')}]: {structured_data.get('document_type')} - {structured_data.get('summary', '')[:80]}")
        except Exception as e:
            print(f"Background Document processing failed: {e}")

        if not structured_data:
            structured_data = {
                "document_type": "Medical Scan / Document" if detected_modality != "document" else "Unknown Document",
                "modality": detected_modality,
                "diagnoses": ["Extraction failed — please try again"],
                "medications": [],
                "flagged_values": [],
                "document_date": "Visual Scan",
                "summary": "Could not fully process this visual document. Please try a clearer scan.",
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
        walkin_id = f"PT-{uuid.uuid4().hex[:6].upper()}"
        patient = PatientRecord(
            patient_id=walkin_id,
            patient_name="Walk-In Patient",
            age=0,
            gender="Unknown",
            contact="",
            created_at=datetime.utcnow()
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

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


# ── Finalize Intake & Trigger Comprehensive Synthesis (Spoken Dialogue + Documents) ──
@app.post("/api/finalize-intake")
async def finalize_intake(
    background_tasks: BackgroundTasks,
    patient_id: str = Form(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Skip redundant synthesis if follow-up completion already triggered it (saves ~15-25s GPU time)
    if patient.is_synthesized:
        return {"status": "success", "message": "Synthesis already complete — skipping redundant call"}
    
    background_tasks.add_task(synthesize_and_filter_patient_background, patient.id, patient.abha_id, language, is_ayush)
    return {"status": "success", "message": "Comprehensive synthesis (dialogue + documents) queued"}


# ── On-Demand Resynthesize Endpoint (Doctor Dashboard & Error Recovery) ──
@app.post("/api/resynthesize")
@app.get("/api/resynthesize")
async def resynthesize_patient(
    background_tasks: BackgroundTasks,
    patient_id: str,
    language: str = "English",
    is_ayush: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    use_ayush = patient.is_ayush if is_ayush is None else is_ayush
    print(f"🔄 Manual/Doctor triggered re-synthesis for patient {patient.patient_id} ({patient.patient_name})...")
    background_tasks.add_task(synthesize_and_filter_patient_background, patient.id, patient.abha_id, language, use_ayush)
    return {"status": "success", "message": f"Synthesis queued for patient {patient_id}", "patient_id": patient_id}



# ── Red Flag Check ──
@app.get("/api/red-flag-check")
async def red_flag_check(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        return {"has_red_flags": False, "flags": [], "message": "Patient not found"}

    # Fast-Path: Zero-latency check using recorded clinical indicators
    text_to_check = f"{patient.chief_complaint or ''} {patient.hpi or ''}".lower()
    
    red_flag_terms = [
        ("chest pain", "Severe substernal chest discomfort"),
        ("सीने में दर्द", "सीने में तेज दर्द / दिल का दौरा"),
        ("heart attack", "Suspected acute coronary syndrome"),
        ("radiat", "Pain radiating to left arm/jaw"),
        ("breath", "Severe respiratory distress / shortness of breath"),
        ("सांस", "सांस लेने में अत्यधिक तकलीफ़"),
        ("unconscious", "Loss of consciousness or syncope"),
        ("बेहोश", "बेहोशी या चक्कर आना"),
        ("seizure", "Active seizures or neurological episode"),
        ("stroke", "Acute stroke symptoms / weakness on one side"),
        ("heavy bleeding", "Severe hemorrhage or acute blood loss"),
        ("खून", "अत्यधिक रक्तस्राव / खून की उल्टी"),
    ]

    matched_flags = []
    for term, label in red_flag_terms:
        if term in text_to_check:
            matched_flags.append(label)

    if patient.is_emergency or len(matched_flags) > 0:
        return {
            "has_red_flags": True,
            "flags": matched_flags if matched_flags else ["High-acuity symptoms requiring urgent triage assessment"],
            "message": "Potential urgent red-flag condition detected. Immediate clinical assessment recommended."
        }

    return {
        "has_red_flags": False,
        "flags": [],
        "message": "Safety check completed — no urgent flags detected."
    }


# ── Specialty Matching ──
@app.get("/api/specialty-match")
async def specialty_match(patient_id: str, language: str = "English", db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        return {"specialty": "General Medicine", "reason": "Default", "confidence": "Low"}

    if patient.is_ayush:
        return {
            "specialty": "AYUSH Medicine",
            "reason": "Holistic Ayurvedic & Traditional Medicine OPD",
            "confidence": "High"
        }

    cat = patient.symptom_category or detect_symptom_category(f"{patient.chief_complaint or ''} {patient.hpi or ''}")
    
    specialty_map = {
        "chest_pain": {
            "specialty": "Cardiology",
            "reason": "Evaluation of acute chest discomfort and cardiovascular parameters",
            "confidence": "High"
        },
        "stomach_pain": {
            "specialty": "Gastroenterology",
            "reason": "Assessment of abdominal, gastrointestinal, or digestive symptoms",
            "confidence": "High"
        },
        "headache": {
            "specialty": "Neurology",
            "reason": "Cranial, neurological, and migraine evaluation",
            "confidence": "High"
        },
        "fever": {
            "specialty": "General Medicine",
            "reason": "Acute febrile illness, infectious disease, and vitals assessment",
            "confidence": "High"
        },
        "joint_pain": {
            "specialty": "Orthopedics",
            "reason": "Musculoskeletal, bone, and joint examination",
            "confidence": "High"
        }
    }

    if cat in specialty_map:
        return specialty_map[cat]

    return {
        "specialty": "General Medicine",
        "reason": "Primary comprehensive medical consultation",
        "confidence": "Medium"
    }


# ── Patient Queue ──
@app.get("/api/patients")
async def get_patients(db: Session = Depends(get_db)):
    patients = db.query(PatientRecord).order_by(PatientRecord.id.desc()).all()
    return [{
        "patient_id": p.patient_id,
        "patient_name": p.patient_name or "Patient",
        "age": p.age,
        "gender": p.gender,
        "abha_id": p.abha_id,
        "is_ayush": bool(p.is_ayush),
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

    # Auto-clean and persist HPI if it contains redundant cross-section leakage
    cleaned_hpi = clean_hpi_text(patient.hpi) if patient.hpi else "None reported"
    if patient.hpi and cleaned_hpi != patient.hpi:
        patient.hpi = cleaned_hpi
        try:
            db.commit()
        except Exception:
            pass

    impression = {}
    if patient.clinical_impression_json:
        try:
            parsed = json.loads(patient.clinical_impression_json)
            if isinstance(parsed, dict):
                impression = parsed
        except Exception:
            impression = {}

    sanitized_flagged_lab_values = patient.flagged_lab_values or "[]"
    try:
        parsed_docs = json.loads(sanitized_flagged_lab_values)
        if isinstance(parsed_docs, list):
            for doc in parsed_docs:
                if isinstance(doc, dict) and "flagged_values" in doc and isinstance(doc["flagged_values"], list):
                    cleaned_items = []
                    for v in doc["flagged_values"]:
                        v_clean = sanitize_clinical_flagged_value(v)
                        if v_clean:
                            cleaned_items.append(v_clean)
                    doc["flagged_values"] = cleaned_items
            sanitized_flagged_lab_values = json.dumps(parsed_docs)
    except Exception:
        pass

    return {
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name or "Patient",
        "age": patient.age or "",
        "gender": patient.gender or "",
        "phone": patient.phone or "",
        "abha_id": patient.abha_id,
        "is_ayush": bool(patient.is_ayush),
        "chief_complaint": patient.chief_complaint or "Not recorded",
        "hpi": cleaned_hpi,
        "is_emergency": patient.is_emergency,
        "severity": patient.severity or "Unknown",
        "duration": patient.duration or "Unknown",
        "past_medical_history": patient.past_medical_history or "None reported",
        "family_history": patient.family_history or "None reported",
        "personal_history": patient.personal_history or "None reported",
        "allergies": patient.allergies or "None reported",
        "review_of_systems": patient.review_of_systems or "None reported",
        "clinical_impression": impression,
        "prakriti": patient.prakriti or "Not assessed",
        "vikriti": patient.vikriti or "Not assessed",
        "agni": patient.agni or "Not assessed",
        "flagged_lab_values": sanitized_flagged_lab_values,
        "is_synthesized": bool(patient.is_synthesized),
        "created_at": patient.created_at,
    }


# ── Patient History (ABHA-linked past visits) ──
@app.get("/api/patient-history")
async def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    """Returns past visit history strictly for this patient's linked ABHA profile."""
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient or not patient.abha_id:
        return {"relevant_history": [], "other_history": [], "filter_status": "no_abha", "abha_id": None}
    
    # Ensure profile history is seeded if not present
    seed_abha_history(patient.abha_id, db)
    
    past_visits = db.query(VisitHistory).filter(VisitHistory.abha_id == patient.abha_id).all()
    if not past_visits:
        return {"relevant_history": [], "other_history": [], "filter_status": "no_history", "abha_id": patient.abha_id}
    
    # Load this patient's isolated relevance mapping
    relevance_map = {}
    if patient.abha_relevance_json:
        try:
            relevance_map = json.loads(patient.abha_relevance_json)
        except Exception:
            relevance_map = {}
    
    filter_complete = bool(patient.is_synthesized) or bool(relevance_map) or (patient.abha_relevance_json is not None and patient.abha_relevance_json != "")
    
    relevant = []
    other = []
    for v in past_visits:
        rel_info = relevance_map.get(str(v.id), {})
        is_rel = rel_info.get("is_relevant", False)
        reason = rel_info.get("reason", "")
        
        visit_data = {
            "id": v.id,
            "visit_date": v.visit_date,
            "chief_complaint": v.chief_complaint,
            "diagnoses": json.loads(v.diagnoses) if v.diagnoses else [],
            "medications": json.loads(v.medications) if v.medications else [],
            "flagged_values": json.loads(v.flagged_values) if v.flagged_values else [],
            "summary": v.summary,
            "specialty": v.specialty,
            "is_relevant": is_rel,
            "relevance_reason": reason
        }
        if is_rel:
            relevant.append(visit_data)
        else:
            other.append(visit_data)
    
    # Sort by date descending
    relevant.sort(key=lambda x: x["visit_date"], reverse=True)
    other.sort(key=lambda x: x["visit_date"], reverse=True)
    
    return {
        "relevant_history": relevant,
        "other_history": other,
        "filter_status": "complete" if filter_complete else "processing",
        "abha_id": patient.abha_id,
        "patient_name": patient.patient_name
    }


@app.post("/api/demo-data")
async def demo_data(background_tasks: BackgroundTasks, abha_id: Optional[str] = "12-3456-7890-1234", db: Session = Depends(get_db)):
    profile = ABHA_PROFILES.get(abha_id, ABHA_PROFILES["12-3456-7890-1234"])
    pt_id = f"PT-DEMO-{str(uuid.uuid4())[:4].upper()}"
    
    demo_doc = {
        "document_type": "Lipid Profile & ECG Report",
        "modality": "ecg",
        "diagnoses": ["Hyperlipidemia", "CAD Status Post-PCI"],
        "medications": ["Atorvastatin 40mg OD", "Aspirin 75mg OD"],
        "flagged_values": ["LDL Cholesterol: 145 mg/dL (High)", "Total Cholesterol: 230 mg/dL (High)"],
        "document_date": datetime.now().strftime("%Y-%m-%d"),
        "summary": "Follow-up lab report showing elevated LDL cholesterol and stable cardiac rhythm.",
        "file_url": "",
        "raw_text": "Demo cardiology follow-up document"
    }

    demo_impression = {
        "clinical_synthesis": [
            "58-year-old male presenting with acute crushing retrosternal chest pain radiating to left arm with diaphoresis of 2 hours duration.",
            "Historical ABHA records confirm prior STEMI (2024 PCI LAD stenting) and uncontrolled T2DM (HbA1c 8.2%).",
            "Elevated lipid profile (LDL 145 mg/dL) and active presentation strongly indicate recurrent acute coronary syndrome / stent thrombosis."
        ],
        "probable_diagnoses": [
            {
                "condition": "Acute Coronary Syndrome / Recurrent NSTEMI vs Stent Thrombosis",
                "likelihood": "High",
                "supporting_evidence": "Crushing substernal pain radiating to left arm with diaphoresis, past STEMI with LAD stent in 2024, uncontrolled diabetes (HbA1c 8.2%)."
            },
            {
                "condition": "Unstable Angina Pectoris",
                "likelihood": "Medium",
                "supporting_evidence": "Exertional chest discomfort with high cardiovascular risk profile and uncontrolled hyperlipidemia."
            },
            {
                "condition": "Acute Gastroesophageal Reflux Disease (GERD) with Esophageal Spasm",
                "likelihood": "Low",
                "supporting_evidence": "Can mimic substernal chest pressure, but severe cardiac risk factors mandate treating as ACS until ruled out."
            }
        ],
        "suggested_investigations": [
            "Stat 12-lead ECG",
            "Serial Cardiac Biomarkers (Troponin I & CK-MB at 0h, 3h)",
            "Bedside 2D Echocardiography (Wall Motion Assessment)",
            "Coronary Angiography consideration"
        ],
        "critical_rule_outs": [
            "Acute Aortic Dissection",
            "Pulmonary Embolism",
            "Tension Pneumothorax"
        ]
    }

    patient = PatientRecord(
        patient_id=pt_id,
        abha_id=profile["abha_id"],
        patient_name=profile["name"],
        age=str(profile["age"]),
        gender=profile["gender"],
        phone=profile["phone"],
        chief_complaint="Severe retrosternal chest pain with left arm radiation and sweating",
        hpi="• Started 2 hours ago while walking\n• Crushing substernal pressure, severity 8/10\n• Accompanied by diaphoresis and mild nausea",
        is_emergency=True,
        severity="High",
        duration="2 hours",
        past_medical_history="• Myocardial Infarction in 2024 (Stented LAD)\n• Type 2 Diabetes Mellitus",
        family_history="• Father had premature CAD at age 52",
        personal_history="• Non-smoker, vegetarian diet",
        allergies="• None reported",
        review_of_systems="• No fever\n• Shortness of breath on exertion",
        clinical_impression_json=json.dumps(demo_impression),
        prakriti="Not assessed",
        vikriti="Not assessed",
        agni="Not assessed",
        flagged_lab_values=json.dumps([demo_doc]),
        is_synthesized=True,
        created_at=datetime.now().strftime("%I:%M %p")
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # Seed ABHA history and trigger AI filter in background
    seed_abha_history(profile["abha_id"], db)
    background_tasks.add_task(filter_history_background, patient.id, profile["abha_id"], patient.chief_complaint)

    return {"status": "success", "patient_id": pt_id, "patient_name": profile["name"], "abha_id": profile["abha_id"]}


if __name__ == "__main__":
    print("🏥 Starting MediKiosk v2 Backend (Offline Mode)...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["uploads/*", "*.db", "*.db*", "*.log", "data/*"]
    )
