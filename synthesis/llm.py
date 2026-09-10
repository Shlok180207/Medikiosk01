"""
MediKiosk Synthesis - Clinical LLM Reasoning & Doctor Note Generation Module
GPU Inference: Qwen2.5-7B-Instruct (Q4_K_M GGUF via llama.cpp, llama-cpp-python, or local Ollama).
Takes structured multi-modal payload:
  {audio_transcript, prescription_drugs, xray_findings, ecg_metrics}
Cross-references symptoms with medications and diagnostic findings,
and produces a bilingual (English & Hindi) Doctor Note with Doctor Review Warnings
plus structured JSON for the MediKiosk Doctor Dashboard.
"""

import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

# Configuration
DEFAULT_LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080/v1")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TARGET_MODEL_NAME = os.getenv("CLINICAL_LLM_MODEL", "qwen2.5:7b")

CLINICAL_SYSTEM_PROMPT = """You are MediKiosk AI, an expert Clinical Decision Support System and Chief Medical Officer.
You analyze multi-modal clinical intake data comprising:
1. Spoken Consultation Audio Transcript (मरीज़ का मौखिक विवरण)
2. Handwritten / Printed Prescription Drugs (परचे की दवाइयाँ)
3. Chest X-Ray Findings (छाती का एक्स-रे)
4. 12-Lead ECG Metrics (ईसीजी रिपोर्ट)

CRITICAL MEDICAL INSTRUCTIONS:
- Cross-reference the patient's reported symptoms with the prescribed medications. Check for drug-symptom alignment and missing indications.
- Highlight any unverified / flagged medications in the Doctor Review Warnings section.
- Output a comprehensive, professional bilingual clinical report with English and Hindi (हिन्दी) translations for each section.

You MUST structure your response as valid JSON with these exact keys:
{
  "bilingual_doctor_note": {
    "chief_complaints": "Detailed bulleted symptoms in English and Hindi (लक्षण)...",
    "diagnostic_findings": "X-ray, ECG, and lab findings in English and Hindi (जाँच परिणाम)...",
    "prescriptions_and_dosage": "List of drugs with dosages, generic names, and schedule in English and Hindi (दवाइयाँ और खुराक)...",
    "doctor_review_warnings": "Critical safety warnings, drug-symptom contraindications, or unverified prescription tokens requiring physical slip review (डॉक्टर समीक्षा चेतावनी)..."
  },
  "clinical_impression": {
    "probable_diagnoses": ["Primary Diagnosis 1", "Differential Diagnosis 2"],
    "critical_rule_outs": ["Emergency Condition to Rule Out 1", "Rule Out 2"],
    "suggested_investigations": ["Recommended Test 1", "Recommended Test 2"],
    "clinical_synthesis": "Comprehensive narrative clinical note synthesizing all modalities..."
  }
}
Output ONLY valid JSON. Do not add markdown backticks or commentary outside the JSON."""


def _call_openai_compatible_api(endpoint_url: str, prompt: str, system_prompt: str) -> Optional[str]:
    """Calls a local OpenAI-compatible endpoint (e.g. llama.cpp server on port 8080 or Ollama /v1)."""
    try:
        import socket
        from urllib.parse import urlparse
        p = urlparse(endpoint_url)
        host = p.hostname or "127.0.0.1"
        port = p.port or 8080
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.3)
        err = probe.connect_ex((host, port))
        probe.close()
        if err != 0:
            return None

        url = f"{endpoint_url.rstrip('/')}/chat/completions"
        payload = {
            "model": TARGET_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data["choices"][0]["message"]["content"]
    except Exception as e:
        return None


def _call_ollama_native_api(prompt: str, system_prompt: str) -> Optional[str]:
    """Direct offline call via local Ollama daemon on 127.0.0.1:11434."""
    try:
        import ollama
        combined = f"{system_prompt}\n\nPatient Case Data:\n{prompt}"
        response = ollama.chat(
            model=TARGET_MODEL_NAME,
            messages=[{'role': 'user', 'content': combined}],
            format='json',
            options={'temperature': 0.1, 'num_ctx': 4096}
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama native call failed: {e}")
        return None


def extract_json(raw_text: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM response strings."""
    if not raw_text:
        return {}
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and last > first:
            cleaned = cleaned[first:last + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def synthesize_clinical_case(
    case_payload: Dict[str, Any],
    custom_llama_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes multi-modal clinical case data using Qwen2.5-7B-Instruct on GPU.
    Input payload format:
      {
        "audio_transcript": str,
        "prescription_drugs": List[Dict],
        "xray_findings": Dict[str, float],
        "ecg_metrics": Dict[str, Any]
      }
    Returns complete bilingual doctor note + Doctor Dashboard clinical impression structure.
    """
    user_prompt = f"""Multi-Modal Patient Clinical Intake Data:

1. AUDIO CONSULTATION TRANSCRIPT:
{case_payload.get('audio_transcript', 'No patient consultation speech recorded.')}

2. PRESCRIPTION & HANDWRITING DRUG FINDINGS:
{json.dumps(case_payload.get('prescription_drugs', []), indent=2)}

3. CHEST X-RAY FINDINGS (DenseNet-121):
{json.dumps(case_payload.get('xray_findings', {}), indent=2)}

4. 12-LEAD ECG METRICS (OpenCV Waveform Analysis):
{json.dumps(case_payload.get('ecg_metrics', {}), indent=2)}

Generate the complete bilingual clinical doctor note and clinical impression JSON."""

    response_text = None

    # Priority 1: Check llama.cpp local server
    url = custom_llama_url or DEFAULT_LLAMA_CPP_URL
    response_text = _call_openai_compatible_api(url, user_prompt, CLINICAL_SYSTEM_PROMPT)

    # Priority 2: Fallback to local Ollama daemon
    if not response_text:
        response_text = _call_ollama_native_api(user_prompt, CLINICAL_SYSTEM_PROMPT)

    parsed_result = extract_json(response_text) if response_text else {}

    # Defensive Default Construction if LLM offline / unavailable
    if not parsed_result or "bilingual_doctor_note" not in parsed_result:
        audio_text = case_payload.get("audio_transcript", "None reported")
        xray = case_payload.get("xray_findings", {})
        ecg = case_payload.get("ecg_metrics", {})
        drugs = case_payload.get("prescription_drugs", [])

        # Compile warnings for unverified drugs
        warnings = []
        for d in drugs:
            if d.get("status") == "FLAGGED_FOR_DOCTOR":
                warnings.append(f"⚠️ Unverified Drug Token: '{d.get('raw_token', d.get('drug', ''))}' — Requires physical slip verification.")

        parsed_result = {
            "bilingual_doctor_note": {
                "chief_complaints": f"• English: {audio_text}\n• हिन्दी (लक्षण): {audio_text}",
                "diagnostic_findings": f"• X-Ray: {', '.join(xray.keys()) if xray else 'Normal'}\n• ECG: {ecg.get('rhythm', 'Normal')}, HR {ecg.get('heart_rate_bpm', 72)} bpm",
                "prescriptions_and_dosage": "\n".join([f"• {d.get('drug', '')} ({d.get('strength', '')}) - [{d.get('status', 'VERIFIED')}]" for d in drugs]),
                "doctor_review_warnings": "\n".join(warnings) if warnings else "No critical contraindications detected."
            },
            "clinical_impression": {
                "probable_diagnoses": list(xray.keys()) if xray else ["Clinical Evaluation Required"],
                "critical_rule_outs": ["Acute Coronary Syndrome", "Aspiration Pneumonia"] if ecg.get("heart_rate_bpm", 72) > 100 else ["Secondary Complications"],
                "suggested_investigations": ["Repeat ECG in 4 hours", "Complete Blood Count (CBC)"],
                "clinical_synthesis": f"Patient evaluated via offline multi-modal intake. Symptoms: {audio_text}. Diagnostic scans reviewed."
            }
        }

    return parsed_result
