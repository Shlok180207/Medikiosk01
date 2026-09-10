"""
MediKiosk Synthesis - Clinical Case Summarizer
GPU Inference: Qwen2.5-7B-Instruct (Q4_K_M GGUF via llama.cpp or local Ollama).
Connects to local llama-server on http://127.0.0.1:8080/v1 (or Ollama on 127.0.0.1:11434).

Accepts compiled JSON schema:
{
  "audio_transcript": "...",
  "radiology_findings": {...},
  "ecg_analysis": {...},
  "printed_report_impressions": "...",
  "prescription_drugs": [...]
}
Returns structured SOAP clinical impression and bilingual Doctor Note.
"""

import os
import json
from typing import Dict, Any, Optional
from synthesis.llm import synthesize_clinical_case, DEFAULT_LLAMA_CPP_URL, DEFAULT_OLLAMA_URL

def summarize_clinical_case(
    case_payload: Dict[str, Any],
    custom_llama_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes the multi-modal clinical case payload into a structured clinical report.
    Args:
      case_payload: Dict conforming to:
        - audio_transcript (str)
        - radiology_findings (dict)
        - ecg_analysis (dict)
        - printed_report_impressions (str or list)
        - prescription_drugs (list)
      custom_llama_url: Optional override URL for llama.cpp server (defaults to 127.0.0.1:8080/v1)
    Returns:
      Structured clinical synthesis containing:
        - bilingual_doctor_note (chief_complaints, diagnostic_findings, prescriptions, warnings)
        - clinical_impression (probable_diagnoses, rule_outs, suggested_investigations, clinical_synthesis)
        - status ("success" or "fallback_offline")
    """
    # Normalize keys for the synthesis engine
    audio_transcript = case_payload.get("audio_transcript", "")
    radiology_findings = case_payload.get("radiology_findings", {})
    ecg_analysis = case_payload.get("ecg_analysis", {})
    printed_report = case_payload.get("printed_report_impressions", "")
    prescription_drugs = case_payload.get("prescription_drugs", [])

    # Format prescription drugs if strings are provided
    formatted_drugs = []
    for d in prescription_drugs:
        if isinstance(d, str):
            formatted_drugs.append({"drug": d, "strength": "", "status": "VERIFIED"})
        elif isinstance(d, dict):
            formatted_drugs.append(d)

    normalized_payload = {
        "audio_transcript": audio_transcript,
        "prescription_drugs": formatted_drugs,
        "xray_findings": radiology_findings.get("pathologies", radiology_findings) if isinstance(radiology_findings, dict) else {},
        "ecg_metrics": ecg_analysis.get("metrics", ecg_analysis) if isinstance(ecg_analysis, dict) else {},
        "printed_report_impressions": printed_report
    }

    result = synthesize_clinical_case(normalized_payload, custom_llama_url=custom_llama_url)
    return result


def generate_clinical_summary(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for summarize_clinical_case."""
    return summarize_clinical_case(case_payload)
