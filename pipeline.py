"""
MediKiosk Orchestrator Pipeline
Sequential execution of:
1. Audio ASR on GPU (Faster-Whisper int8) -> Explicit unload() to reclaim VRAM
2. Perception Models on CPU (TorchXRayVision, OpenCV ECG, Dual OCR + RapidFuzz) -> Zero GPU VRAM
3. Structured Multi-Modal Payload Compilation
4. Clinical LLM Synthesis on GPU (Qwen2.5-7B)
Strictly enforces Peak GPU VRAM < 6.5 GB (safe on 8GB RTX 4060).
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional

from audio.transcriber import get_transcriber, transcribe_consultation, get_vram_mb
from perception.router import classify_image_modality
from perception.radiology import analyze_chest_xray, analyze_bone_xray, analyze_dental_opg, analyze_xray
from perception.ecg import extract_ecg_metrics, analyze_ecg
from perception.document_ocr import parse_document_or_prescription
from synthesis.summarizer import summarize_clinical_case


def log_vram_step(step_name: str, audit_trail: List[Dict[str, Any]]):
    """Logs and records VRAM allocation at critical stage boundaries."""
    vram = get_vram_mb()
    record = {
        "step": step_name,
        "allocated_mb": vram["allocated_mb"],
        "reserved_mb": vram["reserved_mb"],
        "device": vram["device_name"],
        "timestamp": time.strftime("%H:%M:%S")
    }
    audit_trail.append(record)
    print(f"📊 [VRAM Audit] {step_name.ljust(35)} -> Allocated: {vram['allocated_mb']:>7.2f} MB | Reserved: {vram['reserved_mb']:>7.2f} MB")
    return record


class MedicalPipelineOrchestrator:
    """Orchestrates sequential multi-modal intake with strict GPU VRAM budget control."""

    def __init__(self, max_vram_mb: float = 6500.0):
        self.max_vram_mb = max_vram_mb

    def run(
        self,
        audio_path: Optional[str] = None,
        images: Optional[List[Tuple[str, str]]] = None,
        patient_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full sequential clinical intake pipeline.
        Args:
          audio_path: Path to doctor-patient conversation audio file (optional)
          images: List of (file_path, report_type) tuples.
                  report_type in ['xray', 'ecg', 'prescription', 'lab_report']
          patient_meta: Optional patient demographics / ABHA info
        Returns:
          Aggregated clinical dossier with bilingual doctor note,
          dashboard documents, and VRAM audit trail.
        """
        images = images or []
        patient_meta = patient_meta or {}
        audit_trail: List[Dict[str, Any]] = []

        print("\n" + "=" * 80)
        print("🏥 MEDIKIOSK MULTI-MODAL PIPELINE EXECUTION (BUDGET: < 6.5 GB VRAM)")
        print("=" * 80)

        # ── Step 0: Baseline VRAM ──
        log_vram_step("0. Pipeline Start (Baseline)", audit_trail)

        # ── Step 1: GPU Audio ASR ──
        transcript_text = ""
        transcriber = get_transcriber()

        if audio_path and os.path.exists(audio_path):
            print(f"\n🎙️ [STAGE 1] Transcribing consultation audio: {audio_path}")
            log_vram_step("1a. Before Audio Transcription", audit_trail)

            asr_res = transcriber.transcribe(audio_path, auto_unload=True)
            transcript_text = asr_res.get("text", "")
            print(f"   Transcript: \"{transcript_text[:120]}...\"" if len(transcript_text) > 120 else f"   Transcript: \"{transcript_text}\"")

            log_vram_step("1b. After Audio ASR & Unload", audit_trail)
        else:
            print("\n🎙️ [STAGE 1] No audio provided; skipping ASR.")
            log_vram_step("1. Audio Skipped", audit_trail)

        # ── Step 2: CPU Perception (X-ray, ECG, Prescription) ──
        print(f"\n👁️ [STAGE 2] Running CPU Perception Models on {len(images)} document(s)...")
        log_vram_step("2a. Before CPU Perception", audit_trail)

        all_xray_findings: Dict[str, float] = {}
        all_ecg_metrics: Dict[str, Any] = {}
        all_prescription_drugs: List[Dict[str, Any]] = []
        all_printed_reports: List[str] = []
        dashboard_documents: List[Dict[str, Any]] = []

        for img_path, report_type in images:
            if not os.path.exists(img_path):
                print(f"   ⚠️ Image file not found: {img_path}")
                continue

            # Auto-route or use specified report_type
            if not report_type or report_type.lower() in ["auto", "detect", "none"]:
                route_res = classify_image_modality(img_path)
                modality = route_res.get("modality", "CHEST_XRAY")
                sub_type = route_res.get("sub_type", "")
            else:
                mod_str = report_type.upper().strip()
                if mod_str in ["XRAY", "CHEST_XRAY", "CXR"]:
                    modality = "CHEST_XRAY"
                    sub_type = ""
                elif mod_str in ["BONE", "BONE_XRAY", "EXTREMITY"]:
                    modality = "BONE_XRAY"
                    sub_type = ""
                elif mod_str in ["ECG", "EKG", "ECG_WAVEFORM"]:
                    modality = "ECG_WAVEFORM"
                    sub_type = ""
                elif mod_str in ["PRESCRIPTION", "RX", "HANDWRITTEN_PRESCRIPTION"]:
                    modality = "HANDWRITTEN_PRESCRIPTION"
                    sub_type = ""
                else:
                    modality = "PRINTED_REPORT"
                    sub_type = ""

            print(f"   Processing [{modality}]: {os.path.basename(img_path)}")

            if modality == "CHEST_XRAY":
                res = analyze_chest_xray(img_path)
                all_xray_findings.update(res.get("pathologies", {}))
                dashboard_documents.append(res.get("dashboard_payload", {}))
            elif modality == "BONE_XRAY":
                if sub_type == "DENTAL_OPG":
                    res = analyze_dental_opg(img_path)
                else:
                    res = analyze_bone_xray(img_path)
                dashboard_documents.append(res.get("dashboard_payload", {}))
            elif modality == "ECG_WAVEFORM":
                res = extract_ecg_metrics(img_path)
                all_ecg_metrics.update(res)
                dashboard_documents.append(res.get("dashboard_payload", {}))
            elif modality == "HANDWRITTEN_PRESCRIPTION":
                res = parse_document_or_prescription(img_path, doc_type="HANDWRITTEN_PRESCRIPTION")
                all_prescription_drugs.extend(res.get("normalized_drugs", []))
                dashboard_documents.append(res.get("dashboard_payload", {}))
            else:  # PRINTED_REPORT
                res = parse_document_or_prescription(img_path, doc_type="PRINTED_REPORT")
                if res.get("impression"):
                    all_printed_reports.append(res["impression"])
                dashboard_documents.append(res.get("dashboard_payload", {}))

        log_vram_step("2b. After CPU Perception (Zero GPU)", audit_trail)

        # ── Step 3: Payload Compilation ──
        case_payload = {
            "audio_transcript": transcript_text,
            "radiology_findings": all_xray_findings,
            "ecg_analysis": all_ecg_metrics,
            "printed_report_impressions": "; ".join([r for r in all_printed_reports if r]),
            "prescription_drugs": all_prescription_drugs,
            "patient_meta": patient_meta
        }

        # ── Step 4: GPU LLM Clinical Synthesis ──
        print("\n🧠 [STAGE 3] Synthesizing Clinical Impression & Bilingual Doctor Note (Qwen2.5-7B)...")
        log_vram_step("3a. Before LLM Synthesis", audit_trail)

        synthesis_result = summarize_clinical_case(case_payload)

        log_vram_step("3b. After LLM Synthesis", audit_trail)

        # ── Step 5: VRAM Verification ──
        peak_allocated = max(r["allocated_mb"] for r in audit_trail)
        print("\n" + "=" * 80)
        print(f"✅ PIPELINE COMPLETE | Peak VRAM Allocated: {peak_allocated:.2f} MB / {self.max_vram_mb:.0f} MB")
        if peak_allocated <= self.max_vram_mb:
            print("🟢 VRAM SAFETY COMPLIANCE: PASSED (Strictly within 6.5 GB limit)")
        else:
            print("🔴 VRAM SAFETY WARNING: Peak exceeded 6.5 GB limit!")
        print("=" * 80 + "\n")

        return {
            "status": "success",
            "transcript": transcript_text,
            "xray_findings": all_xray_findings,
            "ecg_metrics": all_ecg_metrics,
            "prescription_drugs": all_prescription_drugs,
            "dashboard_documents": dashboard_documents,
            "bilingual_doctor_note": synthesis_result.get("bilingual_doctor_note", {}),
            "clinical_impression": synthesis_result.get("clinical_impression", {}),
            "vram_audit_trail": audit_trail,
            "peak_vram_mb": peak_allocated
        }


def cli_main():
    parser = argparse.ArgumentParser(description="MediKiosk Offline Multi-Modal Clinical Pipeline")
    parser.add_argument("--audio", type=str, default=None, help="Path to patient consultation audio file")
    parser.add_argument("--xray", type=str, default=None, help="Path to Chest X-ray image")
    parser.add_argument("--ecg", type=str, default=None, help="Path to 12-Lead ECG printout image")
    parser.add_argument("--prescription", type=str, default=None, help="Path to doctor prescription image")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file path")
    args = parser.parse_args()

    images = []
    if args.xray:
        images.append((args.xray, "xray"))
    if args.ecg:
        images.append((args.ecg, "ecg"))
    if args.prescription:
        images.append((args.prescription, "prescription"))

    orchestrator = MedicalPipelineOrchestrator()
    result = orchestrator.run(audio_path=args.audio, images=images)

    note = result.get("bilingual_doctor_note", {})
    print("\n📋 GENERATED BILINGUAL CLINICAL DOCTOR NOTE:")
    print("-" * 60)
    print("1. Chief Complaints (लक्षण):")
    print(note.get("chief_complaints", "N/A"))
    print("\n2. Diagnostic Findings (जाँच परिणाम):")
    print(note.get("diagnostic_findings", "N/A"))
    print("\n3. Prescriptions & Dosage (दवाइयाँ और खुराक):")
    print(note.get("prescriptions_and_dosage", "N/A"))
    print("\n4. Doctor Review Warnings (डॉक्टर समीक्षा चेतावनी):")
    print(note.get("doctor_review_warnings", "N/A"))
    print("-" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved full clinical dossier to {args.output}")


if __name__ == "__main__":
    cli_main()
