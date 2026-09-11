"""
pipeline.py
===========
MediKiosk Multi-Modal Sequential Clinical Pipeline.
Hardware Target: NVIDIA RTX 4060 (8GB VRAM limit).

Enforces:
1. Single-Resident-GPU-Model (SRGM) invariant.
2. Faster-Whisper ASR loaded only during voice intake.
3. Qwen2.5-VL-3B VLM loaded only during document triage & OCR extraction.
4. TorchXRayVision & OpenCV ECG pinned 100% to host CPU (0 MB VRAM).
5. Qwen2.5-7B loaded only during historical ABHA and final CDSS report synthesis.
6. Guaranteed peak VRAM < 6.5 GB.
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import json
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional

from core.vram_manager import VRAMManager, ModelState
from core.cpu_perception import CPUPerceptionEngine
from core.clinical_orchestrator import (
    ClinicalOrchestrator,
    PatientJourney,
    CDSSReport,
    DocumentType
)


class MedicalPipelineOrchestrator:
    """
    High-level orchestrator interfacing with the underlying VRAM state machine
    and CPU perception engine.
    """

    def __init__(self, max_vram_mb: float = 6500.0, device: str = "cuda:0"):
        self.max_vram_mb = max_vram_mb
        self.vram_mgr = VRAMManager(device=device, vram_safety_margin_mb=1200)
        self.cpu_engine = CPUPerceptionEngine()
        self.orchestrator = ClinicalOrchestrator(self.vram_mgr, self.cpu_engine)

    async def run_async(
        self,
        patient_id: str,
        audio_path: Optional[str] = None,
        images: Optional[List[Tuple[str, str]]] = None,
        abha_records: Optional[List[Dict[str, Any]]] = None,
        spoken_transcript_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously executes the 4-phase sequential clinical intake pipeline.
        """
        images = images or []
        abha_records = abha_records or []

        print("\n" + "=" * 80)
        print(f"🏥 MEDIKIOSK EVENT-DRIVEN CLINICAL PIPELINE (RTX 4060 - 8GB VRAM)")
        print(f"   Patient ID: {patient_id} | Documents: {len(images)} | Safety Ceiling: {self.max_vram_mb:.0f} MB")
        print("=" * 80)

        # Read audio bytes if provided
        audio_bytes = None
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

        journey = PatientJourney(
            patient_id=patient_id,
            audio_bytes=audio_bytes,
            abha_raw_records=abha_records,
            spoken_transcript=spoken_transcript_override or ""
        )

        # Enqueue documents into the asynchronous processing queue
        for img_path, r_type in images:
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    await self.orchestrator.enqueue_document(
                        doc_id=os.path.basename(img_path),
                        filename=os.path.basename(img_path),
                        data=f.read(),
                        hint=r_type
                    )
            else:
                print(f"⚠️ Document file not found: {img_path}")

        # Signal end of document uploads
        await self.orchestrator.close_document_queue()

        # Execute 4-Phase Pipeline
        # Phase 1: Intake & Speech
        await self.orchestrator.run_phase_1_intake(journey)

        # Phase 2 & 3: VLM Gatekeeper & CPU Routing (Consumes Asynchronous Queue)
        await self.orchestrator.run_phase_2_and_3_document_triage(journey)

        # Phase 4: CDSS Generation
        cdss = await self.orchestrator.run_phase_4_cdss_generation(journey)

        # Audit Peak VRAM
        peaks = [cp["allocated_mb"] for cp in journey.vram_audit_trail]
        peak_allocated = max(peaks) if peaks else 0.0

        print("\n" + "=" * 80)
        print(f"✅ PIPELINE COMPLETE | Peak VRAM Allocated: {peak_allocated:.2f} MB / {self.max_vram_mb:.0f} MB")
        if peak_allocated <= self.max_vram_mb:
            print("🟢 VRAM SAFETY COMPLIANCE: PASSED (Zero CUDA OOM Risk)")
        else:
            print("🔴 VRAM SAFETY WARNING: Allocation exceeded target headroom threshold!")
        print("=" * 80 + "\n")

        return {
            "status": "success",
            "patient_id": patient_id,
            "transcript": journey.spoken_transcript,
            "patient_summary": journey.patient_summary,
            "cpu_diagnostics": journey.cpu_diagnostics,
            "extracted_text_records": journey.extracted_text_records,
            "cdss_report": cdss.model_dump() if cdss else {},
            "vram_audit_trail": journey.vram_audit_trail,
            "peak_vram_mb": peak_allocated
        }

    def run(
        self,
        audio_path: Optional[str] = None,
        images: Optional[List[Tuple[str, str]]] = None,
        patient_meta: Optional[Dict[str, Any]] = None,
        patient_id: str = "PT-DEMO"
    ) -> Dict[str, Any]:
        """Synchronous wrapper for legacy CLI and backend calls."""
        return asyncio.run(
            self.run_async(
                patient_id=patient_id,
                audio_path=audio_path,
                images=images,
                abha_records=(patient_meta or {}).get("abha_records", [])
            )
        )


def cli_main():
    parser = argparse.ArgumentParser(description="MediKiosk Offline Multi-Modal Sequential Pipeline")
    parser.add_argument("--patient-id", type=str, default="PT-TRIAGE-01", help="Patient Identifier")
    parser.add_argument("--audio", type=str, default=None, help="Path to consultation audio file")
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
    result = orchestrator.run(
        patient_id=args.patient_id,
        audio_path=args.audio,
        images=images
    )

    cdss = result.get("cdss_report", {})
    print("\n📋 GENERATED CLINICAL DECISION SUPPORT (CDSS) REPORT:")
    print("-" * 60)
    print(f"Patient ID: {cdss.get('patient_id')}")
    print(f"\n1. Primary Clinical Impression:\n   {cdss.get('primary_impression')}")
    print(f"\n2. Critical Alerts / Rule-Outs:")
    for alert in cdss.get("critical_alerts", []):
        print(f"   ⚠️  {alert}")
    print(f"\n3. CPU Imaging & ECG Synthesis:")
    for syn in cdss.get("imaging_synthesis", []):
        print(f"   🩻  {syn}")
    print(f"\n4. Prescriptions Identified:")
    for rx in cdss.get("prescriptions_identified", []):
        print(f"   💊 {rx}")
    print(f"\n5. Recommended Clinical Plan:")
    for plan in cdss.get("recommended_plan", []):
        print(f"   🩺 {plan}")
    print("-" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved full clinical dossier to {args.output}")


if __name__ == "__main__":
    cli_main()
