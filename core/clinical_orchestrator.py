"""
core/clinical_orchestrator.py
=============================
Asynchronous, Event-Driven Clinical Triage Pipeline.
Orchestrates Phase 1 (Intake), Phase 2 (Gatekeeping), Phase 3 (Routing), and Phase 4 (CDSS).
Enforces the Single-Resident-GPU-Model (SRGM) invariant with real-time VRAM telemetry auditing.

Hardware Target: NVIDIA RTX 4060 (8GB VRAM limit).
Zero CUDA OOM Guarantee via strict sequential model lifecycle management.
"""

import asyncio
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field
import torch

from core.vram_manager import ModelState, VRAMManager
from core.cpu_perception import CPUPerceptionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ClinicalOrchestrator")


class DocumentType(str, Enum):
    """Document modalities classified by the visual gatekeeper."""
    CHEST_XRAY = "CHEST_XRAY"
    ECG_STRIP = "ECG_STRIP"
    HANDWRITTEN_RX = "HANDWRITTEN_RX"
    LAB_REPORT = "LAB_REPORT"
    UNKNOWN = "UNKNOWN"


class CDSSReport(BaseModel):
    """Structured Clinical Decision Support System Report."""
    patient_id: str
    primary_impression: str = Field(description="Primary clinical differential and etiology")
    differential_diagnoses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top 2-3 differentials with likelihood"
    )
    critical_alerts: List[str] = Field(
        default_factory=list,
        description="Urgent red flags requiring emergency stabilization"
    )
    imaging_synthesis: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Corroboration from CPU imaging/ECG"
    )
    prescriptions_identified: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Extracted active medicines"
    )
    recommended_plan: List[str] = Field(
        default_factory=list,
        description="Diagnostics, referrals, or precautions"
    )


@dataclass
class DocumentQueueItem:
    """Asynchronous queue payload for uploaded clinical media."""
    doc_id: str
    filename: str
    data: bytes
    hint: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class PatientJourney:
    """End-to-end clinical intake state for a patient session."""
    patient_id: str
    audio_bytes: Optional[bytes] = None
    abha_raw_records: List[Dict[str, Any]] = field(default_factory=list)
    extracted_text_records: List[Dict[str, Any]] = field(default_factory=list)
    cpu_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    spoken_transcript: str = ""
    patient_summary: str = ""
    final_cdss: Optional[CDSSReport] = None
    vram_audit_trail: List[Dict[str, Any]] = field(default_factory=list)


class ClinicalOrchestrator:
    """
    Principal Clinical Orchestrator.
    Manages the asynchronous queue and sequential state transitions.
    Guarantees that at any single point in time, only ONE model occupies GPU memory.
    """

    def __init__(self, vram_mgr: VRAMManager, cpu_engine: CPUPerceptionEngine):
        self.vram = vram_mgr
        self.cpu = cpu_engine
        self.doc_queue: asyncio.Queue[Optional[DocumentQueueItem]] = asyncio.Queue()

    def _record_vram_checkpoint(self, stage_name: str, journey: PatientJourney):
        """Records a timestamped VRAM audit record for hardware safety compliance."""
        telemetry = self.vram.get_vram_telemetry()
        checkpoint = {
            "stage": stage_name,
            "allocated_mb": telemetry["allocated_mb"],
            "reserved_mb": telemetry["reserved_mb"],
            "free_mb": telemetry["free_mb"],
            "total_mb": telemetry["total_mb"],
            "timestamp": time.strftime("%H:%M:%S")
        }
        journey.vram_audit_trail.append(checkpoint)
        logger.info(
            f"📊 [VRAM Checkpoint] {stage_name.ljust(35)} -> "
            f"Alloc: {telemetry['allocated_mb']:>7.2f} MB | "
            f"Res: {telemetry['reserved_mb']:>7.2f} MB | "
            f"Free: {telemetry['free_mb']:>7.2f} MB"
        )

    # =========================================================================
    # PHASE 1: Voice Intake & History Synthesis
    # =========================================================================
    async def run_phase_1_intake(self, journey: PatientJourney):
        """
        Phase 1:
        1. Load Whisper -> Transcribe patient audio -> Unload Whisper & sweep VRAM.
        2. Load Qwen2.5-7B -> Synthesize ABHA history -> Unload Qwen2.5-7B & sweep VRAM.
        """
        logger.info("\n" + "=" * 75)
        logger.info(f"🎙️  [PHASE 1: INTAKE] Starting for Patient: {journey.patient_id}")
        logger.info("=" * 75)
        self._record_vram_checkpoint("1.0 Phase 1 Start (IDLE)", journey)

        # ── Step 1A: Patient Audio Processing (Whisper) ──
        if journey.audio_bytes:
            bundle = await self.vram.transition_to(ModelState.WHISPER)
            self._record_vram_checkpoint("1.1 Whisper Loaded (CUDA)", journey)

            logger.info("Transcribing patient speech on CUDA...")
            if bundle.get("backend") == "ctranslate2":
                whisper_model = bundle["model"]
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(journey.audio_bytes)
                    tmp_path = tmp.name

                try:
                    segments, _ = await asyncio.to_thread(
                        whisper_model.transcribe,
                        tmp_path,
                        beam_size=3,
                        language="en"
                    )
                    journey.spoken_transcript = " ".join([seg.text for seg in segments]).strip()
                finally:
                    import os
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
            else:
                # Simulation / Mock fallback
                journey.spoken_transcript = (
                    "Patient complains of severe chest pressure radiating to left arm for 3 hours, "
                    "with shortness of breath and cold diaphoresis."
                )

            logger.info(f"Transcript generated: \"{journey.spoken_transcript}\"")
            self._record_vram_checkpoint("1.2 Speech Transcribed", journey)

            # Unload Whisper immediately
            await self.vram.transition_to(ModelState.IDLE)
            self._record_vram_checkpoint("1.3 Whisper Evicted (IDLE)", journey)

        # ── Step 1B: Historical ABHA Synthesis (Qwen2.5-7B) ──
        bundle = await self.vram.transition_to(ModelState.LLM_REASONING)
        self._record_vram_checkpoint("1.4 Qwen2.5-7B Loaded (CUDA)", journey)

        prompt = f"""You are an expert Clinical Intake AI.
Review the patient's presenting complaints and prior ABHA visit history.
Patient Presenting Complaint: {journey.spoken_transcript or "Acute clinical evaluation"}
Past ABHA Visit Records: {json.dumps(journey.abha_raw_records, ensure_ascii=False)}

TASK:
Provide a concise 2-sentence clinical summary correlating past chronic medical history with today's presenting symptoms.
"""
        logger.info("Generating ABHA baseline summary with Qwen2.5-7B...")
        if bundle.get("backend") == "ollama":
            client = bundle["client"]
            res = await asyncio.to_thread(
                client.generate,
                model=bundle["model_name"],
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 180}
            )
            journey.patient_summary = res["response"].strip()
        elif bundle.get("backend") == "transformers_bnb4":
            model = bundle["model"]
            tokenizer = bundle["tokenizer"]
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                return_tensors="pt"
            ).to(self.vram.device)
            with torch.no_grad():
                out = await asyncio.to_thread(model.generate, **inputs, max_new_tokens=180, temperature=0.1)
            journey.patient_summary = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        else:
            # High-fidelity clinical simulation
            journey.patient_summary = (
                "58-year-old male with 6-year history of Type-2 Diabetes Mellitus and Essential Hypertension "
                "presenting with acute retrosternal chest pain suspicious for acute coronary syndrome."
            )

        logger.info(f"ABHA Baseline Summary: {journey.patient_summary}")

        # Explicit unload back to IDLE to guarantee 100% free VRAM for document phase
        await self.vram.transition_to(ModelState.IDLE)
        self._record_vram_checkpoint("1.5 Phase 1 Complete (Swept & Clean)", journey)

    # =========================================================================
    # PHASE 2 & 3: Asynchronous Document Queue, VLM Gatekeeper & CPU Routing
    # =========================================================================
    async def enqueue_document(
        self,
        doc_id: str,
        filename: str,
        data: bytes,
        hint: Optional[str] = None
    ):
        """Allows event-driven ingestion of documents into the asynchronous queue."""
        item = DocumentQueueItem(doc_id=doc_id, filename=filename, data=data, hint=hint)
        await self.doc_queue.put(item)
        logger.info(f"📥 [Async Queue] Enqueued document: {filename} (ID: {doc_id})")

    async def close_document_queue(self):
        """Sends sentinel to terminate the document processing consumer loop."""
        await self.doc_queue.put(None)

    async def _classify_modality_gatekeeper(
        self,
        bundle: Dict[str, Any],
        doc_bytes: bytes,
        filename: str,
        hint: Optional[str] = None
    ) -> DocumentType:
        """
        Phase 2: Qwen2.5-VL-3B inspects the image as an anatomical/modality gatekeeper.
        Classifies strictly into: CHEST_XRAY, ECG_STRIP, HANDWRITTEN_RX, LAB_REPORT.
        """
        gatekeeper_prompt = (
            "Examine this medical image carefully. Classify it strictly into exactly ONE category: "
            "CHEST_XRAY, ECG_STRIP, HANDWRITTEN_RX, LAB_REPORT. "
            "Output ONLY the category name and nothing else."
        )

        detected_tag = DocumentType.UNKNOWN
        try:
            if bundle.get("backend") == "ollama":
                client = bundle["client"]
                res = await asyncio.to_thread(
                    client.generate,
                    model=bundle["model_name"],
                    prompt=gatekeeper_prompt,
                    images=[doc_bytes],
                    options={"temperature": 0.0, "num_predict": 16}
                )
                raw_out = res["response"].strip().upper()
                for dt in DocumentType:
                    if dt.value in raw_out:
                        return dt

            elif bundle.get("backend") == "transformers_bnb4":
                processor = bundle["processor"]
                model = bundle["model"]
                image = Image.open(io.BytesIO(doc_bytes)).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": gatekeeper_prompt}]
                }]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text_in], images=[image], return_tensors="pt").to(self.vram.device)
                with torch.no_grad():
                    gen_ids = await asyncio.to_thread(model.generate, **inputs, max_new_tokens=16)
                raw_out = processor.batch_decode(
                    gen_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip().upper()
                for dt in DocumentType:
                    if dt.value in raw_out:
                        return dt
            else:
                # Simulation / Heuristic classification
                fn_lower = filename.lower()
                hint_lower = (hint or "").lower()
                if "xray" in fn_lower or "cxr" in fn_lower or "xray" in hint_lower:
                    return DocumentType.CHEST_XRAY
                elif "ecg" in fn_lower or "ekg" in fn_lower or "ecg" in hint_lower:
                    return DocumentType.ECG_STRIP
                elif "rx" in fn_lower or "prescription" in fn_lower or "prescription" in hint_lower:
                    return DocumentType.HANDWRITTEN_RX
                elif "lab" in fn_lower or "blood" in fn_lower or "report" in fn_lower or "lab" in hint_lower:
                    return DocumentType.LAB_REPORT

                # Zero-VRAM CPU perception router fallback
                try:
                    from perception.router import classify_image_modality
                    router_res = classify_image_modality(doc_bytes)
                    mod = router_res.get("modality", "")
                    if mod == "CHEST_XRAY":
                        return DocumentType.CHEST_XRAY
                    elif mod == "ECG_WAVEFORM":
                        return DocumentType.ECG_STRIP
                except Exception:
                    pass

                return DocumentType.LAB_REPORT
        except Exception as e:
            logger.warning(f"VLM Gatekeeper classification error for {filename}: {e}")

        # Fallback via perception router
        try:
            from perception.router import classify_image_modality
            router_res = classify_image_modality(doc_bytes)
            if router_res.get("modality") == "CHEST_XRAY":
                return DocumentType.CHEST_XRAY
            elif router_res.get("modality") == "ECG_WAVEFORM":
                return DocumentType.ECG_STRIP
        except Exception:
            pass

        return DocumentType.HANDWRITTEN_RX

    async def _extract_vlm_text_json(
        self,
        bundle: Dict[str, Any],
        doc_bytes: bytes,
        filename: str,
        modality: DocumentType
    ) -> Dict[str, Any]:
        """
        Uses currently resident Qwen2.5-VL-3B to extract structured JSON from prescriptions or lab reports.
        """
        extract_prompt = (
            f"You are a clinical OCR extraction specialist. The document is a {modality.value}. "
            "Extract all visible medications (with dose, route, frequency) and flagged abnormal laboratory values. "
            "Return valid JSON with keys: 'medications', 'flagged_values', 'summary'."
        )

        # Specialized Tabular Lab Pipeline for dense multi-row reports
        if modality == DocumentType.LAB_REPORT:
            try:
                from perception.lab import LabReportPipeline, ConstrainedVLMExtractor
                backend_type = "ollama" if bundle.get("backend") == "ollama" else ("transformers" if bundle.get("backend") == "transformers_bnb4" else "mock")
                extractor = ConstrainedVLMExtractor(backend=backend_type)
                pipeline = LabReportPipeline(extractor=extractor)
                lab_res = await asyncio.to_thread(pipeline.process, doc_bytes, filename)
                return {
                    "medications": [],
                    "flagged_values": lab_res.flagged_abnormalities,
                    "all_tests": [t.dict() for t in lab_res.tests],
                    "summary": lab_res.summary
                }
            except Exception as le:
                logger.warning(f"LabReportPipeline fallback ({le}). Proceeding with standard VLM extraction.")

        try:
            if bundle.get("backend") == "ollama":
                res = await asyncio.to_thread(
                    bundle["client"].generate,
                    model=bundle["model_name"],
                    prompt=extract_prompt,
                    images=[doc_bytes],
                    options={"temperature": 0.1, "num_predict": 450}
                )
                raw_text = res["response"].strip()
            elif bundle.get("backend") == "transformers_bnb4":
                processor = bundle["processor"]
                image = Image.open(io.BytesIO(doc_bytes)).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": extract_prompt}]
                }]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text_in], images=[image], return_tensors="pt").to(self.vram.device)
                with torch.no_grad():
                    gen_ids = await asyncio.to_thread(bundle["model"].generate, **inputs, max_new_tokens=450)
                raw_text = processor.batch_decode(
                    gen_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            else:
                # Simulation JSON output
                if modality == DocumentType.HANDWRITTEN_RX:
                    raw_text = json.dumps({
                        "medications": [
                            {"name": "Aspirin", "dosage": "75mg", "frequency": "OD", "indication": "Cardioprotection"},
                            {"name": "Atorvastatin", "dosage": "40mg", "frequency": "HS", "indication": "Dyslipidemia"},
                            {"name": "Metformin", "dosage": "500mg", "frequency": "BD", "indication": "Glycemic control"}
                        ],
                        "flagged_values": [],
                        "summary": "Routine maintenance prescription for cardiovascular and metabolic risk factors."
                    })
                else:
                    raw_text = json.dumps({
                        "medications": [],
                        "flagged_values": [
                            {"test": "Troponin-I", "value": "1.42", "unit": "ng/mL", "reference": "< 0.04", "status": "CRITICAL_HIGH"},
                            {"test": "Creatinine", "value": "1.1", "unit": "mg/dL", "reference": "0.7 - 1.2", "status": "NORMAL"}
                        ],
                        "summary": "Elevated Troponin-I indicating myocardial injury."
                    })

            # JSON extraction
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            return json.loads(raw_text)
        except Exception as e:
            logger.warning(f"VLM JSON parsing fallback for {filename}: {e}")
            return {"raw_text": raw_text if 'raw_text' in locals() else str(e)}

    async def run_phase_2_and_3_document_triage(self, journey: PatientJourney):
        """
        Phase 2 & 3:
        1. Loads Qwen2.5-VL-3B (4-bit) ONCE into GPU.
        2. Consumes incoming items from the asynchronous queue.
        3. Visual Gatekeeper classifies document modality.
        4. Triage Routing:
           - CHEST_XRAY / ECG_STRIP -> Dispatched to host CPU (TorchXRayVision / OpenCV) - ZERO VRAM!
           - HANDWRITTEN_RX / LAB_REPORT -> Extracted using resident Qwen2.5-VL-3B.
        5. Drains queue, sweeps and unloads Qwen2.5-VL-3B back to IDLE.
        """
        logger.info("\n" + "=" * 75)
        logger.info(f"👁️  [PHASE 2 & 3: DOCUMENT TRIAGE & ROUTING] Consuming Asynchronous Queue")
        logger.info("=" * 75)

        # Load VLM Gatekeeper into GPU
        bundle = await self.vram.transition_to(ModelState.VLM_GATEKEEPER)
        self._record_vram_checkpoint("2.1 Qwen2.5-VL-3B Resident (CUDA)", journey)

        doc_count = 0
        while True:
            item = await self.doc_queue.get()
            if item is None:
                # Sentinel signaling end of document queue
                self.doc_queue.task_done()
                break

            doc_count += 1
            logger.info(f"⚡ Processing Queue Item #{doc_count}: '{item.filename}'")

            # Step 1: Visual Gatekeeper Classification
            modality = await self._classify_modality_gatekeeper(
                bundle=bundle,
                doc_bytes=item.data,
                filename=item.filename,
                hint=item.hint
            )
            logger.info(f"🔍 [Gatekeeper Decision] Modality Identified -> [{modality.value}]")

            # Step 2: Triage Routing
            if modality == DocumentType.CHEST_XRAY:
                logger.info("🔀 Routing to CPU TorchXRayVision (CheXNet DenseNet-121) | 0 MB VRAM...")
                cxr_res = await self.cpu.analyze_chest_xray(item.data, filename=item.filename)
                journey.cpu_diagnostics.append(cxr_res)

            elif modality == DocumentType.ECG_STRIP:
                logger.info("🔀 Routing to CPU OpenCV Grid Stripper & SciPy Waveform Engine | 0 MB VRAM...")
                ecg_res = await self.cpu.analyze_ecg_strip(item.data)
                journey.cpu_diagnostics.append(ecg_res)

            else:
                # HANDWRITTEN_RX or LAB_REPORT -> Processed via resident Qwen2.5-VL
                logger.info(f"📄 Processing text/table via resident Qwen2.5-VL-3B on GPU [{modality.value}]...")
                extracted = await self._extract_vlm_text_json(
                    bundle=bundle,
                    doc_bytes=item.data,
                    filename=item.filename,
                    modality=modality
                )
                journey.extracted_text_records.append({
                    "filename": item.filename,
                    "modality": modality.value,
                    "data": extracted
                })

            self.doc_queue.task_done()

        # Teardown VLM tenant and sweep VRAM
        await self.vram.transition_to(ModelState.IDLE)
        self._record_vram_checkpoint("2.2 Phase 2/3 Complete (Flushed & IDLE)", journey)

    # =========================================================================
    # PHASE 4: CDSS Generation
    # =========================================================================
    async def run_phase_4_cdss_generation(self, journey: PatientJourney) -> CDSSReport:
        """
        Phase 4:
        1. Reload Qwen2.5-7B into GPU memory.
        2. Feed all multi-modal clinical signals (Audio Transcript, ABHA history,
           CPU radiological/cardiographic findings, and VLM medication/lab extractions).
        3. Synthesize comprehensive Clinical Decision Support (CDSS) Report.
        4. Unload Qwen2.5-7B and leave GPU in clean IDLE state.
        """
        logger.info("\n" + "=" * 75)
        logger.info(f"🧠 [PHASE 4: CDSS GENERATION] Synthesizing Clinical Dossier")
        logger.info("=" * 75)

        bundle = await self.vram.transition_to(ModelState.LLM_REASONING)
        self._record_vram_checkpoint("3.1 Qwen2.5-7B Loaded for CDSS (CUDA)", journey)

        cdss_prompt = f"""You are a Principal Medical Decision Support Specialist.
Synthesize the complete multi-modal clinical dossier for Patient {journey.patient_id}.

PRESENTING CLINICAL SPEECH INTAKE:
{journey.spoken_transcript or "Patient intake completed."}

PATIENT HISTORICAL ABHA CONTEXT:
{journey.patient_summary or "No prior chronic medical conditions recorded."}

CPU-ISOLATED OBJECTIVE IMAGING & ECG ALERTS (HIGH PRECISION):
{json.dumps(journey.cpu_diagnostics, indent=2, ensure_ascii=False)}

PARSED PRESCRIPTION SLIPS & LAB REPORTS:
{json.dumps(journey.extracted_text_records, indent=2, ensure_ascii=False)}

CRITICAL REASONING DIRECTIVES:
1. Prioritize acute symptoms reported today and cross-corroborate with CPU imaging/ECG abnormalities and active uploaded medications.
2. Formulate top 2 differential diagnoses with likelihoods.
3. List urgent red flags or rule-out conditions requiring immediate attention.

OUTPUT REQUIREMENT:
Return ONLY valid JSON with this exact schema:
{{
  "patient_id": "{journey.patient_id}",
  "primary_impression": "Primary diagnosis statement",
  "differential_diagnoses": [
    {{"condition": "Condition 1", "likelihood": "High|Medium|Low"}},
    {{"condition": "Condition 2", "likelihood": "Medium|Low"}}
  ],
  "critical_alerts": ["Rule out condition or acute alert"],
  "imaging_synthesis": ["Corroborating findings from CXR or ECG"],
  "prescriptions_identified": ["Medication name and dose"],
  "recommended_plan": ["Next clinical steps"]
}}"""

        logger.info("Synthesizing final CDSS with Qwen2.5-7B...")
        raw_cdss = ""
        try:
            if bundle.get("backend") == "ollama":
                res = await asyncio.to_thread(
                    bundle["client"].generate,
                    model=bundle["model_name"],
                    prompt=cdss_prompt,
                    options={"temperature": 0.1, "num_predict": 650}
                )
                raw_cdss = res["response"].strip()
            elif bundle.get("backend") == "transformers_bnb4":
                tokenizer = bundle["tokenizer"]
                model = bundle["model"]
                messages = [{"role": "user", "content": cdss_prompt}]
                inputs = tokenizer(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    return_tensors="pt"
                ).to(self.vram.device)
                with torch.no_grad():
                    out = await asyncio.to_thread(model.generate, **inputs, max_new_tokens=650, temperature=0.1)
                raw_cdss = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            else:
                # Simulation realistic CDSS output
                raw_cdss = json.dumps({
                    "patient_id": journey.patient_id,
                    "primary_impression": "Acute Coronary Syndrome (Suspected STEMI / NSTEMI) in patient with chronic CAD risk factors.",
                    "differential_diagnoses": [
                        {"condition": "Acute Myocardial Infarction", "likelihood": "High"},
                        {"condition": "Unstable Angina Pectoris", "likelihood": "Medium"},
                        {"condition": "Acute Aortic Dissection", "likelihood": "Low"}
                    ],
                    "critical_alerts": [
                        "CRITICAL: Elevated cardiac biomarker (Troponin-I 1.42 ng/mL)",
                        "URGENT: ST-segment deviations noted on ECG rhythm strip",
                        "RED FLAG: Acute diaphoresis and radiating left arm pain"
                    ],
                    "imaging_synthesis": [
                        "Chest X-Ray: Cardiomegaly detected (CTR > 0.52); no pneumothorax or acute alveolar consolidation.",
                        "ECG: ST-elevation pattern in anterior leads with tachycardia."
                    ],
                    "prescriptions_identified": [
                        "Aspirin 75mg OD",
                        "Atorvastatin 40mg HS",
                        "Metformin 500mg BD"
                    ],
                    "recommended_plan": [
                        "Immediate 12-lead continuous telemetry and Cardiology emergency consultation",
                        "Administer dual antiplatelet loading dose if indicated and not contraindicated",
                        "Stat echocardiogram and cardiac catheterization laboratory activation"
                    ]
                })

            if "```json" in raw_cdss:
                raw_cdss = raw_cdss.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_cdss:
                raw_cdss = raw_cdss.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_cdss)
            journey.final_cdss = CDSSReport(**parsed)
        except Exception as e:
            logger.error(f"CDSS generation error: {e}. Falling back to defensive clinical report.")
            journey.final_cdss = CDSSReport(
                patient_id=journey.patient_id,
                primary_impression=journey.patient_summary or "Clinical triage synthesis completed.",
                differential_diagnoses=[{"condition": "Acute evaluation required", "likelihood": "High"}],
                critical_alerts=["Bedside clinical assessment required immediately."],
                imaging_synthesis=[str(d.get("summary", "")) for d in journey.cpu_diagnostics],
                prescriptions_identified=[],
                recommended_plan=["Immediate physician review."]
            )

        # Teardown LLM back to IDLE
        await self.vram.transition_to(ModelState.IDLE)
        self._record_vram_checkpoint("3.2 Pipeline Complete (Clean IDLE)", journey)

        return journey.final_cdss
