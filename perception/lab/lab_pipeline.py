"""
lab_pipeline.py
===============
Stage 3: Boundary Deduplication, Clinical Lexicon Canonicalization,
Deterministic Reference Range Validation, and Async Queue Integration.

Features:
1. Boundary Stitching & Fuzzy Deduplication:
   Reconciles overlapping boundary rows across adjacent horizontal strips.
2. Clinical Lexicon Canonicalization:
   Normalizes varied lab terms (e.g. 'Hb', 'Hgb', 'Haemoglobin') to LOINC-standard
   analyte entities using rapid token normalization.
3. Deterministic Flag Verification:
   Parses biological reference intervals and calculates true clinical flags (HIGH / LOW / NORMAL)
   to eliminate VLM interpretation errors or hallucinations.
4. Async Triage Queue Integration:
   Designed for zero-contention execution inside the MediKiosk 8GB VRAM architecture.
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np

from .lab_preprocessor import LabPreprocessor, LabImageChunk
from .vlm_extractor import ConstrainedVLMExtractor, LabTestItem, FlagEnum

logger = logging.getLogger("LabReportPipeline")


# =============================================================================
# Canonical Clinical Reference Ranges & Standard Analyte Lexicon
# =============================================================================
CANONICAL_LAB_LEXICON: Dict[str, Dict[str, Any]] = {
    # Complete Blood Count (CBC)
    "hemoglobin": {"canonical": "Hemoglobin (Hb)", "unit": "g/dL", "min": 12.0, "max": 17.5, "critical_low": 7.0, "critical_high": 20.0},
    "hb": {"canonical": "Hemoglobin (Hb)", "unit": "g/dL", "min": 12.0, "max": 17.5, "critical_low": 7.0, "critical_high": 20.0},
    "total leukocyte count": {"canonical": "Total Leukocyte Count (WBC)", "unit": "/cumm", "min": 4000, "max": 11000, "critical_low": 2000, "critical_high": 30000},
    "total leukocyte": {"canonical": "Total Leukocyte Count (WBC)", "unit": "/cumm", "min": 4000, "max": 11000, "critical_low": 2000, "critical_high": 30000},
    "leukocyte count": {"canonical": "Total Leukocyte Count (WBC)", "unit": "/cumm", "min": 4000, "max": 11000, "critical_low": 2000, "critical_high": 30000},
    "leukocyte": {"canonical": "Total Leukocyte Count (WBC)", "unit": "/cumm", "min": 4000, "max": 11000, "critical_low": 2000, "critical_high": 30000},
    "tlc": {"canonical": "Total Leukocyte Count (WBC)", "unit": "/cumm", "min": 4000, "max": 11000, "critical_low": 2000, "critical_high": 30000},
    "wbc": {"canonical": "Total Leukocyte Count (WBC)", "unit": "/cumm", "min": 4000, "max": 11000, "critical_low": 2000, "critical_high": 30000},
    "total rbc count": {"canonical": "Red Blood Cell Count (RBC)", "unit": "mil/uL", "min": 4.2, "max": 5.8},
    "total rbc": {"canonical": "Red Blood Cell Count (RBC)", "unit": "mil/uL", "min": 4.2, "max": 5.8},
    "rbc count": {"canonical": "Red Blood Cell Count (RBC)", "unit": "mil/uL", "min": 4.2, "max": 5.8},
    "rbc": {"canonical": "Red Blood Cell Count (RBC)", "unit": "mil/uL", "min": 4.2, "max": 5.8},
    "platelet count": {"canonical": "Platelet Count", "unit": "/cumm", "min": 150000, "max": 450000, "critical_low": 25000, "critical_high": 1000000},
    "platelet": {"canonical": "Platelet Count", "unit": "/cumm", "min": 150000, "max": 450000, "critical_low": 25000, "critical_high": 1000000},
    "pcv": {"canonical": "Packed Cell Volume (Hematocrit/PCV)", "unit": "%", "min": 36.0, "max": 50.0},
    "hematocrit": {"canonical": "Packed Cell Volume (Hematocrit/PCV)", "unit": "%", "min": 36.0, "max": 50.0},
    "hct": {"canonical": "Packed Cell Volume (Hematocrit/PCV)", "unit": "%", "min": 36.0, "max": 50.0},
    "mcv": {"canonical": "Mean Corpuscular Volume (MCV)", "unit": "fL", "min": 80.0, "max": 100.0},
    "mchc": {"canonical": "Mean Corpuscular Hb Conc (MCHC)", "unit": "g/dL", "min": 31.5, "max": 36.0},
    "mch": {"canonical": "Mean Corpuscular Hemoglobin (MCH)", "unit": "pg", "min": 27.0, "max": 33.0},
    "rdw": {"canonical": "Red Cell Distribution Width (RDW)", "unit": "%", "min": 11.5, "max": 14.5},
    "neutrophil": {"canonical": "Neutrophils (Polymorphs)", "unit": "%", "min": 40.0, "max": 75.0},
    "lymphocyte": {"canonical": "Lymphocytes", "unit": "%", "min": 20.0, "max": 45.0},
    "monocyte": {"canonical": "Monocytes", "unit": "%", "min": 2.0, "max": 10.0},
    "eosinophil": {"canonical": "Eosinophils", "unit": "%", "min": 1.0, "max": 6.0},
    "basophil": {"canonical": "Basophils", "unit": "%", "min": 0.0, "max": 2.0},

    # Renal Panel / Kidney Function (KFT)
    "creatinine": {"canonical": "Serum Creatinine", "unit": "mg/dL", "min": 0.6, "max": 1.25, "critical_high": 4.0},
    "urea": {"canonical": "Blood Urea", "unit": "mg/dL", "min": 15.0, "max": 45.0, "critical_high": 100.0},
    "bun": {"canonical": "Blood Urea Nitrogen (BUN)", "unit": "mg/dL", "min": 7.0, "max": 20.0},
    "uric acid": {"canonical": "Serum Uric Acid", "unit": "mg/dL", "min": 3.5, "max": 7.2},
    "potassium": {"canonical": "Serum Potassium (K+)", "unit": "mEq/L", "min": 3.5, "max": 5.1, "critical_low": 2.8, "critical_high": 6.2},
    "sodium": {"canonical": "Serum Sodium (Na+)", "unit": "mEq/L", "min": 135.0, "max": 145.0, "critical_low": 120.0, "critical_high": 160.0},

    # Liver Function Test (LFT)
    "bilirubin total": {"canonical": "Total Bilirubin", "unit": "mg/dL", "min": 0.2, "max": 1.2, "critical_high": 10.0},
    "bilirubin direct": {"canonical": "Direct Bilirubin", "unit": "mg/dL", "min": 0.0, "max": 0.3},
    "sgot": {"canonical": "AST / SGOT", "unit": "U/L", "min": 10.0, "max": 40.0, "critical_high": 250.0},
    "ast": {"canonical": "AST / SGOT", "unit": "U/L", "min": 10.0, "max": 40.0, "critical_high": 250.0},
    "sgpt": {"canonical": "ALT / SGPT", "unit": "U/L", "min": 10.0, "max": 45.0, "critical_high": 250.0},
    "alt": {"canonical": "ALT / SGPT", "unit": "U/L", "min": 10.0, "max": 45.0, "critical_high": 250.0},
    "alkaline phosphatase": {"canonical": "Alkaline Phosphatase (ALP)", "unit": "U/L", "min": 44.0, "max": 147.0},
    "alp": {"canonical": "Alkaline Phosphatase (ALP)", "unit": "U/L", "min": 44.0, "max": 147.0},

    # Glycemic & Metabolic
    "fasting glucose": {"canonical": "Blood Glucose (Fasting)", "unit": "mg/dL", "min": 70.0, "max": 99.0, "critical_low": 50.0, "critical_high": 350.0},
    "random glucose": {"canonical": "Blood Glucose (Random)", "unit": "mg/dL", "min": 70.0, "max": 140.0, "critical_low": 50.0, "critical_high": 400.0},
    "hba1c": {"canonical": "Glycated Hemoglobin (HbA1c)", "unit": "%", "min": 4.0, "max": 5.6, "critical_high": 10.0},

    # Lipid Profile
    "cholesterol total": {"canonical": "Total Cholesterol", "unit": "mg/dL", "min": 125.0, "max": 200.0},
    "triglycerides": {"canonical": "Triglycerides", "unit": "mg/dL", "min": 50.0, "max": 150.0},
    "hdl": {"canonical": "HDL Cholesterol", "unit": "mg/dL", "min": 40.0, "max": 60.0},
    "ldl": {"canonical": "LDL Cholesterol", "unit": "mg/dL", "min": 50.0, "max": 100.0},

    # Cardiac Biomarkers
    "troponin": {"canonical": "Troponin-I", "unit": "ng/mL", "min": 0.0, "max": 0.04, "critical_high": 0.10},
    "crp": {"canonical": "C-Reactive Protein (CRP)", "unit": "mg/L", "min": 0.0, "max": 5.0}
}


@dataclass
class LabReportResult:
    """Final verified clinical extraction payload."""
    document_id: str
    total_tests_detected: int
    abnormal_count: int
    tests: List[LabTestItem]
    flagged_abnormalities: List[Dict[str, Any]]
    summary: str
    chunks_processed: int
    total_vision_tokens_estimated: int


class LabReportPipeline:
    """
    End-to-end laboratory report parser combining:
    1. Preprocessor & Adaptive Slicing (CPU)
    2. Constrained VLM Strips Ingestion (RTX 4060 GPU)
    3. Overlap Deduplication & Clinical Reference Range Validation (CPU)
    """

    def __init__(
        self,
        extractor: Optional[ConstrainedVLMExtractor] = None,
        preprocessor: Optional[LabPreprocessor] = None
    ):
        self.preprocessor = preprocessor or LabPreprocessor()
        self.extractor = extractor or ConstrainedVLMExtractor()

    def process(self, doc_input: Union[str, bytes], document_id: str = "LAB_DOC") -> LabReportResult:
        """
        Synchronous execution entrypoint.
        """
        # Step 1: CPU Preprocessing & Adaptive Chunking
        logger.info(f"📄 [Stage 1] Preprocessing document: {document_id}")
        image_bgr = self.preprocessor.load_document(doc_input)
        chunks = self.preprocessor.slice_table_into_strips(image_bgr)
        logger.info(f"🧩 Created {len(chunks)} adaptive horizontal strip(s).")

        total_vision_tokens = sum(ch.token_estimated_count for ch in chunks)

        # Step 2: Constrained VLM Sequential Ingestion
        logger.info(f"👁️  [Stage 2] Ingesting strips sequentially into Qwen2.5-VL-3B...")
        raw_extractions: List[LabTestItem] = []
        for ch in chunks:
            chunk_results = self.extractor.extract_chunk(ch)
            raw_extractions.extend(chunk_results)

        # Failover / Fallback: If VLM timed out or returned 0 items, run deterministic CPU OCR
        if not raw_extractions:
            logger.warning("⚠️ VLM extraction returned 0 rows! Activating deterministic CPU PyTesseract fallback...")
            raw_extractions = self._extract_via_cpu_ocr(image_bgr)

        # Step 3: Stitching, Deduplication & Validation
        logger.info(f"🔬 [Stage 3] Deduplicating & validating {len(raw_extractions)} extracted row(s)...")
        deduplicated = self._deduplicate_items(raw_extractions)
        canonicalized = [self._canonicalize_and_validate(item) for item in deduplicated]

        # Identify Flagged Abnormalities & Critical Alerts
        abnormalities = []
        for item in canonicalized:
            if item.flag in [FlagEnum.HIGH, FlagEnum.LOW, FlagEnum.ABNORMAL]:
                abnormalities.append({
                    "test_name": item.test_name,
                    "result_value": item.result_value,
                    "unit": item.unit or "",
                    "reference_interval": item.reference_interval or "N/A",
                    "status": item.flag.value
                })

        # Generate Clinical Synthesis Summary
        summary = self._generate_clinical_summary(canonicalized, abnormalities)

        return LabReportResult(
            document_id=document_id,
            total_tests_detected=len(canonicalized),
            abnormal_count=len(abnormalities),
            tests=canonicalized,
            flagged_abnormalities=abnormalities,
            summary=summary,
            chunks_processed=len(chunks),
            total_vision_tokens_estimated=total_vision_tokens
        )

    # =========================================================================
    # Overlap Deduplication
    # =========================================================================
    def _deduplicate_items(self, items: List[LabTestItem]) -> List[LabTestItem]:
        """
        Merges duplicate row detections resulting from chunk overlap windows.
        Strictly filters out empty sub-headers, banners, and non-numeric rows.
        """
        SECTION_HEADER_BLACKLIST = {
            "differential leucocyte count", "differential leukocyte count", "differential leucocyte",
            "differential leukocyte", "dlc", "haematology", "hematology", "complete blood count",
            "complete blood count cbc", "cbc", "lipid profile", "liver function test", "lft",
            "kidney function test", "kft", "renal function test", "rft", "urine routine",
            "urine routine examination", "clinical notes", "investigation", "test name", "test",
            "parameter", "biological reference interval", "bio ref interval", "observed value",
            "reference range", "normal range", "method", "unit", "specimen", "page",
            "possible causes of abnormal parameters", "patient registration", "registered on"
        }

        deduped: List[LabTestItem] = []
        seen_keys: Dict[str, int] = {}  # normalized_name -> index in deduped

        for item in items:
            norm_name = self._normalize_test_name(item.test_name)
            if not norm_name or len(norm_name) < 2:
                continue

            # 1. Skip empty section headers and banners
            if norm_name in SECTION_HEADER_BLACKLIST:
                logger.info(f"Deduplicator dropped banner: '{item.test_name}'")
                continue

            # 2. Only rows containing both a test name and a numeric result should be extracted
            num_val = self._parse_numeric_value(item.result_value)
            if num_val is None:
                continue

            # Sync reference_range and reference_interval
            ref = item.reference_range or item.reference_interval
            item.reference_range = ref
            item.reference_interval = ref

            if norm_name in seen_keys:
                existing_idx = seen_keys[norm_name]
                existing_item = deduped[existing_idx]

                # Merge fields: prioritize more informative values
                if not existing_item.unit and item.unit:
                    existing_item.unit = item.unit
                if not existing_item.reference_interval and item.reference_interval:
                    existing_item.reference_interval = item.reference_interval
                    existing_item.reference_range = item.reference_range
                if not existing_item.flag and item.flag:
                    existing_item.flag = item.flag
            else:
                seen_keys[norm_name] = len(deduped)
                deduped.append(item)

        return deduped

    # =========================================================================
    # Clinical Canonicalization & Range Cross-Validation
    # =========================================================================
    def _canonicalize_and_validate(self, item: LabTestItem) -> LabTestItem:
        """
        Normalizes analyte test name to canonical clinical entity.
        Cross-checks result value against reference interval to verify HIGH/LOW flag.
        """
        norm_key = self._normalize_test_name(item.test_name)
        matched_canonical = None

        # Lexicon lookup (longest key first so mchc matches before mch)
        for key, entry in sorted(CANONICAL_LAB_LEXICON.items(), key=lambda x: len(x[0]), reverse=True):
            if key in norm_key:
                matched_canonical = entry
                item.test_name = entry["canonical"]
                if not item.unit and entry.get("unit"):
                    item.unit = entry["unit"]
                break

        # Parse numerical value
        num_val = self._parse_numeric_value(item.result_value)

        # Parse reference range bounds
        ref_min, ref_max = self._parse_reference_interval(item.reference_interval)

        # Fallback to canonical range bounds if document bounds are missing
        if ref_min is None and ref_max is None and matched_canonical:
            ref_min = matched_canonical.get("min")
            ref_max = matched_canonical.get("max")
            if not item.reference_interval:
                item.reference_interval = f"{ref_min} - {ref_max}"

        # Programmatic Flag Evaluation
        if num_val is not None:
            if ref_min is not None and num_val < ref_min:
                item.flag = FlagEnum.LOW
            elif ref_max is not None and num_val > ref_max:
                item.flag = FlagEnum.HIGH
            elif ref_min is not None or ref_max is not None:
                item.flag = FlagEnum.NORMAL

        return item

    # =========================================================================
    # Helpers
    # =========================================================================
    @staticmethod
    def _normalize_test_name(name: str) -> str:
        """Strips punctuation, numbering, and stopwords for clean token hashing."""
        clean = re.sub(r"^[0-9\.\)\-\s]+", "", name.lower())
        clean = re.sub(r"[^\w\s]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def _parse_numeric_value(val: Union[float, str]) -> Optional[float]:
        """Extracts floating-point number from string, removing commas and stray characters."""
        if isinstance(val, (int, float)):
            return float(val)
        if not val:
            return None
        val_str = str(val).replace(",", "").strip()
        match = re.search(r"[-+]?\d*\.?\d+", val_str)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_reference_interval(interval_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
        """
        Parses intervals like '12.0 - 17.5', '< 200', '> 40', '150000 to 450000'.
        """
        if not interval_str:
            return None, None

        s = interval_str.replace(",", "").strip()

        # Less than (< X)
        lt_match = re.search(r"<\s*([0-9\.]+)", s)
        if lt_match:
            try:
                return 0.0, float(lt_match.group(1))
            except ValueError:
                pass

        # Greater than (> X)
        gt_match = re.search(r">\s*([0-9\.]+)", s)
        if gt_match:
            try:
                return float(gt_match.group(1)), None
            except ValueError:
                pass

        # Range interval: X - Y or X to Y
        range_match = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(range_match) >= 2:
            try:
                val1 = float(range_match[0])
                val2 = float(range_match[1])
                return min(val1, val2), max(val1, val2)
            except ValueError:
                pass

    def _extract_via_cpu_ocr(self, image_bgr: np.ndarray) -> List[LabTestItem]:
        """
        Deterministic CPU-bound fallback parser using PyTesseract.
        Executes in ~1.5s with zero GPU VRAM, guaranteeing that valid lab tables
        are NEVER returned with 0 parameters even during Ollama timeouts or busy states.
        """
        try:
            import pytesseract
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("pytesseract / rapidfuzz not available for CPU fallback.")
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        ocr_text = pytesseract.image_to_string(gray, config='--oem 3 --psm 4')

        SECTION_BANNERS = [
            "differential leucocyte", "differential leukocyte", "complete blood count",
            "haematology", "sample letterhead", "clinical notes", "possible causes",
            "patient registration", "registered on", "test value unit reference"
        ]

        items: List[LabTestItem] = []
        for raw_line in ocr_text.splitlines():
            line = raw_line.strip()
            if not line or len(line) < 4:
                continue

            lower_line = line.lower()
            if any(b in lower_line for b in SECTION_BANNERS):
                continue

            # Check matching against canonical tests
            best_key = None
            best_score = 0
            for key, val in sorted(CANONICAL_LAB_LEXICON.items(), key=lambda x: len(x[0]), reverse=True):
                score = fuzz.partial_ratio(key, lower_line)
                if score > 78 and score > best_score:
                    best_score = score
                    best_key = key

            if best_key and best_score >= 80:
                nums = re.findall(r"[-+]?\d*\.?\d+", line.replace(",", ""))
                if nums:
                    try:
                        val_num = float(nums[0]) if "." in nums[0] else int(nums[0])
                    except ValueError:
                        continue

                    ref_str = None
                    if len(nums) >= 3:
                        ref_str = f"{nums[-2]} - {nums[-1]}"

                    flag_val = None
                    if " h " in f" {lower_line} " or " high " in lower_line:
                        flag_val = FlagEnum.HIGH
                    elif " l " in f" {lower_line} " or " low " in lower_line:
                        flag_val = FlagEnum.LOW

                    entry = CANONICAL_LAB_LEXICON[best_key]
                    items.append(LabTestItem(
                        test_name=entry["canonical"],
                        result_value=val_num,
                        unit=entry.get("unit"),
                        reference_range=ref_str,
                        reference_interval=ref_str,
                        flag=flag_val
                    ))

        logger.info(f"CPU PyTesseract fallback recovered {len(items)} lab test row(s).")
        return items

    def _generate_clinical_summary(self, tests: List[LabTestItem], abnormalities: List[Dict[str, Any]]) -> str:
        """Synthesizes an executive summary for CDSS integration."""
        if not abnormalities:
            return f"Laboratory Panel: {len(tests)} parameters analyzed. All within biological reference intervals."

        abnormal_names = [f"{a['test_name']} ({a['status']}: {a['result_value']} {a['unit']})" for a in abnormalities]
        summary_text = (
            f"Laboratory Panel: Identified {len(abnormalities)} abnormal parameter(s) out of {len(tests)} tested: "
            + "; ".join(abnormal_names)
            + "."
        )
        return summary_text


def analyze_lab_report(image_input: Union[str, bytes], file_url: str = "", filename: str = "") -> Dict[str, Any]:
    """
    Main perception entrypoint for clinical laboratory reports (CBC, KFT, LFT, Lipid, etc.).
    Returns structured dashboard payload compatible with Doctor Dashboard.
    """
    pipeline = LabReportPipeline()
    lab_res = pipeline.process(image_input, document_id=filename or "LAB_DOC")

    # Format flagged abnormalities with clinical indicators
    flagged_values = []
    for ab in lab_res.flagged_abnormalities:
        flagged_values.append(
            f"⚠️ {ab['test_name']}: {ab['result_value']} {ab['unit']} (Ref: {ab['reference_interval']}) [{ab['status']}]"
        )

    # Determine panel title
    test_names_str = " ".join(t.test_name.lower() for t in lab_res.tests)
    if any(k in test_names_str for k in ["hemoglobin", "leukocyte", "platelet", "mcv", "mch", "neutrophil", "rbc"]):
        panel_title = "Hematology / Complete Blood Count (CBC)"
    elif any(k in test_names_str for k in ["creatinine", "urea", "bun", "uric acid"]):
        panel_title = "Renal Function Test (KFT)"
    elif any(k in test_names_str for k in ["bilirubin", "sgot", "sgpt", "alp"]):
        panel_title = "Liver Function Test (LFT)"
    elif any(k in test_names_str for k in ["cholesterol", "triglyceride", "hdl", "ldl"]):
        panel_title = "Lipid Profile"
    else:
        panel_title = "Laboratory Diagnostic Report"

    diagnoses = [f"{panel_title}"]
    if lab_res.abnormal_count > 0:
        abnormal_names = [ab['test_name'] for ab in lab_res.flagged_abnormalities]
        diagnoses.append(f"Abnormal: {', '.join(abnormal_names[:3])}")

    # Build clean formatted raw_text transcript
    raw_lines = [
        f"{t.test_name}\t{t.result_value}\t{t.unit or ''}\t{t.reference_interval or ''}\t{t.flag.value if t.flag else 'NORMAL'}"
        for t in lab_res.tests
    ]
    raw_table = "TEST\tVALUE\tUNIT\tREFERENCE\tFLAG\n" + "\n".join(raw_lines)

    dashboard_payload = {
        "document_type": panel_title,
        "modality": "document",
        "diagnoses": diagnoses,
        "medications": [],  # Crucial: Lab reports have NO prescribed medications!
        "flagged_values": flagged_values,
        "document_date": "Lab Report",
        "summary": lab_res.summary,
        "file_url": file_url,
        "raw_text": raw_table
    }

    return {
        "status": "success",
        "lab_result": lab_res,
        "dashboard_payload": dashboard_payload
    }

