"""
MediKiosk Perception - Document OCR & Prescription Parsing Module
100% CPU-Bound: Offline OCR, Sauvola Adaptive Binarization & RapidFuzz Drug Normalization.
Zero GPU VRAM allocation.

Functions:
- parse_document_or_prescription(image_path: str, doc_type: str) -> dict
"""

import os
import re
import json
import cv2
import numpy as np
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Dict, Any, List, Optional, Tuple, Union
from perception.prescription import (
    sauvola_adaptive_threshold,
    deskew_image,
    normalize_drugs,
    load_drug_lexicon
)

def extract_text_from_pdf(pdf_input: Union[str, bytes]) -> str:
    """Extracts digital text from PDF files using PyMuPDF (fitz)."""
    try:
        import fitz
        if isinstance(pdf_input, bytes):
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        else:
            doc = fitz.open(pdf_input)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        return ""

# Common lab test reference intervals for automatic clinical flagging
COMMON_LAB_RANGES = {
    "hemoglobin": {"name": "Hemoglobin (Hb)", "unit": "g/dL", "min": 12.0, "max": 17.5},
    "hb": {"name": "Hemoglobin (Hb)", "unit": "g/dL", "min": 12.0, "max": 17.5},
    "tlc": {"name": "Total Leukocyte Count (TLC/WBC)", "unit": "/cumm", "min": 4000, "max": 11000},
    "wbc": {"name": "Total Leukocyte Count (TLC/WBC)", "unit": "/cumm", "min": 4000, "max": 11000},
    "platelet": {"name": "Platelet Count", "unit": "/cumm", "min": 150000, "max": 450000},
    "creatinine": {"name": "Serum Creatinine", "unit": "mg/dL", "min": 0.6, "max": 1.2},
    "urea": {"name": "Blood Urea", "unit": "mg/dL", "min": 15, "max": 45},
    "bilirubin": {"name": "Total Bilirubin", "unit": "mg/dL", "min": 0.2, "max": 1.2},
    "glucose": {"name": "Blood Glucose (Fasting)", "unit": "mg/dL", "min": 70, "max": 100},
    "sugar": {"name": "Blood Sugar (Random)", "unit": "mg/dL", "min": 70, "max": 140},
    "hba1c": {"name": "HbA1c", "unit": "%", "min": 4.0, "max": 5.7},
}


def run_cpu_ocr(image_path: str) -> str:
    """Executes CPU OCR via PyMuPDF (for PDFs) or PaddleOCR / pytesseract fallback."""
    if image_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(image_path)
        if text and len(text.strip()) > 30:
            return text

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]
    if w < 1500:
        scale = 1600 / w
        img = cv2.resize(img, (1600, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Attempt PaddleOCR (CPU mode)
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        result = ocr.ocr(img, cls=True)
        lines = []
        if result and result[0]:
            for item in result[0]:
                text_part = item[1][0]
                lines.append(text_part)
        extracted = "\n".join(lines)
        if len(extracted.strip()) > 10:
            return extracted
    except Exception:
        pass

    # Fallback to pytesseract with CLAHE contrast enhancement
    try:
        import pytesseract
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        text_psm4 = pytesseract.image_to_string(contrast, config='--oem 3 --psm 4')
        text_psm3 = pytesseract.image_to_string(contrast, config='--oem 3 --psm 3')
        return text_psm4 if len(text_psm4.split('\n')) >= len(text_psm3.split('\n')) else text_psm3
    except Exception:
        pass

    return ""


def parse_printed_report(image_path: str, raw_text: str = "") -> Dict[str, Any]:
    """
    Parses printed diagnostic reports (CT/MRI/Ultrasound/Lab/PFT):
    - Isolates 'IMPRESSION:', 'FINDINGS:', and 'CONCLUSION:' sections.
    - Extracts numerical lab parameters and flags out-of-range values.
    """
    if not raw_text:
        raw_text = run_cpu_ocr(image_path)

    # 1. Extract Section Impressions (CT / MRI / Ultrasound / Biopsy)
    impression_text = ""
    findings_text = ""

    # Match Impression
    imp_match = re.search(r"(?:IMPRESSION|CONCLUSION|DIAGNOSIS|FINAL IMPRESSION)\s*[:\-]?\s*(.*?)(?=(?:RECOMMENDATION|NOTE|ADVICE|CORRELATION|$|\n\n[A-Z]))", raw_text, re.DOTALL | re.IGNORECASE)
    if imp_match:
        impression_text = imp_match.group(1).strip()
        # Clean multiple newlines
        impression_text = re.sub(r"\s+", " ", impression_text)[:500]

    # Match Findings
    find_match = re.search(r"(?:FINDINGS|OBSERVATIONS|DESCRIPTION)\s*[:\-]?\s*(.*?)(?=(?:IMPRESSION|CONCLUSION|RECOMMENDATION|$|\n\n[A-Z]))", raw_text, re.DOTALL | re.IGNORECASE)
    if find_match:
        findings_text = find_match.group(1).strip()
        findings_text = re.sub(r"\s+", " ", findings_text)[:500]

    # 2. Extract and Flag Numerical Lab Metrics
    flagged_lab_values = []
    diagnoses = []

    for key, ref in COMMON_LAB_RANGES.items():
        # Match pattern like "Hemoglobin: 9.5 g/dL" or "Hb 8.2" or "Platelet Count 95,000"
        pattern = rf"(?:{key})\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                name = ref["name"]
                unit = ref["unit"]
                min_v, max_v = ref["min"], ref["max"]

                if val < min_v:
                    flag_msg = f"Low {name}: {val} {unit} (Ref: {min_v}-{max_v})"
                    flagged_lab_values.append(flag_msg)
                    if "Hemoglobin" in name:
                        diagnoses.append("Anemia (Low Hemoglobin)")
                    elif "Platelet" in name:
                        diagnoses.append("Thrombocytopenia (Low Platelets)")
                    elif "Leukocyte" in name or "WBC" in name:
                        diagnoses.append("Leukopenia")
                elif val > max_v:
                    flag_msg = f"Elevated {name}: {val} {unit} (Ref: {min_v}-{max_v})"
                    flagged_lab_values.append(flag_msg)
                    if "Leukocyte" in name or "WBC" in name:
                        diagnoses.append("Leukocytosis (Suspected Active Infection / Inflammation)")
                    elif "Creatinine" in name:
                        diagnoses.append("Elevated Serum Creatinine (Renal Impairment)")
                    elif "Glucose" in name or "Sugar" in name:
                        diagnoses.append("Hyperglycemia")
                    elif "Bilirubin" in name:
                        diagnoses.append("Hyperbilirubinemia (Jaundice)")
            except Exception:
                pass

    if impression_text:
        diagnoses.insert(0, f"Clinical Impression: {impression_text[:120]}")
    elif not diagnoses:
        diagnoses.append("Printed Diagnostic Study Reviewed")

    summary = (
        f"Printed Diagnostic Report: {impression_text if impression_text else 'Structured test report processed.'} "
        f"{f'Flagged {len(flagged_lab_values)} abnormal lab parameter(s).' if flagged_lab_values else 'Parameters within normal baseline limits.'}"
    )

    dashboard_payload = {
        "document_type": "Printed Diagnostic / Lab Report",
        "modality": "document",
        "diagnoses": diagnoses,
        "medications": [],
        "flagged_values": flagged_lab_values,
        "document_date": "Report Scan",
        "summary": summary,
        "file_url": f"/{image_path}" if not image_path.startswith("/") else image_path,
        "raw_text": raw_text[:2000]
    }

    return {
        "doc_type": "PRINTED_REPORT",
        "impression": impression_text,
        "findings": findings_text,
        "flagged_lab_values": flagged_lab_values,
        "diagnoses": diagnoses,
        "raw_text": raw_text,
        "dashboard_payload": dashboard_payload
    }


def parse_handwritten_prescription(image_path: str) -> Dict[str, Any]:
    """
    Parses handwritten doctor prescription slips:
    1. Preprocessing: Sauvola adaptive thresholding heals faint ballpoint pen strokes.
    2. OCR: Runs CPU OCR.
    3. Drug Normalization: RapidFuzz matching against data/indian_drug_lexicon.json.
       Tokens with score < 80 are explicitly flagged for physician verification.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load prescription image: {image_path}")

    # Step 1: Sauvola Adaptive Binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    deskewed = deskew_image(gray)
    binarized = sauvola_adaptive_threshold(deskewed, window_size=25, k=0.18)

    # Step 2: OCR on Binarized Pen Strokes
    # Save temporary binarized image for OCR pass
    temp_bin_path = f"{image_path}_sauvola.png"
    cv2.imwrite(temp_bin_path, binarized)
    try:
        raw_text = run_cpu_ocr(temp_bin_path)
    finally:
        if os.path.exists(temp_bin_path):
            try:
                os.remove(temp_bin_path)
            except Exception:
                pass

    if not raw_text:
        # Fallback to raw image OCR
        raw_text = run_cpu_ocr(image_path)

    # Step 3: RapidFuzz Drug Normalization against Indian Lexicon
    drug_items = normalize_drugs(raw_text)

    verified_meds = []
    flagged_unverified = []

    for item in drug_items:
        drug_name = item.get("drug", "")
        strength = item.get("strength", "")
        status = item.get("status", "NORMALIZED")
        score = item.get("score", 0)
        raw_tok = item.get("raw_token", "")

        med_entry = f"{drug_name} {strength}".strip()
        verified_meds.append(med_entry)

        if status == "FLAGGED_FOR_DOCTOR" or score < 80:
            flagged_unverified.append(
                f"⚠️ Unverified Rx Token: '{raw_tok}' -> Matched '{drug_name}' (Confidence: {score}%)"
            )

    diagnoses = ["Doctor Consultation Slip / Prescription"]
    summary = (
        f"Handwritten Prescription (Sauvola CPU): Identified {len(drug_items)} candidate medication(s). "
        f"{len(flagged_unverified)} token(s) flagged for physical slip verification."
    )

    dashboard_payload = {
        "document_type": "Handwritten Prescription",
        "modality": "document",
        "diagnoses": diagnoses,
        "medications": verified_meds,
        "flagged_values": flagged_unverified,
        "document_date": "Prescription Scan",
        "summary": summary,
        "file_url": f"/{image_path}" if not image_path.startswith("/") else image_path,
        "raw_text": raw_text[:2000]
    }

    return {
        "doc_type": "HANDWRITTEN_PRESCRIPTION",
        "medications": verified_meds,
        "normalized_drugs": drug_items,
        "flagged_for_doctor": flagged_unverified,
        "raw_text": raw_text,
        "dashboard_payload": dashboard_payload
    }


def parse_document_or_prescription(image_path: str, doc_type: str = "PRINTED_REPORT") -> Dict[str, Any]:
    """
    Main dispatch function:
    - doc_type in ['PRINTED_REPORT', 'HANDWRITTEN_PRESCRIPTION']
    """
    doc_upper = doc_type.upper()
    if "HANDWRITTEN" in doc_upper or "PRESCRIPTION" in doc_upper:
        return parse_handwritten_prescription(image_path)
    else:
        return parse_printed_report(image_path)
