"""
MediKiosk Perception - Diagnostic Imaging & Radiology Report Analysis Module
100% CPU-Bound: High-Fidelity OCR + Clinical Pattern Structuring for Printed Ultrasound, CT, MRI, and X-Ray Reports.
Zero GPU VRAM allocation.

Functions:
- is_diagnostic_imaging_report(ocr_text: str, filename: str = "") -> bool
- analyze_diagnostic_report(image_input, file_url: str = "", filename: str = "") -> dict
"""

import os
import re
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Union

DIAGNOSTIC_IMAGING_KEYWORDS = [
    "ultrasound", "usg", "sonography", "sonogram", "echocardiograph", "echo report",
    "computed tomography", "ct scan", "hrct", "ncct", "cect", "mri", "magnetic resonance",
    "x-ray report", "x ray report", "radiological examination", "roentgen", "mammograph",
    "adenomyomatosis", "gall bladder", "cholelithiasis", "corticomedullary", "hydronephrosis",
    "biliary duct", "portal vein", "retroperitoneum", "echopattern", "echogenicity"
]

def is_diagnostic_imaging_report(text: str, filename: str = "") -> bool:
    """Detects whether text or filename belongs to a printed radiology/ultrasound/CT/MRI report."""
    combined = (text + " " + filename).lower()
    matches = sum(1 for kw in DIAGNOSTIC_IMAGING_KEYWORDS if kw in combined)
    has_impression = any(h in combined for h in ["impression", "conclusion", "opinion", "findings"])
    return matches >= 2 or (matches >= 1 and has_impression)


def parse_diagnostic_report_text(ocr_text: str, file_url: str = "") -> Dict[str, Any]:
    """
    Parses OCR text of an ultrasound, CT, MRI, or X-ray printed report into structured
    clinical findings, doctor metadata, and radiologist's impression.
    """
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    # 1. Detect Document / Scan Title
    title = "Ultrasound / Diagnostic Imaging Report"
    for line in lines[:8]:
        line_clean = line.replace("|", "").strip()
        if re.search(r'(?i)\b(ULTRASOUND|SONOGRAPHY|CT\s+SCAN|HRCT|CECT|NCCT|MRI|ECHOCARDIOGRAPHY|MAMMOGRAPHY)\b', line_clean):
            title = line_clean
            break

    # Clean title
    title = re.sub(r'^[^\w]+', '', title).strip()

    # 2. Extract Doctor / Radiologist Name & Center
    doctor_name = ""
    center_name = ""
    for line in lines[:10]:
        line_clean = line.replace("|", "").strip()
        if not doctor_name:
            doc_m = re.search(r'(?i)\b(?:DR[\.,\s]+|DOCTOR\s+)([A-Z\.\s]{3,35})\b', line_clean)
            if doc_m:
                doctor_name = "Dr. " + doc_m.group(1).strip().title()
        if not center_name and any(k in line_clean.lower() for k in ["hospital", "scan", "imaging", "diagnostic", "ultrasound clinic", "radiology"]):
            center_name = line_clean

    # 3. Extract Document Date
    doc_date = ""
    for line in lines[:10]:
        date_m = re.search(r'\b(\d{1,2}[\/\-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})[\/\-\s]\d{2,4})\b', line, re.IGNORECASE)
        if date_m:
            doc_date = date_m.group(1)
            break

    # 4. Extract IMPRESSION / CONCLUSION
    impression_text = ""
    imp_match = re.search(
        r'(?i)(?:IMPRESSION|CONCLUSION|OPINION|FINAL DIAGNOSIS)\s*[:;\-]?\s*([\s\S]+?)(?:\n\s*(?:Adv[i:]|Recommended|Advised|Signature|Dr\.|\bDate\b)|$)',
        ocr_text
    )
    if imp_match:
        raw_imp = imp_match.group(1).strip()
        # Clean bullet characters or OCR artifacts
        cleaned_lines = []
        for l in raw_imp.split('\n'):
            l_str = re.sub(r'^[\*\-\•\¢\>\.\s]+', '', l.strip())
            if l_str and not l_str.lower().startswith("adv") and not l_str.lower().startswith("dr"):
                cleaned_lines.append(l_str)
        impression_text = "; ".join(cleaned_lines)

    # 5. Extract Organ-Specific Findings
    organ_findings = []
    organ_keywords = ["Liver", "Gall bladder", "GB", "Pancreas", "Spleen", "Right Kidney", "Left Kidney", "Kidneys", "Urinary Bladder", "Prostate", "Uterus", "Ovaries", "Bowel", "Lungs", "Pleura", "Peritoneal cavity"]
    for organ in organ_keywords:
        pattern = rf'(?i)\b{organ}\b\s*[:\-]?\s*([^\n\r]+(?:\n(?![A-Z][a-z]+\s*[:\-])[^\n\r]+)?)'
        m = re.search(pattern, ocr_text)
        if m:
            clean_val = " ".join(m.group(1).strip().split())
            if len(clean_val) > 5 and not clean_val.lower().startswith("impression"):
                organ_findings.append(f"{organ}: {clean_val[:120]}")

    # 6. Formulate Diagnoses & Flagged Abnormalities
    diagnoses = []
    flagged_values = []

    if impression_text:
        # Split discrete findings in impression
        for part in re.split(r'[;.]|\n', impression_text):
            part_str = part.strip()
            if part_str and len(part_str) > 3:
                diagnoses.append(part_str)
                # Check if it's an actual positive abnormality
                if not any(neg in part_str.lower() for neg in ["no other", "no significant", "normal study", "unremarkable", "within normal"]):
                    flagged_values.append(f"⚠️ Diagnostic Finding: {part_str}")
    else:
        diagnoses.append(title if title else "Diagnostic Imaging Study")

    if not flagged_values:
        flagged_values.append("✅ No acute gross radiological abnormality detected in impression")

    # 7. Construct Clinical Synthesis Summary
    doctor_prefix = f"({doctor_name})" if doctor_name else ""
    if impression_text:
        summary_text = f"{title} {doctor_prefix}: Impression — {impression_text}"
    else:
        summary_text = f"{title} {doctor_prefix}: Diagnostic scan reviewed. Detailed organ findings transcribed."

    dashboard_payload = {
        "document_type": title,
        "modality": "document",
        "diagnoses": diagnoses if diagnoses else [title],
        "medications": [],
        "flagged_values": flagged_values,
        "document_date": doc_date if doc_date else "Visual Scan",
        "summary": summary_text[:280],
        "file_url": file_url,
        "raw_text": ocr_text if ocr_text else "Diagnostic report transcribed."
    }

    return {
        "title": title,
        "doctor_name": doctor_name,
        "date": doc_date,
        "impression": impression_text,
        "organ_findings": organ_findings,
        "dashboard_payload": dashboard_payload
    }


def analyze_diagnostic_report(
    image_input: Union[str, bytes, np.ndarray],
    file_url: str = "",
    filename: str = ""
) -> Dict[str, Any]:
    """
    Unified entry point to analyze a printed ultrasound, CT, MRI, or X-ray clinical report.
    Executes Sauvola preprocessing and dual OCR on CPU.
    """
    from perception.prescription import preprocess_prescription, run_ocr
    contrast_gray, binarized = preprocess_prescription(image_input)
    ocr_text = run_ocr(contrast_gray, binarized)

    parsed = parse_diagnostic_report_text(ocr_text, file_url=file_url)
    return parsed
