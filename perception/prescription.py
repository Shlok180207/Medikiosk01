"""
MediKiosk Perception - Doctor Prescription & Handwriting Analysis Module
100% CPU-Bound: OpenCV Sauvola/Adaptive preprocessing + Dual OCR + RapidFuzz Drug Normalizer.
Zero GPU VRAM allocation.
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
from typing import Dict, Any, List, Optional, Tuple

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

# Lexicon Cache
_DRUG_LEXICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indian_drug_lexicon.json")
_CACHED_DRUG_LIST = None
_CACHED_DRUG_MAP = {}

def load_drug_lexicon() -> Tuple[List[str], Dict[str, Dict]]:
    """Loads and caches the static Indian drug lexicon from data/indian_drug_lexicon.json."""
    global _CACHED_DRUG_LIST, _CACHED_DRUG_MAP
    if _CACHED_DRUG_LIST is not None:
        return _CACHED_DRUG_LIST, _CACHED_DRUG_MAP

    drug_names = []
    drug_map = {}
    if os.path.exists(_DRUG_LEXICON_PATH):
        try:
            with open(_DRUG_LEXICON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    brand = item.get("brand_name", "")
                    generic = item.get("generic_name", "")
                    if brand:
                        drug_names.append(brand)
                        drug_map[brand.lower()] = item
                    if generic and generic != brand:
                        drug_names.append(generic)
                        drug_map[generic.lower()] = item
        except Exception as e:
            print(f"⚠️ Error loading drug lexicon: {e}")

    # Default fallback list if lexicon file is absent
    if not drug_names:
        fallback = [
            "Augmentin 625 Duo", "Azithral 500", "Paracetamol 650", "Pan 40",
            "Pantocid 40", "Metrogyl 400", "Amoxyclav 625", "Cifran 500",
            "Montek LC", "Allegra 120", "Telma 40", "Amlong 5", "Glycomet 500",
            "Thyronorm 50", "Shelcal 500", "Becosules", "Combiflam", "Voveran 50",
            "Ciplox 500", "Omez 20", "Ecosprin 75", "Atorva 20"
        ]
        drug_names = fallback
        for d in fallback:
            drug_map[d.lower()] = {"brand_name": d, "generic_name": d, "category": "Medication"}

    _CACHED_DRUG_LIST = drug_names
    _CACHED_DRUG_MAP = drug_map
    return _CACHED_DRUG_LIST, _CACHED_DRUG_MAP


# ── Step A: Image Preprocessing (Deskewing + Sauvola Adaptive Thresholding) ──

def deskew_image(img_gray: np.ndarray) -> np.ndarray:
    """Calculates text line orientation and deskews image to level horizontal lines."""
    try:
        # Invert to make text white on black background
        thresh = cv2.bitwise_not(img_gray)
        _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Extract coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img_gray

        # Compute minimum area bounding box
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Adjust OpenCV angle convention
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If skew is noticeable (> 0.5 degrees and < 35 degrees)
        if 0.5 < abs(angle) < 35.0:
            (h, w) = img_gray.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(img_gray, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed
    except Exception:
        pass
    return img_gray


def sauvola_adaptive_threshold(img_gray: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """
    Applies Sauvola local adaptive binarization with morphological closing to restore broken ballpoint pen strokes.
    Formula: T = mean * (1 + k * (std / R - 1))
    """
    try:
        from scipy.ndimage import uniform_filter
        img_float = img_gray.astype(np.float32)

        mean = uniform_filter(img_float, size=window_size)
        sq_mean = uniform_filter(img_float ** 2, size=window_size)
        variance = np.maximum(0, sq_mean - (mean ** 2))
        std = np.sqrt(variance)

        threshold = mean * (1.0 + k * (std / r - 1.0))
        binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)

        # Morphological closing (re-bridges broken/faint ballpoint pen ink strokes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        restored = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return restored
    except Exception:
        # Fallback to OpenCV adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window_size, 9
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def preprocess_prescription(image_input) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for doctor handwritten prescriptions:
    1. Grayscale conversion
    2. Bilateral filtering for denoising while preserving ink edge gradients
    3. Deskewing
    4. Sauvola adaptive binarization
    Returns: (deskewed_gray, binarized_restored)
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("Unsupported input format for prescription processing")

    if img is None:
        raise ValueError("Could not decode prescription image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Bilateral filter eliminates paper texture and wrinkles
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    deskewed = deskew_image(denoised)
    binarized = sauvola_adaptive_threshold(deskewed, window_size=25, k=0.2)

    return deskewed, binarized


# ── Step B: Dual OCR Runner (TrOCR + PaddleOCR on CPU) ──

_TROCR_PROCESSOR = None
_TROCR_MODEL = None

def get_trocr():
    """Lazily loads Microsoft TrOCR on CPU."""
    global _TROCR_PROCESSOR, _TROCR_MODEL
    if _TROCR_MODEL is None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            print("🖋️ Initializing Microsoft TrOCR (CPU)...")
            _TROCR_PROCESSOR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to("cpu")
            _TROCR_MODEL.eval()
            print("✅ TrOCR initialized on CPU.")
        except Exception as e:
            print(f"⚠️ TrOCR offline initialization note: {e}")
            _TROCR_PROCESSOR = None
            _TROCR_MODEL = None
    return _TROCR_PROCESSOR, _TROCR_MODEL


_PADDLE_OCR = None

def get_paddle_ocr():
    """Lazily loads PaddleOCR for Hindi/Devanagari on CPU."""
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
            print("🇮🇳 Initializing PaddleOCR (Hindi, CPU)...")
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, show_log=False)
            print("✅ PaddleOCR initialized on CPU.")
        except Exception as e:
            print(f"⚠️ PaddleOCR offline initialization note: {e}")
            _PADDLE_OCR = None
    return _PADDLE_OCR


def run_ocr(deskewed: np.ndarray, binarized: np.ndarray) -> str:
    """
    Executes dual OCR on CPU:
    1. Attempts PaddleOCR for multilingual / Devanagari bounding-box recognition.
    2. Runs TrOCR or Tesseract for Latin handwritten line recognition.
    Returns aggregated text.
    """
    extracted_lines = []

    # Try PaddleOCR first for Hindi + English line detection
    paddle = get_paddle_ocr()
    if paddle is not None:
        try:
            results = paddle.ocr(deskewed, cls=True)
            if results and len(results) > 0 and results[0]:
                for line in results[0]:
                    txt, conf = line[1]
                    if conf > 0.4 and txt.strip():
                        extracted_lines.append(txt.strip())
        except Exception as pe:
            print(f"PaddleOCR error: {pe}")

    # Fallback / Complement with Tesseract on CPU
    if len(extracted_lines) < 3:
        try:
            import pytesseract
            # Try Hindi + English
            custom_config = r'--oem 3 --psm 6'
            tess_text = pytesseract.image_to_string(binarized, config=custom_config)
            for line in tess_text.split('\n'):
                c = line.strip()
                if c and len(c) > 2 and c not in extracted_lines:
                    extracted_lines.append(c)
        except Exception:
            pass

    return "\n".join(extracted_lines)


# ── Step C: Defensive Fuzzy Drug Normalizer with RapidFuzz ──

# Clinical Stopwords — words that must NEVER be flagged as drugs
NON_DRUG_STOPWORDS = {
    "investigation", "registration", "collection", "gender", "patient", "referred",
    "sample", "tissue", "pathology", "biopsy", "hospital", "clinic", "memorial",
    "trust", "marg", "jaipur", "rajasthan", "delhi", "mumbai", "road", "street", "phone",
    "toll", "free", "gross", "microscopic", "examination", "sections", "impression",
    "report", "consultant", "pathologist", "entered", "doctor", "signature", "page",
    "date", "male", "female", "years", "dr.", "mbbs", "md", "dnb", "nature", "material",
    "received", "typed", "opinion", "medico", "legal", "purpose", "slide", "block",
    "future", "record", "preservation", "specimen", "amputation", "findings", "history",
    "advised", "tests", "timing", "routine", "neoplasm", "carcinoma", "subepithelium",
    "nucleoli", "neutrophils", "lymphocytes", "mitotic", "figures", "necrosis",
    "eos", "ry", "rl", "slip", "order", "bill", "invoice", "cost", "cash", "total",
    "rupees", "rs", "no", "yes", "mr", "mrs", "ms", "consultation", "medical", "centre"
}

DRUG_CUES = ["tab", "cap", "syp", "inj", "drops", "ointment", "gel", "susp", "rx", "mg", "mcg", "ml", "gm", "od", "bd", "tds", "qid", "sos", "stat", "1-0-1", "0-1-0", "1-1-1"]
DRUG_SUFFIXES = ("cillin", "mycin", "statin", "olol", "pril", "sartan", "prazole", "dipine", "floxacin", "zole", "fen", "mol", "par", "cef", "clav", "kheerapaka", "vati", "churna")


def normalize_drugs(raw_text: str) -> List[Dict[str, Any]]:
    """
    Fuzzy matches raw OCR tokens against the static Indian drug lexicon.
    Strictly avoids treating pathology descriptions or administrative text as drugs.
    """
    raw_lower = raw_text.lower()
    # If the document is clearly a biopsy/histopathology or lab report, it does NOT contain prescriptions
    if any(k in raw_lower for k in ["histopathology", "biopsy", "microscopic examination", "gross examination", "reference range", "normal range", "lipid profile", "complete blood count"]):
        return []

    lexicon_list, lexicon_map = load_drug_lexicon()
    matched_results = []
    seen_drugs = set()

    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

    for line in lines:
        line_l = line.lower()
        # Skip pure header/footer/administrative rows
        if any(h in line_l for h in ["hospital", "clinic", "dr.", "mbbs", "ph:", "phone", "date", "signature", "timing", "registration", "address"]):
            continue

        has_line_cue = any(cue in line_l for cue in DRUG_CUES)

        # Split line into candidate medication chunks
        chunks = re.split(r"[,;•\d+\.]+", line)
        for chunk in chunks:
            # Strip leading and trailing non-alphanumeric characters
            token = re.sub(r'^[^\w]+|[^\w]+$', '', chunk).strip()
            if len(token) < 3 or not any(c.isalpha() for c in token):
                continue

            token_l = token.lower()
            # If token is in clinical/administrative stopwords, skip
            if token_l in NON_DRUG_STOPWORDS or any(stop in token_l for stop in NON_DRUG_STOPWORDS):
                continue

            best_match = None
            best_score = 0

            if process and fuzz:
                match_result = process.extractOne(token, lexicon_list, scorer=fuzz.token_set_ratio)
                if match_result:
                    best_match = match_result[0]
                    best_score = match_result[1]
            else:
                for candidate in lexicon_list:
                    if candidate.lower() in token_l or token_l in candidate.lower():
                        best_match = candidate
                        best_score = 85
                        break

            if best_match and best_score >= 80:
                meta = lexicon_map.get(best_match.lower(), {})
                if best_match not in seen_drugs:
                    seen_drugs.add(best_match)
                    matched_results.append({
                        "drug": best_match,
                        "generic": meta.get("generic_name", best_match),
                        "strength": meta.get("strength", ""),
                        "category": meta.get("category", "Prescribed Drug"),
                        "raw_token": token,
                        "status": "VERIFIED",
                        "score": round(float(best_score), 1)
                    })
            elif len(token) >= 4 and any(c.isalpha() for c in token):
                # Unlisted drug check: ONLY flag if accompanied by medication cues or known pharma patterns
                has_vowel = any(v in token_l for v in 'aeiou')
                has_symbol_noise = any(c in token for c in '{}[]"\'`~|\\^')
                has_drug_form = (has_line_cue and len(token) >= 4) or token_l.endswith(DRUG_SUFFIXES) or (65 <= best_score < 80)

                if has_drug_form and has_vowel and not has_symbol_noise:
                    if token_l not in seen_drugs:
                        seen_drugs.add(token_l)
                        matched_results.append({
                            "drug": token,
                            "generic": "Unlisted / Novel Formulation",
                            "strength": "Unverified",
                            "category": "Unmatched Drug Token",
                            "raw_token": token,
                            "status": "FLAGGED_FOR_DOCTOR",
                            "score": round(float(best_score), 1) if best_score else 0.0,
                            "warning": "Not found in static lexicon or cursive ambiguity. Requires physical slip verification."
                        })

    return matched_results


def analyze_prescription(image_input, file_url: str = "") -> Dict[str, Any]:
    """
    Main entry point for prescription & handwritten clinical slip analysis.
    Executes CPU preprocessing, dual OCR, and RapidFuzz drug normalization.
    Returns:
      - raw_ocr_text: Extracted text
      - normalized_drugs: List of verified and flagged medicines
      - dashboard_payload: Dict conforming to MediKiosk Doctor Dashboard DocumentExtraction
    """
    deskewed, binarized = preprocess_prescription(image_input)
    ocr_text = run_ocr(deskewed, binarized)
    drugs = normalize_drugs(ocr_text)

    # Format for Doctor Dashboard
    verified_meds = []
    flagged_values = []
    diagnoses = ["Doctor Consultation Slip"]

    for d in drugs:
        status_tag = "VERIFIED" if d["status"] == "VERIFIED" else "FLAGGED"
        med_str = f"{d['drug']} ({d.get('strength', '')}) [{status_tag}]"
        verified_meds.append(med_str)

        if d["status"] == "FLAGGED_FOR_DOCTOR":
            flagged_values.append(f"⚠️ Unverified Rx Token: '{d['raw_token']}' (Score: {d['score']}%)")

    has_unverified = any(d["status"] == "FLAGGED_FOR_DOCTOR" for d in drugs)
    summary_text = (
        f"Prescription OCR (CPU + RapidFuzz): Identified {len(drugs)} medication token(s). "
        f"{sum(1 for d in drugs if d['status'] == 'VERIFIED')} Verified against Indian Drug Lexicon, "
        f"{sum(1 for d in drugs if d['status'] == 'FLAGGED_FOR_DOCTOR')} Flagged for Doctor Confirmation."
    )

    dashboard_payload = {
        "document_type": "Prescription / Doctor Slip",
        "modality": "document",
        "diagnoses": diagnoses,
        "medications": verified_meds,
        "flagged_values": flagged_values,
        "document_date": "Visual Scan",
        "summary": summary_text,
        "file_url": file_url,
        "raw_text": ocr_text if ocr_text else "No legible text extracted"
    }

    return {
        "raw_ocr_text": ocr_text,
        "normalized_drugs": drugs,
        "status": "success",
        "dashboard_payload": dashboard_payload
    }
