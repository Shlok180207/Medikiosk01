import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json
import re
import tempfile
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

_qwen_vl_model = None
_qwen_vl_processor = None

def get_qwen_vl():
    global _qwen_vl_model, _qwen_vl_processor
    if _qwen_vl_model is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        print("⚡ Loading Qwen2.5-VL-3B-Instruct on CUDA...")
        _qwen_vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=dtype,
            device_map="cuda"
        )
        _qwen_vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("[OK] Qwen2.5-VL-3B-Instruct ready on CUDA.")
    return _qwen_vl_model, _qwen_vl_processor

def run_qwen_vl_extraction(image_bytes: bytes, prompt: str, max_new_tokens: int = 512) -> dict:
    model, processor = get_qwen_vl()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON
        clean_text = output_text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        elif clean_text.startswith("{") and clean_text.endswith("}"):
            pass
        else:
            first_brace = clean_text.find("{")
            last_brace = clean_text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                clean_text = clean_text[first_brace:last_brace+1]

        data = json.loads(clean_text)
        data["raw_generation"] = output_text
        return data
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# Safety Guardrail (from main.py)
def enforce_clinical_safety_guardrail(data: dict, raw_context: str = "") -> dict:
    doc_type = str(data.get("document_type", "")).lower()
    raw_lower = (raw_context + " " + json.dumps(data)).lower()

    is_prescription = any(k in doc_type for k in ["prescription", "consultation slip", "rx", "doctor slip", "clinic visit"]) or \
                      (any(k in raw_lower for k in ["dr.", "clinic", "rx", "hospital", "prescription", "consultant"]) and
                       not any(k in raw_lower for k in ["reference range", "lipid profile", "cbc", "spirometry", "kft", "lft"]))

    if is_prescription:
        data["document_type"] = "Prescription / Doctor Consultation Slip"
        data["modality"] = "document"
        data["flagged_values"] = []

        unmentioned_high_stakes = [
            "tuberculosis", "cancer", "carcinoma", "malignancy", "leukemia",
            "lymphoma", "asthma", "fev1", "copd", "hiv", "hepatitis", "infarction",
            "stroke", "fracture"
        ]
        cleaned_diagnoses = []
        for diag in data.get("diagnoses", []):
            d_lower = str(diag).lower()
            if any(term in d_lower for term in unmentioned_high_stakes):
                if not any(term in raw_lower for term in unmentioned_high_stakes):
                    continue
            cleaned_diagnoses.append(diag)
        data["diagnoses"] = cleaned_diagnoses

        safety_advisory = "⚠️ Clinical Safety Notice: Cursive doctor handwriting requires pharmacist verification before dispensing medications."
        existing_summary = data.get("summary", "").strip()
        if safety_advisory not in existing_summary:
            data["summary"] = f"{existing_summary} {safety_advisory}".strip() if existing_summary else safety_advisory

    return data


with open("uploads/304ad8313efd471cad72784ac7f0ce62.webp", "rb") as f:
    file_bytes = f.read()

prompt = """You are an expert Chief Medical Officer and Clinical Pharmacologist.
Analyze this medical document or doctor prescription image.
Transcribe and extract the clinical details into structured JSON:

```json
{
  "document_type": "Prescription / Doctor Consultation Slip",
  "modality": "document",
  "diagnoses": ["Chief symptoms, complaints, or diagnoses written on the slip"],
  "medications": ["Each prescribed medicine with strength/dosage, frequency, and duration"],
  "flagged_values": [],
  "document_date": "Date printed on document (YYYY-MM-DD or DD/MM/YYYY)",
  "summary": "Full clinical summary stating clinic name, doctor name, patient name, age, gender, vital signs (BP, HR, Temp, SpO2), and list of prescribed medicines"
}
```

CLINICAL SAFETY MANDATES:
1. For prescriptions, 'flagged_values' MUST be an empty list []. Do not extract lab reference ranges from a prescription!
2. Do NOT hallucinate unmentioned diseases (like Tuberculosis, Cancer, COPD, Asthma). Extract only written symptoms/complaints.
3. In 'summary', provide actual clinic, doctor, patient vitals, and medications.
4. Output ONLY valid JSON."""

result = run_qwen_vl_extraction(file_bytes, prompt, max_new_tokens=450)
safe_result = enforce_clinical_safety_guardrail(result, result.get("raw_generation", ""))

print("="*60)
print("FINAL ENFORCED CLINICAL EXTRACTION:")
print("="*60)
print(json.dumps(safe_result, indent=2))
