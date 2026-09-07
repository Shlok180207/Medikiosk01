import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import time
import json
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

def run_qwen_vl_inference(image_bytes: bytes, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
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
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

with open("uploads/304ad8313efd471cad72784ac7f0ce62.webp", "rb") as f:
    img_bytes = f.read()

prompt = """You are an expert Chief Medical Officer and Clinical Pharmacologist.
Analyze this doctor prescription or medical document image.
Extract the clinical findings and output ONLY valid JSON matching this exact JSON schema:

```json
{
  "document_type": "Prescription / Doctor Consultation Slip",
  "modality": "document",
  "diagnoses": ["Chief complaint, symptoms, or diagnoses"],
  "medications": ["Medication name, dose, frequency, duration"],
  "flagged_values": [],
  "document_date": "YYYY-MM-DD or date printed",
  "summary": "Full summary with clinic name, doctor name, patient name/age/gender, vitals (BP, HR, SpO2, Temp), and prescribed medications"
}
```

CLINICAL SAFETY RULES:
1. For prescriptions, set 'flagged_values': [] (prescriptions do not have lab reference ranges).
2. Do NOT hallucinate unmentioned diseases (like Tuberculosis or Cancer). Only extract symptoms/complaints actually written.
3. In 'summary', write the actual clinic name, doctor name, patient details, vitals, and medications.
4. Output ONLY valid JSON."""

t0 = time.time()
res = run_qwen_vl_inference(img_bytes, prompt, max_new_tokens=400)
print(f"Extraction finished in {round(time.time() - t0, 2)}s")
print(res)
