import os
import time
import json
import torch
from PIL import Image
import io
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

img_path = os.path.abspath("uploads/304ad8313efd471cad72784ac7f0ce62.webp")
print(f"Testing Qwen 2.5-VL (3B) Direct JSON Extraction on: {img_path}")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

t0 = time.time()
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=dtype,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print(f"Loaded in {round(time.time() - t0, 2)}s")

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
  "summary": "Concise summary including clinic name, doctor name, patient name/age/gender, vitals (BP, HR, SpO2, Temp), and prescribed medications"
}
```

CLINICAL SAFETY RULES:
1. For prescriptions, set 'flagged_values': [] (prescriptions do not have lab reference ranges).
2. Do NOT hallucinate unmentioned diseases (like Tuberculosis or Cancer). Only extract symptoms/complaints actually written.
3. Output ONLY the JSON object, nothing else."""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": prompt}
        ]
    }
]

t1 = time.time()
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

t2 = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

gen_time = round(time.time() - t2, 2)
print(f"Generation completed in {gen_time}s!")
print("\n" + "="*50)
print("RAW GENERATED OUTPUT:")
print("="*50)
print(output_text)
print("="*50)

# Validate JSON parsing
try:
    # Strip markdown fences if present
    clean_text = output_text.strip()
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_text:
        clean_text = clean_text.split("```")[1].split("```")[0].strip()
    parsed = json.loads(clean_text)
    print("SUCCESSFULLY PARSED JSON:")
    print(json.dumps(parsed, indent=2))
except Exception as e:
    print("JSON Parse Error:", e)
