import os
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

img_path = os.path.abspath("uploads/304ad8313efd471cad72784ac7f0ce62.webp")
print(f"Testing Qwen 2.5-VL (3B) PyTorch inference on: {img_path}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Choose bfloat16 for RTX 40-series (Ada Lovelace natively supports bfloat16)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using torch_dtype: {dtype}")

t0 = time.time()
print("Loading model and processor from HuggingFace...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=dtype,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print(f"Model loaded in {round(time.time() - t0, 2)}s")

prompt = """You are an expert Chief Medical Officer and Clinical Pharmacologist.
Analyze this doctor prescription / clinical consultation slip and extract:
1. Clinic Name & Doctor Name
2. Patient Name, Age, Gender, Date
3. Recorded Vitals (BP, Pulse, Temperature, SpO2, etc.)
4. Patient Symptoms / Chief Complaints
5. Prescribed Medications (name, strength/dosage, frequency, duration, instructions)
6. Any other clinical advice or follow-up instructions.

Output structured clinical findings clearly."""

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
)
inputs = inputs.to("cuda")

print(f"Preprocessing completed in {round(time.time() - t1, 2)}s. Generating response...")
t2 = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

total_gen_time = round(time.time() - t2, 2)
print(f"Generation completed in {total_gen_time}s!")
print("\n" + "="*50)
print("EXTRACTED PRESCRIPTION CONTENT:")
print("="*50)
print(output_text)
print("="*50)
