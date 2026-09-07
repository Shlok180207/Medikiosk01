import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from faster_whisper import WhisperModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

print("Testing Whisper + Qwen 2.5-VL Coexistence in 8GB VRAM...")
print("1. Loading Faster-Whisper large-v3 on CUDA (int8_float16)...")
whisper = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
print("[OK] Whisper loaded.")

print(f"Allocated after Whisper: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
print(f"Reserved after Whisper: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("2. Loading Qwen2.5-VL-3B-Instruct on CUDA...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=dtype,
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print("[OK] Qwen 2.5-VL loaded successfully alongside Whisper!")
    print(f"Allocated total: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Reserved total: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
except Exception as e:
    print("[ERROR] Failed with exception:", e)

