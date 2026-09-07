import subprocess
import tempfile
import os
import json
import time

img_path = os.path.abspath("uploads/304ad8313efd471cad72784ac7f0ce62.webp")

prompt = """You are an expert Chief Medical Officer and Clinical Pharmacologist.
Analyze this doctor prescription image.
Extract the clinical findings and output ONLY valid JSON matching this schema:
{
  "document_type": "Prescription / Doctor Consultation Slip",
  "modality": "document",
  "diagnoses": ["Chief complaint or symptoms"],
  "medications": ["Medication name, strength, dosage"],
  "flagged_values": [],
  "document_date": "YYYY-MM-DD",
  "summary": "Full summary with clinic name, doctor, patient vitals, and prescribed medications"
}
Output ONLY valid JSON."""

with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as pf:
    pf.write(prompt)
    prompt_file = pf.name

try:
    python_exe = os.path.abspath("venv/Scripts/python.exe")
    script_path = os.path.abspath("scripts/extract_doc_vl.py")
    cmd = [python_exe, script_path, "--image", img_path, "--prompt_file", prompt_file, "--max_tokens", "400"]
    
    t0 = time.time()
    print("Calling isolated Qwen 2.5-VL runner...")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    print(f"Runner completed in {round(time.time() - t0, 2)}s (Exit code: {proc.returncode})")
    out = proc.stdout
    if "---OUTPUT_START---" in out and "---OUTPUT_END---" in out:
        result_text = out.split("---OUTPUT_START---")[1].split("---OUTPUT_END---")[0].strip()
        print("\nExtracted Output:")
        print(result_text)
    else:
        print("Raw stdout:", out)
        print("Stderr:", proc.stderr)
finally:
    if os.path.exists(prompt_file):
        os.remove(prompt_file)
