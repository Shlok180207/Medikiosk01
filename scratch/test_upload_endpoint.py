import time
import requests
import json

url = "http://127.0.0.1:8000/api/process-document"
patient_id = "PT-BB39"
img_path = "uploads/304ad8313efd471cad72784ac7f0ce62.webp"

print(f"Uploading {img_path} for {patient_id} to {url}...")
with open(img_path, "rb") as f:
    files = {"file": ("prescription_test.webp", f, "image/webp")}
    data = {"patient_id": patient_id}
    res = requests.post(url, files=files, data=data)

print("Immediate response status:", res.status_code)
print("Immediate response payload:", res.json())

print("\nWaiting for background processing with Qwen 2.5-VL to complete...")
for i in range(45):
    time.sleep(3)
    p_res = requests.get(f"http://127.0.0.1:8000/api/patient-summary?patient_id={patient_id}")
    if p_res.status_code == 200:
        p_data = p_res.json()
        val = p_data.get("flagged_lab_values", "[]")
        docs = json.loads(val) if isinstance(val, str) else val
        if docs and isinstance(docs, list):
            latest_doc = docs[-1]
            if isinstance(latest_doc, dict) and latest_doc.get("document_type") != "Processing...":
                print(f"\n[OK] Document Processed Successfully in {(i+1)*3}s!")
                print("="*60)
                print("Document Type:", latest_doc.get("document_type"))
                print("Diagnoses:", latest_doc.get("diagnoses"))
                print("Medications:", latest_doc.get("medications"))
                print("Flagged Values:", latest_doc.get("flagged_values"))
                print("Summary:", latest_doc.get("summary"))
                print("="*60)
                break
    print(f"[{i*3}s] Background task working...")
