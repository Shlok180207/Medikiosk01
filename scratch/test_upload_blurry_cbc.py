import requests
import io
import time
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# 1. Create a simulated phone camera photo of a CBC & Kidney Function Test report
img = Image.new('RGB', (800, 450), color='#F8F7F3') # slightly warm off-white paper
draw = ImageDraw.Draw(img)

report_lines = [
    "DIAGNOSTIC PATHOLOGY LABORATORY",
    "PATIENT: Sneha Patel | AGE/GENDER: 28 Y / F | DATE: 05-SEP-2026",
    "----------------------------------------------------------------",
    "TEST NAME                     OBSERVED      REFERENCE RANGE     UNITS",
    "Hemoglobin                    8.4           12.0 - 15.5         g/dL",
    "Total Leucocyte Count (WBC)   13200         4000 - 10000        /cumm",
    "RBC Count                     3.3           3.8 - 4.8           mill/cumm",
    "Platelet Count                68000         150000 - 450000     /cumm",
    "Packed Cell Volume (PCV)      27.2          36.0 - 46.0         %",
    "MCV                           76.5          80.0 - 100.0        fL",
    "Serum Creatinine              2.1           0.6 - 1.2           mg/dL",
    "Serum Bilirubin Total         1.9           0.2 - 1.2           mg/dL",
    "----------------------------------------------------------------",
    "CLINICAL IMPRESSION & REMARKS:",
    "Microcytic hypochromic blood picture with thrombocytopenia and leukocytosis.",
    "Rx: Tab Ferrous Ascorbate + Folic Acid OD, IV Ceftriaxone 1g IV BD, Nephrology follow-up."
]

y = 20
for line in report_lines:
    draw.text((30, y), line, fill='#1A1A1A')
    y += 26

# Simulate smartphone photo imperfections: slight camera defocus blur + contrast drop
blurry = img.filter(ImageFilter.GaussianBlur(radius=1.1))
blurry = ImageEnhance.Contrast(blurry).enhance(0.85)

buf = io.BytesIO()
blurry.save(buf, format='PNG')
image_bytes = buf.getvalue()

print(f"Generated blurry CBC report image: {len(image_bytes)} bytes")

# 2. Upload to the FastAPI endpoint for PT-1EFC
files = {'file': ('blurry_cbc_report.png', image_bytes, 'image/png')}
data = {'patient_id': 'PT-1EFC'}

resp = requests.post("http://localhost:8000/api/process-document", files=files, data=data)
print("Upload response:", resp.status_code, resp.json())

# Wait 8 seconds for background worker to process through OCR and Qwen
print("Waiting for background AI processing...")
time.sleep(12)

# Check patient record
summary_resp = requests.get("http://localhost:8000/api/patient-summary?patient_id=PT-1EFC")
print("Status:", summary_resp.status_code)
data = summary_resp.json()
print("\n=== UPDATED PATIENT FLAG LAB VALUES ===")
import json
flagged_raw = data.get('patient', {}).get('flagged_lab_values')
if flagged_raw:
    parsed = json.loads(flagged_raw)
    for doc in parsed:
        print("Doc Type:", doc.get('document_type'), "| Modality:", doc.get('modality'))
        print("Diagnoses:", doc.get('diagnoses'))
        print("Medications:", doc.get('medications'))
        print("Flagged Values:", doc.get('flagged_values'))
        print("Summary:", doc.get('summary')[:100], "...\n")
