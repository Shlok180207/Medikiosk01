import requests
import io
import time
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# 1. Create a simulated phone photo of a Liver Function Test (LFT)
img = Image.new('RGB', (800, 420), color='#FBFBFA')
draw = ImageDraw.Draw(img)

lft_lines = [
    "APOLLO DIAGNOSTICS - HEPATIC FUNCTION PANEL",
    "PATIENT: Rajesh Kumar | AGE/GENDER: 42 Y / M | DATE: 05-SEP-2026",
    "----------------------------------------------------------------",
    "TEST NAME                     OBSERVED      REFERENCE INTERVAL  UNITS",
    "Total Bilirubin               3.8           0.2 - 1.2           mg/dL",
    "Direct Bilirubin              2.4           0.0 - 0.3           mg/dL",
    "SGOT / AST                    145           10 - 40             U/L",
    "SGPT / ALT                    182           10 - 45             U/L",
    "Alkaline Phosphatase (ALP)    320           40 - 130            U/L",
    "Total Protein                 6.1           6.4 - 8.3           g/dL",
    "Serum Albumin                 2.9           3.5 - 5.2           g/dL",
    "----------------------------------------------------------------",
    "IMPRESSION: Acute hepatocellular injury with cholestatic jaundice.",
    "Rx: Tab Ursodeoxycholic acid 300mg BD, IV Fluids, avoid hepatotoxic drugs."
]

y = 20
for line in lft_lines:
    draw.text((30, y), line, fill='#222222')
    y += 28

# Add phone camera defocus blur + contrast degradation
blurry = img.filter(ImageFilter.GaussianBlur(radius=1.2))
blurry = ImageEnhance.Contrast(blurry).enhance(0.85)

buf = io.BytesIO()
blurry.save(buf, format='PNG')
image_bytes = buf.getvalue()

# 2. Upload to FastAPI
files = {'file': ('blurry_lft_report.png', image_bytes, 'image/png')}
data = {'patient_id': 'PT-1EFC'}

resp = requests.post("http://localhost:8000/api/process-document", files=files, data=data)
print("Upload status:", resp.status_code, resp.json().get('status'))

print("Waiting for background AI processing...")
time.sleep(15)

# 3. Check updated patient record
r = requests.get("http://localhost:8000/api/patient-summary?patient_id=PT-1EFC").json()
import json
flagged = json.loads(r.get('flagged_lab_values', '[]'))
latest_doc = flagged[-1]
print("\n=== LATEST DOCUMENT EXTRACTION ===")
print("Document Type:", latest_doc.get('document_type'))
print("Modality:", latest_doc.get('modality'))
print("Diagnoses:", latest_doc.get('diagnoses'))
print("Medications:", latest_doc.get('medications'))
print("Flagged Values:")
for v in latest_doc.get('flagged_values', []):
    print("  -", v)
print("Summary:", latest_doc.get('summary'))
