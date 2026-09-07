import sys, os
sys.path.insert(0, os.path.abspath('.'))
from main import process_document_background, SessionLocal, PatientRecord
import json

db = SessionLocal()
patient = db.query(PatientRecord).filter(PatientRecord.patient_id == "PT-0FEF").first()
print("Found patient:", patient.patient_id, patient.patient_name, "id in db:", patient.id)

# Read the uploaded file
image_path = "uploads/7c6b94633fe644859f473cf19b6b81ca.jpeg"
with open(image_path, "rb") as f:
    file_bytes = f.read()

# Clear the old flawed extraction from flagged_lab_values
patient.flagged_lab_values = "[]"
db.commit()

# Re-run processing with the new physiological decimal restoration prompt
print("Processing document with updated physiological rules...")
process_document_background(
    file_bytes=file_bytes,
    filename="asthma.jpeg",
    content_type="image/jpeg",
    file_url=f"http://localhost:8000/{image_path}",
    patient_id_db=patient.id
)

db.refresh(patient)
docs = json.loads(patient.flagged_lab_values)
print("\n=== RE-EXTRACTED PFT RECORD ===")
print(json.dumps(docs, indent=2))
