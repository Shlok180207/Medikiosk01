import ollama

# Simulating what Tesseract produces from a blurry smartphone camera photo of a CBC + Biochemistry report:
# Notice: decimals are dropped!
# - "82 g/dL" instead of "8.2"
# - "34 mill/cumm" instead of "3.4"
# - "12 mg/dL" instead of "1.2"
# - "24 mg/dL" instead of "2.4"
# - "776 fL" instead of "77.6"
# - Reference intervals: "120 - 155 g/dL" instead of "12.0 - 15.5"
blurry_dropped_decimals_ocr = """
PATHOLOGY & CLINICAL LABS
PATIENT: Rahul Sharma, 45 Y / Male
DATE: 05/09/2026

COMPLETE BLOOD COUNT & RENAL PROFILE
Test Description            Observed Value    Reference Interval    Unit
Hemoglobin                  82                120 - 155             g/dL
Total Leucocyte Count (WBC) 13800             4000 - 10000          /cumm
RBC Count                   34                38 - 48               mill/cumm
Platelet Count              55000             150000 - 450000       /cumm
MCV                         776               800 - 1000            fL
Serum Creatinine            24                06 - 12               mg/dL
Serum Bilirubin (Total)     18                02 - 12               mg/dL

PHYSICIAN NOTES:
Patient presenting with high fever, severe pallor, petechiae on lower limbs, and oliguria.
Rx: Tab Cefixime 200mg BD, IV Fluids, urgent nephrology and hematology consultation.
"""

prompt = f"""You are an expert Chief Medical Officer and Clinical Pathologist.
Analyze this medical laboratory diagnostic report (CBC, Renal Profile, LFT, etc.).

CRITICAL CLINICAL & OCR RESILIENCE RULES:
1. DECIMAL RECONSTRUCTION & PHYSIOLOGICAL SANITY:
   - Smartphone photos of paper reports frequently drop 1-pixel decimal points during OCR.
   - For example:
     * Hemoglobin '82' with unit 'g/dL' or range '120 - 155' represents 8.2 g/dL (normal 12.0 - 15.5 g/dL).
     * RBC '34' with unit 'mill/cumm' represents 3.4 mill/cumm (normal 3.8 - 4.8 mill/cumm).
     * MCV '776' with unit 'fL' represents 77.6 fL (normal 80.0 - 100.0 fL).
     * Serum Creatinine '24' with unit 'mg/dL' represents 2.4 mg/dL (normal 0.6 - 1.2 mg/dL).
     * Serum Bilirubin '18' with unit 'mg/dL' represents 1.8 mg/dL (normal 0.2 - 1.2 mg/dL).
     * FEV1/FVC lung volumes '472' or '197' in Liters represent 4.72 L and 1.97 L.
   - ALWAYS reconstruct the physiologically authentic decimal value based on standard human physiological units. NEVER output physiologically impossible values like 82 g/dL hemoglobin or 24 mg/dL creatinine!

2. ACCURATE EVALUATION AGAINST REFERENCE INTERVALS:
   - Compare every measured value against normal clinical reference limits.
   - Flag EVERY abnormal parameter (e.g. Low Hemoglobin, Leukocytosis / High WBC, Thrombocytopenia / Low Platelets, Elevated Creatinine / Renal Impairment, Hyperbilirubinemia).
   - In 'flagged_values', output the corrected physiological value, proper unit, and whether it is Low or High.

3. CLINICAL DIAGNOSES:
   - Identify primary clinical syndromes and conditions (e.g. Microcytic Anemia, Acute Kidney Injury / Renal Impairment, Leukocytosis / Sepsis, Thrombocytopenia).

4. MEDICATIONS:
   - Extract prescribed medications with dosage and frequency.

Document Content:
{blurry_dropped_decimals_ocr}

Extract into JSON:
{{
  "document_type": "string",
  "modality": "document",
  "diagnoses": ["string"],
  "medications": ["string"],
  "flagged_values": ["string"],
  "document_date": "string",
  "summary": "string"
}}
"""

res = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.1})
print('=== DROPPED DECIMALS CBC/RENAL RESULT ===')
print(res['message']['content'])
