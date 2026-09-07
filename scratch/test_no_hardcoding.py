import ollama
import json

raw_pft_text = """
Credit Valley Pulmonary Diagnostics
PFT Lab Report
Date: 10/09/15
Age 19 Gender: Male
Wt 70.8 (Kg) Ht 187 (cm) BMI: 20.28
Diagnosis: Asthma
Medication: post inhalation of 4 puffs of Salbutamol

Spirometry Ref (Normal Range) Pre %Ref Post %Ref %Chg
FVC Liters (4.4 - 6.5) 5.47 98 5.22 93 -5
FEV1 Liters (3.8 - 5.6) 4.72 98 4.38 91 -7
FEV1/FVC % (82 - 97) 38 46 42 51 +9
FEF25-75% L/sec (3.6 - 7.3) 3.83 70 5.28 96 +38
FEF25% L/sec (5.8 - 11.0) 8.46 95 10.33 116 +22
FEF50% L/sec (4.3 - 8.7) 4.31 64 6.22 92 +44
FEF75% L/sec (1.7 - 4.7) 1.22 41 1.71 58 +40
PEF L/sec (7.2 - 14.1) 10.51 88 14.43 121 +37

Lung Volumes
TLC Liters (5.5 - 7.9) 7.52 105
VC Liters (4.4 - 6.5) 5.55 100
IC Liters (3.1 - 4.9) 3.75 92
FRCPL Liters (2.3 - 4.1) 3.76 115
ERV Liters (1.7 - 2.4) 1.72 85
RV Liters (0.7 - 1.8) 1.97 121
"""

raw_cbc_text = """
DIAGNOSTIC PATHOLOGY LABORATORY
PATIENT: Sneha Patel | AGE/GENDER 28 Y/F | DATE 05-SEP-2026

TEST NAME                     OBSERVED      REFERENCE RANGE     UNITS
Hemoglobin                    8.4           12.0 - 15.5         g/dL
Total Leucocyte Count (WBC)   13200         4000 - 10000        /cumm
RBC Count                     3.3           3.8 - 4.8           mill/cumm
Platelet Count                68000         150000 - 450000     /cumm
Packed Cell Volume (PCV)      27.2          36.0 - 46.0         %
MCV                           76.5          80.0 - 100.0        fL
Serum Creatinine              2.1           0.6 - 1.2           mg/dL
Serum Bilirubin Total         1.9           0.2 - 1.2           mg/dL

CLINICAL IMPRESSION & REMARKS:
Microcytic hypochromic blood picture with thrombocytopenia and leukocytosis.
Rx: Tab Ferrous Ascorbate + Folic Acid OD, IV Ceftriaxone 1g IV BD, Nephrology follow-up.
"""

# Universal Clinical Prompt - ZERO hardcoded test names, ZERO hardcoded ranges
def run_clinical_structuring(doc_text, doc_name):
    prompt = f"""You are an expert Chief Medical Officer and Clinical Pathologist.
Analyze the following medical diagnostic report text and extract structured clinical findings into JSON.

CLINICAL DECISION SUPPORT RULES (APPLIES UNIVERSALLY TO ALL MEDICAL REPORTS):
1. REFERENCE RANGE EVALUATION:
   - For every test parameter, compare the observed measured value against the provided reference/normal range.
   - If any parameter is below the lower reference limit, flag it as '(Low)'.
   - If any parameter is above the upper reference limit, flag it as '(High)'.
   - If any parameter shows significant post-medication reversibility or clinical abnormality, flag it.
   - In 'flagged_values', list each abnormal finding with the test name, measured value, units, status (Low or High), and reference range.
   - CRITICAL SAFETY: NEVER state "results are within normal limits" if abnormal values or active diagnoses exist!

2. CLINICAL DIAGNOSES:
   - Extract all stated diagnoses, pathological impressions, or clinical syndromes (e.g. Asthma, Microcytic Anemia, Thrombocytopenia, Leukocytosis).

3. MEDICATIONS & DOSAGES:
   - Extract all prescribed medications, inhalers, dosages, route, and frequency.

4. ACCURACY & DECIMAL INTEGRITY:
   - Always preserve exact numbers, decimals, and units as printed in the document.

Document Content:
{doc_text}

Output ONLY valid JSON matching this schema:
{{
  "document_type": "string (e.g. PFT Report, CBC Report, Renal Function Test, Prescription)",
  "modality": "document",
  "diagnoses": ["string"],
  "medications": ["string"],
  "flagged_values": ["string"],
  "document_date": "string",
  "summary": "string"
}}
"""
    res = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.1})
    return res['message']['content']

print("=== TESTING PFT WITHOUT ANY HARDCODED REGEX ===")
pft_out = run_clinical_structuring(raw_pft_text, "PFT")
print(pft_out)

print("\n=== TESTING CBC WITHOUT ANY HARDCODED REGEX ===")
cbc_out = run_clinical_structuring(raw_cbc_text, "CBC")
print(cbc_out)
