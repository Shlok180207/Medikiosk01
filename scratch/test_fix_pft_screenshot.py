import ollama
import json

# Exact OCR text extracted from the user's uploaded PFT file (uploads/660530fccef843958f1006549932eab6.jpeg)
ocr_pft_text = """
Credit Valley Pulmonary Diagnostics
2300 Eglinton Avenue West, Suite 512,
Mississauga, ON L5M 2V8

PFT Lab Report
Diagnosis: Asthma
Dyspnea Rest: No Dyspnea Exercise: No Cough: No Persistent: No
Date: 10/09/15
Age: 19  Gender: Male  Wt: 70.8 (Kg)  Ht: 187 (cm)  BMI: 20.28
Race: East Indian/Arab
Test Done By: David Grisebach RRT
Physician: Dr. Emad Amer

Spirometry Ref (Normal Range) Pre %Ref Post %Ref %Chg
FVC Liters 5.44 (44-65) 547 101 522 96 -5
FEV1 Liters 4.72 (38-56) 438 93 473 100 +8
FEV1/FVC % 88 (78.2 - 97.2) 80 90
FEF25-75% L/sec 3.44 (3.6 - 7.3) 383 70 528 97 +38
FEF25% L/sec 8.40 (5.8 - 11.0) 1033 123 1042 124 +1
FEF50% L/sec 6.22 (5.7 - 6.7) 439 71 614 99 +40
FEF75% L/sec 3.21 (1.7 - 4.7) 227 71 313 98 +37
PEF L/sec 11.13 (8.2 - 14.1) 1051 94 1042 94 -1

Lung Volumes
TLC Liters 6.53 (55-79) 752 115
VC Liters 5.44 (44-65) 555 102
IC Liters 4.07 (33-48) 376 92
FRCPL Liters 3.95 (33-46) 376 95
ERV Liters 2.03 (17-24) 172 84
RV Liters 1.24 (07-18) 197 159

Medication: Post inhalation of 4 puffs of Salbutamol.
"""

prompt = f"""You are an expert Pulmonologist, Chief Medical Officer, and Clinical Pathologist.
Analyze this medical Pulmonary Function Test (PFT) diagnostic report.

CRITICAL CLINICAL PHYSIOLOGY & DECIMAL RECONSTRUCTION RULES:
1. Low-resolution scans and small-font tables drop 1-pixel decimal points during OCR.
2. In human respiratory physiology, adult lung volumes and flow rates (FVC, FEV1, FEF, PEF, TLC, VC, RV) are ALWAYS on the scale of 0.5 to 8.0 Liters or L/sec:
   - '547' in Liters is ALWAYS 5.47 L (a human chest CANNOT hold 547 Liters!).
   - '473' or '438' in Liters is ALWAYS 4.73 L or 4.38 L.
   - '383' or '528' in L/sec is ALWAYS 3.83 L/sec or 5.28 L/sec.
   - '1033' or '1051' in L/sec is ALWAYS 10.33 L/sec or 10.51 L/sec.
   - '752' in Liters is 7.52 L; '555' is 5.55 L; '197' is 1.97 L; '172' is 1.72 L.
   - Reference intervals like '(44-65)', '(38-56)', '(55-79)', '(17-24)', '(07-18)' represent 4.4 - 6.5 L, 3.8 - 5.6 L, 5.5 - 7.9 L, 1.7 - 2.4 L, and 0.7 - 1.8 L.
3. NEVER output physiologically absurd values like '547 L' or '473 L' or 'Ref: 64-65'! Always restore the true physiological decimals.

4. PARAMETER EVALUATION AGAINST REFERENCE INTERVALS:
   - Compare measured values against their reference ranges.
   - If any parameter is below the lower reference limit (e.g. FEV1/FVC ratio 80% vs 82-97%, FEF25-75% 3.83 vs 3.6-7.3, FEF50% 4.39 vs 5.7-6.7), flag it as '(Low)'.
   - If any parameter is above the upper reference limit (e.g. RV 1.97 L vs 0.7 - 1.8 L), flag it as '(High)'.
   - If any parameter shows significant bronchodilator reversibility (e.g. FEF25-75% +38% improvement post-Salbutamol, FEF50% +40%), highlight this clinical response.
   - In 'flagged_values', list each finding with test name, measured value, units, status (Low/High), and reference range.

5. DIAGNOSES & MEDICATIONS:
   - Extract primary diagnosis (Asthma) and medications (Salbutamol 4 puffs).

Document Text:
{ocr_pft_text}

Output ONLY valid JSON:
{{
  "document_type": "PFT Lab Report",
  "modality": "document",
  "diagnoses": ["string"],
  "medications": ["string"],
  "flagged_values": ["string"],
  "document_date": "string",
  "summary": "string"
}}
"""

res = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.1})
print("=== QWEN-2.5 7B RECONSTRUCTION RESULT ===")
print(res['message']['content'])
