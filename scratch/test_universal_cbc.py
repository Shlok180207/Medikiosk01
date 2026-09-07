import ollama

# Sample messy, slightly noisy OCR text of a CBC report (such as might come from a blurry smartphone photo)
blurry_cbc_ocr = """
METROPOLIS HEALTHCARE LABS
PATIENT: Anjali Verma, 34 Y / Female
DATE: 04/09/2026
REF BY: Dr. R. K. Gupta

COMPLETE BLOOD COUNT (CBC) REPORT
Test Description            Observed Value    Reference Interval    Unit
Hemoglobin                  8.2               12.0 - 15.5           g/dL
Total Leucocyte Count (WBC) 14500             4000 - 10000          /cumm
RBC Count                   3.4               3.8 - 4.8             mill/cumm
Platelet Count              65000             150000 - 450000       /cumm
PCV / Packed Cell Volume    26.4              36.0 - 46.0           %
MCV                         77.6              80.0 - 100.0          fL
MCH                         24.1              27.0 - 32.0           pg
MCHC                        31.0              32.0 - 36.0           g/dL

DIFFERENTIAL LEUCOCYTE COUNT (DLC)
Neutrophils                 78                40 - 70               %
Lymphocytes                 16                20 - 40               %
Eosinophils                 03                01 - 06               %
Monocytes                   03                02 - 08               %
Basophils                   00                00 - 01               %

PERIPHERAL BLOOD SMEAR:
RBCs show moderate microcytosis and hypochromia with mild anisopoikilocytosis.
WBCs show neutrophilic leukocytosis with toxic granulation.
Platelets reduced on smear (Thrombocytopenia).
"""

prompt = f"""You are an expert Chief Medical Officer and Clinical Pathologist.
Analyze this medical laboratory diagnostic report (CBC, Biochemistry, LFT, KFT, Lipid, PFT, Urine, or Biopsy).

CLINICAL REASONING RULES:
1. Examine EVERY test parameter and compare the Observed Value against the Reference Interval.
2. If ANY parameter is BELOW the lower reference limit (e.g. Low Hemoglobin, Low Platelets, Low MCV) or ABOVE the upper reference limit (e.g. High WBC / Leukocytosis, High Neutrophils), you MUST list it in 'flagged_values' with the observed value, unit, and whether it is Low or High.
3. Formulate the primary clinical diagnoses or hematological/pathological impressions based on the abnormal findings (e.g. Microcytic Hypochromic Anemia, Neutrophilic Leukocytosis / Bacterial Infection, Thrombocytopenia).
4. Extract any prescribed medications or clinical recommendations.
5. Provide a clear, actionable clinical summary highlighting key urgent risks.

Report Content:
{blurry_cbc_ocr}

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
print('=== CBC EXTRACTION RESULT ===')
print(res['message']['content'])
