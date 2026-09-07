import re

def repair_medical_ocr_decimals(text: str) -> str:
    """
    Clinically restores dropped decimal points from low-resolution medical tables
    across all clinical domains (CBC, LFT, KFT, Lipid, Thyroid, PFT, Spirometry).
    """
    lines = text.split('\n')
    repaired_lines = []

    # Common medical laboratory units & test markers
    LAB_UNITS = [
        'G/DL', 'GM/DL', 'MG/DL', 'UG/DL', 'FL', 'PG', '/CUMM', 'MILL/CUMM', 'MIL/CUMM', 
        '/UL', 'UIU/ML', 'MIU/ML', 'MMOL/L', 'MEQ/L', 'IU/L', 'U/L', 'NG/ML', 'PG/ML', 
        'LITERS', 'L/SEC', 'LSEC', 'CMH2O', 'SEC', 'MM/HR', '%'
    ]
    LAB_KEYWORDS = [
        'HEMOGLOBIN', 'HB', 'WBC', 'RBC', 'LEUCOCYTE', 'PLATELET', 'MCV', 'MCH', 'MCHC', 'PCV', 
        'HEMATOCRIT', 'CREATININE', 'BILIRUBIN', 'SGOT', 'SGPT', 'ALT', 'AST', 'ALP', 'UREA', 
        'BUN', 'URIC', 'CHOLESTEROL', 'TRIGLYCERIDE', 'HDL', 'LDL', 'VLDL', 'TSH', 'T3', 'T4', 
        'GLUCOSE', 'HBA1C', 'CALCIUM', 'SODIUM', 'POTASSIUM', 'CHLORIDE', 
        'FEV', 'FVC', 'FEF', 'PEF', 'TLC', 'RV', 'DIFFUSING', 'SPIROMETRY', 'PFT'
    ]

    for line in lines:
        upper = line.upper()
        is_lab_or_pft = any(u in upper for u in LAB_UNITS) or any(k in upper for k in LAB_KEYWORDS)

        if is_lab_or_pft:
            # 1. Repair bracketed ranges e.g. (44-65), (38-56), (06-12), (02-12), (120-155), (800-1000), (244-364)
            def fix_range(m):
                a, b = m.group(1), m.group(2)
                # If a starts with '0' e.g. 06 -> 0.6, 02 -> 0.2, 07 -> 0.7
                if len(a) == 2 and '.' not in a:
                    if a[0] == '0':
                        a = f"0.{a[1]}"
                    elif any(u in upper for u in ['LITERS', 'L/SEC', 'LSEC', 'MILL/CUMM', 'MIL/CUMM']):
                        a = f"{a[0]}.{a[1]}"
                elif len(a) == 3 and '.' not in a:
                    # e.g. 120 -> 12.0 for g/dL, or 800 -> 80.0 for fL, or 244 -> 2.44 for Liters
                    if any(u in upper for u in ['LITERS', 'L/SEC', 'LSEC']):
                        a = f"{a[0]}.{a[1:]}"
                    else:
                        a = f"{a[:2]}.{a[2]}"
                elif len(a) == 4 and '.' not in a and any(u in upper for u in ['FL', '%']):
                    # 800-1000 for fL -> 80.0 - 100.0
                    a = f"{a[:2]}.{a[2:]}"

                if len(b) == 2 and '.' not in b:
                    if b[0] == '0':
                        b = f"0.{b[1]}"
                    elif any(u in upper for u in ['LITERS', 'L/SEC', 'LSEC', 'MILL/CUMM', 'MIL/CUMM']):
                        b = f"{b[0]}.{b[1]}"
                elif len(b) == 3 and '.' not in b:
                    if any(u in upper for u in ['LITERS', 'L/SEC', 'LSEC']):
                        if b.startswith(('10', '11', '12', '13', '14')):
                            b = f"{b[:2]}.{b[2]}"
                        else:
                            b = f"{b[0]}.{b[1:]}"
                    else:
                        b = f"{b[:2]}.{b[2]}"
                elif len(b) == 4 and '.' not in b and any(u in upper for u in ['FL', '%', 'G/DL']):
                    b = f"{b[:3]}.{b[3]}" if b.startswith('100') else f"{b[:2]}.{b[2:]}"

                return f"({a} - {b})"

            line = re.sub(r'[\(\[]\s*([0-9]{2,4})\s*[-~]\s*([0-9]{2,4})\s*[\)\]]', fix_range, line)

            # 2. Repair standalone missing decimals for known unit contexts
            tokens = line.split()
            new_tokens = []
            for t in tokens:
                # PFT volumes in Liters or L/sec (472 -> 4.72, 197 -> 1.97, 438 -> 4.38, 383 -> 3.83)
                if any(u in upper for u in ['LITERS', 'L/SEC', 'LSEC', 'FEV', 'FVC', 'FEF']):
                    if t.isdigit() and len(t) == 3 and t not in ['100', '200', '300']:
                        new_tokens.append(f"{t[0]}.{t[1:]}")
                    elif t.isdigit() and len(t) == 4 and t.startswith(('10', '11', '12', '13', '14', '15')):
                        new_tokens.append(f"{t[:2]}.{t[2:]}")
                    else:
                        new_tokens.append(t)
                else:
                    new_tokens.append(t)
            line = ' '.join(new_tokens)

        repaired_lines.append(line)
    return '\n'.join(repaired_lines)


# Test cases:
test_pft = """
FVC (Liters) 472 (38-56)
FEV1 (Liters) 197 (244-364)
FEF 25-75 (L/sec) 124 (17-24)
"""

test_cbc = """
Hemoglobin 8.2 (120-155) g/dL
Total Leucocyte Count (WBC) 14500 (4000-10000) /cumm
RBC Count 3.4 (38-48) mill/cumm
MCV 77.6 (800-1000) fL
Serum Creatinine 2.4 (06-12) mg/dL
Serum Bilirubin Total 1.8 (02-12) mg/dL
"""

print("=== REPAIRED PFT ===")
print(repair_medical_ocr_decimals(test_pft))
print("\n=== REPAIRED CBC & BIOCHEMISTRY ===")
print(repair_medical_ocr_decimals(test_cbc))
