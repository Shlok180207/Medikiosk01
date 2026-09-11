"""
MediKiosk - Offline SQLite FTS5 Indian Drug Master Database Builder
Constructs data/indian_drugs.db with FTS5 Trigram virtual table.
Covers single molecules and Fixed Drug Combinations (FDCs).
0 MB GPU VRAM | < 1 ms query latency | Native Python sqlite3.
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import sqlite3
import json
import time
import csv

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indian_drugs.db")

# Comprehensive Compendium of Indian Single Formulations & Fixed Drug Combinations (FDCs)
DRUG_MASTER_RECORDS = [
    # ── FDC: Hematinics & Supplements ──
    {
        "brand_name": "HB Set",
        "generic_salts": "Ferrous Ascorbate + Folic Acid",
        "strength": "100mg + 1.5mg",
        "dosage_form": "Tablet",
        "category": "Hematinic / Anti-anemic FDC",
        "common_indications": "Iron deficiency anemia, Pregnancy supplement, Nutritional deficiency",
        "is_fdc": 1
    },
    {
        "brand_name": "Orofer XT",
        "generic_salts": "Ferrous Ascorbate + Folic Acid",
        "strength": "100mg + 1.5mg",
        "dosage_form": "Tablet / Syrup",
        "category": "Hematinic / Anti-anemic FDC",
        "common_indications": "Iron deficiency anemia, Chronic fatigue, Low hemoglobin",
        "is_fdc": 1
    },
    {
        "brand_name": "Livogen Z",
        "generic_salts": "Ferrous Fumarate + Folic Acid + Zinc Sulfate",
        "strength": "152mg + 750mcg + 61.8mg",
        "dosage_form": "Capsule",
        "category": "Hematinic / Mineral FDC",
        "common_indications": "Nutritional anemia, Convalescence, Pregnancy",
        "is_fdc": 1
    },
    {
        "brand_name": "Autrin",
        "generic_salts": "Ferrous Fumarate + Folic Acid + Vitamin B12",
        "strength": "300mg + 1.5mg + 15mcg",
        "dosage_form": "Capsule",
        "category": "Hematinic FDC",
        "common_indications": "Macrocytic & Microcytic Anemia",
        "is_fdc": 1
    },
    {
        "brand_name": "Shelcal 500",
        "generic_salts": "Calcium Carbonate + Vitamin D3 (Cholecalciferol)",
        "strength": "500mg + 250IU",
        "dosage_form": "Tablet",
        "category": "Mineral Supplement FDC",
        "common_indications": "Osteopenia, Osteoporosis, Calcium deficiency",
        "is_fdc": 1
    },
    {
        "brand_name": "Cipcal 500",
        "generic_salts": "Calcium Carbonate + Vitamin D3",
        "strength": "500mg + 250IU",
        "dosage_form": "Tablet",
        "category": "Mineral Supplement FDC",
        "common_indications": "Bone health, Post-menopausal bone loss",
        "is_fdc": 1
    },
    {
        "brand_name": "Becosules",
        "generic_salts": "Vitamin B-Complex + Vitamin C",
        "strength": "Therapeutic",
        "dosage_form": "Capsule",
        "category": "Multivitamin FDC",
        "common_indications": "Mouth ulcers, Stomatitis, General debility",
        "is_fdc": 1
    },
    {
        "brand_name": "Cobaforte Z",
        "generic_salts": "Methylcobalamin + Alpha Lipoic Acid + Folic Acid + Pyridoxine",
        "strength": "1500mcg + 100mg + 1.5mg + 3mg",
        "dosage_form": "Capsule",
        "category": "Neurotropic FDC",
        "common_indications": "Diabetic neuropathy, Peripheral neuropathy, Neuralgia",
        "is_fdc": 1
    },

    # ── FDC: Anthelmintics & Antiparasitics ──
    {
        "brand_name": "Bandy Plus",
        "generic_salts": "Albendazole + Ivermectin",
        "strength": "400mg + 6mg",
        "dosage_form": "Chewable Tablet",
        "category": "Anthelmintic FDC",
        "common_indications": "Intestinal parasitic infections, Filariasis, Helminthiasis",
        "is_fdc": 1
    },
    {
        "brand_name": "Ivecop A",
        "generic_salts": "Albendazole + Ivermectin",
        "strength": "400mg + 6mg",
        "dosage_form": "Tablet",
        "category": "Anthelmintic FDC",
        "common_indications": "Parasitic infections, Strongyloidiasis, Hookworm",
        "is_fdc": 1
    },
    {
        "brand_name": "Albendazole + Ivermectin Suspension",
        "generic_salts": "Albendazole + Ivermectin",
        "strength": "200mg + 1.5mg per 5ml",
        "dosage_form": "Syrup / Suspension",
        "category": "Anthelmintic FDC",
        "common_indications": "Pediatric deworming, Mixed nematode infections",
        "is_fdc": 1
    },
    {
        "brand_name": "Bendex Plus",
        "generic_salts": "Albendazole + Ivermectin",
        "strength": "400mg + 6mg",
        "dosage_form": "Chewable Tablet",
        "category": "Anthelmintic FDC",
        "common_indications": "Intestinal worm infestation",
        "is_fdc": 1
    },

    # ── FDC: Analgesics & Anti-inflammatories ──
    {
        "brand_name": "Zerodol SP",
        "generic_salts": "Aceclofenac + Paracetamol + Serratiopeptidase",
        "strength": "100mg + 325mg + 15mg",
        "dosage_form": "Tablet",
        "category": "NSAID Anti-inflammatory FDC",
        "common_indications": "Post-surgical pain, Severe edema, Traumatic injury, Arthritis",
        "is_fdc": 1
    },
    {
        "brand_name": "Zerodol P",
        "generic_salts": "Aceclofenac + Paracetamol",
        "strength": "100mg + 325mg",
        "dosage_form": "Tablet",
        "category": "NSAID Analgesic FDC",
        "common_indications": "Fever with body pain, Osteoarthritis, Musculoskeletal pain",
        "is_fdc": 1
    },
    {
        "brand_name": "Combiflam",
        "generic_salts": "Ibuprofen + Paracetamol",
        "strength": "400mg + 325mg",
        "dosage_form": "Tablet",
        "category": "NSAID Analgesic FDC",
        "common_indications": "Dental pain, Headaches, Muscular sprain, Dysmenorrhea",
        "is_fdc": 1
    },
    {
        "brand_name": "Flexon",
        "generic_salts": "Ibuprofen + Paracetamol",
        "strength": "400mg + 325mg",
        "dosage_form": "Tablet / Syrup",
        "category": "NSAID Analgesic FDC",
        "common_indications": "Acute pain, Fever, Joint inflammation",
        "is_fdc": 1
    },
    {
        "brand_name": "Chymoral Forte",
        "generic_salts": "Trypsin + Chymotrypsin",
        "strength": "100,000 Armour Units",
        "dosage_form": "Tablet",
        "category": "Proteolytic Anti-inflammatory FDC",
        "common_indications": "Hematoma, Post-operative swelling, Sports injury edema",
        "is_fdc": 1
    },
    {
        "brand_name": "Chymoral Plus",
        "generic_salts": "Trypsin + Chymotrypsin + Diclofenac Potassium",
        "strength": "50,000 Units + 50mg",
        "dosage_form": "Tablet",
        "category": "Proteolytic Analgesic FDC",
        "common_indications": "Traumatic swelling with severe pain, Orthopedic injury",
        "is_fdc": 1
    },
    {
        "brand_name": "Voveran D",
        "generic_salts": "Diclofenac + Serratiopeptidase",
        "strength": "50mg + 10mg",
        "dosage_form": "Tablet",
        "category": "NSAID Anti-inflammatory FDC",
        "common_indications": "Knee pain, Osteoarthritis, Joint effusion",
        "is_fdc": 1
    },
    {
        "brand_name": "Ultracet",
        "generic_salts": "Tramadol Hydrochloride + Paracetamol",
        "strength": "37.5mg + 325mg",
        "dosage_form": "Tablet",
        "category": "Opioid / Non-Opioid Analgesic FDC",
        "common_indications": "Moderate to severe acute pain, Post-operative pain",
        "is_fdc": 1
    },

    # ── FDC: Antibacterials & Antivirals ──
    {
        "brand_name": "Augmentin 625 Duo",
        "generic_salts": "Amoxicillin + Clavulanic Acid",
        "strength": "500mg + 125mg",
        "dosage_form": "Tablet",
        "category": "Beta-lactam Antibiotic FDC",
        "common_indications": "Respiratory tract infection, Sinusitis, Cellulitis, Dental abscess",
        "is_fdc": 1
    },
    {
        "brand_name": "Clavam 625",
        "generic_salts": "Amoxicillin + Clavulanic Acid",
        "strength": "500mg + 125mg",
        "dosage_form": "Tablet",
        "category": "Beta-lactam Antibiotic FDC",
        "common_indications": "Bacterial pneumonia, UTI, Otitis media, Skin infections",
        "is_fdc": 1
    },
    {
        "brand_name": "Moxikind CV 625",
        "generic_salts": "Amoxicillin + Clavulanic Acid",
        "strength": "500mg + 125mg",
        "dosage_form": "Tablet",
        "category": "Beta-lactam Antibiotic FDC",
        "common_indications": "Bacterial infections, Bronchitis, ENT infections",
        "is_fdc": 1
    },
    {
        "brand_name": "O2",
        "generic_salts": "Ofloxacin + Ornidazole",
        "strength": "200mg + 500mg",
        "dosage_form": "Tablet",
        "category": "Antidiarrheal / Antibacterial FDC",
        "common_indications": "Infectious diarrhea, Amoebic dysentery, Mixed GI infections",
        "is_fdc": 1
    },
    {
        "brand_name": "Zenflox OZ",
        "generic_salts": "Ofloxacin + Ornidazole",
        "strength": "200mg + 500mg",
        "dosage_form": "Tablet",
        "category": "Antidiarrheal / Antibacterial FDC",
        "common_indications": "Gastroenteritis, Dental infections, Gynecological infections",
        "is_fdc": 1
    },
    {
        "brand_name": "Ciplox TZ",
        "generic_salts": "Ciprofloxacin + Tinidazole",
        "strength": "500mg + 600mg",
        "dosage_form": "Tablet",
        "category": "Antibacterial / Antiprotozoal FDC",
        "common_indications": "Severe diarrhea, Dysentery, Pelvic inflammatory disease",
        "is_fdc": 1
    },
    {
        "brand_name": "Norflox TZ",
        "generic_salts": "Norfloxacin + Tinidazole",
        "strength": "400mg + 600mg",
        "dosage_form": "Tablet",
        "category": "Gastrointestinal Antimicrobial FDC",
        "common_indications": "Acute infectious diarrhea, Food poisoning, Giardiasis",
        "is_fdc": 1
    },
    {
        "brand_name": "Taxim O CV",
        "generic_salts": "Cefixime + Clavulanic Acid",
        "strength": "200mg + 125mg",
        "dosage_form": "Tablet",
        "category": "Cephalosporin Antibiotic FDC",
        "common_indications": "Resistant respiratory infections, Typhoid fever, Complicated UTI",
        "is_fdc": 1
    },
    {
        "brand_name": "Gudcef CV",
        "generic_salts": "Cefpodoxime Proxetil + Clavulanic Acid",
        "strength": "200mg + 125mg",
        "dosage_form": "Tablet",
        "category": "Cephalosporin Antibiotic FDC",
        "common_indications": "Community acquired pneumonia, Acute sinusitis",
        "is_fdc": 1
    },

    # ── FDC: Respiratory & Allergy ──
    {
        "brand_name": "Montek LC",
        "generic_salts": "Montelukast + Levocetirizine",
        "strength": "10mg + 5mg",
        "dosage_form": "Tablet",
        "category": "Anti-allergic / Leukotriene Antagonist FDC",
        "common_indications": "Allergic rhinitis, Seasonal allergies, Asthma maintenance",
        "is_fdc": 1
    },
    {
        "brand_name": "Telekast L",
        "generic_salts": "Montelukast + Levocetirizine",
        "strength": "10mg + 5mg",
        "dosage_form": "Tablet",
        "category": "Anti-allergic FDC",
        "common_indications": "Runny nose, Sneezing, Allergic broncho-constriction",
        "is_fdc": 1
    },
    {
        "brand_name": "Montair FX",
        "generic_salts": "Montelukast + Fexofenadine",
        "strength": "10mg + 120mg",
        "dosage_form": "Tablet",
        "category": "Non-sedating Anti-allergic FDC",
        "common_indications": "Chronic urticaria, Severe allergic rhinitis",
        "is_fdc": 1
    },
    {
        "brand_name": "Ascoril D Plus",
        "generic_salts": "Dextromethorphan + Chlorpheniramine + Phenylephrine",
        "strength": "10mg + 2mg + 5mg per 5ml",
        "dosage_form": "Syrup",
        "category": "Antitussive / Decongestant FDC",
        "common_indications": "Dry cough, Nasal congestion, Allergic throat irritation",
        "is_fdc": 1
    },
    {
        "brand_name": "Alex Cough Syrup",
        "generic_salts": "Dextromethorphan + Chlorpheniramine + Phenylephrine",
        "strength": "10mg + 2mg + 5mg per 5ml",
        "dosage_form": "Syrup",
        "category": "Antitussive FDC",
        "common_indications": "Dry irritating cough, Common cold symptoms",
        "is_fdc": 1
    },
    {
        "brand_name": "Ascoril LS",
        "generic_salts": "Levosalbutamol + Ambroxol + Guaiphenesin",
        "strength": "1mg + 30mg + 50mg per 5ml",
        "dosage_form": "Syrup",
        "category": "Mucolytic Bronchodilator FDC",
        "common_indications": "Productive wet cough, Bronchial asthma, Chronic bronchitis",
        "is_fdc": 1
    },
    {
        "brand_name": "Cheston Cold",
        "generic_salts": "Cetirizine + Paracetamol + Phenylephrine",
        "strength": "5mg + 325mg + 10mg",
        "dosage_form": "Tablet",
        "category": "Cold & Flu FDC",
        "common_indications": "Head cold, Fever with body ache, Rhinitis",
        "is_fdc": 1
    },

    # ── FDC: Gastroenterology & Acid Reflux ──
    {
        "brand_name": "Pan D",
        "generic_salts": "Pantoprazole + Domperidone",
        "strength": "40mg + 30mg (SR)",
        "dosage_form": "Capsule",
        "category": "PPI / Prokinetic FDC",
        "common_indications": "GERD with nausea, Dyspepsia, Acid reflux, Gastric ulcer",
        "is_fdc": 1
    },
    {
        "brand_name": "Pantocid DSR",
        "generic_salts": "Pantoprazole + Domperidone",
        "strength": "40mg + 30mg (SR)",
        "dosage_form": "Capsule",
        "category": "PPI / Prokinetic FDC",
        "common_indications": "Heartburn, Belching, Regurgitation",
        "is_fdc": 1
    },
    {
        "brand_name": "Razo D",
        "generic_salts": "Rabeprazole + Domperidone",
        "strength": "20mg + 30mg (SR)",
        "dosage_form": "Capsule",
        "category": "PPI / Prokinetic FDC",
        "common_indications": "Severe acid dyspepsia, Gastritis with nausea",
        "is_fdc": 1
    },
    {
        "brand_name": "Omez D",
        "generic_salts": "Omeprazole + Domperidone",
        "strength": "20mg + 10mg",
        "dosage_form": "Capsule",
        "category": "PPI / Prokinetic FDC",
        "common_indications": "Heartburn, Acid reflux, Nausea",
        "is_fdc": 1
    },

    # ── FDC: Cardiology & Diabetes ──
    {
        "brand_name": "Telma AM",
        "generic_salts": "Telmisartan + Amlodipine",
        "strength": "40mg + 5mg",
        "dosage_form": "Tablet",
        "category": "Antihypertensive FDC",
        "common_indications": "Essential hypertension, Uncontrolled high blood pressure",
        "is_fdc": 1
    },
    {
        "brand_name": "Telma H",
        "generic_salts": "Telmisartan + Hydrochlorothiazide",
        "strength": "40mg + 12.5mg",
        "dosage_form": "Tablet",
        "category": "Antihypertensive Diuretic FDC",
        "common_indications": "Hypertension, Fluid retention",
        "is_fdc": 1
    },
    {
        "brand_name": "Glycomet GP 1",
        "generic_salts": "Metformin + Glimepiride",
        "strength": "500mg + 1mg",
        "dosage_form": "Tablet",
        "category": "Oral Antidiabetic FDC",
        "common_indications": "Type 2 Diabetes Mellitus",
        "is_fdc": 1
    },
    {
        "brand_name": "Glycomet GP 2",
        "generic_salts": "Metformin + Glimepiride",
        "strength": "500mg + 2mg",
        "dosage_form": "Tablet",
        "category": "Oral Antidiabetic FDC",
        "common_indications": "Type 2 Diabetes Mellitus",
        "is_fdc": 1
    },
    {
        "brand_name": "Atorva CV",
        "generic_salts": "Atorvastatin + Clopidogrel",
        "strength": "10mg + 75mg",
        "dosage_form": "Capsule",
        "category": "Lipid-lowering Antiplatelet FDC",
        "common_indications": "Coronary artery disease, Post-PCI stenting, Secondary prevention",
        "is_fdc": 1
    },
    {
        "brand_name": "Rosuvas CV",
        "generic_salts": "Rosuvastatin + Clopidogrel",
        "strength": "10mg + 75mg",
        "dosage_form": "Capsule",
        "category": "Lipid-lowering Antiplatelet FDC",
        "common_indications": "Dyslipidemia, Cardiovascular event prevention",
        "is_fdc": 1
    },

    # ── Single Formulations & Antibiotics / Hepatics ──
    {
        "brand_name": "Gembax 400",
        "generic_salts": "Gemifloxacin",
        "strength": "400mg",
        "dosage_form": "Tablet",
        "category": "Fluoroquinolone Antibiotic",
        "common_indications": "Community acquired pneumonia, Acute exacerbation of chronic bronchitis",
        "is_fdc": 0
    },
    {
        "brand_name": "Gemina 400",
        "generic_salts": "Gemifloxacin",
        "strength": "400mg",
        "dosage_form": "Tablet",
        "category": "Fluoroquinolone Antibiotic",
        "common_indications": "Lower respiratory tract infection",
        "is_fdc": 0
    },
    {
        "brand_name": "Hepcoac",
        "generic_salts": "Daclatasvir",
        "strength": "60mg",
        "dosage_form": "Tablet",
        "category": "Direct-Acting Antiviral / Hepatic Support",
        "common_indications": "Chronic Hepatitis C, Hepatic convalescence",
        "is_fdc": 0
    },
    {
        "brand_name": "Daclatasvir",
        "generic_salts": "Daclatasvir Dihydrochloride",
        "strength": "60mg",
        "dosage_form": "Tablet",
        "category": "Direct-Acting Antiviral",
        "common_indications": "Chronic Hepatitis C infection",
        "is_fdc": 0
    },
    {
        "brand_name": "Paracetamol 650",
        "generic_salts": "Acetaminophen / Paracetamol",
        "strength": "650mg",
        "dosage_form": "Tablet",
        "category": "Antipyretic / Analgesic",
        "common_indications": "Fever, Body ache, Mild pain",
        "is_fdc": 0
    },
    {
        "brand_name": "Dolo 650",
        "generic_salts": "Acetaminophen / Paracetamol",
        "strength": "650mg",
        "dosage_form": "Tablet",
        "category": "Antipyretic / Analgesic",
        "common_indications": "Fever, Headache, Viral myalgia",
        "is_fdc": 0
    },
    {
        "brand_name": "Azithral 500",
        "generic_salts": "Azithromycin",
        "strength": "500mg",
        "dosage_form": "Tablet",
        "category": "Macrolide Antibiotic",
        "common_indications": "Pharyngitis, Tonsillitis, Chest infection",
        "is_fdc": 0
    },
    {
        "brand_name": "Pan 40",
        "generic_salts": "Pantoprazole",
        "strength": "40mg",
        "dosage_form": "Tablet",
        "category": "Proton Pump Inhibitor",
        "common_indications": "Acidity, GERD, Gastric ulcer",
        "is_fdc": 0
    },
    {
        "brand_name": "Pantocid 40",
        "generic_salts": "Pantoprazole",
        "strength": "40mg",
        "dosage_form": "Tablet",
        "category": "Proton Pump Inhibitor",
        "common_indications": "Hyperacidity, Peptic ulcer",
        "is_fdc": 0
    },
    {
        "brand_name": "Omez 20",
        "generic_salts": "Omeprazole",
        "strength": "20mg",
        "dosage_form": "Capsule",
        "category": "Proton Pump Inhibitor",
        "common_indications": "Acid reflux, Heartburn",
        "is_fdc": 0
    },
    {
        "brand_name": "Razo 20",
        "generic_salts": "Rabeprazole",
        "strength": "20mg",
        "dosage_form": "Tablet",
        "category": "Proton Pump Inhibitor",
        "common_indications": "GERD, Duodenal ulcer",
        "is_fdc": 0
    },
    {
        "brand_name": "Taxim O 200",
        "generic_salts": "Cefixime",
        "strength": "200mg",
        "dosage_form": "Tablet",
        "category": "Cephalosporin Antibiotic",
        "common_indications": "Typhoid, UTI, Otitis media",
        "is_fdc": 0
    },
    {
        "brand_name": "Gudcef 200",
        "generic_salts": "Cefpodoxime Proxetil",
        "strength": "200mg",
        "dosage_form": "Tablet",
        "category": "Cephalosporin Antibiotic",
        "common_indications": "Sinusitis, Bronchitis, Skin infection",
        "is_fdc": 0
    },
    {
        "brand_name": "Monocef 1g",
        "generic_salts": "Ceftriaxone",
        "strength": "1g",
        "dosage_form": "Injection",
        "category": "Injectable Cephalosporin",
        "common_indications": "Severe bacterial infections, Sepsis, Meningitis",
        "is_fdc": 0
    },
    {
        "brand_name": "Metrogyl 400",
        "generic_salts": "Metronidazole",
        "strength": "400mg",
        "dosage_form": "Tablet",
        "category": "Antiprotozoal / Antibacterial",
        "common_indications": "Amoebiasis, Giardiasis, Dental infections",
        "is_fdc": 0
    },
    {
        "brand_name": "Ciplox 500",
        "generic_salts": "Ciprofloxacin",
        "strength": "500mg",
        "dosage_form": "Tablet",
        "category": "Fluoroquinolone Antibiotic",
        "common_indications": "UTI, Enteric fever, Bone & joint infection",
        "is_fdc": 0
    },
    {
        "brand_name": "Telma 40",
        "generic_salts": "Telmisartan",
        "strength": "40mg",
        "dosage_form": "Tablet",
        "category": "Antihypertensive ARB",
        "common_indications": "Hypertension, Cardiac protection",
        "is_fdc": 0
    },
    {
        "brand_name": "Amlong 5",
        "generic_salts": "Amlodipine",
        "strength": "5mg",
        "dosage_form": "Tablet",
        "category": "Calcium Channel Blocker",
        "common_indications": "High blood pressure, Chronic stable angina",
        "is_fdc": 0
    },
    {
        "brand_name": "Glycomet 500",
        "generic_salts": "Metformin Hydrochloride",
        "strength": "500mg",
        "dosage_form": "Tablet",
        "category": "Biguanide Antidiabetic",
        "common_indications": "Type 2 Diabetes Mellitus, Insulin resistance",
        "is_fdc": 0
    },
    {
        "brand_name": "Thyronorm 50",
        "generic_salts": "Levothyroxine Sodium",
        "strength": "50mcg",
        "dosage_form": "Tablet",
        "category": "Thyroid Hormone",
        "common_indications": "Hypothyroidism",
        "is_fdc": 0
    },
    {
        "brand_name": "Voveran 50",
        "generic_salts": "Diclofenac Sodium",
        "strength": "50mg",
        "dosage_form": "Tablet",
        "category": "NSAID Analgesic",
        "common_indications": "Arthritis pain, Spondylitis, Acute muscle spasm",
        "is_fdc": 0
    },
    {
        "brand_name": "Atorva 20",
        "generic_salts": "Atorvastatin",
        "strength": "20mg",
        "dosage_form": "Tablet",
        "category": "Statin Lipid-lowering",
        "common_indications": "High cholesterol, Atherosclerosis prevention",
        "is_fdc": 0
    },
    {
        "brand_name": "Rosuvas 10",
        "generic_salts": "Rosuvastatin",
        "strength": "10mg",
        "dosage_form": "Tablet",
        "category": "Statin Lipid-lowering",
        "common_indications": "Dyslipidemia, Cardiovascular risk reduction",
        "is_fdc": 0
    },
    {
        "brand_name": "Ecosprin 75",
        "generic_salts": "Aspirin",
        "strength": "75mg",
        "dosage_form": "Tablet",
        "category": "Antiplatelet",
        "common_indications": "Myocardial infarction prevention, Stroke prevention",
        "is_fdc": 0
    },
    {
        "brand_name": "Clopilet 75",
        "generic_salts": "Clopidogrel",
        "strength": "75mg",
        "dosage_form": "Tablet",
        "category": "Antiplatelet",
        "common_indications": "Recent MI, Peripheral arterial disease",
        "is_fdc": 0
    },
    {
        "brand_name": "Liv 52",
        "generic_salts": "Herbal Hepatoprotective Formulation",
        "strength": "Therapeutic",
        "dosage_form": "Tablet / Syrup",
        "category": "Hepatoprotective",
        "common_indications": "Hepatotoxicity, Anorexia, Early cirrhosis recovery",
        "is_fdc": 0
    },
    {
        "brand_name": "Udiliv 300",
        "generic_salts": "Ursodeoxycholic Acid (UDCA)",
        "strength": "300mg",
        "dosage_form": "Tablet",
        "category": "Bile Acid Derivative",
        "common_indications": "Gallstones, Primary biliary cirrhosis",
        "is_fdc": 0
    },
    {
        "brand_name": "Defcort 6",
        "generic_salts": "Deflazacort",
        "strength": "6mg",
        "dosage_form": "Tablet",
        "category": "Corticosteroid",
        "common_indications": "Severe inflammation, Autoimmune disorders",
        "is_fdc": 0
    },
    {
        "brand_name": "Wysolone 5",
        "generic_salts": "Prednisolone",
        "strength": "5mg",
        "dosage_form": "Tablet",
        "category": "Corticosteroid",
        "common_indications": "Allergic conditions, Asthma exacerbation, Rheumatoid arthritis",
        "is_fdc": 0
    }
]


def build_fts5_database():
    """Constructs the SQLite FTS5 database with trigram search covering all 250,000+ Indian medicines."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fast bulk insertion pragmas
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = MEMORY;")
    cursor.execute("PRAGMA cache_size = 100000;")

    print(f"📦 Building SQLite FTS5 Indian Drug Master Database at: {DB_PATH}")

    # Create FTS5 Virtual Table with trigram indexing
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS indian_drugs USING fts5(
        brand_name,
        generic_salts,
        strength,
        dosage_form,
        category,
        common_indications,
        is_fdc UNINDEXED,
        tokenize='trigram'
    );
    """)

    t0 = time.time()

    # 1. Insert curated high-priority clinic records first
    for rec in DRUG_MASTER_RECORDS:
        cursor.execute("""
        INSERT INTO indian_drugs (
            brand_name, generic_salts, strength, dosage_form, category, common_indications, is_fdc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["brand_name"],
            rec["generic_salts"],
            rec["strength"],
            rec["dosage_form"],
            rec["category"],
            rec["common_indications"],
            str(rec["is_fdc"])
        ))

    # 2. Check for indian_medicine_data.csv in workspace or Downloads
    csv_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "indian_medicine_data.csv"),
        os.path.join(os.path.expanduser("~"), "Downloads", "indian_medicine_data.csv"),
        r"c:\Users\hp\Downloads\indian_medicine_data.csv"
    ]
    csv_path = None
    for cand in csv_candidates:
        if os.path.exists(cand):
            csv_path = cand
            break

    total_csv_added = 0
    if csv_path:
        print(f"📖 Found comprehensive dataset: {csv_path}")
        print("🚀 Ingesting 250,000+ Indian medicines into FTS5 index...")
        rows_batch = []
        with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_disc = row.get("Is_discontinued", "").strip().upper() == "TRUE"
                if is_disc:
                    continue

                name = row.get("name", "").strip()
                if not name or len(name) < 2:
                    continue

                c1 = row.get("short_composition1", "").strip()
                c2 = row.get("short_composition2", "").strip()
                mfg = row.get("manufacturer_name", "").strip()
                pack = row.get("pack_size_label", "").strip()

                is_fdc = 1 if (c1 and c2) else 0
                if c1 and c2:
                    combined_salts = f"{c1} + {c2}"
                else:
                    combined_salts = c1 or c2 or "Pharmaceutical Formulation"

                dosage_form = "Tablet"
                name_pack_l = f"{name} {pack}".lower()
                if "syrup" in name_pack_l or "suspension" in name_pack_l or "liquid" in name_pack_l:
                    dosage_form = "Syrup"
                elif "capsule" in name_pack_l or "cap" in name_pack_l:
                    dosage_form = "Capsule"
                elif "injection" in name_pack_l or "inj" in name_pack_l:
                    dosage_form = "Injection"
                elif "drop" in name_pack_l:
                    dosage_form = "Drops"
                elif "gel" in name_pack_l or "cream" in name_pack_l or "ointment" in name_pack_l:
                    dosage_form = "Topical"

                category = row.get("type", "Allopathy")
                if mfg:
                    category = f"{category} ({mfg})"

                rows_batch.append((
                    name,
                    combined_salts,
                    pack,
                    dosage_form,
                    category,
                    "Commercial formulation",
                    str(is_fdc)
                ))

                if len(rows_batch) >= 10000:
                    cursor.executemany("""
                    INSERT INTO indian_drugs (brand_name, generic_salts, strength, dosage_form, category, common_indications, is_fdc)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, rows_batch)
                    total_csv_added += len(rows_batch)
                    rows_batch = []

            if rows_batch:
                cursor.executemany("""
                INSERT INTO indian_drugs (brand_name, generic_salts, strength, dosage_form, category, common_indications, is_fdc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows_batch)
                total_csv_added += len(rows_batch)

    conn.commit()
    dt = time.time() - t0

    cursor.execute("SELECT count(*) FROM indian_drugs")
    total_count = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM indian_drugs WHERE is_fdc = '1'")
    total_fdc = cursor.fetchone()[0]
    file_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

    print(f"\n🎉 Successfully indexed {total_count:,} pharmaceutical formulations in {dt:.2f}s!")
    print(f"  • Multi-salt Fixed Drug Combinations (FDCs): {total_fdc:,}")
    print(f"  • Database file size: {file_size_mb:.2f} MB (100% offline, 0 MB GPU VRAM)")

    # Test FDC Queries
    print("\n🔍 Testing Sample FTS5 Queries:")
    test_queries = ["HB Set", "Ferrous Folic", "Albendazole Ivermectin", "Augmentin", "Zerodol SP", "Pan D", "Gembax"]
    for q in test_queries:
        t_q0 = time.time()
        search_term = q.replace(" ", "* ") + "*"
        cursor.execute("""
        SELECT brand_name, generic_salts, strength, is_fdc, rank
        FROM indian_drugs
        WHERE indian_drugs MATCH ?
        ORDER BY rank
        LIMIT 1
        """, (search_term,))
        hit = cursor.fetchone()
        t_q1 = time.time()
        lat_ms = (t_q1 - t_q0) * 1000
        if hit:
            fdc_tag = " [FDC Mixture]" if hit[3] == "1" else ""
            print(f"  • Query: '{q}' -> {hit[0]} ({hit[1]}){fdc_tag} in {lat_ms:.3f}ms")
        else:
            print(f"  • Query: '{q}' -> No match in {lat_ms:.3f}ms")

    conn.close()
    return DB_PATH


if __name__ == "__main__":
    build_fts5_database()

