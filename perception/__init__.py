"""
MediKiosk Perception Package (100% CPU-Bound)
Exports:
  - analyze_xray (torchxrayvision DenseNet-121)
  - analyze_ecg (OpenCV HSV grid isolation + SciPy peak detection)
  - analyze_prescription (Sauvola preprocessing + dual OCR + RapidFuzz drug normalizer)
"""

from .xray import analyze_xray
from .ecg import analyze_ecg
from .prescription import analyze_prescription, normalize_drugs

__all__ = ["analyze_xray", "analyze_ecg", "analyze_prescription", "normalize_drugs"]
