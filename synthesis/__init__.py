"""
MediKiosk Synthesis Package (GPU-Inference)
Exports:
  - synthesize_clinical_case
  - CLINICAL_SYSTEM_PROMPT
"""

from .llm import synthesize_clinical_case, CLINICAL_SYSTEM_PROMPT

__all__ = ["synthesize_clinical_case", "CLINICAL_SYSTEM_PROMPT"]
