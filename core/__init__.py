"""
core package
============
Sequential, Event-Driven Clinical Triage Pipeline for 8GB VRAM (NVIDIA RTX 4060).
"""

from core.vram_manager import VRAMManager, ModelState
from core.cpu_perception import CPUPerceptionEngine
from core.clinical_orchestrator import (
    ClinicalOrchestrator,
    PatientJourney,
    CDSSReport,
    DocumentType
)

__all__ = [
    "VRAMManager",
    "ModelState",
    "CPUPerceptionEngine",
    "ClinicalOrchestrator",
    "PatientJourney",
    "CDSSReport",
    "DocumentType"
]
