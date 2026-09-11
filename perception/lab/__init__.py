"""
MediKiosk Tabular Laboratory Perception Module
=============================================
High-performance, memory-efficient laboratory report extraction pipeline.
Guarantees robust tabular association and deterministic 8GB VRAM constraint adherence.
"""

from .lab_preprocessor import LabPreprocessor, LabImageChunk
from .vlm_extractor import ConstrainedVLMExtractor, LabTestItem, FlagEnum
from .lab_pipeline import LabReportPipeline, LabReportResult, analyze_lab_report

__all__ = [
    "LabPreprocessor",
    "LabImageChunk",
    "ConstrainedVLMExtractor",
    "LabTestItem",
    "FlagEnum",
    "LabReportPipeline",
    "LabReportResult",
    "analyze_lab_report",
]
