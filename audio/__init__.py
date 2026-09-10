"""
MediKiosk Audio Package (GPU-Managed with VRAM Lifecycle)
Exports:
  - AudioTranscriber
  - get_transcriber
  - get_vram_mb
"""

from .transcriber import AudioTranscriber, get_transcriber, get_vram_mb

__all__ = ["AudioTranscriber", "get_transcriber", "get_vram_mb"]
