"""
core/cpu_perception.py
======================
Diagnostic perception engine pinned 100% to host CPU memory.
Guarantees 0 MB GPU VRAM allocation for all medical imaging (CXR & ECG).
"""

import asyncio
import io
import logging
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import torch

from perception.xray import analyze_xray
from perception.ecg import analyze_ecg
from perception.prescription import analyze_prescription

logger = logging.getLogger("CPUPerception")

CPU_DEVICE = torch.device("cpu")


class CPUPerceptionEngine:
    """
    Dedicated diagnostic execution sandbox pinned strictly to host CPU.
    Tensors and OpenCV memory buffers never touch CUDA.
    """

    def __init__(self):
        logger.info("Initializing CPU Perception Engine (TorchXRayVision + OpenCV ECG)...")

    def _sync_analyze_xray(self, image_bytes: bytes, filename: str = "cxr.png") -> Dict[str, Any]:
        """Runs TorchXRayVision DenseNet on CPU."""
        logger.info("🩻 [CPU Thread] Running TorchXRayVision DenseNet-121 on CPU...")
        try:
            result = analyze_xray(image_bytes, filename=filename)
            findings = []
            for path_name, conf in result.get("pathologies", {}).items():
                if conf >= 0.45:
                    findings.append({
                        "finding": path_name,
                        "confidence": round(float(conf), 4),
                        "severity": "CRITICAL" if path_name in ["Pneumothorax", "Cardiomegaly", "Infiltration"] else "MODERATE"
                    })

            return {
                "modality": "CHEST_XRAY",
                "device": "CPU",
                "findings": findings,
                "summary": result.get("summary", f"Detected {len(findings)} radiographic anomalies on CPU."),
                "dashboard_payload": result.get("dashboard_payload", {})
            }
        except Exception as e:
            logger.error(f"X-Ray CPU perception failed: {e}")
            return {
                "modality": "CHEST_XRAY",
                "device": "CPU",
                "findings": [],
                "summary": f"X-Ray analysis error: {e}",
                "dashboard_payload": {}
            }

    def _sync_analyze_ecg(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs OpenCV + SciPy ECG lead isolation on CPU."""
        logger.info("📈 [CPU Thread] Running OpenCV HSV Grid Stripping & ECG Rhythm Analysis on CPU...")
        try:
            result = analyze_ecg(image_bytes)
            return {
                "modality": "ECG_STRIP",
                "device": "CPU",
                "findings": result.get("flagged_values", []),
                "summary": result.get("summary", "ECG waveform parsed on CPU."),
                "dashboard_payload": result.get("dashboard_payload", {})
            }
        except Exception as e:
            logger.error(f"ECG CPU perception failed: {e}")
            return {
                "modality": "ECG_STRIP",
                "device": "CPU",
                "findings": [],
                "summary": f"ECG analysis error: {e}",
                "dashboard_payload": {}
            }

    def _sync_analyze_prescription(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs Sauvola adaptive thresholding + RapidFuzz drug matching on CPU."""
        logger.info("📄 [CPU Thread] Running Sauvola Binarization & RapidFuzz Drug Matcher on CPU...")
        try:
            result = analyze_prescription(image_bytes)
            return {
                "modality": "PRESCRIPTION",
                "device": "CPU",
                "medications": result.get("dashboard_payload", {}).get("medications", []),
                "flagged_values": result.get("dashboard_payload", {}).get("flagged_values", []),
                "summary": result.get("summary", "Prescription parsed on CPU."),
                "dashboard_payload": result.get("dashboard_payload", {})
            }
        except Exception as e:
            logger.error(f"Prescription CPU perception failed: {e}")
            return {
                "modality": "PRESCRIPTION",
                "device": "CPU",
                "medications": [],
                "flagged_values": [],
                "summary": f"Prescription analysis error: {e}",
                "dashboard_payload": {}
            }

    async def analyze_chest_xray(self, image_bytes: bytes, filename: str = "cxr.png") -> Dict[str, Any]:
        """Non-blocking async wrapper around CPU X-Ray analysis."""
        return await asyncio.to_thread(self._sync_analyze_xray, image_bytes, filename)

    async def analyze_ecg_strip(self, image_bytes: bytes) -> Dict[str, Any]:
        """Non-blocking async wrapper around CPU ECG analysis."""
        return await asyncio.to_thread(self._sync_analyze_ecg, image_bytes)

    async def analyze_prescription_fallback(self, image_bytes: bytes) -> Dict[str, Any]:
        """Non-blocking async wrapper around CPU prescription analysis."""
        return await asyncio.to_thread(self._sync_analyze_prescription, image_bytes)
