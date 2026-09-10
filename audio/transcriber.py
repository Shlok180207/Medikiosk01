"""
MediKiosk Audio - Faster-Whisper Speech-to-Text Module
GPU-Managed with Explicit VRAM Lifecycle Management:
- Model runs on CUDA (int8) for ultra-fast Hindi/English consultation transcription (~0.5s).
- Explicit unload() completely frees VRAM (del model, gc.collect(), torch.cuda.empty_cache())
  to guarantee zero lingering GPU allocation before LLM synthesis begins.
"""

import os
import gc
import tempfile
import threading
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None

def get_vram_mb() -> Dict[str, float]:
    """Returns GPU VRAM allocation and reservation in Megabytes."""
    if torch and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "device_name": torch.cuda.get_device_name(0)
        }
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "device_name": "CPU/None"}


class AudioTranscriber:
    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device if (torch and torch.cuda.is_available() and device == "cuda") else "cpu"
        self.compute_type = compute_type
        self.model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        """Loads Faster-Whisper pipeline onto the specified device."""
        with self._lock:
            if self.model is None:
                vram_before = get_vram_mb()
                print(f"⚡ Loading Faster-Whisper ({self.model_size}, {self.compute_type}) on {self.device}...")
                print(f"   [VRAM Before ASR Load]: {vram_before['allocated_mb']} MB allocated")
                try:
                    from faster_whisper import WhisperModel
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type
                    )
                    vram_after = get_vram_mb()
                    print(f"✅ Faster-Whisper loaded. [VRAM After Load]: {vram_after['allocated_mb']} MB allocated.")
                except Exception as e:
                    print(f"⚠️ Failed to load on {self.device}: {e}. Falling back to CPU...")
                    from faster_whisper import WhisperModel
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribe(
        self,
        audio_input,
        language: Optional[str] = None,
        beam_size: int = 5,
        auto_unload: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribes audio file path or raw bytes.
        Returns:
          - text: Full transcribed string
          - language: Detected/specified language
          - segments: Detailed segment timings
        """
        self._ensure_loaded()
        temp_path = None

        try:
            if isinstance(audio_input, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(audio_input)
                    temp_path = tf.name
                target_file = temp_path
            elif isinstance(audio_input, str):
                target_file = audio_input
            else:
                raise ValueError("audio_input must be a file path (str) or raw audio bytes.")

            segments, info = self.model.transcribe(
                target_file,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )

            text_segments = []
            for seg in segments:
                text_segments.append(seg.text.strip())

            full_transcript = " ".join(text_segments).strip()
            detected_lang = info.language if hasattr(info, "language") else (language or "unknown")

            result = {
                "text": full_transcript,
                "language": detected_lang,
                "language_probability": round(float(info.language_probability), 3) if hasattr(info, "language_probability") else 1.0,
                "status": "success"
            }
            return result

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            if auto_unload:
                self.unload()

    def unload(self):
        """
        Deloads the Whisper model from memory and clears CUDA cache.
        Guarantees zero lingering VRAM allocation before LLM synthesis.
        """
        with self._lock:
            if self.model is not None:
                print("🧹 Explicitly deloading Faster-Whisper from memory to reclaim VRAM...")
                del self.model
                self.model = None
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                vram_now = get_vram_mb()
                print(f"✅ Faster-Whisper deloaded. [VRAM Now]: {vram_now['allocated_mb']} MB (Baseline reset).")


# Global singleton transcriber
_GLOBAL_TRANSCRIBER = None

def get_transcriber() -> AudioTranscriber:
    global _GLOBAL_TRANSCRIBER
    if _GLOBAL_TRANSCRIBER is None:
        _GLOBAL_TRANSCRIBER = AudioTranscriber(model_size="medium", device="cuda", compute_type="int8")
    return _GLOBAL_TRANSCRIBER


def transcribe_consultation(audio_path: str) -> str:
    """
    GPU-managed consultation ASR:
    - Loads faster-whisper on CUDA (int8).
    - Extracts Hindi/English bilingual dialogue.
    - Immediately calls del model, torch.cuda.empty_cache(), and gc.collect() upon completion.
    """
    transcriber = get_transcriber()
    res = transcriber.transcribe(audio_path, auto_unload=True)
    return res.get("text", "").strip()

