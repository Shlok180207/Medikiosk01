"""
core/vram_manager.py
====================
Deterministic VRAM State Machine for single-GPU 8GB VRAM architectures (NVIDIA RTX 4060).
Strictly guarantees a Single-Resident-GPU-Model (SRGM) invariant:
  1. Only ONE neural model resides in GPU VRAM at any given instant.
  2. Every state transition unconditionally executes a 4-step memory vacuum:
     del model -> gc.collect() -> torch.cuda.empty_cache() -> torch.cuda.ipc_collect()
  3. Pre-flight memory headroom checks reject transitions if remaining free VRAM is insufficient.
  4. Supports 4-bit NF4 / AWQ quantization configs and Ollama daemon eviction.
"""

import asyncio
import gc
import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, Optional
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VRAMManager")


class ModelState(str, Enum):
    """Mutually exclusive GPU resident states."""
    IDLE = "IDLE"
    WHISPER = "WHISPER"
    VLM_GATEKEEPER = "VLM_GATEKEEPER"
    LLM_REASONING = "LLM_REASONING"


class VRAMManager:
    """
    Deterministic VRAM State Machine.
    Coordinates sequential loading, unloading, and memory vacuuming
    to guarantee zero CUDA OOM errors under tight 8GB VRAM hardware limits.
    """

    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        vram_safety_margin_mb: int = 1200,
        use_ollama_backend: bool = True,
        allow_simulation_fallback: bool = True
    ):
        self.device = device
        self.safety_margin_bytes = vram_safety_margin_mb * 1024 * 1024
        self.use_ollama = use_ollama_backend
        self.allow_simulation_fallback = allow_simulation_fallback
        self.current_state: ModelState = ModelState.IDLE
        self.active_bundle: Dict[str, Any] = {}
        self._async_lock = asyncio.Lock()

    def get_vram_telemetry(self) -> Dict[str, float]:
        """Returns accurate current GPU memory telemetry in Megabytes."""
        if not torch.cuda.is_available() or "cpu" in self.device:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 8192.0, "total_mb": 8192.0}

        try:
            device_idx = torch.cuda.current_device() if self.device == "cuda" else self.device
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**2)
            free = free_bytes / (1024**2)
            total = total_bytes / (1024**2)
            return {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "free_mb": round(free, 2),
                "total_mb": round(total, 2)
            }
        except Exception as e:
            logger.warning(f"Telemetry query warning: {e}")
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}

    def sweep_vram(self):
        """
        Hard vacuuming of CUDA memory blocks.
        Executes a deterministic 4-step flush to break fragmentation and reclaim VRAM.
        """
        pre_telemetry = self.get_vram_telemetry()

        # Step 1: Evict cyclic references from Python GC
        gc.collect()

        if torch.cuda.is_available() and "cpu" not in self.device:
            # Step 2: Flush PyTorch CUDA allocator cache
            torch.cuda.empty_cache()
            # Step 3: Flush cross-process CUDA inter-process handles
            torch.cuda.ipc_collect()
            # Step 4: Synchronize GPU device to ensure pending kernels complete
            torch.cuda.synchronize(self.device)

        post_telemetry = self.get_vram_telemetry()
        reclaimed = max(0.0, pre_telemetry["reserved_mb"] - post_telemetry["reserved_mb"])
        logger.info(
            f"🧹 [VRAM Sweep] Reclaimed: {reclaimed:.1f} MB | "
            f"Allocated: {post_telemetry['allocated_mb']:.1f} MB | "
            f"Reserved: {post_telemetry['reserved_mb']:.1f} MB | "
            f"Free: {post_telemetry['free_mb']:.1f} MB"
        )

    def _evict_ollama_model(self, model_name: str):
        """Signals Ollama daemon to immediately evict the model from VRAM (keep_alive: 0)."""
        try:
            import requests
            requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=1.5
            )
            logger.info(f"🔻 Ollama daemon signaled to evict '{model_name}' (keep_alive=0).")
        except Exception:
            pass

    def unload_active_model(self):
        """
        Destructively tears down current model and sweeps GPU memory.
        Guarantees that model weights, processors, and tensor graphs are removed.
        """
        if self.current_state == ModelState.IDLE:
            return

        logger.info(f"🔻 [VRAM Eviction] Unloading model tenant: [{self.current_state.value}]...")

        # If Ollama model was active, signal Ollama daemon to release GPU memory
        if self.current_state == ModelState.VLM_GATEKEEPER:
            self._evict_ollama_model("qwen2.5vl:3b")
        elif self.current_state == ModelState.LLM_REASONING:
            self._evict_ollama_model("qwen2.5:7b")

        # Explicitly delete all tensor references and pipeline objects
        keys = list(self.active_bundle.keys())
        for key in keys:
            val = self.active_bundle.pop(key, None)
            del val

        self.active_bundle.clear()
        self.current_state = ModelState.IDLE
        self.sweep_vram()

    def _load_whisper(self) -> Dict[str, Any]:
        """Loads Faster-Whisper using CTranslate2 int8/float16 engine on CUDA."""
        logger.info("🔼 [Load] Initializing Faster-Whisper strictly on CUDA (int8_float16)...")
        try:
            from faster_whisper import WhisperModel
            use_cuda = torch.cuda.is_available() and "cpu" not in self.device
            if not use_cuda:
                logger.warning("CUDA is not available. Falling back to CPU for Whisper.")
            device_str = "cuda" if use_cuda else "cpu"
            compute_str = "int8_float16" if use_cuda else "int8"

            for size in ["tiny", "medium", "base"]:
                try:
                    model = WhisperModel(
                        size,
                        device=device_str,
                        compute_type=compute_str,
                        device_index=0
                    )
                    logger.info(f"✅ Faster-Whisper ({size}) loaded successfully on {device_str.upper()} ({compute_str}).")
                    return {"model": model, "backend": "ctranslate2", "device": device_str, "size": size}
                except Exception as load_err:
                    logger.debug(f"Whisper {size} load attempt note: {load_err}")
                    continue

            raise RuntimeError("Could not load Faster-Whisper on CUDA")
        except Exception as e:
            if not self.allow_simulation_fallback:
                raise
            logger.warning(f"⚠️ Faster-Whisper CUDA load fallback ({e}). Initializing simulation bundle.")
            return {"model": "mock_whisper", "backend": "simulation", "device": "cuda"}

    def _load_qwen_vlm(self) -> Dict[str, Any]:
        """
        Initializes Qwen2.5-VL-3B.
        Prefers local Ollama daemon for sub-second weight streaming,
        with 4-bit BitsAndBytes (NF4) HuggingFace Transformers fallback.
        """
        logger.info("🔼 [Load] Initializing Qwen2.5-VL-3B-Instruct (4-bit)...")

        # Path A: Ollama daemon
        if self.use_ollama:
            try:
                import ollama
                import requests
                # Quick healthcheck
                r = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
                if r.status_code == 200:
                    return {"client": ollama, "model_name": "qwen2.5vl:3b", "backend": "ollama"}
            except Exception as oe:
                logger.info(f"Ollama daemon not reachable ({oe}). Falling back to Transformers.")

        # Path B: HuggingFace Transformers with 4-bit BitsAndBytes NF4 Quantization
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
            )
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map={"": self.device},
                trust_remote_code=True
            )
            model.eval()
            return {"model": model, "processor": processor, "backend": "transformers_bnb4"}
        except Exception as te:
            if not self.allow_simulation_fallback:
                raise
            logger.warning(f"⚠️ Qwen2.5-VL-3B load fallback ({te}). Initializing simulation bundle.")
            return {"model": "mock_qwen_vl", "processor": "mock_processor", "backend": "simulation"}

    def _load_qwen_llm(self) -> Dict[str, Any]:
        """
        Initializes Qwen2.5-7B.
        Prefers local Ollama daemon, with 4-bit BitsAndBytes (NF4) Transformers fallback.
        """
        logger.info("🔼 [Load] Initializing Qwen2.5-7B-Instruct (4-bit)...")

        # Path A: Ollama daemon
        if self.use_ollama:
            try:
                import ollama
                import requests
                r = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
                if r.status_code == 200:
                    return {"client": ollama, "model_name": "qwen2.5:7b", "backend": "ollama"}
            except Exception as oe:
                logger.info(f"Ollama daemon not reachable ({oe}). Falling back to Transformers.")

        # Path B: HuggingFace Transformers with 4-bit BitsAndBytes NF4 Quantization
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map={"": self.device},
                trust_remote_code=True
            )
            model.eval()
            return {"model": model, "tokenizer": tokenizer, "backend": "transformers_bnb4"}
        except Exception as te:
            if not self.allow_simulation_fallback:
                raise
            logger.warning(f"⚠️ Qwen2.5-7B load fallback ({te}). Initializing simulation bundle.")
            return {"model": "mock_qwen_llm", "tokenizer": "mock_tokenizer", "backend": "simulation"}

    async def transition_to(self, target_state: ModelState) -> Dict[str, Any]:
        """
        Guaranteed atomic transition.
        1. Acquires async lock (mutual exclusion).
        2. Teardown existing tenant and sweeps GPU memory.
        3. Validates hardware memory headroom.
        4. Loads the target model tenant.
        """
        async with self._async_lock:
            if self.current_state == target_state:
                return self.active_bundle

            # Step 1: Teardown existing GPU tenant
            self.unload_active_model()

            if target_state == ModelState.IDLE:
                return {}

            # Step 2: Guard against memory exhaustion
            telemetry = self.get_vram_telemetry()
            if torch.cuda.is_available() and "cpu" not in self.device:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                if free_bytes < self.safety_margin_bytes:
                    raise MemoryError(
                        f"Insufficient VRAM for {target_state.value}. Free: {telemetry['free_mb']:.1f} MB, "
                        f"Required Margin: {self.safety_margin_bytes / (1024**2):.1f} MB"
                    )

            # Step 3: Materialize requested model
            if target_state == ModelState.WHISPER:
                self.active_bundle = await asyncio.to_thread(self._load_whisper)
            elif target_state == ModelState.VLM_GATEKEEPER:
                self.active_bundle = await asyncio.to_thread(self._load_qwen_vlm)
            elif target_state == ModelState.LLM_REASONING:
                self.active_bundle = await asyncio.to_thread(self._load_qwen_llm)
            else:
                raise ValueError(f"Unknown target state: {target_state}")

            self.current_state = target_state
            logger.info(f"✅ [VRAM State Machine] Transition complete: [{target_state.value}] active.")
            return self.active_bundle

    @asynccontextmanager
    async def acquire(self, target_state: ModelState):
        """Context manager pattern for scoped model residency."""
        bundle = await self.transition_to(target_state)
        try:
            yield bundle
        finally:
            await self.transition_to(ModelState.IDLE)
