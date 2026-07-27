from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("VT_DATA_DIR", "data"))
    output_dir: Path = Path(os.getenv("VT_OUTPUT_DIR", "data/outputs"))
    whisper_model: str = os.getenv("VT_WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("VT_WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("VT_WHISPER_COMPUTE_TYPE", "int8")
    en_indic_model: str = os.getenv(
        "VT_EN_INDIC_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M"
    )
    indic_en_model: str = os.getenv(
        "VT_INDIC_EN_MODEL", "ai4bharat/indictrans2-indic-en-dist-200M"
    )
    odia_asr_model: str = os.getenv(
        "VT_ODIA_ASR_MODEL",
        "ai4bharat/indicconformer_stt_or_hybrid_ctc_rnnt_large",
    )
    openvoice_dir: Path = Path(os.getenv("VT_OPENVOICE_DIR", "checkpoints_v2"))
    metrics_file: Path = Path(os.getenv("VT_METRICS_FILE", "data/metrics.jsonl"))
    seamless_model: str = os.getenv(
        "VT_SEAMLESS_MODEL", "facebook/hf-seamless-m4t-medium"
    )
    seamless_device: str = os.getenv("VT_SEAMLESS_DEVICE", "cpu")
    seamless_dtype: str = os.getenv("VT_SEAMLESS_DTYPE", "float32")
    qwen_model: str = os.getenv(
        "VT_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    )
    qwen_device: str = os.getenv("VT_QWEN_DEVICE", "cuda:0")
    qwen_dtype: str = os.getenv("VT_QWEN_DTYPE", "float32")
    qwen_attention: str = os.getenv("VT_QWEN_ATTENTION", "eager")
    qwen_seed: int = int(os.getenv("VT_QWEN_SEED", "42"))
    max_upload_bytes: int = int(os.getenv("VT_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    min_audio_seconds: float = float(os.getenv("VT_MIN_AUDIO_SECONDS", "3"))
    max_audio_seconds: float = float(os.getenv("VT_MAX_AUDIO_SECONDS", "15"))
    inference_wait_seconds: float = float(
        os.getenv("VT_INFERENCE_WAIT_SECONDS", "5")
    )
    request_timeout_seconds: float = float(
        os.getenv("VT_REQUEST_TIMEOUT_SECONDS", "600")
    )
    api_token: str = os.getenv("VT_API_TOKEN", "")
    cors_origins: str = os.getenv(
        "VT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
