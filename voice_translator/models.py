from __future__ import annotations

from dataclasses import asdict, dataclass

LANGUAGES = {
    "en": {"name": "English", "indic": "eng_Latn", "whisper": "en", "tts": "EN"},
    "hi": {"name": "Hindi", "indic": "hin_Deva", "whisper": "hi", "tts": "HI"},
}


@dataclass(frozen=True)
class StageTimings:
    asr_ms: float
    translation_ms: float
    tts_ms: float
    total_ms: float


@dataclass(frozen=True)
class TranslationResult:
    transcript: str
    translation: str
    source_language: str
    target_language: str
    audio_path: str
    timings: StageTimings

    def as_dict(self) -> dict:
        value = asdict(self)
        value["timings"] = asdict(self.timings)
        return value


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
