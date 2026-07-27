from __future__ import annotations

import time
import uuid
from pathlib import Path

from .config import Settings
from .metrics import MetricsRecorder
from .models import LANGUAGES, PipelineError, StageTimings, TranslationResult
from .stages import ASR, Translator, VoiceSynthesizer


class VoiceTranslationPipeline:
    def __init__(
        self,
        asr: ASR,
        translator: Translator,
        synthesizer: VoiceSynthesizer,
        settings: Settings,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self.asr = asr
        self.translator = translator
        self.synthesizer = synthesizer
        self.settings = settings
        self.metrics = metrics or MetricsRecorder(settings.metrics_file)

    def run(
        self, audio_path: Path, source: str, target: str
    ) -> TranslationResult:
        if source not in LANGUAGES or target not in LANGUAGES:
            raise ValueError("Supported language codes are 'en' and 'hi'")
        if source == target:
            raise ValueError("Source and target languages must differ")
        self.settings.prepare()
        started = time.perf_counter()
        values = {"asr_ms": 0.0, "translation_ms": 0.0, "tts_ms": 0.0}
        stage = "asr"
        try:
            tick = time.perf_counter()
            transcript = self.asr.transcribe(audio_path, source)
            values["asr_ms"] = (time.perf_counter() - tick) * 1000

            stage = "translation"
            tick = time.perf_counter()
            translation = self.translator.translate(transcript, source, target)
            values["translation_ms"] = (time.perf_counter() - tick) * 1000

            stage = "tts"
            output_path = self.settings.output_dir / f"{uuid.uuid4().hex}.wav"
            tick = time.perf_counter()
            self.synthesizer.synthesize(
                translation, target, audio_path, output_path
            )
            values["tts_ms"] = (time.perf_counter() - tick) * 1000
        except Exception as exc:
            timings = StageTimings(
                **values, total_ms=(time.perf_counter() - started) * 1000
            )
            self.metrics.record(source, target, timings, False, stage)
            raise PipelineError(stage, str(exc)) from exc

        timings = StageTimings(
            **values, total_ms=(time.perf_counter() - started) * 1000
        )
        self.metrics.record(source, target, timings, True)
        return TranslationResult(
            transcript=transcript,
            translation=translation,
            source_language=source,
            target_language=target,
            audio_path=str(output_path),
            timings=timings,
        )


def build_pipeline(settings: Settings | None = None) -> VoiceTranslationPipeline:
    from .stages import IndicTrans2Translator, OpenVoiceSynthesizer, WhisperASR

    settings = settings or Settings()
    return VoiceTranslationPipeline(
        WhisperASR(settings),
        IndicTrans2Translator(settings),
        OpenVoiceSynthesizer(settings),
        settings,
    )
