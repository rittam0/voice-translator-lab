from __future__ import annotations

import json

import pytest

from voice_translator.config import Settings
from voice_translator.metrics import MetricsRecorder
from voice_translator.models import PipelineError
from voice_translator.pipeline import VoiceTranslationPipeline

from .test_stages import FakeASR, FakeTTS, FakeTranslator


def build(tmp_path, translator=None):
    settings = Settings(
        data_dir=tmp_path,
        output_dir=tmp_path / "outputs",
        metrics_file=tmp_path / "metrics.jsonl",
    )
    return VoiceTranslationPipeline(
        FakeASR(),
        translator or FakeTranslator(),
        FakeTTS(),
        settings,
        MetricsRecorder(settings.metrics_file),
    )


def test_pipeline_records_each_stage_latency(wav_sample, tmp_path):
    pipeline = build(tmp_path)
    result = pipeline.run(wav_sample, "en", "hi")
    assert result.translation.startswith("नमस्ते")
    assert result.timings.total_ms >= (
        result.timings.asr_ms
        + result.timings.translation_ms
        + result.timings.tts_ms
    )
    assert pipeline.settings.metrics_file.is_file()
    metric = json.loads(pipeline.settings.metrics_file.read_text().splitlines()[0])
    assert metric["success"] is True
    assert set(("asr_ms", "translation_ms", "tts_ms", "total_ms")) <= metric.keys()


def test_rejects_same_language(wav_sample, tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        build(tmp_path).run(wav_sample, "en", "en")


def test_stage_failure_is_identified_and_logged(wav_sample, tmp_path):
    class BrokenTranslator:
        def translate(self, text, source, target):
            raise RuntimeError("model unavailable")

    pipeline = build(tmp_path, BrokenTranslator())
    with pytest.raises(PipelineError) as caught:
        pipeline.run(wav_sample, "en", "hi")
    assert caught.value.stage == "translation"
    metric = json.loads(pipeline.settings.metrics_file.read_text().splitlines()[0])
    assert metric["success"] is False
    assert metric["failed_stage"] == "translation"
