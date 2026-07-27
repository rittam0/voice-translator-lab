from io import BytesIO
import asyncio

from starlette.datastructures import Headers, UploadFile

import voice_translator.api as api
from voice_translator.config import Settings
from voice_translator.pipeline import VoiceTranslationPipeline

from .test_stages import FakeASR, FakeTTS, FakeTranslator


def test_health_does_not_load_models():
    assert api.health() == {
        "status": "ok",
        "models": "lazy",
    }


def test_legacy_pipeline_remains_available_off_primary_v1_endpoint(wav_sample, tmp_path):
    settings = Settings(
        output_dir=tmp_path / "outputs",
        metrics_file=tmp_path / "metrics.jsonl",
    )
    fake = VoiceTranslationPipeline(
        FakeASR(), FakeTranslator(), FakeTTS(), settings
    )
    original = api.get_pipeline
    try:
        api.get_pipeline = lambda: fake
        upload = UploadFile(
            BytesIO(wav_sample.read_bytes()),
            filename="sample.wav",
            headers=Headers({"content-type": "audio/wav"}),
        )
        body = api._translate_legacy(upload, "en", "hi")
        assert body["translation"].startswith("नमस्ते")
        assert body["audio_url"].endswith(".wav")
    finally:
        api.get_pipeline = original
