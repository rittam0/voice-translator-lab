"""Opt-in acceptance test requiring checkpoints, ffmpeg, and a human recording."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice_translator.pipeline import build_pipeline


@pytest.mark.skipif(
    not os.getenv("VT_REAL_AUDIO"),
    reason="set VT_REAL_AUDIO to a clear English or Hindi speech clip",
)
def test_real_models_end_to_end():
    source = os.getenv("VT_REAL_SOURCE", "en")
    target = "hi" if source == "en" else "en"
    result = build_pipeline().run(Path(os.environ["VT_REAL_AUDIO"]), source, target)
    assert result.transcript.strip()
    assert result.translation.strip()
    output = Path(result.audio_path)
    assert output.read_bytes()[:4] == b"RIFF"
    assert result.timings.asr_ms > 0
    assert result.timings.translation_ms > 0
    assert result.timings.tts_ms > 0
