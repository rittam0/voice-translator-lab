"""Opt-in real-model tests; never counted as lightweight verification."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from voice_translator.audio_validation import normalize_and_validate
from voice_translator.config import Settings
from voice_translator.v1_pipeline import build_v1_pipeline


@pytest.mark.parametrize(
    ("source", "environment_variable"),
    [
        ("en", "VT_REAL_AUDIO_EN"),
        ("hi", "VT_REAL_AUDIO_HI"),
        ("ory", "VT_REAL_AUDIO_ORY"),
    ],
)
@pytest.mark.parametrize("target", ["ja", "fr"])
def test_real_v1_route(source, environment_variable, target):
    value = os.getenv(environment_variable)
    if not value:
        pytest.skip(f"set {environment_variable} to a 3–15 second human recording")
    settings = Settings()
    with tempfile.TemporaryDirectory(prefix="real-japanese-test-") as directory:
        work_dir = Path(directory)
        normalized = work_dir / "normalized.wav"
        info = normalize_and_validate(Path(value), normalized, settings)
        result, audio = build_v1_pipeline(settings).run(
            normalized, source, target, info, work_dir, uuid.uuid4().hex
        )
    assert result["translated_text"].strip()
    assert result["english_reference"].strip()
    assert result["target_language"] == target
    assert audio.startswith(b"RIFF")
