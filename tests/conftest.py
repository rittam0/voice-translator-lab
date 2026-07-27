from __future__ import annotations

import wave
from pathlib import Path

import pytest


@pytest.fixture
def wav_sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 1_600)
    return path
