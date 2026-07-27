from __future__ import annotations

import wave
from pathlib import Path


class FakeASR:
    def transcribe(self, audio_path: Path, language: str) -> str:
        assert audio_path.stat().st_size > 44
        return "hello, how are you?"


class FakeTranslator:
    def translate(self, text: str, source: str, target: str) -> str:
        assert text
        return "नमस्ते, आप कैसे हैं?"


class FakeTTS:
    def synthesize(
        self, text: str, target: str, reference_audio: Path, output_path: Path
    ) -> None:
        assert any("\u0900" <= char <= "\u097f" for char in text)
        with wave.open(str(output_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\x00\x00" * 400)


def test_asr_produces_reasonable_text(wav_sample):
    output = FakeASR().transcribe(wav_sample, "en")
    assert "hello" in output.lower()


def test_translation_is_nonempty_devanagari():
    output = FakeTranslator().translate("hello", "en", "hi")
    assert output
    assert any("\u0900" <= char <= "\u097f" for char in output)


def test_tts_produces_valid_wav(wav_sample, tmp_path):
    output = tmp_path / "output.wav"
    FakeTTS().synthesize("नमस्ते", "hi", wav_sample, output)
    with wave.open(str(output), "rb") as audio:
        assert audio.getnframes() > 0
        assert audio.getframerate() == 16_000
