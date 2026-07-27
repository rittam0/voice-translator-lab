from __future__ import annotations

import io
import math
import struct
import sys
import types
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

import voice_translator.api as api
from voice_translator.audio_validation import AudioInfo, normalize_and_validate
from voice_translator.config import Settings
from voice_translator.prototype_errors import PrototypeError
from voice_translator.v1_pipeline import (
    OdiaV2TextStage,
    QwenVoiceStage,
    TranslationTexts,
    V1Pipeline,
    VoiceOutput,
    join_wav_files,
    split_sentences,
)


def wav_bytes(seconds: float = 3.2, amplitude: float = 0.2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        frames = [
            int(amplitude * 32767 * math.sin(2 * math.pi * 220 * i / 16_000))
            for i in range(round(seconds * 16_000))
        ]
        audio.writeframes(struct.pack(f"<{len(frames)}h", *frames))
    return output.getvalue()


def write_wav(path: Path, seconds: float = 0.3) -> None:
    path.write_bytes(wav_bytes(seconds, 0.1))


class FakeText:
    model_id = "fake-seamless"

    def __init__(self):
        self.calls = 0

    def translate(self, audio: Path, source: str, target: str):
        assert source in {"en", "hi"}
        self.calls += 1
        transcript = "Exact English transcript." if source == "en" else "नमस्ते दुनिया।"
        return TranslationTexts(
            transcript,
            transcript if source == "en" else "Hello world.",
            "First sentence. Second sentence." if target == "fr" else "最初です。次です。",
            10.0, 20.0, 100.0 if self.calls == 1 else 0.0,
            self.calls == 1, {"target_translation_ms": 20.0},
            {"source_asr": "fake-asr", "target_translation": "fake-seamless"},
        )


class FakeOdia:
    model_id = "fake-odia-v2"

    def __init__(self):
        self.calls = 0

    def translate(self, audio: Path, source: str, target: str):
        assert source == "ory"
        self.calls += 1
        return TranslationTexts(
            "ମୋର ନାମ ରବି।", "My name is Ravi.",
            "Bonjour." if target == "fr" else "こんにちは。",
            12.0, 18.0, 50.0 if self.calls == 1 else 0.0,
            self.calls == 1,
            {"odia_asr_ms": 12.0, "odia_to_english_ms": 8.0,
             "target_translation_ms": 10.0},
            {"odia_asr": "fake-indicconformer",
             "odia_to_english": "fake-indictrans",
             "target_translation": "fake-seamless"},
        )


class FakeVoice:
    model_id = "fake-qwen"

    def __init__(self):
        self.calls = []

    def synthesize(
        self, text, language, reference_audio, reference_text, output_path, seed,
        *, x_vector_only_mode=False,
    ):
        self.calls.append(
            (language, reference_text, seed, x_vector_only_mode, text)
        )
        write_wav(output_path)
        return VoiceOutput(
            30.0, 5.0, 200.0 if len(self.calls) == 1 else 0.0,
            len(self.calls) == 1,
            "speaker_only" if x_vector_only_mode else "transcript_conditioned",
            [{"index": 0, "text": text, "attempts": [{"attempt": 1, "seed": seed}],
              "retried": False}],
        )


def build_fake(tmp_path: Path):
    return V1Pipeline(
        FakeText(), FakeVoice(), Settings(inference_wait_seconds=0.1), FakeOdia()
    )


@pytest.mark.parametrize(
    ("source", "target", "mode"),
    [
        ("en", "ja", "transcript_conditioned"),
        ("en", "fr", "transcript_conditioned"),
        ("hi", "ja", "speaker_only"),
        ("hi", "fr", "speaker_only"),
        ("ory", "ja", "transcript_conditioned"),
        ("ory", "fr", "transcript_conditioned"),
    ],
)
def test_all_six_routes_and_conditioning(tmp_path, source, target, mode):
    pipeline = build_fake(tmp_path)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(wav_bytes())
    result, audio = pipeline.run(
        input_path, source, target, AudioInfo(3200, 16000, 2800, 0.2),
        tmp_path, "request",
    )
    assert result["conditioning_mode"] == mode
    call = pipeline.voice_stage.calls[0]
    if source == "hi":
        assert call[1] is None and call[3] is True
    else:
        assert call[1] == result["source_transcript"] and call[3] is False
    assert audio[:4] == b"RIFF"
    assert result["timings"]["voice_segments"]


def test_odia_production_selects_v2_not_direct_baseline(tmp_path):
    pipeline = build_fake(tmp_path)
    source = tmp_path / "input.wav"
    source.write_bytes(wav_bytes())
    pipeline.run(
        source, "ory", "ja", AudioInfo(3200, 16000, 2800, 0.2),
        tmp_path, "request",
    )
    assert pipeline.odia_stage.calls == 1
    assert pipeline.text_stage.calls == 0


def test_odia_injected_asr_and_translation_success(tmp_path):
    class ASR:
        model_id = "asr"
        _model = object()
        last_load_ms = 0

        def transcribe(self, path):
            return "ଓଡ଼ିଆ ପାଠ"

    class ToEnglish:
        model_id = "translator"
        _bundle = object()
        last_load_ms = 0

        def translate(self, text):
            assert text == "ଓଡ଼ିଆ ପାଠ"
            return "Odia text"

    class Target:
        model_id = "target"

        def translate_english(self, text, target):
            assert text == "Odia text"
            return "Texte français", 3.0, 0.0, False

    result = OdiaV2TextStage(ASR(), ToEnglish(), Target()).translate(
        tmp_path / "audio.wav", "ory", "fr"
    )
    assert result.source_transcript == "ଓଡ଼ିଆ ପାଠ"
    assert result.english_reference == "Odia text"
    assert result.translated_text == "Texte français"


@pytest.mark.parametrize(
    ("broken", "stage"),
    [("asr", "odia_asr"), ("translation", "odia_to_english"),
     ("target", "target_translation")],
)
def test_odia_stage_failure_is_identified(tmp_path, broken, stage):
    class ASR:
        model_id = "asr"
        _model = object()
        def transcribe(self, path):
            if broken == "asr":
                raise PrototypeError("odia_asr", "failed", "ASR failed")
            return "text"

    class English:
        model_id = "english"
        _bundle = object()
        def translate(self, text):
            if broken == "translation":
                raise PrototypeError("odia_to_english", "failed", "Translation failed")
            return "English"

    class Target:
        model_id = "target"
        def translate_english(self, text, target):
            if broken == "target":
                raise RuntimeError("target internals")
            return "Japanese", 1.0, 0.0, False

    with pytest.raises(PrototypeError) as caught:
        OdiaV2TextStage(ASR(), English(), Target()).translate(
            tmp_path / "audio.wav", "ory", "ja"
        )
    assert caught.value.stage == stage


def test_sentence_split_preserves_first_and_last():
    assert split_sentences("First sentence. Second sentence! Third?") == [
        "First sentence.", "Second sentence!", "Third?"
    ]
    assert split_sentences("最初です。次です。") == ["最初です。", "次です。"]


def test_wav_join_preserves_first_and_inserts_silence(tmp_path):
    first, second, joined = (
        tmp_path / "first.wav", tmp_path / "second.wav", tmp_path / "joined.wav"
    )
    write_wav(first, 0.2)
    write_wav(second, 0.3)
    join_wav_files([first, second], joined, silence_ms=200)
    assert 690 <= _duration(joined) <= 710
    with wave.open(str(joined), "rb") as audio:
        prefix = audio.readframes(50)
    with wave.open(str(first), "rb") as audio:
        assert prefix == audio.readframes(50)


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate() * 1000


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False
    @staticmethod
    def manual_seed(seed):
        pass


def _install_fake_soundfile(monkeypatch):
    def write(path, values, sample_rate):
        amplitude = 0.1 if values else 0.0
        Path(path).write_bytes(wav_bytes(len(values) / sample_rate, amplitude))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(write=write))


def test_short_segment_retries_once_and_all_sentences_generate(tmp_path, monkeypatch):
    _install_fake_soundfile(monkeypatch)

    class Model:
        def __init__(self):
            self.calls = []
        def create_voice_clone_prompt(self, **kwargs):
            return "prompt"
        def generate_voice_clone(self, text, **kwargs):
            self.calls.append(text)
            if len(self.calls) == 1:
                return [[0.1] * 20], 16000
            return [[0.1] * 3200], 16000

    stage = QwenVoiceStage(Settings())
    stage._model, stage._torch = Model(), FakeTorch()
    reference = tmp_path / "reference.wav"
    write_wav(reference)
    result = stage.synthesize(
        "First sentence. Second sentence.", "fr", reference, "Reference.",
        tmp_path / "output.wav", 42,
    )
    assert stage._model.calls == [
        "First sentence.", "First sentence.", "Second sentence."
    ]
    assert result.segment_timings[0]["retried"] is True
    assert result.segment_timings[0]["attempts"][1]["seed"] == 43
    assert _duration(tmp_path / "output.wav") >= 590


def test_invalid_segment_retries_at_most_once_then_errors(tmp_path, monkeypatch):
    _install_fake_soundfile(monkeypatch)

    class Model:
        calls = 0
        def create_voice_clone_prompt(self, **kwargs):
            return "prompt"
        def generate_voice_clone(self, **kwargs):
            self.calls += 1
            return [[]], 16000

    stage = QwenVoiceStage(Settings())
    stage._model, stage._torch = Model(), FakeTorch()
    reference = tmp_path / "reference.wav"
    write_wav(reference)
    with pytest.raises(PrototypeError) as caught:
        stage.synthesize(
            "Sentence.", "ja", reference, "Reference.",
            tmp_path / "output.wav", 42,
        )
    assert stage._model.calls == 2
    assert caught.value.code == "segment_generation_failed"


def test_cold_and_warm_loading_are_separated(tmp_path):
    pipeline = build_fake(tmp_path)
    source = tmp_path / "input.wav"
    source.write_bytes(wav_bytes())
    info = AudioInfo(3200, 16000, 2800, 0.2)
    cold, _ = pipeline.run(source, "en", "ja", info, tmp_path, "one")
    warm, _ = pipeline.run(source, "en", "fr", info, tmp_path, "two")
    assert cold["cold_start"] is True
    assert cold["timings"]["model_loading_ms"] == 300
    assert warm["cold_start"] is False
    assert warm["timings"]["model_loading_ms"] == 0


@pytest.mark.parametrize(("source", "target"), [("de", "ja"), ("en", "de")])
def test_unsupported_pair(source, target, tmp_path):
    with pytest.raises(PrototypeError) as caught:
        build_fake(tmp_path).run(
            tmp_path / "missing.wav", source, target,
            AudioInfo(3200, 16000, 2800, 0.2), tmp_path, "request",
        )
    assert caught.value.code == "unsupported_language_pair"


@pytest.mark.parametrize(
    ("seconds", "amplitude", "code"),
    [(1.0, 0.2, "audio_too_short"), (16.0, 0.2, "audio_too_long"),
     (3.2, 0.0, "insufficient_voice")],
)
def test_duration_and_voice_validation(tmp_path, seconds, amplitude, code):
    source, normalized = tmp_path / "source.wav", tmp_path / "normalized.wav"
    source.write_bytes(wav_bytes(seconds, amplitude))
    with pytest.raises(PrototypeError) as caught:
        normalize_and_validate(source, normalized, Settings())
    assert caught.value.code == code


def _upload(content=wav_bytes(), mime="audio/wav"):
    return UploadFile(
        io.BytesIO(content), filename="recording.wav",
        headers=Headers({"content-type": mime}),
    )


@pytest.mark.parametrize("source", ["en", "hi", "ory"])
def test_complete_api_contract_and_audio_lifecycle(tmp_path, monkeypatch, source):
    fake = build_fake(tmp_path)
    monkeypatch.setattr(api, "get_v1_pipeline", lambda: fake)
    body = api._translate_audio_sync(_upload(), source, "fr")
    required = {
        "request_id", "source_language", "target_language", "source_transcript",
        "english_reference", "translated_text", "audio_mime_type", "audio_url",
        "models", "conditioning_mode", "cold_start", "timings",
        "process_memory_mb", "gpu_vram_mb",
    }
    assert required <= body.keys()
    assert "audio_base64" not in body
    name = body["audio_url"].rsplit("/", 1)[-1]
    response = api.translated_audio(name)
    assert response.body[:4] == b"RIFF"
    api._store_audio("evict.wav", b"RIFF")
    assert api.translated_audio(name).body[:4] == b"RIFF"


def test_api_structured_errors_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "get_v1_pipeline", lambda: build_fake(tmp_path))
    created = []
    original = api.tempfile.TemporaryDirectory
    class Tracking(original):
        def __enter__(self):
            name = super().__enter__()
            created.append(Path(name))
            return name
    monkeypatch.setattr(api.tempfile, "TemporaryDirectory", Tracking)
    with pytest.raises(HTTPException) as caught:
        api._translate_audio_sync(_upload(b"not wav"), "ory", "ja")
    assert caught.value.detail["code"] == "invalid_audio"
    assert created and all(not path.exists() for path in created)


def test_bounded_upload_rejects_size_limit(tmp_path):
    with pytest.raises(PrototypeError) as caught:
        api._copy_upload_bounded(_upload(b"x" * 1025), tmp_path / "upload", 1024)
    assert caught.value.code == "audio_too_large"
