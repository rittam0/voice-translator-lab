from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Protocol

from .config import Settings
from .models import LANGUAGES


class ASR(Protocol):
    def transcribe(self, audio_path: Path, language: str) -> str: ...


class Translator(Protocol):
    def translate(self, text: str, source: str, target: str) -> str: ...


class VoiceSynthesizer(Protocol):
    def synthesize(
        self, text: str, target: str, reference_audio: Path, output_path: Path
    ) -> None: ...


class WhisperASR:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self.last_load_ms = 0.0

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    started = time.perf_counter()
                    self._model = WhisperModel(
                        self.settings.whisper_model,
                        device=self.settings.whisper_device,
                        compute_type=self.settings.whisper_compute_type,
                    )
                    self.last_load_ms = (time.perf_counter() - started) * 1000
        return self._model

    def transcribe(self, audio_path: Path, language: str) -> str:
        segments, _ = self._load().transcribe(
            str(audio_path),
            language=LANGUAGES[language]["whisper"],
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise ValueError("No speech was detected in the recording")
        return text


class IndicTrans2Translator:
    """CPU-compatible distilled IndicTrans2 adapter, loaded by direction."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._bundles: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def _load(self, direction: str):
        if direction not in self._bundles:
            with self._lock:
                if direction not in self._bundles:
                    import torch
                    from IndicTransToolkit.processor import IndicProcessor
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                    model_id = (
                        self.settings.en_indic_model
                        if direction == "en-indic"
                        else self.settings.indic_en_model
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, trust_remote_code=True
                    )
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    ).eval()
                    self._bundles[direction] = (
                        tokenizer,
                        model,
                        IndicProcessor(inference=True),
                        torch,
                    )
        return self._bundles[direction]

    def translate(self, text: str, source: str, target: str) -> str:
        direction = "en-indic" if source == "en" else "indic-en"
        tokenizer, model, processor, torch = self._load(direction)
        source_code = LANGUAGES[source]["indic"]
        target_code = LANGUAGES[target]["indic"]
        batch = processor.preprocess_batch(
            [text], src_lang=source_code, tgt_lang=target_code
        )
        encoded = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        with torch.inference_mode():
            generated = model.generate(
                **encoded, num_beams=5, num_return_sequences=1, max_length=256
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translated = processor.postprocess_batch(decoded, lang=target_code)[0].strip()
        if not translated:
            raise ValueError("Translation model returned empty text")
        return translated


class OpenVoiceSynthesizer:
    """MeloTTS base voice + OpenVoice v2 tone-colour conversion."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._converter = None
        self._speakers: dict[str, object] = {}
        self._lock = threading.Lock()

    def _load_converter(self):
        if self._converter is None:
            with self._lock:
                if self._converter is None:
                    from openvoice.api import ToneColorConverter

                    converter_dir = self.settings.openvoice_dir / "converter"
                    converter = ToneColorConverter(
                        str(converter_dir / "config.json"), device="cpu"
                    )
                    converter.load_ckpt(str(converter_dir / "checkpoint.pth"))
                    self._converter = converter
        return self._converter

    def _load_speaker(self, target: str):
        language = LANGUAGES[target]["tts"]
        if language not in self._speakers:
            with self._lock:
                if language not in self._speakers:
                    if target == "en":
                        from melo.api import TTS

                        self._speakers[language] = TTS(
                            language=language, device="cpu"
                        )
                    else:
                        from transformers import AutoTokenizer, VitsModel

                        self._speakers[language] = (
                            AutoTokenizer.from_pretrained("facebook/mms-tts-hin"),
                            VitsModel.from_pretrained("facebook/mms-tts-hin").eval(),
                        )
        return self._speakers[language]

    def synthesize(
        self, text: str, target: str, reference_audio: Path, output_path: Path
    ) -> None:
        from openvoice import se_extractor

        converter = self._load_converter()
        speaker = self._load_speaker(target)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_path = output_path.with_suffix(".base.wav")
        target_se, _ = se_extractor.get_se(
            str(reference_audio), converter, vad=True
        )
        if target == "en":
            speaker_ids = speaker.hps.data.spk2id
            speaker_key = next(iter(speaker_ids))
            speaker.tts_to_file(
                text, speaker_ids[speaker_key], str(base_path), speed=1.0, quiet=True
            )
        else:
            import soundfile as sf
            import torch

            tokenizer, model = speaker
            inputs = tokenizer(text, return_tensors="pt")
            with torch.inference_mode():
                waveform = model(**inputs).waveform[0].cpu().float().numpy()
            sf.write(base_path, waveform, model.config.sampling_rate)
        source_se, _ = se_extractor.get_se(str(base_path), converter, vad=False)
        converter.convert(
            audio_src_path=str(base_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(output_path),
            message="@voice-translator",
        )
        base_path.unlink(missing_ok=True)
        if not output_path.exists() or output_path.stat().st_size < 44:
            raise ValueError("TTS did not produce valid audio")


class PassthroughSynthesizer:
    """Explicit development fallback; never enabled in production automatically."""

    def synthesize(
        self, text: str, target: str, reference_audio: Path, output_path: Path
    ) -> None:
        shutil.copyfile(reference_audio, output_path)
