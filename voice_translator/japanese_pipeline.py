from __future__ import annotations

import base64
import math
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .audio_validation import AudioInfo, wav_duration_ms
from .config import Settings
from .prototype_errors import PrototypeError

SEAMLESS_SOURCE_CODES = {"en": "eng", "hi": "hin"}


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def process_memory_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 3)
    except (OSError, ValueError, IndexError):
        pass
    return None


@dataclass(frozen=True)
class JapaneseGeneration:
    translated_text: str
    english_reference: str
    text_generation_ms: float
    speech_generation_ms: float
    reference_translation_ms: float | None
    model_load_ms: float
    cold_start: bool


@dataclass(frozen=True)
class ConversionTimings:
    speaker_embedding_ms: float
    voice_conversion_ms: float
    model_load_ms: float
    cold_start: bool


class JapaneseGenerator(Protocol):
    model_id: str

    def generate(
        self, source_transcript: str, source_language: str, output_path: Path
    ) -> JapaneseGeneration: ...


class Converter(Protocol):
    model_id: str

    def convert(
        self, base_audio: Path, reference_audio: Path, output_audio: Path
    ) -> ConversionTimings: ...


class SeamlessJapaneseGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.seamless_model
        self._processor = None
        self._model = None
        self._torch = None
        self._load_lock = threading.Lock()
        self._last_load_ms = 0.0

    def _load(self) -> tuple[object, object, object, bool, float]:
        cold = self._model is None
        load_ms = 0.0
        if cold:
            with self._load_lock:
                cold = self._model is None
                if cold:
                    started = time.perf_counter()
                    import torch
                    from transformers import AutoProcessor

                    if "v2" in self.model_id:
                        from transformers import SeamlessM4Tv2Model as ModelClass
                    else:
                        from transformers import SeamlessM4TModel as ModelClass
                    dtype = getattr(torch, self.settings.seamless_dtype)
                    processor = AutoProcessor.from_pretrained(self.model_id)
                    model = ModelClass.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    ).to(self.settings.seamless_device)
                    model.eval()
                    self._processor, self._model, self._torch = processor, model, torch
                    load_ms = elapsed_ms(started)
                    self._last_load_ms = load_ms
        return self._processor, self._model, self._torch, cold, load_ms

    def generate(
        self, source_transcript: str, source_language: str, output_path: Path
    ) -> JapaneseGeneration:
        if source_language not in SEAMLESS_SOURCE_CODES:
            raise PrototypeError(
                "translation",
                "unsupported_source",
                "Source language must be English or Hindi.",
            )
        try:
            processor, model, torch, cold, load_ms = self._load()
        except Exception as exc:
            raise PrototypeError(
                "model_loading",
                "seamless_load_failed",
                "The Japanese translation model could not be loaded.",
                status_code=503,
            ) from exc
        inputs = processor(
            text=source_transcript,
            src_lang=SEAMLESS_SOURCE_CODES[source_language],
            return_tensors="pt",
        ).to(self.settings.seamless_device)
        try:
            started = time.perf_counter()
            with torch.inference_mode():
                text_tokens = model.generate(
                    **inputs, tgt_lang="jpn", generate_speech=False
                )
            translated_text = _decode_generation(processor, text_tokens)
            text_ms = elapsed_ms(started)
            if not translated_text.strip():
                raise ValueError("empty Japanese text")

            reference_ms = None
            if source_language == "en":
                english_reference = source_transcript.strip()
            else:
                started = time.perf_counter()
                with torch.inference_mode():
                    reference_tokens = model.generate(
                        **inputs, tgt_lang="eng", generate_speech=False
                    )
                english_reference = _decode_generation(processor, reference_tokens)
                reference_ms = elapsed_ms(started)
                if not english_reference.strip():
                    raise ValueError("empty English reference")

            started = time.perf_counter()
            with torch.inference_mode():
                speech_output = model.generate(**inputs, tgt_lang="jpn")
            waveform = _extract_waveform(speech_output).detach().cpu().float().numpy()
            speech_ms = elapsed_ms(started)
            sampling_rate = int(model.config.sampling_rate)
            _write_float_wav(output_path, waveform, sampling_rate)
        except PrototypeError:
            raise
        except Exception as exc:
            raise PrototypeError(
                "seamless",
                "generation_failed",
                "Japanese translation or speech generation failed.",
                status_code=503,
            ) from exc
        return JapaneseGeneration(
            translated_text=translated_text.strip(),
            english_reference=english_reference.strip(),
            text_generation_ms=text_ms,
            speech_generation_ms=speech_ms,
            reference_translation_ms=reference_ms,
            model_load_ms=load_ms,
            cold_start=cold,
        )


class OpenVoiceJapaneseConverter:
    model_id = "OpenVoice V2 converter"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._converter = None
        self._load_lock = threading.Lock()

    def _load(self):
        cold = self._converter is None
        load_ms = 0.0
        if cold:
            with self._load_lock:
                cold = self._converter is None
                if cold:
                    converter_dir = self.settings.openvoice_dir / "converter"
                    config = converter_dir / "config.json"
                    checkpoint = converter_dir / "checkpoint.pth"
                    missing = [
                        str(path)
                        for path in (config, checkpoint)
                        if not path.is_file() or path.stat().st_size == 0
                    ]
                    if missing:
                        raise PrototypeError(
                            "model_loading",
                            "openvoice_checkpoint_missing",
                            "OpenVoice converter checkpoints are not installed.",
                            status_code=503,
                        )
                    started = time.perf_counter()
                    try:
                        from openvoice.api import (
                            OpenVoiceBaseClass,
                            ToneColorConverter,
                        )

                        # Pinned OpenVoice forwards `enable_watermark` to its base
                        # constructor, which rejects the argument. Constructing the
                        # converter base explicitly disables optional watermarking
                        # without changing tone-colour conversion.
                        converter = ToneColorConverter.__new__(ToneColorConverter)
                        OpenVoiceBaseClass.__init__(
                            converter, str(config), device="cpu"
                        )
                        converter.watermark_model = None
                        converter.version = getattr(
                            converter.hps, "_version_", "v1"
                        )
                        converter.load_ckpt(str(checkpoint))
                    except Exception as exc:
                        raise PrototypeError(
                            "model_loading",
                            "openvoice_load_failed",
                            "OpenVoice could not load its converter checkpoint.",
                            status_code=503,
                        ) from exc
                    self._converter = converter
                    load_ms = elapsed_ms(started)
        return self._converter, cold, load_ms

    def convert(
        self, base_audio: Path, reference_audio: Path, output_audio: Path
    ) -> ConversionTimings:
        converter, cold, load_ms = self._load()
        try:
            started = time.perf_counter()
            # Input audio is already duration/voice validated and normalized.
            # Calling OpenVoice's se_extractor would redundantly run its legacy
            # Whisper/VAD stack (and its `vad=False` branch still invokes Whisper).
            # The converter's native encoder is the actual embedding operation.
            target_se = converter.extract_se([str(reference_audio)])
            source_se = converter.extract_se([str(base_audio)])
            embedding_ms = elapsed_ms(started)
            started = time.perf_counter()
            converter.convert(
                audio_src_path=str(base_audio),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_audio),
                message="@voice-translator",
            )
            conversion_ms = elapsed_ms(started)
        except Exception as exc:
            raise PrototypeError(
                "voice_conversion",
                "conversion_failed",
                "Best-effort voice conversion failed.",
                status_code=503,
            ) from exc
        if not output_audio.is_file() or output_audio.stat().st_size < 44:
            raise PrototypeError(
                "voice_conversion",
                "invalid_converted_audio",
                "Voice conversion did not produce valid audio.",
                status_code=503,
            )
        return ConversionTimings(
            speaker_embedding_ms=embedding_ms,
            voice_conversion_ms=conversion_ms,
            model_load_ms=load_ms,
            cold_start=cold,
        )


class JapanesePrototypePipeline:
    def __init__(
        self,
        asr,
        generator: JapaneseGenerator,
        converter: Converter,
        settings: Settings,
    ) -> None:
        self.asr = asr
        self.generator = generator
        self.converter = converter
        self.settings = settings
        self._inference_lock = threading.Lock()

    def run(
        self,
        normalized_audio: Path,
        source_language: str,
        audio_info: AudioInfo,
        work_dir: Path,
        request_id: str,
    ) -> dict:
        if source_language not in SEAMLESS_SOURCE_CODES:
            raise PrototypeError(
                "request",
                "unsupported_source",
                "Source language must be English or Hindi.",
            )
        if not self._inference_lock.acquire(timeout=self.settings.inference_wait_seconds):
            raise PrototypeError(
                "queue",
                "inference_busy",
                "Another translation is running. Please try again shortly.",
                status_code=429,
            )
        total_started = time.perf_counter()
        memory_before = process_memory_mb()
        base_audio = work_dir / "japanese-base.wav"
        final_audio = work_dir / "japanese-voice.wav"
        try:
            asr_load_before = getattr(self.asr, "_model", object()) is None
            started = time.perf_counter()
            try:
                source_transcript = self.asr.transcribe(
                    normalized_audio, source_language
                )
            except Exception as exc:
                raise PrototypeError(
                    "asr",
                    "transcription_failed",
                    "Speech transcription failed.",
                    status_code=503,
                ) from exc
            asr_ms = elapsed_ms(started)
            if not source_transcript.strip():
                raise PrototypeError(
                    "asr", "no_speech", "No speech was transcribed."
                )

            generation = self.generator.generate(
                source_transcript, source_language, base_audio
            )
            conversion = self.converter.convert(
                base_audio, normalized_audio, final_audio
            )
            final_bytes = final_audio.read_bytes()
            if final_bytes[:4] != b"RIFF":
                raise PrototypeError(
                    "voice_conversion",
                    "invalid_wav",
                    "Voice conversion returned invalid WAV audio.",
                    status_code=503,
                )
            total_ms = elapsed_ms(total_started)
            output_duration_ms = wav_duration_ms(final_audio)
            timings = {
                "asr_ms": asr_ms,
                "japanese_text_generation_ms": generation.text_generation_ms,
                "japanese_speech_generation_ms": generation.speech_generation_ms,
                "speaker_embedding_ms": conversion.speaker_embedding_ms,
                "voice_conversion_ms": conversion.voice_conversion_ms,
                "total_ms": total_ms,
                "input_duration_ms": round(audio_info.duration_ms, 3),
                "output_duration_ms": round(output_duration_ms, 3),
                "real_time_factor": round(
                    total_ms / audio_info.duration_ms, 4
                ),
            }
            if generation.reference_translation_ms is not None:
                timings["english_reference_translation_ms"] = (
                    generation.reference_translation_ms
                )
            load_ms = generation.model_load_ms + conversion.model_load_ms
            if asr_load_before:
                load_ms += float(getattr(self.asr, "last_load_ms", 0.0))
            if load_ms:
                timings["model_loading_ms"] = round(load_ms, 3)
            return {
                "request_id": request_id,
                "source_language": source_language,
                "target_language": "ja",
                "source_transcript": source_transcript.strip(),
                "translated_text": generation.translated_text,
                "english_reference": generation.english_reference,
                "audio_mime_type": "audio/wav",
                "audio_base64": base64.b64encode(final_bytes).decode("ascii"),
                "models": {
                    "asr": getattr(
                        getattr(self.asr, "settings", None),
                        "whisper_model",
                        "injected-test-double",
                    ),
                    "translation_and_speech": self.generator.model_id,
                    "voice_conversion": self.converter.model_id,
                },
                "timings": timings,
                "cold_start": bool(
                    asr_load_before or generation.cold_start or conversion.cold_start
                ),
                "process_memory_mb": {
                    "before": memory_before,
                    "after": process_memory_mb(),
                },
            }
        finally:
            self._inference_lock.release()


def _decode_generation(processor, output) -> str:
    tokens = output[0]
    if getattr(tokens, "ndim", 1) > 1:
        tokens = tokens[0]
    values = tokens.detach().cpu().tolist() if hasattr(tokens, "detach") else tokens
    return processor.decode(values, skip_special_tokens=True)


def _extract_waveform(output):
    if hasattr(output, "waveform"):
        waveform = output.waveform
    else:
        waveform = output[0]
    while getattr(waveform, "ndim", 1) > 1:
        waveform = waveform[0]
    return waveform


def _write_float_wav(path: Path, waveform, sampling_rate: int) -> None:
    values = waveform.tolist() if hasattr(waveform, "tolist") else list(waveform)
    if not values:
        raise ValueError("empty generated waveform")
    pcm = bytearray()
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite generated waveform")
        numeric = max(-1.0, min(1.0, numeric))
        pcm.extend(int(numeric * 32767).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sampling_rate)
        audio.writeframes(bytes(pcm))


def build_japanese_pipeline(
    settings: Settings | None = None,
) -> JapanesePrototypePipeline:
    from .stages import WhisperASR

    settings = settings or Settings()
    return JapanesePrototypePipeline(
        WhisperASR(settings),
        SeamlessJapaneseGenerator(settings),
        OpenVoiceJapaneseConverter(settings),
        settings,
    )
