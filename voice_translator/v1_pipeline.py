from __future__ import annotations

import hashlib
import math
import re
import shutil
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .audio_validation import AudioInfo, wav_duration_ms
from .config import Settings
from .japanese_pipeline import process_memory_mb
from .prototype_errors import PrototypeError
from .stages import WhisperASR

SOURCE_CODES = {"en": "eng", "hi": "hin", "ory": "ory"}
TARGETS = {
    "ja": {"seamless": "jpn", "qwen": "Japanese"},
    "fr": {"seamless": "fra", "qwen": "French"},
}
ODIA_ASR_MODEL = "ai4bharat/indicconformer_stt_or_hybrid_ctc_rnnt_large"
ODIA_TRANSLATION_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1024**2, 3)
    except Exception:
        pass
    return None


def split_sentences(text: str) -> list[str]:
    """Split without dropping punctuation or the first/last fragment."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s*", compact)
    return [piece.strip() for piece in pieces if piece.strip()]


def join_wav_files(parts: list[Path], output: Path, silence_ms: int = 200) -> None:
    if not parts:
        raise ValueError("no WAV segments")
    params = None
    chunks: list[bytes] = []
    for part in parts:
        with wave.open(str(part), "rb") as audio:
            current = (
                audio.getnchannels(), audio.getsampwidth(), audio.getframerate()
            )
            if params is None:
                params = current
            if current != params:
                raise ValueError("WAV segment formats differ")
            frames = audio.readframes(audio.getnframes())
            if not frames:
                raise ValueError("empty WAV segment")
            chunks.append(frames)
    channels, width, rate = params
    silence = b"\0" * round(rate * silence_ms / 1000) * channels * width
    with wave.open(str(output), "wb") as joined:
        joined.setnchannels(channels)
        joined.setsampwidth(width)
        joined.setframerate(rate)
        joined.writeframes(silence.join(chunks))


def _valid_segment(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44 or wav_duration_ms(path) < 120:
        return False
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            if audio.getsampwidth() != 2 or len(frames) < 2:
                return False
        samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        return max((abs(value) for value in samples), default=0) >= 32
    except (wave.Error, EOFError, struct.error):
        return False


def _clean_reference(source: Path, output: Path) -> Path:
    """Use the full <=10s clip or the highest-energy 8s window for longer audio."""
    with wave.open(str(source), "rb") as audio:
        channels, width, rate = (
            audio.getnchannels(), audio.getsampwidth(), audio.getframerate()
        )
        frames = audio.readframes(audio.getnframes())
    if channels != 1 or width != 2 or len(frames) <= rate * width * 10:
        shutil.copyfile(source, output)
        return output
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    window = rate * 8
    step = rate
    best_start, best_energy = 0, -1.0
    for start in range(0, max(1, len(samples) - window + 1), step):
        segment = samples[start : start + window]
        energy = sum(value * value for value in segment) / max(1, len(segment))
        if energy > best_energy:
            best_start, best_energy = start, energy
    selected = samples[best_start : best_start + window]
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(struct.pack(f"<{len(selected)}h", *selected))
    return output


@dataclass(frozen=True)
class TranslationTexts:
    source_transcript: str
    english_reference: str
    translated_text: str
    asr_ms: float
    translation_ms: float
    model_load_ms: float
    cold_start: bool
    stage_timings: dict[str, float] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceOutput:
    generation_ms: float
    prompt_ms: float
    model_load_ms: float
    cold_start: bool
    conditioning_mode: str
    segment_timings: list[dict] = field(default_factory=list)


class TextStage(Protocol):
    model_id: str

    def translate(self, audio_path: Path, source: str, target: str) -> TranslationTexts: ...


class OdiaASRStage:
    model_id = ODIA_ASR_MODEL

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self.last_load_ms = 0.0

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    started = time.perf_counter()
                    from nemo.collections.asr.models import (
                        EncDecHybridRNNTCTCBPEModel,
                    )

                    self._model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
                        self.model_id
                    )
                    self.last_load_ms = _elapsed(started)
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        try:
            result = self._load().transcribe([str(audio_path)])
            value = result[0]
            text = getattr(value, "text", value)
            if isinstance(text, (list, tuple)):
                text = text[0]
            text = str(text).strip()
            if not text:
                raise ValueError("empty Odia transcript")
            return text
        except Exception as exc:
            raise PrototypeError(
                "odia_asr", "odia_asr_failed",
                "Odia speech recognition failed.", status_code=503,
            ) from exc


class OdiaToEnglishStage:
    model_id = ODIA_TRANSLATION_MODEL

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._bundle = None
        self._lock = threading.Lock()
        self.last_load_ms = 0.0

    def _load(self):
        if self._bundle is None:
            with self._lock:
                if self._bundle is None:
                    started = time.perf_counter()
                    import torch
                    from IndicTransToolkit.processor import IndicProcessor
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_id, trust_remote_code=True,
                        torch_dtype=torch.float32, low_cpu_mem_usage=True,
                    ).eval()
                    self._bundle = (
                        tokenizer, model, IndicProcessor(inference=True), torch
                    )
                    self.last_load_ms = _elapsed(started)
        return self._bundle

    def translate(self, text: str) -> str:
        try:
            tokenizer, model, processor, torch = self._load()
            batch = processor.preprocess_batch(
                [text], src_lang="ory_Orya", tgt_lang="eng_Latn"
            )
            encoded = tokenizer(
                batch, truncation=True, padding="longest", return_tensors="pt"
            )
            with torch.inference_mode():
                tokens = model.generate(**encoded, num_beams=5, max_length=256)
            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            result = processor.postprocess_batch(decoded, lang="eng_Latn")[0].strip()
            if not result:
                raise ValueError("empty English translation")
            return result
        except Exception as exc:
            raise PrototypeError(
                "odia_to_english", "odia_translation_failed",
                "Odia-to-English translation failed.", status_code=503,
            ) from exc


class SeamlessTextStage:
    def __init__(self, settings: Settings, asr=None) -> None:
        self.settings = settings
        self.model_id = settings.seamless_model
        self.asr = asr or WhisperASR(settings)
        self._processor = self._model = self._torch = None
        self._lock = threading.Lock()

    def _load(self):
        cold, load_ms = self._model is None, 0.0
        if cold:
            with self._lock:
                cold = self._model is None
                if cold:
                    started = time.perf_counter()
                    import torch
                    from transformers import AutoProcessor
                    if "v2" in self.model_id:
                        from transformers import SeamlessM4Tv2Model as Model
                    else:
                        from transformers import SeamlessM4TModel as Model
                    processor = AutoProcessor.from_pretrained(self.model_id)
                    model = Model.from_pretrained(
                        self.model_id,
                        torch_dtype=getattr(torch, self.settings.seamless_dtype),
                        low_cpu_mem_usage=True,
                    ).to(self.settings.seamless_device).eval()
                    self._processor, self._model, self._torch = processor, model, torch
                    load_ms = _elapsed(started)
        return self._processor, self._model, self._torch, cold, load_ms

    @staticmethod
    def _decode(processor, output) -> str:
        tokens = output[0]
        if getattr(tokens, "ndim", 1) > 1:
            tokens = tokens[0]
        return processor.decode(
            tokens.detach().cpu().tolist(), skip_special_tokens=True
        ).strip()

    def translate_english(self, text: str, target: str) -> tuple[str, float, float, bool]:
        processor, model, torch, cold, load_ms = self._load()
        inputs = processor(text=text, src_lang="eng", return_tensors="pt").to(
            self.settings.seamless_device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, tgt_lang=TARGETS[target]["seamless"], generate_speech=False
            )
        translated = self._decode(processor, output)
        if not translated:
            raise ValueError("empty target translation")
        return translated, _elapsed(started), load_ms, cold

    def translate(self, audio_path: Path, source: str, target: str) -> TranslationTexts:
        if source not in {"en", "hi"}:
            raise PrototypeError(
                "routing", "production_odia_requires_v2",
                "Production Odia requests require the Odia V2 cascade.",
            )
        try:
            asr_cold = getattr(self.asr, "_model", None) is None
            started = time.perf_counter()
            transcript = self.asr.transcribe(audio_path, source).strip()
            asr_ms = _elapsed(started)
            asr_load = (
                float(getattr(self.asr, "last_load_ms", 0.0)) if asr_cold else 0.0
            )
            if source == "en":
                english, reference_ms = transcript, 0.0
            else:
                processor, model, torch, seamless_cold, seamless_load = self._load()
                inputs = processor(
                    text=transcript, src_lang="hin", return_tensors="pt"
                ).to(self.settings.seamless_device)
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, tgt_lang="eng", generate_speech=False
                    )
                english = self._decode(processor, output)
                reference_ms = _elapsed(started)
            translated, target_ms, target_load, target_cold = self.translate_english(
                english, target
            )
            load_ms = asr_load + target_load
            if source == "hi":
                load_ms += seamless_load
                target_cold = target_cold or seamless_cold
            return TranslationTexts(
                transcript, english, translated, asr_ms,
                round(reference_ms + target_ms, 3), load_ms,
                asr_cold or target_cold,
                {"source_asr_ms": asr_ms, "english_reference_ms": reference_ms,
                 "target_translation_ms": target_ms},
                {"source_asr": self.settings.whisper_model,
                 "target_translation": self.model_id},
            )
        except PrototypeError:
            raise
        except Exception as exc:
            raise PrototypeError(
                "translation", "translation_failed",
                "Speech transcription or translation failed.", status_code=503,
            ) from exc

    def translate_direct_audio(self, audio_path: Path, target: str) -> dict:
        """Optional Odia benchmark baseline; production routing never calls this."""
        try:
            import soundfile as sf
            processor, model, torch, cold, load_ms = self._load()
            samples, rate = sf.read(str(audio_path), dtype="float32")
            inputs = processor(
                audios=samples, sampling_rate=rate, return_tensors="pt"
            ).to(self.settings.seamless_device)
            started = time.perf_counter()
            with torch.inference_mode():
                english = self._decode(
                    processor, model.generate(
                        **inputs, tgt_lang="eng", generate_speech=False
                    )
                )
                translated = self._decode(
                    processor, model.generate(
                        **inputs, tgt_lang=TARGETS[target]["seamless"],
                        generate_speech=False,
                    )
                )
            return {
                "english_reference": english, "translated_text": translated,
                "translation_ms": _elapsed(started), "model_loading_ms": load_ms,
                "cold_start": cold, "model": self.model_id,
            }
        except Exception as exc:
            raise PrototypeError(
                "odia_direct_baseline", "baseline_failed",
                "Direct Seamless Odia baseline failed.", status_code=503,
            ) from exc


class OdiaV2TextStage:
    model_id = "Odia V2 cascade"

    def __init__(self, odia_asr, odia_to_english, target_stage: SeamlessTextStage):
        self.odia_asr = odia_asr
        self.odia_to_english = odia_to_english
        self.target_stage = target_stage

    def translate(self, audio_path: Path, source: str, target: str) -> TranslationTexts:
        if source != "ory":
            raise PrototypeError("routing", "wrong_odia_route", "Odia V2 requires ory.")
        asr_cold = getattr(self.odia_asr, "_model", None) is None
        started = time.perf_counter()
        transcript = self.odia_asr.transcribe(audio_path)
        asr_ms = _elapsed(started)
        translation_cold = getattr(self.odia_to_english, "_bundle", None) is None
        started = time.perf_counter()
        english = self.odia_to_english.translate(transcript)
        odia_translation_ms = _elapsed(started)
        try:
            translated, target_ms, target_load, target_cold = (
                self.target_stage.translate_english(english, target)
            )
        except Exception as exc:
            raise PrototypeError(
                "target_translation", "target_translation_failed",
                "English-to-target translation failed.", status_code=503,
            ) from exc
        load_ms = target_load
        if asr_cold:
            load_ms += float(getattr(self.odia_asr, "last_load_ms", 0.0))
        if translation_cold:
            load_ms += float(getattr(self.odia_to_english, "last_load_ms", 0.0))
        return TranslationTexts(
            transcript, english, translated, asr_ms,
            round(odia_translation_ms + target_ms, 3), load_ms,
            asr_cold or translation_cold or target_cold,
            {"odia_asr_ms": asr_ms, "odia_to_english_ms": odia_translation_ms,
             "target_translation_ms": target_ms},
            {"odia_asr": self.odia_asr.model_id,
             "odia_to_english": self.odia_to_english.model_id,
             "target_translation": self.target_stage.model_id},
        )


class QwenVoiceStage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.qwen_model
        self._model = self._torch = None
        self._lock = threading.Lock()
        self._prompt_cache: dict[tuple[str, str | None, bool], object] = {}

    def _load(self):
        cold, load_ms = self._model is None, 0.0
        if cold:
            with self._lock:
                cold = self._model is None
                if cold:
                    started = time.perf_counter()
                    import torch
                    from qwen_tts import Qwen3TTSModel
                    self._model = Qwen3TTSModel.from_pretrained(
                        self.model_id, device_map=self.settings.qwen_device,
                        dtype=getattr(torch, self.settings.qwen_dtype),
                        attn_implementation=self.settings.qwen_attention,
                    )
                    self._torch = torch
                    load_ms = _elapsed(started)
        return self._model, self._torch, cold, load_ms

    def synthesize(
        self, text: str, language: str, reference_audio: Path,
        reference_text: str | None, output_path: Path, seed: int,
        *, x_vector_only_mode: bool = False,
    ) -> VoiceOutput:
        sentences = split_sentences(text)
        if not sentences:
            raise PrototypeError(
                "voice_generation", "empty_target_text",
                "Target translation contained no sentences.", status_code=503,
            )
        try:
            model, torch, cold, load_ms = self._load()
            key = (
                hashlib.sha256(reference_audio.read_bytes()).hexdigest(),
                reference_text, x_vector_only_mode,
            )
            started = time.perf_counter()
            prompt = self._prompt_cache.get(key)
            if prompt is None:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=str(reference_audio), ref_text=reference_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
                self._prompt_cache[key] = prompt
            prompt_ms = _elapsed(started)
            parts, segment_timings = [], []
            import soundfile as sf
            for index, sentence in enumerate(sentences):
                segment_path = output_path.with_name(f"segment-{index}.wav")
                success = False
                attempts = []
                for attempt, attempt_seed in enumerate((seed, seed + 1), start=1):
                    torch.manual_seed(attempt_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(attempt_seed)
                    started = time.perf_counter()
                    try:
                        # Do not pass do_sample: owner-verified default sampling.
                        wavs, sample_rate = model.generate_voice_clone(
                            text=sentence, language=TARGETS[language]["qwen"],
                            voice_clone_prompt=prompt,
                        )
                        sf.write(str(segment_path), wavs[0], sample_rate)
                        valid = _valid_segment(segment_path)
                    except Exception:
                        valid = False
                    attempts.append({
                        "attempt": attempt, "seed": attempt_seed,
                        "generation_ms": _elapsed(started), "valid": valid,
                    })
                    if valid:
                        success = True
                        break
                    segment_path.unlink(missing_ok=True)
                if not success:
                    raise PrototypeError(
                        "voice_generation", "segment_generation_failed",
                        f"Voice generation failed for target sentence {index + 1}.",
                        status_code=503,
                    )
                parts.append(segment_path)
                segment_timings.append({
                    "index": index, "text": sentence, "attempts": attempts,
                    "retried": len(attempts) == 2,
                })
            join_wav_files(parts, output_path, silence_ms=200)
            generation_ms = round(sum(
                attempt["generation_ms"]
                for segment in segment_timings for attempt in segment["attempts"]
            ), 3)
            return VoiceOutput(
                generation_ms, prompt_ms, load_ms, cold,
                "speaker_only" if x_vector_only_mode else "transcript_conditioned",
                segment_timings,
            )
        except PrototypeError:
            raise
        except Exception as exc:
            raise PrototypeError(
                "voice_generation", "qwen_generation_failed",
                "Qwen voice generation failed.", status_code=503,
            ) from exc


class V1Pipeline:
    def __init__(
        self, text_stage: TextStage, voice_stage: QwenVoiceStage,
        settings: Settings, odia_stage: TextStage | None = None,
    ) -> None:
        self.text_stage = text_stage
        self.odia_stage = odia_stage
        self.voice_stage = voice_stage
        self.settings = settings
        self._generation_lock = threading.Lock()

    def run(
        self, audio_path: Path, source: str, target: str, audio_info: AudioInfo,
        work_dir: Path, request_id: str,
    ) -> tuple[dict, bytes]:
        if source not in SOURCE_CODES or target not in TARGETS:
            raise PrototypeError(
                "request", "unsupported_language_pair",
                "Use source en, hi, or ory and target ja or fr.",
            )
        if not self._generation_lock.acquire(timeout=self.settings.inference_wait_seconds):
            raise PrototypeError(
                "queue", "inference_busy",
                "Another translation is running. Please try again shortly.",
                status_code=429,
            )
        total_started = time.perf_counter()
        memory_before, vram_before = process_memory_mb(), gpu_memory_mb()
        try:
            if source == "ory":
                if self.odia_stage is None:
                    raise PrototypeError(
                        "routing", "odia_v2_unavailable",
                        "The Odia V2 stages are not configured.", status_code=503,
                    )
                texts = self.odia_stage.translate(audio_path, source, target)
            else:
                texts = self.text_stage.translate(audio_path, source, target)
            reference = audio_path
            reference_trimmed = False
            if source == "ory" and audio_info.duration_ms > 10_000:
                reference = _clean_reference(
                    audio_path, work_dir / "odia-reference.wav"
                )
                reference_trimmed = True
            output_path = work_dir / "translated.wav"
            # Hindi stays speaker-only because Qwen does not natively support
            # Hindi reference text. Odia V2 deliberately evaluates its real ASR text.
            speaker_only = source == "hi"
            voice = self.voice_stage.synthesize(
                texts.translated_text, target, reference,
                None if speaker_only else texts.source_transcript,
                output_path, self.settings.qwen_seed,
                x_vector_only_mode=speaker_only,
            )
            audio_bytes = output_path.read_bytes()
            if audio_bytes[:4] != b"RIFF":
                raise PrototypeError(
                    "voice_generation", "invalid_wav",
                    "Qwen returned invalid WAV audio.", status_code=503,
                )
            elapsed = _elapsed(total_started)
            loading_ms = round(texts.model_load_ms + voice.model_load_ms, 3)
            warm_ms = round(max(0.0, elapsed - loading_ms), 3)
            timings = {
                "model_loading_ms": loading_ms,
                "inference_ms": warm_ms,
                "asr_ms": texts.asr_ms,
                "translation_ms": texts.translation_ms,
                "qwen_prompt_ms": voice.prompt_ms,
                "qwen_voice_generation_ms": voice.generation_ms,
                "voice_segments": voice.segment_timings,
                "input_duration_ms": round(audio_info.duration_ms, 3),
                "output_duration_ms": round(wav_duration_ms(output_path), 3),
                "real_time_factor": round(warm_ms / audio_info.duration_ms, 4),
                **texts.stage_timings,
            }
            return {
                "request_id": request_id, "source_language": source,
                "target_language": target,
                "source_transcript": texts.source_transcript,
                "english_reference": texts.english_reference,
                "translated_text": texts.translated_text,
                "audio_mime_type": "audio/wav",
                "models": {**texts.models, "voice_generation": self.voice_stage.model_id},
                "conditioning_mode": voice.conditioning_mode,
                "reference_trimmed": reference_trimmed,
                "seed": self.settings.qwen_seed,
                "cold_start": texts.cold_start or voice.cold_start,
                "timings": timings,
                "process_memory_mb": {
                    "before": memory_before, "after": process_memory_mb()
                },
                "gpu_vram_mb": {"before": vram_before, "after": gpu_memory_mb()},
            }, audio_bytes
        finally:
            self._generation_lock.release()


def build_v1_pipeline(settings: Settings | None = None) -> V1Pipeline:
    settings = settings or Settings()
    target_stage = SeamlessTextStage(settings)
    odia_stage = OdiaV2TextStage(
        OdiaASRStage(settings), OdiaToEnglishStage(settings), target_stage
    )
    return V1Pipeline(target_stage, QwenVoiceStage(settings), settings, odia_stage)
