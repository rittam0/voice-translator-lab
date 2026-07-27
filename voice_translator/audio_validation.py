from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .prototype_errors import PrototypeError

ACCEPTED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
}
ACCEPTED_SUFFIXES = {".wav", ".webm", ".ogg", ".oga", ".mp4", ".m4a", ".mp3"}


@dataclass(frozen=True)
class AudioInfo:
    duration_ms: float
    sample_rate: int
    voiced_ms: float
    peak: float


def validate_upload_metadata(
    filename: str | None, content_type: str | None, size: int, settings: Settings
) -> str:
    if size == 0:
        raise PrototypeError("decode", "empty_audio", "The recording is empty.")
    if size > settings.max_upload_bytes:
        raise PrototypeError(
            "decode",
            "audio_too_large",
            "The recording exceeds the 5 MB upload limit.",
            status_code=413,
        )
    suffix = Path(filename or "").suffix.lower()
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if suffix not in ACCEPTED_SUFFIXES or mime not in ACCEPTED_MIME_TYPES:
        raise PrototypeError(
            "decode",
            "unsupported_audio",
            "Use WAV, WebM, Ogg, MP4/M4A, or MP3 audio.",
        )
    return suffix


def normalize_and_validate(
    source_path: Path, output_path: Path, settings: Settings
) -> AudioInfo:
    """Decode to mono 16 kHz PCM WAV and validate duration/voiced content."""
    if source_path.suffix.lower() == ".wav":
        try:
            _normalize_wav_stdlib(source_path, output_path)
        except (wave.Error, EOFError, ValueError, struct.error) as exc:
            raise PrototypeError(
                "decode", "invalid_audio", "The WAV recording could not be decoded."
            ) from exc
    else:
        _normalize_with_ffmpeg(source_path, output_path)
    info = _inspect_pcm_wav(output_path)
    duration = info.duration_ms / 1000
    if duration < settings.min_audio_seconds:
        raise PrototypeError(
            "decode",
            "audio_too_short",
            f"Record at least {settings.min_audio_seconds:g} seconds of speech.",
        )
    if duration > settings.max_audio_seconds:
        raise PrototypeError(
            "decode",
            "audio_too_long",
            f"Keep the recording under {settings.max_audio_seconds:g} seconds.",
        )
    minimum_voiced_ms = min(750.0, info.duration_ms * 0.25)
    if info.voiced_ms < minimum_voiced_ms or info.peak < 0.01:
        raise PrototypeError(
            "decode",
            "insufficient_voice",
            "Not enough clear speech was detected. Speak closer to the microphone.",
        )
    return info


def wav_duration_ms(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate() * 1000
    except (wave.Error, EOFError, ZeroDivisionError):
        return 0.0


def _normalize_wav_stdlib(source_path: Path, output_path: Path) -> None:
    with wave.open(str(source_path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels not in (1, 2) or width != 2 or rate <= 0:
        if shutil.which("ffmpeg"):
            _normalize_with_ffmpeg(source_path, output_path)
            return
        raise ValueError("ffmpeg is required for this WAV encoding")
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    if channels == 2:
        samples = tuple(
            int((samples[index] + samples[index + 1]) / 2)
            for index in range(0, len(samples), 2)
        )
    if rate != 16_000:
        samples = _linear_resample(samples, rate, 16_000)
    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _linear_resample(samples: tuple[int, ...], source_rate: int, target_rate: int):
    if not samples:
        return ()
    output_length = max(1, round(len(samples) * target_rate / source_rate))
    scale = source_rate / target_rate
    result = []
    for output_index in range(output_length):
        position = min(output_index * scale, len(samples) - 1)
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        value = samples[left] * (1 - fraction) + samples[right] * fraction
        result.append(max(-32768, min(32767, round(value))))
    return tuple(result)


def _normalize_with_ffmpeg(source_path: Path, output_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise PrototypeError(
            "decode",
            "ffmpeg_missing",
            "This audio format requires ffmpeg on the backend.",
            status_code=503,
        )
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PrototypeError(
            "decode", "invalid_audio", "The recording could not be decoded."
        ) from exc


def _inspect_pcm_wav(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16_000
            ):
                raise ValueError("normalization invariant failed")
            frames = audio.readframes(audio.getnframes())
            frame_rate = audio.getframerate()
    except (wave.Error, EOFError, ValueError) as exc:
        raise PrototypeError(
            "decode", "invalid_audio", "Normalized audio is invalid."
        ) from exc
    if not frames:
        raise PrototypeError("decode", "empty_audio", "The recording is empty.")
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    normalized = [sample / 32768.0 for sample in samples]
    if not all(math.isfinite(sample) for sample in normalized):
        raise PrototypeError(
            "decode", "non_finite_audio", "The recording contains invalid samples."
        )
    peak = max(abs(sample) for sample in normalized)
    frame_size = 320
    voiced_frames = 0
    total_windows = 0
    for start in range(0, len(normalized), frame_size):
        window = normalized[start : start + frame_size]
        if not window:
            continue
        total_windows += 1
        rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
        if rms >= 0.008:
            voiced_frames += 1
    return AudioInfo(
        duration_ms=len(normalized) / frame_rate * 1000,
        sample_rate=frame_rate,
        voiced_ms=voiced_frames * frame_size / frame_rate * 1000,
        peak=peak,
    )
