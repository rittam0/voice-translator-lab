from __future__ import annotations

import shutil
import asyncio
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .models import PipelineError
from .audio_validation import normalize_and_validate, validate_upload_metadata
from .config import Settings
from .pipeline import VoiceTranslationPipeline, build_pipeline
from .prototype_errors import PrototypeError
from .v1_pipeline import V1Pipeline, build_v1_pipeline

app = FastAPI(title="Voice Translator", version="0.1.0")
_settings = Settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in _settings.cors_origins.split(",") if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
STATIC_DIR = Path(__file__).with_name("static")
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_v1_pipeline: V1Pipeline | None = None
_v1_pipeline_lock = threading.Lock()
_audio_artifacts: OrderedDict[str, bytes] = OrderedDict()
_audio_lock = threading.Lock()
_MAX_AUDIO_ARTIFACTS = 16


@lru_cache(maxsize=1)
def get_pipeline() -> VoiceTranslationPipeline:
    return build_pipeline()


def get_v1_pipeline() -> V1Pipeline:
    global _v1_pipeline
    if _v1_pipeline is None:
        with _v1_pipeline_lock:
            if _v1_pipeline is None:
                _v1_pipeline = build_v1_pipeline(_settings)
    return _v1_pipeline


@app.get("/", include_in_schema=False)
def index():
    if FRONTEND_DIST.is_dir():
        return FileResponse(FRONTEND_DIST / "index.html")
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": "lazy"}


@app.post("/api/translate")
async def translate_audio(
    audio: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    authorization: str | None = Header(None),
) -> dict:
    _require_token(authorization)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _translate_audio_sync, audio, source_language, target_language
            ),
            timeout=_settings.request_timeout_seconds,
        )
    except TimeoutError as exc:
        error = PrototypeError(
            "request", "request_timeout",
            "Translation timed out. Try again after the model is warm.",
            status_code=504,
        )
        raise HTTPException(error.status_code, detail=error.detail("timeout")) from exc


def _translate_audio_sync(
    audio: UploadFile,
    source_language: str,
    target_language: str,
) -> dict:
    return _translate_v1(audio, source_language, target_language)


def _translate_v1(
    audio: UploadFile, source_language: str, target_language: str
) -> dict:
    request_id = uuid.uuid4().hex
    request_started = time.perf_counter()
    settings = get_v1_pipeline().settings
    if source_language not in {"en", "hi", "ory"} or target_language not in {"ja", "fr"}:
        error = PrototypeError(
            "request",
            "unsupported_language_pair",
            "Use source en, hi, or ory and target ja or fr.",
        )
        raise HTTPException(error.status_code, detail=error.detail(request_id))
    try:
        with tempfile.TemporaryDirectory(prefix="voice-translator-") as directory:
            work_dir = Path(directory)
            raw_path = work_dir / "upload"
            size = _copy_upload_bounded(audio, raw_path, settings.max_upload_bytes)
            suffix = validate_upload_metadata(
                audio.filename, audio.content_type, size, settings
            )
            source_path = raw_path.with_suffix(suffix)
            raw_path.rename(source_path)
            normalized_path = work_dir / "normalized.wav"
            decode_started = time.perf_counter()
            audio_info = normalize_and_validate(
                source_path, normalized_path, settings
            )
            decode_ms = round((time.perf_counter() - decode_started) * 1000, 3)
            payload, final_audio = get_v1_pipeline().run(
                normalized_path,
                source_language,
                target_language,
                audio_info,
                work_dir,
                request_id,
            )
            total_ms = round((time.perf_counter() - request_started) * 1000, 3)
            payload["timings"]["decode_normalization_ms"] = decode_ms
            payload["timings"]["total_ms"] = total_ms
            payload["timings"]["real_time_factor"] = round(
                payload["timings"]["inference_ms"] / audio_info.duration_ms, 4
            )
            payload["timings"]["request_total_ms"] = total_ms
            artifact_id = f"{request_id}.wav"
            _store_audio(artifact_id, final_audio)
            payload["audio_url"] = f"/api/audio/{artifact_id}"
            return payload
    except PrototypeError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.detail(request_id)
        ) from exc
    except Exception as exc:
        error = PrototypeError(
            "internal",
            "unexpected_failure",
            "The translation request failed unexpectedly.",
            status_code=500,
        )
        raise HTTPException(
            status_code=error.status_code, detail=error.detail(request_id)
        ) from exc


def _store_audio(name: str, content: bytes) -> None:
    with _audio_lock:
        _audio_artifacts[name] = content
        _audio_artifacts.move_to_end(name)
        while len(_audio_artifacts) > _MAX_AUDIO_ARTIFACTS:
            _audio_artifacts.popitem(last=False)


def _require_token(authorization: str | None) -> None:
    if not _settings.api_token:
        return
    if authorization != f"Bearer {_settings.api_token}":
        raise HTTPException(
            status_code=401,
            detail={
                "stage": "request",
                "code": "unauthorized",
                "message": "A valid API token is required.",
            },
        )


def _copy_upload_bounded(
    audio: UploadFile, destination: Path, max_bytes: int
) -> int:
    size = 0
    with destination.open("wb") as output:
        while chunk := audio.file.read(64 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise PrototypeError(
                    "decode",
                    "audio_too_large",
                    "The recording exceeds the 5 MB upload limit.",
                    status_code=413,
                )
            output.write(chunk)
    return size


def _translate_legacy(
    audio: UploadFile, source_language: str, target_language: str
) -> dict:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        input_path = Path(temporary.name)
        shutil.copyfileobj(audio.file, temporary)
    try:
        if input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Recording is empty")
        result = get_pipeline().run(
            input_path, source_language, target_language
        )
        payload = result.as_dict()
        payload["audio_url"] = f"/api/audio/{Path(result.audio_path).name}"
        payload.pop("audio_path")
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PipelineError as exc:
        raise HTTPException(
            status_code=503,
            detail={"stage": exc.stage, "message": str(exc)},
        ) from exc
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/api/audio/{filename}")
def translated_audio(
    filename: str, authorization: str | None = Header(None)
):
    _require_token(authorization)
    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid audio filename")
    with _audio_lock:
        content = _audio_artifacts.get(filename)
    if content is not None:
        from fastapi.responses import Response

        return Response(
            content=content,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )
    path = get_pipeline().settings.output_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
