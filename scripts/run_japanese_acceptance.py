from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import tempfile
import time
import uuid
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_translator.audio_validation import (
    normalize_and_validate,
    validate_upload_metadata,
)
from voice_translator.config import Settings
from voice_translator.japanese_pipeline import build_japanese_pipeline
from voice_translator.prototype_errors import PrototypeError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real English/Hindi→Japanese prototype acceptance."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source", choices=("en", "hi"), required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/prototype-acceptance")
    )
    args = parser.parse_args()

    settings = Settings()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{args.source}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "result.json"
    started = time.perf_counter()
    try:
        size = args.audio.stat().st_size
        mime = mimetypes.guess_type(args.audio.name)[0] or "audio/wav"
        validate_upload_metadata(args.audio.name, mime, size, settings)
        with tempfile.TemporaryDirectory(prefix="voice-acceptance-") as temporary:
            normalized = Path(temporary) / "normalized.wav"
            info = normalize_and_validate(args.audio, normalized, settings)
            pipeline = build_japanese_pipeline(settings)
            result = pipeline.run(
                normalized, args.source, info, run_dir, run_id
            )
        result["audio_base64"] = "<saved as final-japanese.wav>"
        (run_dir / "japanese-voice.wav").replace(run_dir / "final-japanese.wav")
        result["artifacts"] = {
            "base_japanese_audio": "japanese-base.wav",
            "final_converted_audio": "final-japanese.wav",
            "result": "result.json",
        }
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(report_path.resolve())
    except Exception as exc:
        if isinstance(exc, PrototypeError):
            error = {
                "stage": exc.stage,
                "code": exc.code,
                "message": exc.message,
            }
        else:
            error = {
                "stage": "acceptance",
                "code": "unexpected_failure",
                "message": type(exc).__name__,
            }
        error["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        report_path.write_text(
            json.dumps({"status": "failed", "error": error}, indent=2),
            encoding="utf-8",
        )
        print(f"Acceptance failed; diagnostic saved to {report_path.resolve()}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
