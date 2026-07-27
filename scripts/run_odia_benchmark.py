"""Opt-in real-model Odia baseline/V2 benchmark; never fabricates evidence."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from voice_translator.audio_validation import normalize_and_validate
from voice_translator.config import Settings
from voice_translator.prototype_errors import PrototypeError
from voice_translator.v1_pipeline import build_v1_pipeline, gpu_memory_mb
from voice_translator.japanese_pipeline import process_memory_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", action="append", default=[], metavar="ITEM_ID=PATH",
        help="Repeat for owner-supplied manifest recordings.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/odia-benchmark"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supplied = dict(value.split("=", 1) for value in args.audio)
    manifest = json.loads(
        Path("benchmarks/odia_acceptance_manifest.json").read_text()
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output / stamp
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pipeline = build_v1_pipeline(Settings())
    rows = []
    for item in manifest["items"]:
        item_id = item["id"]
        for target in ("ja", "fr"):
            row = {
                "item_id": item_id, "target": target,
                "owner_meaning_score": "not measured",
                "owner_voice_score": "not measured",
                "direct_status": "skipped", "v2_status": "skipped",
            }
            path_value = supplied.get(item_id)
            if not path_value:
                row["skip_reason"] = "owner recording not supplied"
                rows.append(row)
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="odia-benchmark-") as directory:
                    work = Path(directory)
                    normalized = work / "normalized.wav"
                    info = normalize_and_validate(
                        Path(path_value), normalized, pipeline.settings
                    )
                    started = time.perf_counter()
                    baseline = pipeline.text_stage.translate_direct_audio(
                        normalized, target
                    )
                    baseline_audio = output / f"{item_id}-direct-{target}.wav"
                    baseline_voice = pipeline.voice_stage.synthesize(
                        baseline["translated_text"], target, normalized, None,
                        baseline_audio, pipeline.settings.qwen_seed,
                        x_vector_only_mode=True,
                    )
                    row.update({
                        "direct_status": "completed",
                        "direct_english_reference": baseline["english_reference"],
                        "direct_translated_text": baseline["translated_text"],
                        "direct_wav": str(baseline_audio),
                        "direct_total_ms": round(
                            (time.perf_counter() - started) * 1000, 3
                        ),
                        "direct_translation_ms": baseline["translation_ms"],
                        "direct_voice_ms": baseline_voice.generation_ms,
                    })
                    v2, audio = pipeline.run(
                        normalized, "ory", target, info, work, uuid.uuid4().hex
                    )
                    v2_audio = output / f"{item_id}-v2-{target}.wav"
                    v2_audio.write_bytes(audio)
                    row.update({
                        "v2_status": "completed",
                        "v2_odia_transcript": v2["source_transcript"],
                        "v2_english_reference": v2["english_reference"],
                        "v2_translated_text": v2["translated_text"],
                        "v2_wav": str(v2_audio),
                        "v2_timings": v2["timings"],
                        "v2_models": v2["models"],
                        "process_memory_mb": process_memory_mb(),
                        "gpu_vram_mb": gpu_memory_mb(),
                    })
            except PrototypeError as exc:
                row.update({
                    "failure_stage": exc.stage, "failure_code": exc.code,
                    "failure_message": exc.message,
                })
            except Exception as exc:
                row.update({
                    "failure_stage": "benchmark", "failure_code": "unexpected",
                    "failure_message": str(exc),
                })
            finally:
                (output / f"{item_id}-{target}-intermediate.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2)
                )
            rows.append(row)
    comparison = {
        "created_at": stamp, "manifest": manifest,
        "results": rows, "measurements": "real execution only",
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2)
    )
    columns = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
