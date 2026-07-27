"""Run the six-route real-model acceptance in one persistent Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from voice_translator.audio_validation import normalize_and_validate
from voice_translator.config import Settings
from voice_translator.prototype_errors import PrototypeError
from voice_translator.v1_pipeline import build_v1_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--hindi", type=Path, required=True)
    parser.add_argument("--odia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = build_v1_pipeline(Settings())
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": "real model execution; owner listening pending",
        "routes": [],
    }
    sources = {"en": args.english, "hi": args.hindi, "ory": args.odia}
    normalized_paths = {}
    normalization_dirs = []
    try:
        for source, original in sources.items():
            directory = tempfile.TemporaryDirectory(prefix=f"accept-{source}-")
            normalization_dirs.append(directory)
            normalized = Path(directory.name) / f"{source}.wav"
            info = normalize_and_validate(original, normalized, pipeline.settings)
            normalized_paths[source] = (normalized, info)
        for source in ("en", "hi"):
            for target in ("ja", "fr"):
                route = {"source": source, "target": target, "status": "failed"}
                try:
                    with tempfile.TemporaryDirectory(prefix="accept-route-") as work:
                        result, audio = pipeline.run(
                            normalized_paths[source][0], source, target,
                            normalized_paths[source][1], Path(work), uuid.uuid4().hex,
                        )
                    wav = args.output / f"{source}-to-{target}.wav"
                    wav.write_bytes(audio)
                    result["audio_file"] = wav.name
                    (args.output / f"{source}-to-{target}.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2)
                    )
                    route.update({"status": "completed", "result": result})
                except PrototypeError as exc:
                    route["failure"] = {
                        "stage": exc.stage, "code": exc.code, "message": exc.message
                    }
                except Exception as exc:
                    route["failure"] = {
                        "stage": "acceptance", "code": "unexpected",
                        "message": str(exc),
                    }
                finally:
                    summary["routes"].append(route)
                    (args.output / "summary.partial.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2)
                    )
        for target in ("ja", "fr"):
            baseline = {"target": target, "status": "failed"}
            try:
                baseline.update(
                    pipeline.text_stage.translate_direct_audio(
                        normalized_paths["ory"][0], target
                    )
                )
                wav = args.output / f"ory-direct-to-{target}.wav"
                voice = pipeline.voice_stage.synthesize(
                    baseline["translated_text"], target, normalized_paths["ory"][0],
                    None, wav, pipeline.settings.qwen_seed, x_vector_only_mode=True,
                )
                baseline.update({
                    "status": "completed", "audio_file": wav.name,
                    "voice_generation_ms": voice.generation_ms,
                })
            except PrototypeError as exc:
                baseline["failure"] = {
                    "stage": exc.stage, "code": exc.code, "message": exc.message
                }
            summary.setdefault("odia_direct_baseline", []).append(baseline)
            (args.output / "summary.partial.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2)
            )
        for target in ("ja", "fr"):
            route = {"source": "ory", "target": target, "status": "failed"}
            try:
                with tempfile.TemporaryDirectory(prefix="accept-route-") as work:
                    result, audio = pipeline.run(
                        normalized_paths["ory"][0], "ory", target,
                        normalized_paths["ory"][1], Path(work), uuid.uuid4().hex,
                    )
                wav = args.output / f"ory-to-{target}.wav"
                wav.write_bytes(audio)
                result["audio_file"] = wav.name
                (args.output / f"ory-to-{target}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2)
                )
                speaker_only = args.output / f"ory-to-{target}-qwen-speaker-only.wav"
                pipeline.voice_stage.synthesize(
                    result["translated_text"], target, normalized_paths["ory"][0],
                    None, speaker_only, pipeline.settings.qwen_seed,
                    x_vector_only_mode=True,
                )
                route.update({"status": "completed", "result": result,
                              "speaker_only_audio": speaker_only.name})
            except PrototypeError as exc:
                route["failure"] = {
                    "stage": exc.stage, "code": exc.code, "message": exc.message
                }
            except Exception as exc:
                route["failure"] = {
                    "stage": "acceptance", "code": "unexpected",
                    "message": str(exc),
                }
            finally:
                summary["routes"].append(route)
                (args.output / "summary.partial.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2)
                )
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        return 0
    finally:
        for directory in normalization_dirs:
            directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
