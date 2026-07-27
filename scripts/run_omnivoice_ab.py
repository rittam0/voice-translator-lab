"""Optional isolated OmniVoice experiment. Failure must not block Qwen acceptance."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    command = [
        args.python, "-m", "omnivoice.cli.infer",
        "--model", "k2-fsa/OmniVoice",
        "--text", args.text,
        "--ref_audio", str(args.reference_audio),
        "--ref_text", args.reference_text,
        "--language_id", "ory",
        "--output", str(args.output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    status = {
        "status": "completed" if completed.returncode == 0 else "skipped",
        "language_id": "ory", "command": command,
        "reason": None if completed.returncode == 0 else (
            completed.stderr.strip() or completed.stdout.strip()
            or f"OmniVoice exited {completed.returncode}"
        ),
    }
    args.status.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
