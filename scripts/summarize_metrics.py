from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize successful real runs")
    parser.add_argument("path", nargs="?", default="data/metrics.jsonl")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["success"]]
    if not rows:
        raise SystemExit("No successful measured runs found")
    print(f"runs: {len(rows)}")
    for key in ("asr_ms", "translation_ms", "tts_ms", "total_ms"):
        values = [float(row[key]) for row in rows]
        print(
            f"{key}: mean={statistics.fmean(values):.1f} "
            f"median={statistics.median(values):.1f} "
            f"min={min(values):.1f} max={max(values):.1f}"
        )


if __name__ == "__main__":
    main()
