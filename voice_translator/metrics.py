from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import StageTimings


class MetricsRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(
        self,
        source: str,
        target: str,
        timings: StageTimings,
        success: bool,
        failed_stage: str | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_language": source,
            "target_language": target,
            "success": success,
            "failed_stage": failed_stage,
            **asdict(timings),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
