from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


class FeatureLogger:
    """Append-only logger for plan features and runtime outcomes."""

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append_records(self, records: Iterable[Dict[str, object]]) -> None:
        rows = list(records)
        if not rows:
            return
        df = pd.DataFrame(rows)
        if self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(self.output_path, index=False)
