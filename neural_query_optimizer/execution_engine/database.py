from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class TableStats:
    """Lightweight table statistics for optimizer features/costing."""

    rows: int
    columns: int


class InMemoryDatabase:
    """In-memory table catalog with optional synthetic data generation."""

    def __init__(self) -> None:
        self.tables: Dict[str, pd.DataFrame] = {}

    def add_table(self, name: str, frame: pd.DataFrame) -> None:
        self.tables[name] = frame

    def get_table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Table {name} not found")
        return self.tables[name]

    def table_stats(self, name: str) -> TableStats:
        table = self.get_table(name)
        return TableStats(rows=len(table), columns=len(table.columns))

    def has_index(self, table: str, column: str) -> bool:
        # In this simulator we treat *_id columns as indexed to emulate realistic trade-offs.
        return column.endswith("_id") or column == "id"

    def generate_synthetic(self, num_tables: int, rows_per_table: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        for idx in range(num_tables):
            table_name = f"t{idx + 1}"
            ids = np.arange(1, rows_per_table + 1)
            join_key = rng.integers(1, rows_per_table // 2 + 1, size=rows_per_table)
            value = rng.normal(loc=100 + idx * 5, scale=15, size=rows_per_table)
            category = rng.choice(["A", "B", "C", "D"], size=rows_per_table, p=[0.2, 0.3, 0.25, 0.25])
            frame = pd.DataFrame(
                {
                    "id": ids,
                    "join_id": join_key,
                    "value": value,
                    "category": category,
                }
            )
            self.add_table(table_name, frame)

    def table_names(self) -> List[str]:
        return sorted(self.tables.keys())
