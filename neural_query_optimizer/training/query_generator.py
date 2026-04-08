from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class QueryGenerator:
    """Generate synthetic SQL workloads for optimizer training."""

    table_names: List[str]
    seed: int = 42

    def generate(self, count: int) -> List[str]:
        rng = np.random.default_rng(self.seed)
        sqls: List[str] = []

        for _ in range(count):
            tables = list(rng.choice(self.table_names, size=min(3, len(self.table_names)), replace=False))
            base = tables[0]
            sql = f"SELECT * FROM {base}"

            for tbl in tables[1:]:
                sql += f" JOIN {tbl} ON {base}.join_id = {tbl}.join_id"

            if rng.random() < 0.8:
                threshold = float(rng.uniform(80, 130))
                category = rng.choice(["A", "B", "C", "D"])
                sql += (
                    f" WHERE {base}.value > {threshold:.2f}"
                    f" AND {base}.category = '{category}'"
                )

            sqls.append(sql)

        return sqls
