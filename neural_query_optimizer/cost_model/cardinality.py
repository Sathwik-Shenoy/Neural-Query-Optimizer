from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.utils.types import JoinCondition, Predicate


@dataclass
class CardinalityEstimate:
    """Estimated rows and selectivity after applying an operation."""

    rows: float
    selectivity: float


class CardinalityEstimator:
    """Histogram-lite cardinality estimator for filters and equi-joins."""

    def __init__(self, db: InMemoryDatabase) -> None:
        self.db = db

    def estimate_filter_rows(self, table: str, predicates: Iterable[Predicate]) -> CardinalityEstimate:
        base_rows = float(self.db.table_stats(table).rows)
        selectivity = 1.0

        for pred in predicates:
            if pred.table not in (None, table):
                continue
            column_selectivity = self._predicate_selectivity(table, pred)
            selectivity *= column_selectivity

        selectivity = min(max(selectivity, 1e-4), 1.0)
        return CardinalityEstimate(rows=max(1.0, base_rows * selectivity), selectivity=selectivity)

    def estimate_join_rows(
        self,
        left_rows: float,
        right_rows: float,
        condition: JoinCondition,
    ) -> CardinalityEstimate:
        left_ndv = self._column_ndv(condition.left_table, condition.left_column)
        right_ndv = self._column_ndv(condition.right_table, condition.right_column)

        join_selectivity = 1.0 / max(left_ndv, right_ndv, 1.0)
        est_rows = max(1.0, left_rows * right_rows * join_selectivity)
        return CardinalityEstimate(rows=est_rows, selectivity=join_selectivity)

    def _predicate_selectivity(self, table: str, pred: Predicate) -> float:
        if pred.op == "==":
            ndv = self._column_ndv(table, pred.column)
            return min(max(1.0 / max(ndv, 1.0), 1e-4), 1.0)

        if pred.op in {">", ">=", "<", "<="}:
            bounds = self._numeric_bounds(table, pred.column)
            if bounds is None:
                return 0.33
            min_val, max_val = bounds
            if max_val <= min_val:
                return 0.5

            try:
                threshold = float(pred.value)
            except (TypeError, ValueError):
                return 0.33

            span = max_val - min_val
            if pred.op in {">", ">="}:
                surviving = max(max_val - threshold, 0.0)
            else:
                surviving = max(threshold - min_val, 0.0)
            return min(max(surviving / span, 1e-4), 1.0)

        return 0.5

    def _column_ndv(self, table: str, column: str) -> float:
        frame = self.db.get_table(table)
        if column not in frame.columns:
            return max(1.0, float(len(frame) * 0.5))
        return max(1.0, float(frame[column].nunique(dropna=True)))

    def _numeric_bounds(self, table: str, column: str) -> Optional[tuple[float, float]]:
        frame = self.db.get_table(table)
        if column not in frame.columns:
            return None

        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            return None

        return float(series.min()), float(series.max())
