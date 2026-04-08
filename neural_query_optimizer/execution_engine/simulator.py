from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.utils.types import ExecutionStats, JoinCondition, PhysicalPlanNode, Predicate


@dataclass
class EngineConfig:
    index_scan_bonus: float = 0.65
    hash_join_bonus: float = 0.55
    nested_loop_penalty: float = 1.6


class ExecutionSimulator:
    """Execute physical plans over in-memory data and capture execution statistics."""

    def __init__(self, db: InMemoryDatabase, config: EngineConfig | None = None) -> None:
        self.db = db
        self.config = config or EngineConfig()

    def execute(self, plan: PhysicalPlanNode) -> Tuple[pd.DataFrame, ExecutionStats]:
        start = time.perf_counter()
        result, metrics = self._run_node(plan)
        end = time.perf_counter()

        adjusted_ms = (end - start) * 1000.0 * metrics["time_factor"]
        stats = ExecutionStats(
            execution_time_ms=adjusted_ms,
            rows_scanned=int(metrics["rows_scanned"]),
            peak_memory_bytes=int(metrics["peak_memory"]),
            output_rows=len(result),
        )
        return result, stats

    def _run_node(self, node: PhysicalPlanNode) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if node.operator in {"full_scan", "index_scan"}:
            return self._run_scan(node)

        if node.operator == "join":
            left_df, left_metrics = self._run_node(node.children[0])
            right_df, right_metrics = self._run_node(node.children[1])
            cond: JoinCondition = node.params["condition"]

            left_key = self._qualified_column(left_df, cond.left_table, cond.left_column)
            right_key = self._qualified_column(right_df, cond.right_table, cond.right_column)

            joined = left_df.merge(right_df, left_on=left_key, right_on=right_key, how="inner")

            algorithm = node.params.get("algorithm", "nested_loop")
            algo_factor = self.config.hash_join_bonus if algorithm == "hash_join" else self.config.nested_loop_penalty

            metrics = {
                "rows_scanned": left_metrics["rows_scanned"] + right_metrics["rows_scanned"] + len(left_df) + len(right_df),
                "peak_memory": max(
                    left_metrics["peak_memory"],
                    right_metrics["peak_memory"],
                    self._memory_bytes(joined),
                ),
                "time_factor": left_metrics["time_factor"] + right_metrics["time_factor"] + algo_factor,
            }
            return joined, metrics

        if node.operator == "project":
            child_df, metrics = self._run_node(node.children[0])
            columns = node.params.get("columns")
            if columns:
                projected = self._project_columns(child_df, columns)
            else:
                projected = child_df
            metrics["peak_memory"] = max(metrics["peak_memory"], self._memory_bytes(projected))
            return projected, metrics

        raise ValueError(f"Unsupported operator: {node.operator}")

    def _run_scan(self, node: PhysicalPlanNode) -> Tuple[pd.DataFrame, Dict[str, float]]:
        table_name = node.params["table"]
        predicates: list[Predicate] = node.params.get("predicates", [])

        frame = self.db.get_table(table_name).copy()
        frame = frame.rename(columns={col: f"{table_name}.{col}" for col in frame.columns})

        rows_before = len(frame)
        for pred in predicates:
            col_name = f"{table_name}.{pred.column}"
            if col_name not in frame.columns:
                continue
            frame = self._apply_predicate(frame, col_name, pred.op, pred.value)

        is_index = node.operator == "index_scan"
        factor = self.config.index_scan_bonus if is_index else 1.0

        metrics = {
            "rows_scanned": rows_before,
            "peak_memory": self._memory_bytes(frame),
            "time_factor": factor,
        }
        return frame, metrics

    def _project_columns(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if columns == ["*"]:
            return frame

        resolved = []
        for col in columns:
            if col in frame.columns:
                resolved.append(col)
                continue
            candidates = [name for name in frame.columns if name.endswith(f".{col.split('.')[-1]}")]
            if len(candidates) == 1:
                resolved.append(candidates[0])
        if not resolved:
            return frame
        return frame[resolved]

    def _apply_predicate(self, frame: pd.DataFrame, col: str, op: str, value: object) -> pd.DataFrame:
        if op == "==":
            return frame[frame[col] == value]
        if op == ">":
            return frame[frame[col] > value]
        if op == ">=":
            return frame[frame[col] >= value]
        if op == "<":
            return frame[frame[col] < value]
        if op == "<=":
            return frame[frame[col] <= value]
        return frame

    def _qualified_column(self, frame: pd.DataFrame, table: str, column: str) -> str:
        target = f"{table}.{column}"
        if target in frame.columns:
            return target
        fallback = [name for name in frame.columns if name.endswith(f".{column}")]
        if not fallback:
            raise KeyError(f"Column {table}.{column} is not available in dataframe")
        return fallback[0]

    def _memory_bytes(self, frame: pd.DataFrame) -> int:
        return int(frame.memory_usage(deep=True).sum())
