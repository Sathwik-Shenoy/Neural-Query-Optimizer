from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from neural_query_optimizer.cost_model.cardinality import CardinalityEstimator
from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.utils.types import ParsedQuery, PhysicalPlanNode


@dataclass
class FeatureExtractor:
    """Generate tabular features for ML-based plan ranking."""

    db: InMemoryDatabase

    def __post_init__(self) -> None:
        self.cardinality = CardinalityEstimator(self.db)

    def extract(self, query: ParsedQuery, plan: PhysicalPlanNode) -> Dict[str, float]:
        tables = [query.from_table] + [t for t, _ in query.joins]
        total_rows = sum(self.db.table_stats(t).rows for t in tables)

        join_count = self._count_operator(plan, "join")
        index_scans = self._count_operator(plan, "index_scan")
        full_scans = self._count_operator(plan, "full_scan")
        hash_joins = self._count_join_algo(plan, "hash_join")
        nl_joins = self._count_join_algo(plan, "nested_loop")

        table_filter_rows = 0.0
        rows_after_filter_by_table: Dict[str, float] = {}
        for table in tables:
            table_preds = [p for p in query.predicates if p.table in (None, table)]
            est = self.cardinality.estimate_filter_rows(table, table_preds)
            table_filter_rows += est.rows
            rows_after_filter_by_table[table] = est.rows

        estimated_rows_after_filter = max(1.0, table_filter_rows)
        total_base_rows = max(1.0, float(total_rows))
        selectivity = min(1.0, estimated_rows_after_filter / total_base_rows)

        estimated_join_output_size = self._estimate_join_output_size(query, rows_after_filter_by_table)
        estimated_selectivity_explicit = selectivity

        return {
            "table_count": float(len(tables)),
            "join_count": float(join_count),
            "total_rows": float(total_rows),
            "predicate_count": float(len(query.predicates)),
            "selectivity": float(selectivity),
            "estimated_selectivity": float(selectivity),
            "estimated_selectivity_explicit": float(estimated_selectivity_explicit),
            "estimated_filtered_rows": float(table_filter_rows),
            "estimated_rows_after_filter": float(estimated_rows_after_filter),
            "estimated_output_rows": float(estimated_join_output_size),
            "estimated_join_output_size": float(estimated_join_output_size),
            "index_scan_count": float(index_scans),
            "full_scan_count": float(full_scans),
            "hash_join_count": float(hash_joins),
            "nested_loop_join_count": float(nl_joins),
        }

    def _count_operator(self, node: PhysicalPlanNode, op: str) -> int:
        count = 1 if node.operator == op else 0
        for child in node.children:
            count += self._count_operator(child, op)
        return count

    def _count_join_algo(self, node: PhysicalPlanNode, algo: str) -> int:
        count = 0
        if node.operator == "join" and node.params.get("algorithm") == algo:
            count += 1
        for child in node.children:
            count += self._count_join_algo(child, algo)
        return count

    def _estimate_join_output_size(self, query: ParsedQuery, rows_after_filter: Dict[str, float]) -> float:
        if not query.joins:
            return max(1.0, rows_after_filter.get(query.from_table, 1.0))

        current_rows = max(1.0, rows_after_filter.get(query.from_table, 1.0))
        seen = {query.from_table}

        for right_table, cond in query.joins:
            right_rows = max(1.0, rows_after_filter.get(right_table, 1.0))
            # join_output_size = (left_rows * right_rows) * join_selectivity
            join_est = self.cardinality.estimate_join_rows(current_rows, right_rows, cond)
            current_rows = max(1.0, join_est.rows)
            seen.add(right_table)

        return current_rows
