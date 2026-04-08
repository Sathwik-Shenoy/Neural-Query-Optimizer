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

        selectivity = max(0.01, 1.0 - 0.15 * len(query.predicates))
        table_filter_rows = 0.0
        for table in tables:
            table_preds = [p for p in query.predicates if p.table in (None, table)]
            table_filter_rows += self.cardinality.estimate_filter_rows(table, table_preds).rows

        estimated_output_rows = max(1.0, table_filter_rows * (0.6 ** join_count))

        return {
            "table_count": float(len(tables)),
            "join_count": float(join_count),
            "total_rows": float(total_rows),
            "predicate_count": float(len(query.predicates)),
            "estimated_selectivity": float(selectivity),
            "estimated_filtered_rows": float(table_filter_rows),
            "estimated_output_rows": float(estimated_output_rows),
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
