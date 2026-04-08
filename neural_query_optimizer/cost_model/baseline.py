from __future__ import annotations

from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.utils.types import PhysicalPlanNode


class BaselineCostModel:
    """Rule-based cost model approximating operator-level execution costs."""

    def __init__(self, db: InMemoryDatabase) -> None:
        self.db = db

    def estimate(self, plan: PhysicalPlanNode) -> float:
        return self._estimate_node(plan)

    def _estimate_node(self, node: PhysicalPlanNode) -> float:
        if node.operator in {"full_scan", "index_scan"}:
            table = node.params["table"]
            rows = self.db.table_stats(table).rows
            base = rows * (0.002 if node.operator == "index_scan" else 0.004)
            preds = len(node.params.get("predicates", []))
            return base * (0.9 ** preds)

        if node.operator == "join":
            left = self._estimate_node(node.children[0])
            right = self._estimate_node(node.children[1])
            algo = node.params.get("algorithm", "nested_loop")
            join_penalty = 1.3 if algo == "nested_loop" else 0.9
            return (left + right) * join_penalty

        if node.operator == "project":
            return self._estimate_node(node.children[0]) * 0.98

        raise ValueError(f"Unknown operator in cost model: {node.operator}")
