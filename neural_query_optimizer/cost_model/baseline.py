from __future__ import annotations

from dataclasses import dataclass

from neural_query_optimizer.cost_model.cardinality import CardinalityEstimator
from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.utils.types import PhysicalPlanNode, Predicate


@dataclass
class CostConstants:
    """Calibratable constants for a simplified disk-and-cpu cost model."""

    rows_per_page: int = 128
    seq_page_read_cost: float = 1.0
    random_page_read_cost: float = 4.0
    cpu_tuple_cost: float = 0.01
    cpu_operator_cost: float = 0.002
    hash_cpu_cost: float = 0.004


class BaselineCostModel:
    """Rule-based cost model with explicit I/O and CPU components.

    This is intentionally interpretable and deterministic so it can be compared
    against learned cost prediction.
    """

    def __init__(self, db: InMemoryDatabase, constants: CostConstants | None = None) -> None:
        self.db = db
        self.constants = constants or CostConstants()
        self.cardinality = CardinalityEstimator(db)

    def estimate(self, plan: PhysicalPlanNode) -> float:
        cost, _ = self._estimate_node(plan)
        return cost

    def _estimate_node(self, node: PhysicalPlanNode) -> tuple[float, float]:
        if node.operator in {"full_scan", "index_scan"}:
            table = node.params["table"]
            predicates = node.params.get("predicates", [])
            return self._estimate_scan(table, node.operator, predicates)

        if node.operator == "join":
            left_cost, left_rows = self._estimate_node(node.children[0])
            right_cost, right_rows = self._estimate_node(node.children[1])
            algo = node.params.get("algorithm", "nested_loop")
            condition = node.params["condition"]

            join_card = self.cardinality.estimate_join_rows(left_rows, right_rows, condition)
            if algo == "hash_join":
                op_cpu = (left_rows + right_rows) * (
                    self.constants.cpu_tuple_cost + self.constants.hash_cpu_cost
                )
                op_cost = op_cpu + join_card.rows * self.constants.cpu_operator_cost
            else:
                # Simple nested-loop shape where each left tuple probes right.
                op_cpu = (
                    left_rows * right_rows * self.constants.cpu_operator_cost
                    + join_card.rows * self.constants.cpu_tuple_cost
                )
                right_pages = max(1.0, right_rows / self.constants.rows_per_page)
                op_io = (left_rows * right_pages) * self.constants.seq_page_read_cost
                op_cost = op_cpu + op_io

            return left_cost + right_cost + op_cost, join_card.rows

        if node.operator == "project":
            child_cost, child_rows = self._estimate_node(node.children[0])
            projection_cpu = child_rows * self.constants.cpu_operator_cost
            return child_cost + projection_cpu, child_rows

        raise ValueError(f"Unknown operator in cost model: {node.operator}")

    def _estimate_scan(
        self,
        table: str,
        operator: str,
        predicates: list[Predicate],
    ) -> tuple[float, float]:
        stats = self.db.table_stats(table)
        base_rows = float(stats.rows)
        base_pages = max(1.0, base_rows / self.constants.rows_per_page)

        card = self.cardinality.estimate_filter_rows(table, predicates)
        output_rows = card.rows
        output_pages = max(1.0, output_rows / self.constants.rows_per_page)

        if operator == "full_scan":
            io_cost = base_pages * self.constants.seq_page_read_cost
        else:
            # Index access reduces touched pages when predicates can leverage indexes.
            indexed_preds = sum(
                1 for p in predicates if self.db.has_index(table, p.column) and p.table in (None, table)
            )
            reduction = 0.7 if indexed_preds > 0 else 0.25
            touched_pages = max(1.0, base_pages * (1.0 - reduction) * max(card.selectivity, 0.05))
            io_cost = touched_pages * self.constants.random_page_read_cost + output_pages * 0.2

        cpu_cost = base_rows * self.constants.cpu_tuple_cost + output_rows * self.constants.cpu_operator_cost
        return io_cost + cpu_cost, output_rows
