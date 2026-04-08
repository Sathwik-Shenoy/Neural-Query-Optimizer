from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from neural_query_optimizer.utils.types import JoinCondition, ParsedQuery, PhysicalPlanNode, Predicate


@dataclass
class PlanCandidate:
    """Physical plan candidate with a deterministic identifier."""

    plan_id: str
    plan: PhysicalPlanNode


class PhysicalPlanGenerator:
    """Enumerate physical plan alternatives for a parsed query."""

    def __init__(self, max_join_orders: int = 8) -> None:
        self.max_join_orders = max_join_orders

    def generate(self, query: ParsedQuery) -> List[PlanCandidate]:
        tables = [query.from_table] + [table for table, _ in query.joins]
        predicates_by_table = self._group_predicates(query.predicates)
        join_conditions = self._join_condition_map(query.joins)

        candidates: List[PlanCandidate] = []
        all_orders = itertools.islice(itertools.permutations(tables), self.max_join_orders)

        for order in all_orders:
            scan_choices = ["full_scan", "index_scan"]
            for scan_assignment in itertools.product(scan_choices, repeat=len(order)):
                plan = self._build_left_deep_plan(
                    order=order,
                    scan_assignment=scan_assignment,
                    predicates_by_table=predicates_by_table,
                    join_conditions=join_conditions,
                )
                if plan is None:
                    continue
                join_algos = self._count_join_nodes(plan)
                for algo_choice in itertools.product(["nested_loop", "hash_join"], repeat=join_algos):
                    cloned = self._clone(plan)
                    self._assign_join_algorithms(cloned, list(algo_choice))
                    plan_id = self._plan_id(order, scan_assignment, algo_choice)
                    candidates.append(PlanCandidate(plan_id=plan_id, plan=cloned))

        return candidates

    def _group_predicates(self, predicates: Iterable[Predicate]) -> Dict[str, List[Predicate]]:
        grouped: Dict[str, List[Predicate]] = {}
        for pred in predicates:
            if pred.table is None:
                continue
            grouped.setdefault(pred.table, []).append(pred)
        return grouped

    def _join_condition_map(self, joins: List[tuple[str, JoinCondition]]) -> Dict[frozenset[str], JoinCondition]:
        conditions: Dict[frozenset[str], JoinCondition] = {}
        for _, cond in joins:
            key = frozenset({cond.left_table, cond.right_table})
            conditions[key] = cond
        return conditions

    def _build_left_deep_plan(
        self,
        order: Tuple[str, ...],
        scan_assignment: Tuple[str, ...],
        predicates_by_table: Dict[str, List[Predicate]],
        join_conditions: Dict[frozenset[str], JoinCondition],
    ) -> PhysicalPlanNode | None:
        first_table = order[0]
        plan = PhysicalPlanNode(
            operator=scan_assignment[0],
            params={"table": first_table, "predicates": predicates_by_table.get(first_table, [])},
        )
        seen = {first_table}

        for idx in range(1, len(order)):
            table = order[idx]
            condition = self._find_attachable_condition(seen, table, join_conditions)
            if condition is None:
                return None

            right_scan = PhysicalPlanNode(
                operator=scan_assignment[idx],
                params={"table": table, "predicates": predicates_by_table.get(table, [])},
            )
            plan = PhysicalPlanNode(
                operator="join",
                params={"algorithm": "nested_loop", "condition": condition},
                children=[plan, right_scan],
            )
            seen.add(table)

        return PhysicalPlanNode(operator="project", params={}, children=[plan])

    def _find_attachable_condition(
        self,
        seen_tables: set[str],
        next_table: str,
        join_conditions: Dict[frozenset[str], JoinCondition],
    ) -> JoinCondition | None:
        for seen in seen_tables:
            key = frozenset({seen, next_table})
            if key in join_conditions:
                return join_conditions[key]
        return None

    def _count_join_nodes(self, node: PhysicalPlanNode) -> int:
        total = 1 if node.operator == "join" else 0
        for child in node.children:
            total += self._count_join_nodes(child)
        return total

    def _assign_join_algorithms(self, node: PhysicalPlanNode, choices: List[str]) -> None:
        if node.operator == "join":
            node.params["algorithm"] = choices.pop(0)
        for child in node.children:
            self._assign_join_algorithms(child, choices)

    def _clone(self, node: PhysicalPlanNode) -> PhysicalPlanNode:
        return PhysicalPlanNode(
            operator=node.operator,
            params={**node.params},
            children=[self._clone(child) for child in node.children],
        )

    def _plan_id(
        self,
        order: Tuple[str, ...],
        scans: Tuple[str, ...],
        algos: Tuple[str, ...],
    ) -> str:
        left = "-".join(order)
        middle = "-".join(scans)
        right = "-".join(algos)
        return f"order={left}|scan={middle}|join={right}"
