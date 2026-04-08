from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Predicate:
    """A simple filter predicate over a table column."""

    table: Optional[str]
    column: str
    op: str
    value: Any


@dataclass
class JoinCondition:
    """An equi-join condition represented as left_col == right_col."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str


@dataclass
class ParsedQuery:
    """Internal representation of parsed SQL query."""

    select_columns: List[str]
    from_table: str
    joins: List[tuple[str, JoinCondition]]
    predicates: List[Predicate]


@dataclass
class LogicalPlanNode:
    """Relational algebra node for logical plans."""

    operator: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List["LogicalPlanNode"] = field(default_factory=list)


@dataclass
class PhysicalPlanNode:
    """Physical execution operator node used by the execution simulator."""

    operator: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List["PhysicalPlanNode"] = field(default_factory=list)

    def pretty(self, indent: int = 0) -> str:
        padding = "  " * indent
        label = f"{padding}{self.operator}: {self.params}"
        if not self.children:
            return label
        return "\n".join([label] + [child.pretty(indent + 1) for child in self.children])


@dataclass
class ExecutionStats:
    """Runtime statistics captured for a physical plan execution."""

    execution_time_ms: float
    rows_scanned: int
    peak_memory_bytes: int
    output_rows: int


@dataclass
class PlanEvaluation:
    """A single plan candidate with costs and observed runtime stats."""

    plan_id: str
    plan: PhysicalPlanNode
    baseline_cost: float
    stats: ExecutionStats
