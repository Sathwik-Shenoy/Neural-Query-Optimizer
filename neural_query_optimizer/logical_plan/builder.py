from __future__ import annotations

from neural_query_optimizer.utils.types import LogicalPlanNode, ParsedQuery


class LogicalPlanBuilder:
    """Convert parsed SQL query into a relational algebra tree."""

    def build(self, query: ParsedQuery) -> LogicalPlanNode:
        root_scan = LogicalPlanNode(
            operator="scan",
            params={"table": query.from_table},
        )

        plan = root_scan
        joined_tables = {query.from_table}

        for table, cond in query.joins:
            join_node = LogicalPlanNode(
                operator="join",
                params={
                    "condition": cond,
                    "joined_tables": sorted(joined_tables | {table}),
                },
                children=[
                    plan,
                    LogicalPlanNode(operator="scan", params={"table": table}),
                ],
            )
            plan = join_node
            joined_tables.add(table)

        if query.predicates:
            plan = LogicalPlanNode(
                operator="selection",
                params={"predicates": query.predicates},
                children=[plan],
            )

        plan = LogicalPlanNode(
            operator="projection",
            params={"columns": query.select_columns},
            children=[plan],
        )
        return plan
