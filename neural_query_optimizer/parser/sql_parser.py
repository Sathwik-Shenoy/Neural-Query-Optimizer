from __future__ import annotations

from typing import List, Optional, Tuple

import sqlglot
from sqlglot import exp

from neural_query_optimizer.utils.types import JoinCondition, ParsedQuery, Predicate


class SQLParser:
    """Parse SQL text into a typed internal query representation."""

    def parse(self, sql: str) -> ParsedQuery:
        tree = sqlglot.parse_one(sql)
        if not isinstance(tree, exp.Select):
            raise ValueError("Only SELECT queries are supported")

        select_columns = [self._expr_to_column_name(proj) for proj in tree.expressions]

        from_clause = tree.args.get("from_")
        if from_clause is None or not from_clause.this:
            raise ValueError("FROM clause is required")
        from_table = self._table_name(from_clause.this)

        joins = self._parse_joins(tree)
        predicates = self._parse_where(tree)

        return ParsedQuery(
            select_columns=select_columns,
            from_table=from_table,
            joins=joins,
            predicates=predicates,
        )

    def _expr_to_column_name(self, projection: exp.Expression) -> str:
        if isinstance(projection, exp.Column):
            return projection.sql()
        if isinstance(projection, exp.Alias):
            return projection.alias_or_name
        return projection.sql()

    def _table_name(self, table_expr: exp.Expression) -> str:
        if isinstance(table_expr, exp.Table):
            return table_expr.name
        return table_expr.sql()

    def _parse_joins(self, tree: exp.Select) -> List[Tuple[str, JoinCondition]]:
        joins: List[Tuple[str, JoinCondition]] = []
        for join in tree.args.get("joins", []):
            if not isinstance(join, exp.Join):
                continue
            right_table = self._table_name(join.this)
            on_expr = join.args.get("on")
            if not isinstance(on_expr, exp.EQ):
                raise ValueError("Only equi-joins are supported")
            left_col = self._extract_col_ref(on_expr.this)
            right_col = self._extract_col_ref(on_expr.expression)
            join_cond = JoinCondition(
                left_table=left_col[0],
                left_column=left_col[1],
                right_table=right_col[0],
                right_column=right_col[1],
            )
            joins.append((right_table, join_cond))
        return joins

    def _extract_col_ref(self, expression: exp.Expression) -> tuple[str, str]:
        if not isinstance(expression, exp.Column):
            raise ValueError("Expected column reference in join condition")
        table = expression.table
        if not table:
            raise ValueError("Columns in joins must be table-qualified")
        return table, expression.name

    def _parse_where(self, tree: exp.Select) -> List[Predicate]:
        where = tree.args.get("where")
        if where is None:
            return []
        return self._flatten_predicates(where.this)

    def _flatten_predicates(self, node: exp.Expression) -> List[Predicate]:
        if isinstance(node, exp.And):
            return self._flatten_predicates(node.this) + self._flatten_predicates(node.expression)
        predicate = self._predicate_from_binary(node)
        if predicate is None:
            return []
        return [predicate]

    def _predicate_from_binary(self, node: exp.Expression) -> Optional[Predicate]:
        supported = {
            exp.EQ: "==",
            exp.GT: ">",
            exp.GTE: ">=",
            exp.LT: "<",
            exp.LTE: "<=",
        }
        op = None
        for expr_type, symbol in supported.items():
            if isinstance(node, expr_type):
                op = symbol
                break
        if op is None:
            return None

        left = node.this
        right = node.expression
        if not isinstance(left, exp.Column):
            return None

        table = left.table or None
        col = left.name
        value = self._literal_value(right) if isinstance(right, exp.Literal) else right.sql()
        return Predicate(table=table, column=col, op=op, value=value)

    def _literal_value(self, literal: exp.Literal) -> object:
        if literal.is_string:
            return literal.this
        raw = literal.this
        if "." in raw:
            try:
                return float(raw)
            except ValueError:
                return raw
        try:
            return int(raw)
        except ValueError:
            return raw
