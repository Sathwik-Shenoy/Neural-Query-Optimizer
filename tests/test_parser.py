from neural_query_optimizer.parser.sql_parser import SQLParser


def test_parse_join_and_predicates() -> None:
    parser = SQLParser()
    query = parser.parse(
        "SELECT t1.id FROM t1 JOIN t2 ON t1.join_id = t2.join_id "
        "WHERE t1.value > 100 AND t1.category = 'A'"
    )

    assert query.from_table == "t1"
    assert len(query.joins) == 1
    assert len(query.predicates) == 2
