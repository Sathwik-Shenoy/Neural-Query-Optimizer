from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator


def test_generate_candidates() -> None:
    parser = SQLParser()
    generator = PhysicalPlanGenerator(max_join_orders=2)
    parsed = parser.parse("SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id")

    plans = generator.generate(parsed)
    assert len(plans) > 0
    assert any("hash_join" in p.plan_id for p in plans)
