from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator


def test_generate_candidates() -> None:
    parser = SQLParser()
    generator = PhysicalPlanGenerator(max_join_orders=2)
    parsed = parser.parse("SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id")

    plans = generator.generate(parsed)
    assert len(plans) > 0
    assert any("hash_join" in p.plan_id for p in plans)


def test_dp_join_ordering_prioritizes_filtered_table() -> None:
    db = InMemoryDatabase()
    db.generate_synthetic(num_tables=3, rows_per_table=250, seed=11)

    parser = SQLParser()
    generator = PhysicalPlanGenerator(max_join_orders=3, db=db)
    parsed = parser.parse(
        "SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id "
        "JOIN t3 ON t2.join_id = t3.join_id "
        "WHERE t1.value > 120"
    )

    plans = generator.generate(parsed)
    assert len(plans) > 0
    # The first ranked order is the DP best order used for candidate generation.
    assert "|order=t1" in plans[0].plan_id


def test_bushy_shape_is_generated_for_three_table_joins() -> None:
    db = InMemoryDatabase()
    db.generate_synthetic(num_tables=3, rows_per_table=200, seed=13)

    parser = SQLParser()
    generator = PhysicalPlanGenerator(max_join_orders=3, db=db)
    parsed = parser.parse(
        "SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id "
        "JOIN t3 ON t2.join_id = t3.join_id"
    )

    plans = generator.generate(parsed)
    assert any(p.plan_id.startswith("shape=bushy") for p in plans)
