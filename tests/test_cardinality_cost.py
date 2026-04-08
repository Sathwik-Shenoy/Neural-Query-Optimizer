from neural_query_optimizer.cost_model.baseline import BaselineCostModel
from neural_query_optimizer.cost_model.cardinality import CardinalityEstimator
from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator


def test_cardinality_estimator_reduces_rows_after_filter() -> None:
    db = InMemoryDatabase()
    db.generate_synthetic(num_tables=1, rows_per_table=500, seed=3)

    estimator = CardinalityEstimator(db)
    parser = SQLParser()
    query = parser.parse("SELECT * FROM t1 WHERE t1.value > 130")

    estimate = estimator.estimate_filter_rows("t1", query.predicates)
    assert estimate.rows < db.table_stats("t1").rows
    assert 0.0 < estimate.selectivity <= 1.0


def test_hash_join_estimate_is_cheaper_than_nested_loop_for_large_inputs() -> None:
    db = InMemoryDatabase()
    db.generate_synthetic(num_tables=2, rows_per_table=800, seed=5)

    parser = SQLParser()
    generator = PhysicalPlanGenerator(max_join_orders=1, db=db)
    parsed = parser.parse("SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id")

    candidates = generator.generate(parsed)
    hash_plan = next(c.plan for c in candidates if "join=hash_join" in c.plan_id)
    nested_plan = next(c.plan for c in candidates if "join=nested_loop" in c.plan_id)

    cost_model = BaselineCostModel(db)
    assert cost_model.estimate(hash_plan) < cost_model.estimate(nested_plan)
