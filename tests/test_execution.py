from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.execution_engine.simulator import ExecutionSimulator
from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator


def test_simulation_runs() -> None:
    db = InMemoryDatabase()
    db.generate_synthetic(num_tables=2, rows_per_table=120, seed=1)

    parser = SQLParser()
    generator = PhysicalPlanGenerator(max_join_orders=2)
    parsed = parser.parse("SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id WHERE t1.value > 90")
    plans = generator.generate(parsed)

    simulator = ExecutionSimulator(db)
    output, stats = simulator.execute(plans[0].plan)

    assert stats.execution_time_ms >= 0
    assert stats.rows_scanned > 0
    assert len(output) >= 0
