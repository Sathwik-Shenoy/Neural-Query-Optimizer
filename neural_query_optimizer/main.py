from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import uvicorn

from neural_query_optimizer.cost_model.baseline import BaselineCostModel
from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.execution_engine.simulator import ExecutionSimulator
from neural_query_optimizer.ml_model.features import FeatureExtractor
from neural_query_optimizer.ml_model.inference import LearnedPlanSelector
from neural_query_optimizer.ml_model.model import RandomForestPlanModel
from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator
from neural_query_optimizer.training.pipeline import TrainingPipeline
from neural_query_optimizer.utils.config import load_config


def cmd_train(config_path: str) -> None:
    cfg = load_config(config_path)
    metrics = TrainingPipeline(cfg).run()
    print(json.dumps(metrics, indent=2))


def cmd_optimize(config_path: str, sql: str) -> None:
    cfg = load_config(config_path)
    db = InMemoryDatabase()
    db.generate_synthetic(
        num_tables=int(cfg["training"]["num_tables"]),
        rows_per_table=int(cfg["training"]["rows_per_table"]),
        seed=int(cfg["seed"]),
    )

    parser = SQLParser()
    parsed = parser.parse(sql)
    generator = PhysicalPlanGenerator(db=db)
    candidates = generator.generate(parsed)

    baseline = BaselineCostModel(db)
    simulator = ExecutionSimulator(db)
    extractor = FeatureExtractor(db)
    model = RandomForestPlanModel()
    model.load(str(cfg["training"]["model_path"]))

    rows = []
    for cand in candidates:
        baseline_cost = baseline.estimate(cand.plan)
        feat = extractor.extract(parsed, cand.plan)
        feat["baseline_cost"] = baseline_cost
        pred_cost = float(model.predict(pd.DataFrame([feat]))[0])
        _, stats = simulator.execute(cand.plan)
        rows.append(
            {
                "plan_id": cand.plan_id,
                "baseline_cost": baseline_cost,
                "predicted_cost": pred_cost,
                "actual_cost": float(stats.execution_time_ms),
            }
        )

    df = pd.DataFrame(rows)
    ml_best = df.loc[df["predicted_cost"].idxmin()]
    rule_best = df.loc[df["baseline_cost"].idxmin()]
    actual_best = df.loc[df["actual_cost"].idxmin()]
    improvement = (float(rule_best["actual_cost"]) - float(ml_best["actual_cost"])) / max(
        float(rule_best["actual_cost"]), 1e-9
    )

    print(f"Query: {sql}")
    print("")
    print(f"Plans Evaluated: {len(df)}")
    print("")
    print("Best Plan (ML):")
    print(f"- {ml_best['plan_id']}")
    print(f"- Predicted Cost: {float(ml_best['predicted_cost']):.3f}")
    print(f"- Actual Cost: {float(ml_best['actual_cost']):.3f}")
    print("")
    print("Best Plan (Rule-Based):")
    print(f"- {rule_best['plan_id']}")
    print(f"- Cost: {float(rule_best['baseline_cost']):.3f}")
    print(f"- Actual Cost: {float(rule_best['actual_cost']):.3f}")
    print("")
    print("Actual Best Plan (Oracle):")
    print(f"- {actual_best['plan_id']}")
    print(f"- Actual Cost: {float(actual_best['actual_cost']):.3f}")
    print("")
    print(f"Improvement: {improvement * 100:+.2f}%")


def cmd_serve(config_path: str) -> None:
    cfg = load_config(config_path)
    host = str(cfg["api"]["host"])
    port = int(cfg["api"]["port"])
    uvicorn.run("neural_query_optimizer.api.server:create_app", factory=True, host=host, port=port)


def cmd_visualize(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_path = Path(str(cfg["training"]["dataset_path"]))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run training first.")

    import matplotlib.pyplot as plt

    df = pd.read_csv(dataset_path)
    grouped = df.groupby("query_id").agg(
        baseline_best=("baseline_cost", "min"),
        actual_best=("actual_latency_ms", "min"),
    )
    grouped.plot(kind="bar", figsize=(12, 4), title="Baseline Estimated vs Actual Best Latency")
    plt.tight_layout()

    out_path = Path("artifacts/cost_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural Query Optimizer CLI")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Run synthetic workload training")

    opt = sub.add_parser("optimize", help="Rank plans for a SQL query")
    opt.add_argument("--sql", required=True, help="SQL query string")

    sub.add_parser("serve", help="Start FastAPI inference service")
    sub.add_parser("visualize", help="Plot baseline vs actual cost signals")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args.config)
    elif args.command == "optimize":
        cmd_optimize(args.config, args.sql)
    elif args.command == "serve":
        cmd_serve(args.config)
    elif args.command == "visualize":
        cmd_visualize(args.config)


if __name__ == "__main__":
    main()
