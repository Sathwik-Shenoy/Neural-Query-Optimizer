from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import uvicorn

from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.ml_model.inference import LearnedPlanSelector
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

    selector = LearnedPlanSelector(db=db, model_path=str(cfg["training"]["model_path"]))
    result = selector.rank(sql)
    print(json.dumps(result, indent=2))


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
