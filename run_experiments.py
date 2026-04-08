from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neural_query_optimizer.training.pipeline import TrainingPipeline
from neural_query_optimizer.utils.config import load_config


def summarize_comparisons(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if not rows:
        raise RuntimeError("No comparison rows found")

    baseline_costs = [float(r["rule_based"]["actual_cost"]) for r in rows]
    ml_costs = [float(r["ml_selected"]["actual_cost"]) for r in rows]
    best_costs = [float(r["actual_best"]["actual_cost"]) for r in rows]
    accuracy = sum(int(r["ml_selected"]["plan"] == r["actual_best"]["plan"]) for r in rows) / len(rows)
    improvement = (sum(baseline_costs) - sum(ml_costs)) / max(sum(baseline_costs), 1e-9)

    return {
        "rule_based_best_plan_cost": float(sum(baseline_costs) / len(baseline_costs)),
        "ml_predicted_best_plan_cost": float(sum(ml_costs) / len(ml_costs)),
        "actual_best_plan_cost": float(sum(best_costs) / len(best_costs)),
        "accuracy": float(accuracy),
        "improvement": float(improvement),
    }


def print_summary(summary: dict[str, float]) -> None:
    print(f"Rule-based best plan cost: {summary['rule_based_best_plan_cost']:.3f}")
    print(f"ML predicted best plan cost: {summary['ml_predicted_best_plan_cost']:.3f}")
    print(f"Actual best plan cost: {summary['actual_best_plan_cost']:.3f}")
    print(f"Accuracy: {summary['accuracy'] * 100:.2f}%")
    print(f"Improvement: {summary['improvement'] * 100:+.2f}%")


def print_table(comparison_path: Path) -> None:
    with comparison_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "query_id": row["query_id"],
                "rule_plan": row["rule_based"]["plan"],
                "rule_actual": row["rule_based"]["actual_cost"],
                "ml_plan": row["ml_selected"]["plan"],
                "ml_actual": row["ml_selected"]["actual_cost"],
                "oracle_plan": row["actual_best"]["plan"],
                "oracle_actual": row["actual_best"]["actual_cost"],
            }
        )

    df = pd.DataFrame(flat_rows)
    print("\nComparison table (first 10 rows):")
    print(df.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimizer experiments")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = TrainingPipeline(cfg).run()

    comparison_path = Path(str(cfg["training"].get("comparison_path", "artifacts/plan_comparisons.json")))
    summary = summarize_comparisons(comparison_path)

    print("\n=== Performance Summary ===")
    print_summary(summary)
    print("\n=== Training Metrics ===")
    print(json.dumps(metrics, indent=2))

    print_table(comparison_path)


if __name__ == "__main__":
    main()
