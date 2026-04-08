from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from neural_query_optimizer.cost_model.baseline import BaselineCostModel, CostConstants
from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.execution_engine.simulator import EngineConfig, ExecutionSimulator
from neural_query_optimizer.ml_model.features import FeatureExtractor
from neural_query_optimizer.ml_model.model import ModelRegistry, evaluate_regression
from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator
from neural_query_optimizer.training.query_generator import QueryGenerator


@dataclass
class TrainingArtifacts:
    model_path: str
    dataset_path: str
    metrics_path: str


class TrainingPipeline:
    """End-to-end pipeline for workload generation, plan execution, and model training."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.parser = SQLParser()

        exec_cfg = config["execution"]
        self.engine_config = EngineConfig(
            index_scan_bonus=float(exec_cfg["index_scan_bonus"]),
            hash_join_bonus=float(exec_cfg["hash_join_bonus"]),
            nested_loop_penalty=float(exec_cfg["nested_loop_penalty"]),
        )

        cost_cfg = config.get("cost_model", {})
        self.cost_constants = CostConstants(
            rows_per_page=int(cost_cfg.get("rows_per_page", 128)),
            seq_page_read_cost=float(cost_cfg.get("seq_page_read_cost", 1.0)),
            random_page_read_cost=float(cost_cfg.get("random_page_read_cost", 4.0)),
            cpu_tuple_cost=float(cost_cfg.get("cpu_tuple_cost", 0.01)),
            cpu_operator_cost=float(cost_cfg.get("cpu_operator_cost", 0.002)),
            hash_cpu_cost=float(cost_cfg.get("hash_cpu_cost", 0.004)),
        )

    def run(self) -> Dict[str, float]:
        seed = int(self.config["seed"])
        train_cfg = self.config["training"]

        db = InMemoryDatabase()
        db.generate_synthetic(
            num_tables=int(train_cfg["num_tables"]),
            rows_per_table=int(train_cfg["rows_per_table"]),
            seed=seed,
        )

        simulator = ExecutionSimulator(db, self.engine_config)
        baseline = BaselineCostModel(db, constants=self.cost_constants)
        extractor = FeatureExtractor(db)
        generator = PhysicalPlanGenerator(db=db)

        query_gen = QueryGenerator(table_names=db.table_names(), seed=seed)
        sql_queries = query_gen.generate(int(train_cfg["num_queries"]))

        records: List[Dict[str, object]] = []

        for qid, sql in enumerate(sql_queries):
            parsed = self.parser.parse(sql)
            candidates = generator.generate(parsed)
            for candidate in candidates:
                _, stats = simulator.execute(candidate.plan)
                features = extractor.extract(parsed, candidate.plan)

                records.append(
                    {
                        "query_id": qid,
                        "sql": sql,
                        "plan_id": candidate.plan_id,
                        "baseline_cost": baseline.estimate(candidate.plan),
                        "actual_latency_ms": stats.execution_time_ms,
                        "rows_scanned": stats.rows_scanned,
                        "memory_bytes": stats.peak_memory_bytes,
                        **features,
                    }
                )

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError("Training pipeline produced no records")

        train_query_ids, test_query_ids = self._split_queries(df, float(train_cfg["train_split"]), seed)
        feature_cols = self._feature_columns(df)

        train_df = df[df["query_id"].isin(train_query_ids)]
        test_df = df[df["query_id"].isin(test_query_ids)]

        model = ModelRegistry.create("random_forest")
        model.fit(train_df[feature_cols], train_df["actual_latency_ms"])

        test_pred = model.predict(test_df[feature_cols])
        reg_metrics = evaluate_regression(test_df["actual_latency_ms"], test_pred)

        selection_metrics = self._plan_selection_metrics(test_df, test_pred)

        model_path = str(train_cfg["model_path"])
        dataset_path = str(train_cfg["dataset_path"])
        metrics_path = str(train_cfg["metrics_path"])

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

        # type: ignore[attr-defined]
        model.save(model_path)
        df.to_csv(dataset_path, index=False)

        metrics = {
            **reg_metrics,
            **selection_metrics,
            "train_records": float(len(train_df)),
            "test_records": float(len(test_df)),
            "feature_count": float(len(feature_cols)),
        }

        with Path(metrics_path).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        return metrics

    def _feature_columns(self, df: pd.DataFrame) -> List[str]:
        # Keep only features available at inference time.
        return [
            "table_count",
            "join_count",
            "total_rows",
            "predicate_count",
            "estimated_selectivity",
            "estimated_filtered_rows",
            "estimated_output_rows",
            "index_scan_count",
            "full_scan_count",
            "hash_join_count",
            "nested_loop_join_count",
            "baseline_cost",
        ]

    def _split_queries(self, df: pd.DataFrame, train_split: float, seed: int) -> tuple[list[int], list[int]]:
        qids = sorted(df["query_id"].unique().tolist())
        rng = np.random.default_rng(seed)
        rng.shuffle(qids)
        cut = max(1, int(len(qids) * train_split))
        train_ids = qids[:cut]
        test_ids = qids[cut:] if cut < len(qids) else qids[-1:]
        return train_ids, test_ids

    def _plan_selection_metrics(self, test_df: pd.DataFrame, test_pred: np.ndarray) -> Dict[str, float]:
        eval_df = test_df.copy()
        eval_df["predicted_latency_ms"] = test_pred

        correct = 0
        baseline_correct = 0
        total = 0
        baseline_total = 0.0
        model_total = 0.0
        model_beats_baseline = 0

        for _, group in eval_df.groupby("query_id"):
            oracle_row = group.loc[group["actual_latency_ms"].idxmin()]
            baseline_row = group.loc[group["baseline_cost"].idxmin()]
            model_row = group.loc[group["predicted_latency_ms"].idxmin()]

            total += 1
            correct += int(model_row["plan_id"] == oracle_row["plan_id"])
            baseline_correct += int(baseline_row["plan_id"] == oracle_row["plan_id"])
            baseline_total += float(baseline_row["actual_latency_ms"])
            model_total += float(model_row["actual_latency_ms"])
            model_beats_baseline += int(float(model_row["actual_latency_ms"]) < float(baseline_row["actual_latency_ms"]))

        accuracy = float(correct / max(total, 1))
        baseline_accuracy = float(baseline_correct / max(total, 1))
        improvement = float((baseline_total - model_total) / max(baseline_total, 1e-6))
        win_rate = float(model_beats_baseline / max(total, 1))
        return {
            "plan_selection_accuracy": accuracy,
            "baseline_plan_selection_accuracy": baseline_accuracy,
            "ml_vs_baseline_win_rate": win_rate,
            "latency_improvement_over_baseline": improvement,
            "baseline_mean_latency_ms": float(baseline_total / max(total, 1)),
            "ml_mean_latency_ms": float(model_total / max(total, 1)),
        }
