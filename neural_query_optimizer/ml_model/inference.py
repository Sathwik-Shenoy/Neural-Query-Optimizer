from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from neural_query_optimizer.cost_model.baseline import BaselineCostModel
from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.ml_model.features import FeatureExtractor
from neural_query_optimizer.ml_model.model import RandomForestPlanModel
from neural_query_optimizer.parser.sql_parser import SQLParser
from neural_query_optimizer.physical_plan.generator import PhysicalPlanGenerator


@dataclass
class RankedPlan:
    plan_id: str
    predicted_cost: float
    baseline_cost: float
    pretty_plan: str


class LearnedPlanSelector:
    """Inference-time learned plan ranking service."""

    def __init__(self, db: InMemoryDatabase, model_path: str) -> None:
        self.db = db
        self.parser = SQLParser()
        self.generator = PhysicalPlanGenerator()
        self.extractor = FeatureExtractor(db)
        self.baseline = BaselineCostModel(db)

        self.model = RandomForestPlanModel()
        self.model.load(model_path)

    def rank(self, sql: str) -> Dict[str, object]:
        parsed = self.parser.parse(sql)
        plans = self.generator.generate(parsed)
        if not plans:
            raise ValueError("No feasible plan candidates generated")

        rows: List[Dict[str, float]] = []
        baseline_costs: Dict[str, float] = {}

        for p in plans:
            feat = self.extractor.extract(parsed, p.plan)
            feat["baseline_cost"] = float(self.baseline.estimate(p.plan))
            rows.append(feat)
            baseline_costs[p.plan_id] = feat["baseline_cost"]

        features = pd.DataFrame(rows)
        preds = self.model.predict(features)

        ranked: List[RankedPlan] = []
        for idx, candidate in enumerate(plans):
            ranked.append(
                RankedPlan(
                    plan_id=candidate.plan_id,
                    predicted_cost=float(preds[idx]),
                    baseline_cost=float(baseline_costs[candidate.plan_id]),
                    pretty_plan=candidate.plan.pretty(),
                )
            )

        ranked.sort(key=lambda x: x.predicted_cost)

        return {
            "chosen_plan": ranked[0].plan_id,
            "predicted_cost": ranked[0].predicted_cost,
            "alternatives": [r.__dict__ for r in ranked[:5]],
        }
