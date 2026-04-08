from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class BasePlanModel(ABC):
    """Abstract interface for plan-cost predictors."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class RandomForestPlanModel(BasePlanModel):
    """Default production baseline model for learned cost prediction."""

    n_estimators: int = 250
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=2,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class ModelRegistry:
    """Model factory allowing future upgrade to neural architectures."""

    @staticmethod
    def create(name: str) -> BasePlanModel:
        if name == "random_forest":
            return RandomForestPlanModel()
        if name == "gradient_boosting":
            # Keeps API extensible; can add GradientBoostingRegressor without touching callers.
            return RandomForestPlanModel(n_estimators=180)
        raise ValueError(f"Unsupported model type: {name}")


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))))
    return {
        "mae": mae,
        "mape": mape,
    }


def choose_best_plan(plan_ids: List[str], predictions: np.ndarray) -> str:
    if len(plan_ids) == 0:
        raise ValueError("No plans supplied")
    idx = int(np.argmin(predictions))
    return plan_ids[idx]
