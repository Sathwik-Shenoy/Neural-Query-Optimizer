from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neural_query_optimizer.execution_engine.database import InMemoryDatabase
from neural_query_optimizer.ml_model.inference import LearnedPlanSelector
from neural_query_optimizer.utils.config import load_config


class OptimizeRequest(BaseModel):
    sql: str = Field(..., description="SQL query to optimize")


class OptimizeResponse(BaseModel):
    chosen_plan: str
    predicted_cost: float
    alternatives: list[Dict[str, Any]]


def create_app(config_path: str = "configs/default.yaml") -> FastAPI:
    cfg = load_config(config_path)
    model_path = str(cfg["training"]["model_path"])

    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. Run training first via the CLI."
        )

    db = InMemoryDatabase()
    db.generate_synthetic(
        num_tables=int(cfg["training"]["num_tables"]),
        rows_per_table=int(cfg["training"]["rows_per_table"]),
        seed=int(cfg["seed"]),
    )
    selector = LearnedPlanSelector(db=db, model_path=model_path)

    app = FastAPI(title="Neural Query Optimizer", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/optimize", response_model=OptimizeResponse)
    def optimize(req: OptimizeRequest) -> OptimizeResponse:
        try:
            result = selector.rank(req.sql)
            return OptimizeResponse(**result)
        except Exception as exc:  # pragma: no cover - translated as API error response
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
