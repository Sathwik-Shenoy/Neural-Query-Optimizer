# Neural Query Optimizer: A Learned Cost-Based Query Optimization Engine

Production-style research/engineering project that combines query optimization internals with machine learning.

## Core Capabilities

- SQL parsing to a transparent internal representation
- Logical plan construction (selection, projection, join)
- Physical plan enumeration:
  - scan methods: full scan, index scan
  - join algorithms: nested loop, hash join
  - left-deep join ordering alternatives
- Simulated execution engine with metrics:
  - execution latency (adjusted runtime)
  - rows scanned
  - peak memory bytes
- Rule-based baseline cost model
- Cardinality estimation for predicates and equi-joins
- Cost decomposition with explicit I/O and CPU terms
- Learned model (Random Forest baseline) that predicts plan cost
- End-to-end training pipeline with synthetic workload generation
- FastAPI inference service for live plan ranking
- Experiment logging, saved datasets, and evaluation metrics
- RL extension module (epsilon-greedy bandit) for online refinement

## Architecture

- [neural_query_optimizer/parser/sql_parser.py](neural_query_optimizer/parser/sql_parser.py): SQL to internal AST-like query object
- [neural_query_optimizer/logical_plan/builder.py](neural_query_optimizer/logical_plan/builder.py): relational algebra logical tree
- [neural_query_optimizer/physical_plan/generator.py](neural_query_optimizer/physical_plan/generator.py): physical candidate enumeration
- [neural_query_optimizer/execution_engine/database.py](neural_query_optimizer/execution_engine/database.py): in-memory table catalog
- [neural_query_optimizer/execution_engine/simulator.py](neural_query_optimizer/execution_engine/simulator.py): plan execution + metrics
- [neural_query_optimizer/cost_model/baseline.py](neural_query_optimizer/cost_model/baseline.py): heuristic rule-based estimator
- [neural_query_optimizer/cost_model/cardinality.py](neural_query_optimizer/cost_model/cardinality.py): cardinality/selectivity estimation
- [neural_query_optimizer/ml_model/features.py](neural_query_optimizer/ml_model/features.py): feature extraction
- [neural_query_optimizer/ml_model/model.py](neural_query_optimizer/ml_model/model.py): model interface + RF implementation
- [neural_query_optimizer/ml_model/inference.py](neural_query_optimizer/ml_model/inference.py): learned plan ranking service
- [neural_query_optimizer/training/pipeline.py](neural_query_optimizer/training/pipeline.py): training and evaluation orchestration
- [neural_query_optimizer/api/server.py](neural_query_optimizer/api/server.py): FastAPI endpoints
- [neural_query_optimizer/main.py](neural_query_optimizer/main.py): CLI entrypoint

## Database-Style Optimizer Design

This project intentionally models core DB optimizer ideas instead of random scoring:

1. Cardinality estimation:
  - Filter selectivity from NDV and numeric bounds
  - Join selectivity from equi-join NDV relationship
  - Estimated rows feed every downstream cost decision

2. Cost model (interpretable):
  - `Cost = I/O cost + CPU cost`
  - Scan cost combines page-read cost and tuple processing cost
  - Join cost differentiates nested-loop vs hash join complexity

3. Join order search:
  - Selinger-style dynamic programming for left-deep trees
  - Complexity is exponential in tables (`O(n^2 2^n)`), so alternatives are capped
  - Additional orders are still explored for plan diversity

## Setup

```bash
cd neural_query_optimizer
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

```bash
/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml train
```

Outputs:
- model: [artifacts/plan_selector.joblib](artifacts/plan_selector.joblib)
- dataset: [artifacts/training_dataset.csv](artifacts/training_dataset.csv)
- metrics: [artifacts/training_metrics.json](artifacts/training_metrics.json)

## Run Inference (CLI)

```bash
/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml optimize --sql "SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id WHERE t1.value > 100"
```

Response includes:
- chosen plan id
- predicted cost
- top alternatives with plan strings

## Run API

```bash
/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml serve
```

Endpoints:
- `GET /health`
- `POST /optimize` with payload:

```json
{
  "sql": "SELECT * FROM t1 JOIN t2 ON t1.join_id = t2.join_id WHERE t1.value > 100"
}
```

## Experiments and Visualization

```bash
/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml visualize
```

Generates:
- [artifacts/cost_comparison.png](artifacts/cost_comparison.png)

## Run Full Experiments

```bash
python run_experiments.py --config configs/default.yaml
```

Outputs:
- performance summary with:
  - Rule-based best plan cost
  - ML predicted best plan cost
  - Actual best plan cost
  - Accuracy
  - Improvement
- comparison table for sampled queries
- persisted artifact: [artifacts/plan_comparisons.json](artifacts/plan_comparisons.json)

## Testing

```bash
pytest -q
```

## How ML Improves Optimization

ML here is framed as a **cost-function approximator**, not a replacement for query planning.

1. Rule-based cost model gives interpretable but imperfect estimates.
2. Training pipeline executes candidate plans and captures observed latency.
3. The model learns nonlinear interactions among:
   - table sizes
   - predicate/selectivity features
   - join/scan strategy mix
4. At inference, each candidate gets a predicted cost; optimizer picks lowest predicted cost.

## Evaluation Protocol

Training emits quantitative comparison of rule-based vs ML-guided selection:

- Cost prediction quality: `MAE`, `RMSE`, `R2`, `MAPE`
- Plan quality:
  - `plan_selection_accuracy` (ML matches oracle best plan)
  - `baseline_plan_selection_accuracy`
  - `ml_vs_baseline_win_rate`
- Runtime benefit:
  - `latency_improvement_over_baseline`
  - baseline and ML mean chosen latency
- Cost-gap analysis:
  - `baseline_cost_bias_ms`, `baseline_cost_mae_ms`
  - `ml_cost_bias_ms`, `ml_cost_mae_ms`

These are written to [artifacts/training_metrics.json](artifacts/training_metrics.json).

## Limitations vs Real DBMS

- SQL support is intentionally scoped (SELECT/JOIN/WHERE core)
- Join ordering is left-deep DP only (no bushy plan search)
- Simulator approximates hardware/runtime effects and does not model full buffer/cache behavior
- No transaction/concurrency control
- Statistics model is lightweight compared with production systems

### Explicit Differences vs PostgreSQL

1. No catalog statistics maintenance lifecycle (`ANALYZE`-style refresh).
2. No full histogram/MCV/correlation selectivity model.
3. Join search space is intentionally constrained.
4. No parallel plan generation or parallel execution operators.
5. No buffer manager, page eviction policy, WAL, or transaction subsystem.

## Failure Cases

- Unseen query shapes: ML can mis-rank plans under distribution shift.
- Data skew: NDV-based selectivity can under-estimate hot-key joins.
- Correlated predicates: independence assumptions over/under-estimate cardinality.
- Sparse indexes: index-scan assumptions become optimistic when random I/O dominates.
- Drift over time: model and statistics need periodic retraining/refresh.

## Design Notes

- [DESIGN.md](DESIGN.md)

## Interview Discussion Hooks

- Cost model drift and online feedback loops
- Join-order search explosion and pruning strategies
- Replacing RF with GNN/transformer encoders over plan trees
- Contextual bandits or RL for adaptive plan control
