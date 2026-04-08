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
- [neural_query_optimizer/ml_model/features.py](neural_query_optimizer/ml_model/features.py): feature extraction
- [neural_query_optimizer/ml_model/model.py](neural_query_optimizer/ml_model/model.py): model interface + RF implementation
- [neural_query_optimizer/ml_model/inference.py](neural_query_optimizer/ml_model/inference.py): learned plan ranking service
- [neural_query_optimizer/training/pipeline.py](neural_query_optimizer/training/pipeline.py): training and evaluation orchestration
- [neural_query_optimizer/api/server.py](neural_query_optimizer/api/server.py): FastAPI endpoints
- [main.py](main.py): CLI entrypoint

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

## Testing

```bash
pytest -q
```

## How ML Improves Optimization

1. Baseline rule cost model provides explainable but coarse estimates.
2. Training pipeline executes many candidate plans and records true latencies.
3. The model learns nonlinear interactions among:
   - table sizes
   - predicate/selectivity features
   - join/scan strategy mix
4. At inference, candidates are ranked by predicted latency, often choosing faster plans than the heuristic baseline.

## Limitations vs Real DBMS

- SQL support is intentionally scoped (SELECT/JOIN/WHERE core)
- Only left-deep joins are enumerated
- Simulator approximates hardware/runtime effects and does not model full buffer/cache behavior
- No transaction/concurrency control
- Statistics model is lightweight compared with production systems

## Interview Discussion Hooks

- Cost model drift and online feedback loops
- Join-order search explosion and pruning strategies
- Replacing RF with GNN/transformer encoders over plan trees
- Contextual bandits or RL for adaptive plan control
