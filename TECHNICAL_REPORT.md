# Neural Query Optimizer: Technical Report

## 1. System Architecture

The system is organized as a modular optimizer and learning stack:

- Parser: SQL to internal representation.
- Logical planning: relational algebra operators (selection, projection, join).
- Physical planning: scan and join strategy enumeration over multiple join orders.
- Cost model: interpretable rule-based estimator with explicit I/O and CPU terms.
- Execution simulator: in-memory plan execution for empirical runtime labels.
- ML model: plan-level cost prediction used for ranking candidates.
- Training pipeline: query generation, plan execution, feature extraction, supervised learning, evaluation.
- API/CLI: inference endpoints and experiment automation.

Core loop:

1. Parse query.
2. Enumerate candidate physical plans.
3. Estimate plan costs (rule-based and learned).
4. Choose best candidate.
5. Execute and log observed outcomes.

## 2. Query Optimization Pipeline

For each SQL query:

1. SQL Parser extracts tables, predicates, and join conditions.
2. Physical plan generator creates alternatives across:
   - join shapes: left-deep and bushy (for three-table joins),
   - join algorithms: nested loop and hash join,
   - scan methods: full scan and index scan,
   - join orders: DP-prioritized plus additional alternatives.
3. Rule-based cost model estimates candidate costs.
4. ML model predicts candidate costs from engineered features.
5. Lowest predicted cost plan is selected by ML optimizer.
6. Runtime execution simulator provides actual latency for evaluation.

## 3. Cost Model Explanation (with Formulas)

The baseline model is explicit and interpretable:

$$\text{Cost} = \text{I/O Cost} + \text{CPU Cost}$$

Scan:

$$\text{scan\_cost} = \text{pages\_read} \cdot c_{io} + \text{rows\_scanned} \cdot c_{tuple} + \text{rows\_out} \cdot c_{op}$$

Nested loop join (simplified):

$$\text{nl\_cost} \approx \text{outer\_rows} \cdot \text{inner\_rows} \cdot c_{op} + \text{I/O probes}$$

Hash join:

$$\text{hash\_cost} \approx (\text{outer\_rows} + \text{inner\_rows}) \cdot (c_{tuple} + c_{hash}) + \text{rows\_out} \cdot c_{op}$$

Configurable constants include sequential/random page read and CPU costs.

## 4. Cardinality Estimation Logic

Filter cardinality:

$$\text{rows}_{filter} = \text{rows}_{base} \cdot \text{selectivity}$$

For equality predicates:

$$\text{selectivity}_{eq} \approx \frac{1}{\text{NDV}(col)}$$

For range predicates, selectivity is approximated from numeric bounds.

Join output:

$$\text{join\_output\_size} = (\text{left\_rows} \cdot \text{right\_rows}) \cdot \text{join\_selectivity}$$

with

$$\text{join\_selectivity} \approx \frac{1}{\max(\text{NDV}(left\_key), \text{NDV}(right\_key))}$$

## 5. ML Model Design (Features + Target)

Objective:

- Predict plan execution cost (latency proxy), not plan label.

Target:

- `actual_latency_ms` from simulated execution.

Feature groups:

- Query shape: table count, join count, predicate count.
- Data scale: total rows.
- Selectivity: `selectivity`, `estimated_selectivity_explicit`.
- Cardinality outputs: `estimated_rows_after_filter`, `estimated_join_output_size`.
- Operator mix: scan counts, join algorithm counts.
- Rule prior: `baseline_cost`.

Model:

- Random Forest regressor as robust baseline.

Inference policy:

- Score each candidate with predicted cost.
- Choose candidate with minimum predicted cost.

## 6. Experimental Setup

- Synthetic multi-table datasets generated in-memory.
- SQL workload includes joins and mixed predicates.
- Candidate plans are fully executed for runtime labels.
- Train/test split is query-level to reduce leakage.
- Artifacts generated:
  - model,
  - tabular dataset,
  - per-query plan comparisons,
  - structured experiment dataset with features and errors,
  - metrics JSON.

## 7. Results (with Metrics)

The experiment runner reports:

- Regression quality: MAE, RMSE, R2, MAPE.
- Plan quality:
  - Rule-Based Accuracy,
  - ML Accuracy,
  - ML win-rate over rule-based selection.
- Cost accuracy:
  - baseline and ML bias/MAE,
  - average percentage cost error.
- Runtime impact:
  - latency improvement over rule-based.

## 8. Comparison: ML vs Rule-based Optimizer

The system prints direct side-by-side comparisons:

- Rule-Based Plan Cost,
- ML Predicted Best Plan Cost,
- Actual Best Plan Cost,
- Improvement percentage.

Per-query logs include:

- selected plan IDs,
- estimated/predicted cost,
- actual cost,
- absolute and relative error.

This demonstrates whether ML is adding value beyond heuristics.

## 9. Failure Cases and Limitations

Failure cases:

1. Incorrect selectivity estimation under predicate correlation.
2. Data skew causing join selectivity misestimation.
3. ML degradation on unseen query templates.
4. Overfitting risk on small workloads.

Limitations:

- Simplified execution semantics and storage model.
- Limited operator set compared to production engines.
- No concurrency or transactional behavior.

## 10. Differences from Real-world DB Systems (PostgreSQL)

Compared to PostgreSQL:

1. No catalog statistics lifecycle with ANALYZE-driven maintenance.
2. No histogram/MCV/correlation-rich estimator.
3. Join search is constrained (not full production planner breadth).
4. No parallel planning/execution.
5. No real disk buffer manager, WAL, or transaction engine.

This project intentionally isolates the optimizer+ML loop for explainability and experimentation.

## 11. Key Insights Learned

1. Cost modeling errors dominate bad plan choices; measuring bias and relative error is essential.
2. Explicit selectivity and intermediate join-size features materially improve learned ranking.
3. ML improves plan choice when rule assumptions fail, but robust statistics remain critical.
4. Join-shape diversity (including bushy cases) can uncover substantially better plans.
5. Structured experiment logs turn optimizer behavior into a reproducible data pipeline.