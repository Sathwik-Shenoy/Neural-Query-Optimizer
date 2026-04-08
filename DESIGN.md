# Design Notes: Neural Query Optimizer

## Why Cost Modeling Is Hard

A cost-based optimizer depends on row-count and operator-cost estimation. Both are imperfect:

- Cardinality errors multiply through join trees.
- Correlated predicates violate independence assumptions.
- Data skew invalidates uniformity assumptions.
- Runtime depends on memory, caching, and CPU effects not fully captured by simple formulas.

Even in production DBMS engines, cardinality estimation is a top source of bad plans.

## Why ML Helps

This project frames ML as a **cost approximation layer**, not a replacement for planning.

- Rule-based estimator provides interpretable priors.
- ML learns residual patterns from observed plan runtimes.
- Inference ranks existing plan candidates by predicted cost.

This can reduce systematic heuristic bias while preserving optimizer structure.

### Direct answer: Why ML over rule-based?

- Rule models assume uniform data and independent predicates.
- Real workloads violate those assumptions via skew, correlation, and drift.
- A learned model is trained on observed runtime and can approximate the missing nonlinear effects.
- The optimizer still enumerates and validates candidate plans; ML only improves cost ranking.

In short: ML augments cost estimation where hand-crafted formulas are systematically wrong.

## Where ML Fails

- Query distribution shift: unseen query templates degrade predictions.
- Feature blind spots: missing statistics (skew/correlation) create model bias.
- Feedback delay: stale training data causes drift.
- Outlier runtimes: heavy-tail latency is hard for standard regressors.

## Tradeoffs

- Interpretability vs accuracy:
  - Rule costs are explainable but coarse.
  - ML costs can be better but less interpretable.

- Search quality vs compute:
  - More join orders improve chance of optimal plan.
  - Enumeration overhead grows exponentially.

- Generalization vs specialization:
  - Broad model supports many workloads.
  - Workload-specific models can perform better but are brittle.

## Difference from PostgreSQL-style Optimizer

- PostgreSQL has richer statistics (histograms, MCV lists, correlation), many physical operators, and deeper planner integration.
- This project uses a compact simulator and lightweight statistics.
- The architecture mirrors real optimizer loops (parse -> enumerate -> estimate -> choose -> execute) but simplifies storage and execution internals.

### Explicit limitations vs PostgreSQL

1. No catalog-level statistics subsystem (no ANALYZE lifecycle).
2. No histogram/MCV selectivity model with correlation corrections.
3. Limited join search space (left-deep + small bushy subset only).
4. No parallel query planning/execution.
5. No real buffer manager, disk layout, or concurrency control.

These are deliberate scope cuts to isolate learned cost estimation in a controllable environment.

## Failure Cases to Discuss in Interviews

1. Skewed join keys produce severe cardinality underestimation.
2. Predicate correlation overestimates selectivity reduction.
3. Unseen multi-join templates lower ML ranking accuracy.
4. Small synthetic training set can overfit scan/join heuristic patterns.
5. Cost constants tuned for one hardware profile transfer poorly.

## Cost vs Actual Gap Analysis

For each selected plan, we track:

- `cost_error = estimated_or_predicted_cost - actual_runtime_cost`

Interpretation:

- Positive error: model overestimates cost.
- Negative error: model underestimates cost.

Root causes to discuss:

1. Cache effects are not fully represented in static formulas.
2. Data skew breaks NDV-based selectivity assumptions.
3. Correlated predicates violate independence assumptions.
4. Runtime noise and Python/Pandas overhead distort clean operator-level models.
