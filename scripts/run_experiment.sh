#!/usr/bin/env bash
set -euo pipefail

/opt/homebrew/bin/python3 run_experiments.py --config configs/default.yaml
/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml visualize
