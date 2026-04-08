#!/usr/bin/env bash
set -euo pipefail

/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml train
/opt/homebrew/bin/python3 -m neural_query_optimizer.main --config configs/default.yaml visualize
