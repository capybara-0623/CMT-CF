#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8

preflight=outputs/p6b_bridge_preflight_20260918_v1
run=outputs/p6b_formal_autodl_v1
stats=outputs/p6b_statistics_autodl_v1
tables=outputs/p6b_tables_autodl_v1

python -B -u run_cf_formal_bridge.py search --preflight-dir "$preflight" --run-dir "$run" --authorization outputs/autodl_generation_authorization.json
python -B -u run_cf_formal_bridge.py audit --run-dir "$run"
python -B -u run_cf_formal_bridge.py summarize --run-dir "$run"
python -B -u run_cf_formal_bridge.py verify --run-dir "$run"
python -B -u run_cf_formal_bridge.py statistics --preflight-dir "$preflight" --run-dir "$run" --output-dir "$stats"
python -B -u -m tools.export_p6b_tables --run-dir "$run" --statistics-dir "$stats" --output-dir "$tables"
printf 'Completed: %s\n' "$PWD/$tables/RESULTS.md"
