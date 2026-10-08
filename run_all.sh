#!/usr/bin/env bash
# Reproduce every table and figure of the technical validation from the published CSV.
#
#   export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv
#   bash run_all.sh                 # outputs in results/
#   bash run_all.sh --quick         # 1 seed, 100 bootstrap resamples (smoke test)
#
# Optional, if the two sensitivity-variant CSVs were exported from BigQuery:
#   export GLUCOECG_S90=/path/to/sensitivity_A_90min.csv
#   export GLUCOECG_S150=/path/to/sensitivity_B_150min.csv
set -euo pipefail
cd "$(dirname "$0")"

: "${GLUCOECG_CSV:?Set GLUCOECG_CSV to the path of mimiciv_ecg_glucose_aligned.csv}"
OUT="${OUT:-results}"
PY="${PYTHON:-python3}"
SEEDS="42 1 2 3 4"
BOOT=1000
if [[ "${1:-}" == "--quick" ]]; then SEEDS="42"; BOOT=100; fi

echo "== 1/7 tests (counts against PhysioNet v1.0.0, leak guard, error grids)"
"$PY" -m pytest -q tests

echo "== 2/7 verification of Supplementary Tables S1 and S2"
mkdir -p "$OUT/verification"
"$PY" validation/verify_supplementary_table_s1.py "$GLUCOECG_CSV" > "$OUT/verification/table_s1.txt"
"$PY" validation/verify_supplementary_table_s2.py "$GLUCOECG_CSV" > "$OUT/verification/table_s2.txt"

echo "== 3/7 dataset figures 1-6"
"$PY" validation/generate_figures.py --out "$OUT/figures"
"$PY" validation/figure6_quality_heatmap.py --out "$OUT/figures"

echo "== 4/7 leak-free baselines (Table 7)"
"$PY" validation/baseline_xgboost.py --out "$OUT/baseline" --seeds $SEEDS --bootstrap "$BOOT"

echo "== 5/7 supplementary stratified analyses"
"$PY" validation/baseline_stratified_analysis.py --out "$OUT/supplement" --bootstrap "$BOOT"

echo "== 6/7 pharmacokinetic-window sensitivity"
if [[ -n "${GLUCOECG_S90:-}" && -n "${GLUCOECG_S150:-}" ]]; then
  "$PY" validation/q3_sensitivity_analysis.py --orig "$GLUCOECG_CSV" \
    --s90 "$GLUCOECG_S90" --s150 "$GLUCOECG_S150" --out "$OUT/supplement"
else
  echo "   skipped: this script saw GLUCOECG_S90='${GLUCOECG_S90:-}' GLUCOECG_S150='${GLUCOECG_S150:-}'"
  echo "   (variables must be exported in the shell that starts run_all.sh: export GLUCOECG_S90=...)"
fi

echo "== 7/7 MLP baseline"
if "$PY" -c "import torch" 2>/dev/null; then
  "$PY" validation/baseline_mlp.py --out "$OUT/mlp" --seeds $SEEDS --bootstrap "$BOOT"
else
  echo "   skipped (pip install -r requirements-torch.txt)"
fi

echo "Done. Main table: $OUT/baseline/table7_main.csv ; summary: $OUT/baseline/technical_validation_summary.txt"
