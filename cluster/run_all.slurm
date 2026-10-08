#!/bin/bash
# SLURM job: reproduce every table and figure (tests, Table S1/S2 checks, Figures 1-6,
# Table 7 baselines, supplementary analyses, MLP).
#
#   sbatch cluster/run_all.slurm             # full run: 5 seeds, 1,000 bootstrap resamples
#   sbatch cluster/run_all.slurm --quick     # smoke run: 1 seed, 100 resamples
#
# Edit the four paths below once. Outputs go to $REPO/results/ (git-ignored).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # XGBoost and the bootstrap use all allocated cores
#SBATCH --time=24:00:00
#SBATCH --job-name=glucoecg_run_all
#SBATCH --output=glucoecg_run_all_%j.log
#SBATCH --error=glucoecg_run_all_%j.err

set -euo pipefail

# ── Paths to edit ─────────────────────────────────────────────────────────────
REPO=/path/to/mimic-iv-ext-ecg-glucose
VENV=/path/to/venv
export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv
# Optional, for Table S4 (exports of the two sensitivity SQL variants):
# export GLUCOECG_S90=/path/to/sensitivity_A_90min.csv
# export GLUCOECG_S150=/path/to/sensitivity_B_150min.csv
# ─────────────────────────────────────────────────────────────────────────────

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, started $(date)"
cd "$REPO"
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "Python: $(which python3)  commit: $(git rev-parse --short HEAD)  branch: $(git branch --show-current)"
python3 -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())" || true
echo "CSV SHA-256: $(sha256sum "$GLUCOECG_CSV" | cut -d' ' -f1)"
echo "PhysioNet   : eaea161137acbf4cd913a431e0757206f50f87f99a82d0b6a2c67c1fcd9fe552"

PYTHON=python3 bash run_all.sh "$@"

echo "Finished $(date)"
