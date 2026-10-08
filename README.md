# mimic-iv-ext-ecg-glucose

Code for **MIMIC-IV-Ext-ECG-Glucose v1.0.0**
([PhysioNet, doi:10.13026/nts2-vj76](https://doi.org/10.13026/nts2-vj76)):
437,671 laboratory blood glucose results from 131,771 MIMIC-IV patients, each linked to the
nearest 12-lead diagnostic ECG from MIMIC-IV-ECG recorded within ±240 minutes, with ECG
quality, insulin/dextrose exposure, temporal-alignment and usability annotations.

This repository contains the BigQuery SQL that builds the dataset, a leak-free feature
package, a PyTorch data loader, reference baselines and the scripts behind every table and
figure of the Technical Validation in the accompanying Data Descriptor.

> **Data access.** The CSV is distributed by PhysioNet under credentialed access (PhysioNet
> Credentialed Health Data License 1.5.0). It is **not** in this repository. Never commit
> it, or any file derived from it at row level.

## Quick start

```bash
git clone https://github.com/mdbasit897/mimic-iv-ext-ecg-glucose.git
cd mimic-iv-ext-ecg-glucose
pip install -r requirements.txt            # add requirements-torch.txt for the MLP and data loader
export GLUCOECG_CSV=/path/to/mimiciv_ecg_glucose_aligned.csv
pytest -q                                  # 35 tests, including counts against PhysioNet v1.0.0
bash run_all.sh                            # every table and figure, written to results/
```

`bash run_all.sh --quick` runs one seed and 100 bootstrap resamples (smoke test). The full
run uses 5 seeds and 1,000 patient-level resamples. On a SLURM cluster, edit the paths at the
top of `cluster/run_all.slurm` and run `sbatch cluster/run_all.slurm`.

## Repository layout

| Path | Contents |
| --- | --- |
| `sql/` | `mimiciv_ecg_glucose_pipeline.sql` (identical to PhysioNet v1.0.0), the ±90 and ±150 min sensitivity variants, and `QUERY_GUIDE.md` |
| `glucoecg/` | Shared library: `config` (paths, checksum, run info), `features` (allowed and forbidden inputs, ECG cleaning, causal history), `metrics` (Clarke and Parkes error grids, bootstrap intervals), `modeling`, `plots` |
| `validation/` | Scripts that reproduce the paper: Table S1/S2 checks, Figures 1–6, reference baselines, MLP, stratified and pharmacokinetic-window sensitivity analyses |
| `loaders/` | PyTorch `Dataset` and `DataLoader` (`glucoecg_dataset.py`) and WFDB waveform loading (`load_ecg_waveform.py`) |
| `notebooks/` | `exemplar_usage.ipynb`: a walkthrough of the dataset and a leak-free baseline |
| `tests/` | Error-grid reference points, leak guard, causal-history checks, published counts |
| `run_all.sh`, `cluster/` | Run everything in order, locally or as a SLURM job |

## Columns that must not be model inputs

The CSV keeps some columns for description and stratification that are computed from the
target glucose or from information that only exists after it:

| Group | Columns |
| --- | --- |
| Target-derived | `glycemic_class`, `lab_flag`, `glucose_delta_mg_dl`, `glucose_rate_mg_dl_per_hr`, `glucose_trajectory`, `glucose_z_score`, `glucose_minmax_norm` |
| Whole-record patient statistics (include current and later records) | `patient_mean_glucose`, `patient_std_glucose`, `patient_min_glucose`, `patient_max_glucose`, `glucose_cv_percent`, `time_in_range_pct`, `time_below_range_pct`, `time_above_range_pct` |
| Known only later | `n_glucose_ecg_pairs`, `icu_los_days`, `hospital_expire_flag`, `last_careunit` |
| Intervention annotations (windows include doses given after the draw) | `decoupling_risk_score`, `intervention_status`, `insulin_active`, `fast_insulin_active`, `dextrose_active`, `dual_intervention_active`, `max_insulin_dose_units`, `max_dextrose_amount_ml` |

`glucoecg.features.assert_no_leak()` refuses these columns, and
`glucoecg.features.add_causal_history()` builds history features from strictly earlier
records only.

```python
from glucoecg.features import load_dataset
from glucoecg.modeling import train_feature_set

df = load_dataset("mimiciv_ecg_glucose_aligned.csv")     # cleaning + causal history
model, pred = train_feature_set(df, "ecg_history", train_subset="iu", seed=42)
```

## Evaluation protocol

- **Split:** the `split` column (patient-level `FARM_FINGERPRINT`, 80/10/10); no patient appears in two partitions.
- **Training:** IDEAL+USABLE TRAIN rows, early stopping on IDEAL+USABLE VALIDATION rows, sample weights IDEAL 4, USABLE 2.
- **Testing:** fixed subsets from `glucoecg.features.test_sets` (`iu` primary, `conservative`, `all`, `iu_history`, `iu_no_history`).
- **Uncertainty:** patient-level bootstrap 95% confidence intervals; 5 seeds.
- **Error grids:** Clarke et al. (1987) with the published thresholds, and the Parkes type 1 consensus grid (2000).

Reference result (IDEAL+USABLE test set, n = 12,145): XGBoost with ECG, causal history and
context inputs reaches MAE 30.16 mg/dL (95% CI 29.37–31.01), R² 0.34; predicting the
training-set mean gives 40.04 mg/dL.

## Outputs and privacy

Every script writes `run_info.json` (git commit, CSV SHA-256, package versions) next to its
outputs in `results/`, which is git-ignored. `predictions_test.csv.gz` files contain row-level
identifiers from a credentialed dataset: keep them private.

## Citation

Please cite the dataset (doi:10.13026/nts2-vj76), the accompanying Data Descriptor,
MIMIC-IV v3.1, MIMIC-IV-ECG v1.0 and PhysioNet. `CITATION.cff` gives the software citation.

## License

Code: MIT (see `LICENSE`). Data: PhysioNet Credentialed Health Data License 1.5.0.
