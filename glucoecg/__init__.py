"""
glucoecg: shared, leak-free utilities for MIMIC-IV-Ext-ECG-Glucose v1.0.0.

    config    dataset path, checksum, published counts, run_info.json
    features  allowed / forbidden model inputs, ECG cleaning, causal history
    metrics   regression metrics, Clarke and Parkes grids, patient-level bootstrap
    modeling  XGBoost / Ridge / Random Forest with the paper's hyperparameters
    plots     shared figures
"""

__version__ = "1.1.0"
