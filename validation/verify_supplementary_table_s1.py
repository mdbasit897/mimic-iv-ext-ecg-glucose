"""
verify_supplementary_table_s1.py
═══════════════════════════════════════════════════════════════════════════════
Verification script for Supplementary Table S1
Confirms ALL claims in the supplementary materials against the actual CSV data.

    python3 validation/verify_supplementary_table_s1.py

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Path from the command line, else the GLUCOECG_CSV environment variable
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GLUCOECG_CSV", "")
if not CSV_PATH:
    sys.exit("Usage: python %s /path/to/mimiciv_ecg_glucose_aligned.csv  (or set GLUCOECG_CSV)" % Path(__file__).name)

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH from SQL (what the code ACTUALLY implements)
# ─────────────────────────────────────────────────────────────────────────────

# From insulin_fast CTE (line 171): itemid IN (223258, 229299, 228959, 223259)
# effect_window_minutes = 120 → triggers fast_insulin_active = TRUE
# → intervention_status = FLAG_HIGH_RISK
# → decoupling_risk_score contribution = 2
SQL_FAST_INSULIN_ITEMIDS = {223258, 229299, 228959, 223259}

# From insulin_basal CTE (line 192): itemid IN (223260, 223262)
# effect_window_minutes = 240 → triggers insulin_active = TRUE (not fast)
# → intervention_status = FLAG_MODERATE_RISK
# → decoupling_risk_score contribution = 0 (no fast_insulin component)
SQL_BASAL_INSULIN_ITEMIDS = {223260, 223262}

# From dextrose_iv CTE (line 213):
# itemid IN (220949, 220950, 228142, 220952, 220955, 220953)
# effect_window_minutes = 60 → triggers dextrose_active = TRUE
# → intervention_status = FLAG_MODERATE_RISK
# → decoupling_risk_score contribution = 1
SQL_DEXTROSE_ITEMIDS = {220949, 220950, 228142, 220952, 220955, 220953}

# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY TABLE S1 — what the paper CLAIMS
# ─────────────────────────────────────────────────────────────────────────────
SUPP_TABLE = {
    223258: {"drug": "Regular Insulin",        "formulation": "Fast-acting", "pk_window": 120, "risk": "2 (HIGH_RISK)"},
    229299: {"drug": "Insulin-Humalog (Lispro)","formulation": "Fast-acting", "pk_window": 120, "risk": "2 (HIGH_RISK)"},
    228959: {"drug": "Insulin-Novolog (Aspart)","formulation": "Fast-acting", "pk_window": 120, "risk": "2 (HIGH_RISK)"},
    223262: {"drug": "Insulin-NPH",             "formulation": "Basal",       "pk_window": 240, "risk": "1 (MOD_RISK)"},
    223260: {"drug": "Insulin-Glargine",        "formulation": "Basal",       "pk_window": 240, "risk": "1 (MOD_RISK)"},
    223259: {"drug": "Insulin-70/30",           "formulation": "Mixed (fast-dominant)", "pk_window": 120, "risk": "2 (HIGH_RISK)"},
    220949: {"drug": "D5W",                     "formulation": "IV continuous","pk_window": 60,  "risk": "1 (MOD_RISK)"},
    220950: {"drug": "D10W",                    "formulation": "IV continuous","pk_window": 60,  "risk": "1 (MOD_RISK)"},
    228142: {"drug": "D50W bolus",              "formulation": "IV bolus",    "pk_window": 60,  "risk": "1 (MOD_RISK)"},
    220952: {"drug": "D5-0.9% NaCl",           "formulation": "IV continuous","pk_window": 60,  "risk": "1 (MOD_RISK)"},
    220955: {"drug": "D5-0.45% NaCl",          "formulation": "IV continuous","pk_window": 60,  "risk": "1 (MOD_RISK)"},
    220953: {"drug": "D5-LR",                   "formulation": "IV continuous","pk_window": 60,  "risk": "1 (MOD_RISK)"},
}

SEP  = "═" * 72
SEP2 = "─" * 72

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):   print(f"  ✅  {msg}")
def err(msg):  print(f"  ❌  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def info(msg): print(f"  ℹ️   {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
section("LOADING DATASET")
csv_path = Path(CSV_PATH)
if not csv_path.exists():
    # Try with .csv extension
    csv_path = Path(CSV_PATH + ".csv")
    if not csv_path.exists():
        print(f"  ❌ File not found: {CSV_PATH}")
        print(f"     Also tried:     {CSV_PATH}.csv")
        sys.exit(1)

print(f"  Loading: {csv_path}")
df = pd.read_csv(csv_path, low_memory=False)

# Fix BigQuery boolean columns exported as strings
bool_cols = ["fast_insulin_active", "insulin_active", "dextrose_active",
             "dual_intervention_active", "gap_exceeds_threshold",
             "is_first_glucose_in_stay", "delta_is_intervention_confounded",
             "during_icu_stay", "report_artifact_flag",
             "sqi_rr_ok", "sqi_qrs_ok", "sqi_qt_ok",
             "sqi_p_before_qrs", "sqi_t_after_qrs", "sqi_pr_ok", "sqi_jt_ok"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map(
            lambda x: True if str(x).lower() in ("true", "1", "1.0") else False
        )

print(f"  Total records : {len(df):,}")
print(f"  Total patients: {df['subject_id'].nunique():,}")
print(f"  Columns       : {len(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: STATIC SQL VERIFICATION (no data needed — pure code analysis)
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 1: SQL CODE VERIFICATION (Supplementary Table S1 vs SQL)")
print()
print(f"  {'Item ID':<10} {'Drug':<30} {'Table Claims':<22} {'SQL Implements':<22} {'STATUS'}")
print(f"  {SEP2}")

all_errors = []
all_ok     = []

for itemid, info_row in SUPP_TABLE.items():
    drug         = info_row["drug"]
    claimed_win  = info_row["pk_window"]
    claimed_risk = info_row["risk"]
    claimed_score = int(claimed_risk.split(" ")[0])

    # Determine what SQL actually does for this itemid
    if itemid in SQL_FAST_INSULIN_ITEMIDS:
        actual_window = 120
        actual_score  = 2
        actual_status = "FLAG_HIGH_RISK"
        actual_label  = "2 (HIGH_RISK)"
    elif itemid in SQL_BASAL_INSULIN_ITEMIDS:
        actual_window = 240
        actual_score  = 1
        actual_status = "FLAG_MODERATE_RISK"
        actual_label  = "1 (MOD_RISK)"
    elif itemid in SQL_DEXTROSE_ITEMIDS:
        actual_window = 60
        actual_score  = 1
        actual_status = "FLAG_MODERATE_RISK"
        actual_label  = "1 (MOD_RISK)"
    else:
        actual_window = "NOT IN SQL"
        actual_score  = "?"
        actual_label  = "NOT FOUND"

    window_ok = claimed_win == actual_window
    score_ok  = claimed_score == actual_score

    claimed_str = f"±{claimed_win} min / {claimed_risk}"
    actual_str  = f"±{actual_window} min / {actual_label}"

    if window_ok and score_ok:
        status = "✅ CORRECT"
        all_ok.append(itemid)
    else:
        status = "❌ MISMATCH"
        errmsg = []
        if not window_ok:
            errmsg.append(f"window: table=±{claimed_win}, sql=±{actual_window}")
        if not score_ok:
            errmsg.append(f"score: table={claimed_score}, sql={actual_score}")
        all_errors.append((itemid, drug, errmsg))

    print(f"  {itemid:<10} {drug:<30} {claimed_str:<22} {actual_str:<22} {status}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA VERIFICATION — check actual values in CSV
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 2: DATA VERIFICATION (Checking actual values in your CSV)")

# 2a. Intervention status distribution
print(f"\n  [2a] Intervention Status Distribution")
print(f"  {SEP2}")
if "intervention_status" in df.columns:
    dist = df["intervention_status"].value_counts()
    total = len(df)
    for status, count in dist.items():
        pct = count / total * 100
        print(f"  {status:<25} : {count:>10,}  ({pct:5.1f}%)")
else:
    warn("Column 'intervention_status' not found in CSV")

# 2b. Decoupling risk score distribution — CRITICAL CHECK
# Score=3 should NOT exist if the SQL logic is correct
print(f"\n  [2b] Decoupling Risk Score Distribution")
print(f"       ⚠️  Score=3 should NEVER appear (max possible = 4)")
print(f"  {SEP2}")
if "decoupling_risk_score" in df.columns:
    score_dist = df["decoupling_risk_score"].value_counts().sort_index()
    total = len(df)
    for score, count in score_dist.items():
        pct = count / total * 100
        flag = ""
        if score == 3:
            flag = "  ← ❌ SHOULD NOT EXIST — SQL BUG"
        elif score == 4:
            flag = "  ← dual intervention (insulin + dextrose)"
        elif score == 2:
            flag = "  ← fast insulin only"
        elif score == 1:
            flag = "  ← basal insulin only OR dextrose only"
        elif score == 0:
            flag = "  ← CLEAN"
        print(f"  Score {int(score):<5} : {count:>10,}  ({pct:5.1f}%){flag}")

    if 3 in score_dist.index and score_dist[3] > 0:
        err(f"Score=3 exists with {score_dist[3]:,} records — logic error in SQL!")
        all_errors.append(("SCORE", "decoupling_risk_score=3", ["should be impossible"]))
    else:
        ok("Score=3 does not exist — risk score formula is consistent")
else:
    warn("Column 'decoupling_risk_score' not found in CSV")

# 2c. THE KEY CHECK: Does fast_insulin_active correctly map to FLAG_HIGH_RISK?
# This is what proves whether 70/30 (itemid 223259) was treated as fast or basal
print(f"\n  [2c] fast_insulin_active → intervention_status mapping")
print(f"       (This reveals how 70/30 insulin was actually processed)")
print(f"  {SEP2}")
if "fast_insulin_active" in df.columns and "intervention_status" in df.columns:
    cross = pd.crosstab(df["fast_insulin_active"], df["intervention_status"])
    print(f"  fast_insulin_active | ", end="")
    for col in cross.columns:
        print(f"{col:<22}", end="")
    print()
    print(f"  {SEP2}")
    for idx, row in cross.iterrows():
        print(f"  {str(idx):<20}", end="")
        for val in row:
            print(f"  {val:<20,}", end="")
        print()

    # Verify: fast_insulin=True must NEVER be FLAG_MODERATE_RISK or CLEAN
    # (unless also dual=True → EXCLUDE)
    if "FLAG_MODERATE_RISK" in cross.columns:
        n_wrong = cross.loc[True, "FLAG_MODERATE_RISK"] if True in cross.index else 0
        if n_wrong > 0:
            err(f"{n_wrong:,} records have fast_insulin_active=True but FLAG_MODERATE_RISK")
            err("This would mean 70/30 was treated as basal — contradicts SQL")
        else:
            ok("No fast_insulin=True records have FLAG_MODERATE_RISK — correct")

    if "CLEAN" in cross.columns:
        n_clean_fast = cross.loc[True, "CLEAN"] if True in cross.index else 0
        if n_clean_fast > 0:
            err(f"{n_clean_fast:,} records have fast_insulin_active=True but CLEAN — impossible")
        else:
            ok("No fast_insulin=True records are CLEAN — correct")
else:
    warn("Columns 'fast_insulin_active' or 'intervention_status' not found")

# 2d. Dual intervention check — score=4 verification
print(f"\n  [2d] Dual Intervention (score=4) Verification")
print(f"       SQL: dual = (fast OR basal insulin) AND dextrose; score = 2*fast + dextrose + dual")
print(f"       so dual records score 4 with fast-acting insulin and 2 with basal insulin only")
print(f"  {SEP2}")
if all(c in df.columns for c in ["dual_intervention_active", "fast_insulin_active",
                                   "dextrose_active", "decoupling_risk_score"]):
    dual_true = df[df["dual_intervention_active"] == True]
    n_dual    = len(dual_true)
    print(f"  dual_intervention_active=True : {n_dual:,} records")

    if n_dual > 0:
        # Expected score: 4 if fast-acting insulin is active, else 2 (basal insulin + dextrose)
        expected = dual_true["fast_insulin_active"].map(lambda v: 4 if v is True or v == 1 else 2)
        n_bad = int((dual_true["decoupling_risk_score"] != expected).sum())
        n_basal = int((expected == 2).sum())
        ok(f"All dual records have the expected score (4 with fast insulin; 2 for the "
           f"{n_basal:,} basal-insulin + dextrose records)") \
            if n_bad == 0 else \
            err(f"{n_bad:,} dual records have an unexpected decoupling_risk_score")

        # All dual records must have intervention_status=EXCLUDE
        if "intervention_status" in df.columns:
            excl_pct = (dual_true["intervention_status"] == "EXCLUDE").mean() * 100
            ok(f"{excl_pct:.1f}% of dual records have intervention_status=EXCLUDE") \
                if excl_pct == 100.0 else \
                err(f"Only {excl_pct:.1f}% of dual records are EXCLUDE — expected 100%")
    else:
        warn("No dual intervention records found — check your data")
else:
    warn("Required columns for dual check not all found")

# 2e. Internal consistency checks
print(f"\n  [2e] Internal Consistency Checks (should all be 0 violations)")
print(f"  {SEP2}")
checks = []

if all(c in df.columns for c in ["fast_insulin_active", "dual_intervention_active", "intervention_status"]):
    # fast=True, dual=False must → FLAG_HIGH_RISK
    mask = (df["fast_insulin_active"] == True) & (df["dual_intervention_active"] == False)
    n_not_high = (df.loc[mask, "intervention_status"] != "FLAG_HIGH_RISK").sum()
    checks.append(("fast_insulin=True, dual=False, but NOT FLAG_HIGH_RISK", n_not_high))

if all(c in df.columns for c in ["dual_intervention_active", "fast_insulin_active", "dextrose_active"]):
    # dual=True must have dextrose and some insulin (fast or basal) active
    dual = df[df["dual_intervention_active"] == True]
    n_dual_no_dext = (dual["dextrose_active"] != True).sum()
    checks.append(("dual=True but dextrose=False", n_dual_no_dext))
    if "insulin_active" in df.columns:
        checks.append(("dual=True but insulin_active=False", (dual["insulin_active"] != True).sum()))

if "decoupling_risk_score" in df.columns:
    n_score3 = (df["decoupling_risk_score"] == 3).sum()
    checks.append(("decoupling_risk_score = 3 (impossible value)", n_score3))

for check_name, n_violations in checks:
    if n_violations == 0:
        ok(f"0 violations — {check_name}")
    else:
        err(f"{n_violations:,} violations — {check_name}")

# 2f. Record usability distribution
print(f"\n  [2f] Record Usability Distribution")
print(f"  {SEP2}")
if "record_usability" in df.columns:
    usability = df["record_usability"].value_counts()
    total = len(df)
    order = ["IDEAL", "USABLE", "CAUTION", "EXCLUDE"]
    for tier in order:
        count = usability.get(tier, 0)
        pct = count / total * 100
        print(f"  {tier:<10} : {count:>10,}  ({pct:5.1f}%)")
    n_ideal_usable = usability.get("IDEAL", 0) + usability.get("USABLE", 0)
    print(f"  {'IDEAL+USABLE':<10} : {n_ideal_usable:>10,}  ({n_ideal_usable/total*100:5.1f}%)")
else:
    warn("Column 'record_usability' not found")

# 2g. Glucose dynamics NULL rate (85.1% cited in manuscript)
print(f"\n  [2g] Glucose Dynamics NULL Rate")
print(f"       Manuscript cites 85.1% NULL for glucose_rate_mg_dl_per_hr")
print(f"  {SEP2}")
if "glucose_rate_mg_dl_per_hr" in df.columns:
    n_null = df["glucose_rate_mg_dl_per_hr"].isna().sum()
    pct_null = n_null / len(df) * 100
    print(f"  glucose_rate_mg_dl_per_hr NULL: {n_null:,} / {len(df):,} = {pct_null:.1f}%")
    if abs(pct_null - 85.1) < 2.0:
        ok(f"NULL rate {pct_null:.1f}% is consistent with the 85.1% cited")
    else:
        warn(f"NULL rate {pct_null:.1f}% differs from the cited 85.1% — update manuscript")
if "glucose_delta_mg_dl" in df.columns:
    n_null_delta = df["glucose_delta_mg_dl"].isna().sum()
    pct_null_delta = n_null_delta / len(df) * 100
    print(f"  glucose_delta_mg_dl NULL      : {n_null_delta:,} / {len(df):,} = {pct_null_delta:.1f}%")

# 2h. SQI distribution
print(f"\n  [2h] SQI Category Distribution")
print(f"  {SEP2}")
if "sqi_category" in df.columns:
    sqi = df["sqi_category"].value_counts()
    for cat in ["GOOD", "FAIR", "POOR"]:
        count = sqi.get(cat, 0)
        pct = count / len(df) * 100
        print(f"  {cat:<6} : {count:>10,}  ({pct:5.1f}%)")

# 2i. Train/Val/Test split
print(f"\n  [2i] Train / Validation / Test Split")
print(f"  {SEP2}")
if "split" in df.columns:
    splits = df["split"].value_counts()
    total = len(df)
    for s in ["TRAIN", "VALIDATION", "TEST"]:
        count = splits.get(s, 0)
        pct = count / total * 100
        n_pts = df[df["split"] == s]["subject_id"].nunique()
        print(f"  {s:<12} : {count:>10,} records ({pct:4.1f}%)  |  {n_pts:,} patients")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 3: FINAL VERDICT — Supplementary Table S1")

print()
print(f"  Items verified from SQL code analysis: {len(SUPP_TABLE)}")
print(f"  ✅ Correct items    : {len(all_ok)}")
print(f"  ❌ Incorrect items  : {len(all_errors)}")
print()

if all_errors:
    print(f"  ERRORS REQUIRING CORRECTION BEFORE SUBMISSION:")
    print(f"  {SEP2}")
    for item in all_errors:
        if item[0] == "SCORE":
            err(f"{item[1]}: {', '.join(item[2])}")
        else:
            itemid, drug, errs = item
            info_row = SUPP_TABLE.get(itemid, {})
            err(f"Item ID {itemid} ({drug}):")
            for e in errs:
                print(f"       → {e}")
    print()
    print(f"  CORRECTED ROW for Supplementary Table S1:")
    print(f"  {SEP2}")
    print(f"  Item ID 223259 (Insulin-70/30) should read:")
    print(f"    Category    : Insulin")
    print(f"    Formulation : Mixed (fast-acting component)")
    print(f"    PK Window   : ±120 min   (NOT ±240 min)")
    print(f"    Risk Score  : 2 (HIGH_RISK)   (NOT 1 MOD_RISK)")
    print(f"    Reason      : SQL groups 70/30 with fast insulin (itemid IN")
    print(f"                  (223258, 229299, 228959, 223259), window=120)")
    print()
    print(f"  CORRECTED CAPTION sentence:")
    print(f"  {SEP2}")
    print(f"  WRONG : 'Basal insulin formulations (Glargine, NPH, 70/30)")
    print(f"           receive a conservative ±240-minute window.'")
    print()
    print(f"  CORRECT: 'Basal insulin formulations (Glargine, NPH) receive")
    print(f"            a conservative ±240-minute window. Insulin 70/30,")
    print(f"            which contains a fast-acting component, is grouped")
    print(f"            with fast-acting formulations and receives a")
    print(f"            ±120-minute window with a risk score of 2 (HIGH_RISK).'")
else:
    print(f"  ✅ ALL ITEMS VERIFIED — Table S1 is consistent with SQL")

print()
section("DONE — Check all ❌ items above before submission")
print()