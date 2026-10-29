"""Evaluate the twin pipeline on REAL open CGM data: ShanghaiT2DM (Zhao et al., Sci Data 2023).

Download (Figshare collection):  https://doi.org/10.6084/m9.figshare.c.6310860
Unzip the *T2DM* part, then run:
    python src/eval_real.py "C:/path/to/Shanghai_T2DM"

What it does (CGM + whatever static EHR columns it can find; the dataset has NO wearable steps/HR/sleep):
  A. Zero-shot transfer: model trained ONLY on synthetic data -> tested on real patients.
  B. Within-real 5-fold CV (split by patient): CGM-only vs CGM + EHR.
Results -> results/real_data_results.json   (the script prints what it detected so you can check it)
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # always run from project root
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import CGM_COLS, LOW, THRESH, build_dataset, cgm_features  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]


def find_col(cols, *keys, exclude=()):
    for c in cols:
        lc = str(c).lower()
        if all(k in lc for k in keys) and not any(e in lc for e in exclude):
            return c
    return None


# ---------- 1. locate files ----------
csvs, summaries = [], []
for dp, _, fs in os.walk(ROOT):
    for f in fs:
        p = os.path.join(dp, f)
        fl = f.lower()
        if fl.endswith((".xlsx", ".xls")) or (fl.endswith(".csv") and "summary" in fl):
            summaries.append(p)
        elif fl.endswith(".csv"):
            csvs.append(p)
print(f"found {len(csvs)} CGM csv files, {len(summaries)} summary file(s)")
if not csvs:
    sys.exit("No CSV files found. Check the folder path.")


def pid_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem


# ---------- 2. CGM series -> 15-min grid ----------
series = {}
for p in csvs:
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        print("skip", p, e)
        continue
    tcol = find_col(df.columns, "date") or find_col(df.columns, "time")
    gcol = find_col(df.columns, "cgm")
    if tcol is None or gcol is None:
        print("skip (no date/CGM column):", os.path.basename(p), list(df.columns)[:6])
        continue
    t = pd.to_datetime(df[tcol], errors="coerce")
    g = pd.to_numeric(df[gcol], errors="coerce")
    s = pd.Series(g.values, index=t).dropna()
    s = s[~s.index.isna()].sort_index()
    if len(s) < 100:
        continue
    s = s.resample("15min").mean().interpolate(limit=2)
    if s.median() < 30:  # looks like mmol/L
        s = s * 18.016
    series[pid_of(p)] = s
print(f"usable CGM series: {len(series)}")
if not series:
    sys.exit("No usable series. Send me the printed 'skip' lines.")

# ---------- 3. static EHR from summary ----------
ehr = pd.DataFrame(index=list(series))
if summaries:
    sm = None
    for sp in summaries:
        try:
            sm = pd.read_excel(sp) if sp.lower().endswith(("xlsx", "xls")) else pd.read_csv(sp)
            break
        except Exception as e:  # noqa: BLE001
            print("could not read summary", sp, e)
    if sm is not None:
        idc = find_col(sm.columns, "patient") or sm.columns[0]
        sm["_key"] = sm[idc].astype(str)
        mp = {
            "age": find_col(sm.columns, "age"),
            "sex_male": find_col(sm.columns, "gender") or find_col(sm.columns, "sex"),
            "bmi": find_col(sm.columns, "bmi"),
            "hba1c": find_col(sm.columns, "hba1c", exclude=("ga",)),
            "fasting_glucose": find_col(sm.columns, "fpg") or find_col(sm.columns, "fasting"),
            "egfr": find_col(sm.columns, "egfr"),
            "diabetes_duration_yrs": find_col(sm.columns, "duration"),
        }
        print("static columns detected:", {k: v for k, v in mp.items() if v})
        for k, c in mp.items():
            if c is None:
                continue
            vals = {}
            for pid in series:
                m = sm[sm["_key"].str.contains(re.escape(pid.split("_")[0]), regex=True)]
                if len(m):
                    v = m.iloc[0][c]
                    if k == "sex_male":
                        v = 1.0 if str(v).strip().lower() in ("1", "male", "m") else 0.0
                    vals[pid] = pd.to_numeric(v, errors="coerce")
            ehr[k] = pd.Series(vals)
EHR_REAL = [c for c in ehr.columns if ehr[c].notna().mean() > 0.8]
print("static features used:", EHR_REAL)

# ---------- 4. real feature table ----------
parts = []
for pid, s in series.items():
    f = cgm_features(s)
    f["patient_id"] = pid
    f["hour"] = (s.index.hour + s.index.minute / 60.0).values
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    for c in EHR_REAL:
        f[c] = ehr.loc[pid, c]
    parts.append(f)
real = pd.concat(parts, ignore_index=True).dropna(subset=["g_delta_2h", "label_high", "label_low"])
cols_cgm = CGM_COLS + ["hour_sin", "hour_cos"]
cols_ehr = cols_cgm + EHR_REAL
print(f"real samples: {len(real)} from {real.patient_id.nunique()} patients | "
      f"time >180: {(real.g_now > THRESH).mean():.3f} | time <70: {(real.g_now < LOW).mean():.3f}")

TASKS = {
    "high": ("label_high", lambda d: d[d.g_now <= THRESH]),
    "low": ("label_low", lambda d: d[d.g_now >= LOW]),
}


def model():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200,
                                          l2_regularization=1.0, random_state=0)


def metrics(y, p):
    if y.sum() < 20 or (1 - y).sum() < 20:
        return dict(roc_auc=None, pr_auc=None, note="too few events")
    return dict(roc_auc=round(float(roc_auc_score(y, p)), 4),
                pr_auc=round(float(average_precision_score(y, p)), 4), prevalence=round(float(y.mean()), 4))


out = {"n_patients": int(real.patient_id.nunique()), "n_samples": int(len(real)),
       "static_features_used": EHR_REAL}

# ---------- A. zero-shot from synthetic ----------
print("\nA. Training CGM-only model on SYNTHETIC data, testing on REAL ...")
syn = build_dataset(pd.read_csv("data/ehr.csv"), pd.read_csv("data/wearable.csv"), pd.read_csv("data/sleep.csv"))
syn = syn.dropna(subset=["label_high", "label_low", "g_delta_2h"])
out["synthetic_to_real"] = {}
for t, (lab, sub) in TASKS.items():
    tr, te = sub(syn), sub(real)
    m = model().fit(tr[cols_cgm], tr[lab])
    out["synthetic_to_real"][t] = metrics(te[lab].values, m.predict_proba(te[cols_cgm])[:, 1])
    print(" ", t, out["synthetic_to_real"][t])

# ---------- B. within-real CV by patient ----------
print("\nB. 5-fold patient-level CV on REAL data ...")
out["real_cv"] = {}
for t, (lab, sub) in TASKS.items():
    d = sub(real).reset_index(drop=True)
    out["real_cv"][t] = {}
    for name, cols in (("CGM only", cols_cgm), ("CGM + EHR", cols_ehr)):
        if name == "CGM + EHR" and not EHR_REAL:
            continue
        p = np.full(len(d), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(d, groups=d.patient_id):
            p[te] = model().fit(d.loc[tr, cols], d.loc[tr, lab]).predict_proba(d.loc[te, cols])[:, 1]
        out["real_cv"][t][name] = metrics(d[lab].values, p)
        print(" ", t, name, out["real_cv"][t][name])

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/real_data_results.json", "w"), indent=2)
print("\nsaved results/real_data_results.json")
