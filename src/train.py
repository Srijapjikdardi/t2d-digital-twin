"""Train + evaluate the twin for BOTH adverse events (hyper >180, hypo <70).

Patient-level split (no leakage). Ablation: CGM / +wearable / +EHR.
Saves models, metrics, ROC plots, SHAP global plot.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # always run from project root
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import shap
except ImportError:  # SHAP needs numba; skip plots if it cannot be installed
    shap = None
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_SETS, LOW, THRESH, build_dataset  # noqa: E402

DATA, MODELS, DOCS = "data", "models", "docs"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

ehr = pd.read_csv(f"{DATA}/ehr.csv")
wear = pd.read_csv(f"{DATA}/wearable.csv")
sleep = pd.read_csv(f"{DATA}/sleep.csv")
df = build_dataset(ehr, wear, sleep)
df = df.dropna(subset=["label_high", "label_low", "g_delta_2h", "steps_3h"])

pids = df["patient_id"].unique()
rng = np.random.default_rng(0)
rng.shuffle(pids)
n_test = int(0.25 * len(pids))
test_ids = set(pids[:n_test])

TASKS = {
    "high": dict(label="label_high", subset=lambda d: d[d["g_now"] <= THRESH],
                 text=f"glucose > {THRESH:.0f} mg/dL within 2 h (given now <= {THRESH:.0f})"),
    "low": dict(label="label_low", subset=lambda d: d[d["g_now"] >= LOW],
                text=f"glucose < {LOW:.0f} mg/dL within 2 h (given now >= {LOW:.0f})"),
}

results, bundle, probs_all, tests = {}, {}, {}, {}
for task, cfg in TASKS.items():
    d = cfg["subset"](df)
    train, test = d[~d.patient_id.isin(test_ids)], d[d.patient_id.isin(test_ids)]
    results[task] = {}
    probs_all[task], tests[task] = {}, test
    for name, cols in FEATURE_SETS.items():
        clf = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0
        )
        clf.fit(train[cols], train[cfg["label"]])
        p = clf.predict_proba(test[cols])[:, 1]
        probs_all[task][name] = p
        thr = 0.5 if task == "high" else 0.3  # rarer event -> lower alert threshold
        prec, rec, _, _ = precision_recall_fscore_support(
            test[cfg["label"]], p >= thr, average="binary", zero_division=0)
        results[task][name] = dict(
            roc_auc=round(float(roc_auc_score(test[cfg["label"]], p)), 4),
            pr_auc=round(float(average_precision_score(test[cfg["label"]], p)), 4),
            alert_threshold=thr, precision=round(float(prec), 4), recall=round(float(rec), 4),
        )
        print(task, name, results[task][name])
        if name == "CGM + wearable + EHR (twin)":
            bundle[task] = dict(model=clf, features=cols, threshold=thr,
                                background=train[cols].sample(100, random_state=1))
    results[task]["_meta"] = dict(
        n_train=int(len(train)), n_test=int(len(test)), task=cfg["text"],
        positive_rate=round(float(d[cfg["label"]].mean()), 4),
        random_baseline_pr_auc=round(float(test[cfg["label"]].mean()), 4),
    )

json.dump(results, open(f"{MODELS}/metrics.json", "w"), indent=2)
joblib.dump(bundle, f"{MODELS}/twin_model.joblib")

# ROC ablation plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, task in zip(ax, TASKS):
    for name in FEATURE_SETS:
        fpr, tpr, _ = roc_curve(tests[task][TASKS[task]["label"]], probs_all[task][name])
        a.plot(fpr, tpr, label=f"{name} ({results[task][name]['roc_auc']:.3f})")
    a.plot([0, 1], [0, 1], "k--", lw=0.6)
    a.set(xlabel="False positive rate", ylabel="True positive rate",
          title=("HIGH (>180)" if task == "high" else "LOW (<70)") + ": ablation")
    a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{DOCS}/results.png", dpi=150); plt.close()

# SHAP global importance (mean |SHAP|) for each task
if shap is None:
    print("SHAP not installed: skipping docs/shap_global.png")
    sys.exit(0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for a, task in zip(ax, TASKS):
    b = bundle[task]
    sub = tests[task].sample(min(1500, len(tests[task])), random_state=0)[b["features"]]  # then first 200 explained
    bg = tests[task].sample(100, random_state=1)[b["features"]]
    ex = shap.Explainer(lambda x, m=b["model"]: m.predict_proba(pd.DataFrame(x, columns=b["features"]))[:, 1], bg)
    sv = ex(sub.iloc[:200], silent=True)
    imp = np.abs(sv.values).mean(0)
    order = np.argsort(imp)[-10:]
    a.barh(np.array(b["features"])[order], imp[order], color="#3FA796" if task == "high" else "#6FB3E0")
    a.set(title=f"SHAP: top features, {'HIGH' if task == 'high' else 'LOW'} model", xlabel="mean |SHAP| (probability)")
plt.tight_layout(); plt.savefig(f"{DOCS}/shap_global.png", dpi=150); plt.close()
print("saved")
