"""Train + evaluate the twin model with an ablation (CGM / +wearable / +EHR).

Patient-level split (no leakage). Reports ROC-AUC, PR-AUC, and a simple
alert-threshold operating point. Saves model, metrics, plots.
"""
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_SETS, THRESH, build_dataset  # noqa: E402

DATA, MODELS, DOCS = "data", "models", "docs"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

ehr = pd.read_csv(f"{DATA}/ehr.csv")
wear = pd.read_csv(f"{DATA}/wearable.csv")
sleep = pd.read_csv(f"{DATA}/sleep.csv")
df = build_dataset(ehr, wear, sleep)
df = df.dropna(subset=["label", "g_delta_2h", "steps_3h"])
df = df[df["g_now"] <= THRESH]  # onset prediction: currently not hyperglycaemic
print("samples:", len(df), "| positive rate: %.3f" % df["label"].mean())

pids = df["patient_id"].unique()
rng = np.random.default_rng(0)
rng.shuffle(pids)
n_test = int(0.25 * len(pids))
test_ids = set(pids[:n_test])
train = df[~df.patient_id.isin(test_ids)]
test = df[df.patient_id.isin(test_ids)]

results, models, probs = {}, {}, {}
for name, cols in FEATURE_SETS.items():
    clf = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0
    )
    clf.fit(train[cols], train["label"])
    p = clf.predict_proba(test[cols])[:, 1]
    probs[name], models[name] = p, (clf, cols)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test["label"], p >= 0.5, average="binary", zero_division=0
    )
    results[name] = dict(
        roc_auc=round(float(roc_auc_score(test["label"], p)), 4),
        pr_auc=round(float(average_precision_score(test["label"], p)), 4),
        precision_at_0p5=round(float(prec), 4),
        recall_at_0p5=round(float(rec), 4),
    )
    print(name, results[name])

results["_meta"] = dict(
    n_train=int(len(train)), n_test=int(len(test)),
    train_patients=int(len(pids) - n_test), test_patients=int(n_test),
    positive_rate=round(float(df["label"].mean()), 4),
    task=f"P(glucose > {THRESH:.0f} mg/dL within next 2h | currently <= {THRESH:.0f})",
)
json.dump(results, open(f"{MODELS}/metrics.json", "w"), indent=2)

full = "CGM + wearable + EHR (twin)"
clf, cols = models[full]
joblib.dump(dict(model=clf, features=cols), f"{MODELS}/twin_model.joblib")

# Plots: ROC ablation + permutation importance
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name in FEATURE_SETS:
    fpr, tpr, _ = roc_curve(test["label"], probs[name])
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Ablation: value of fusing streams")
ax[0].legend(fontsize=8)

sub = test.sample(min(4000, len(test)), random_state=0)
pi = permutation_importance(clf, sub[cols], sub["label"], scoring="roc_auc", n_repeats=3, random_state=0)
order = np.argsort(pi.importances_mean)[-12:]
ax[1].barh(np.array(cols)[order], pi.importances_mean[order], color="#2a7f62")
ax[1].set(title="Top features (permutation importance, ROC-AUC drop)")
plt.tight_layout()
plt.savefig(f"{DOCS}/results.png", dpi=150)
print("saved model, metrics, plots")
