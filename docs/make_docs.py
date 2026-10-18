"""Generates docs/architecture.pdf and docs/presentation.pdf (run from repo root)."""
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

G, D, L = "#2a7f62", "#1d2b36", "#e8f3ee"
m = json.load(open("models/metrics.json"))
full, cgm, wear = m["CGM + wearable + EHR (twin)"], m["CGM only"], m["CGM + wearable"]


def box(ax, x, y, w, h, text, fc=L, ec=G, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=D, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=D, lw=1.3))


# ---------- architecture ----------
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
ax.text(6.5, 6.6, "T2D Digital Twin PoC: Architecture", ha="center", fontsize=16, fontweight="bold", color=D)
box(ax, 0.3, 4.6, 3.0, 1.3, "STATIC STREAM\nSynthetic EHR (own simulator)\nage, BMI, HbA1c, eGFR, SBP,\nTCF7L2 alleles, metformin", fs=8.5)
box(ax, 0.3, 2.4, 3.0, 1.5, "DYNAMIC STREAM\nSynthetic wearable / CGM\nglucose 15-min, heart rate,\nsteps, nightly sleep", fs=8.5)
box(ax, 0.3, 0.6, 3.0, 1.0, "Simulator: generate_data.py\nmeals x insulin resistance,\nwalking, sleep, noise", fc="white", fs=8)
arrow(ax, 1.8, 1.6, 1.8, 2.4)
box(ax, 4.2, 3.0, 2.6, 2.2, "FEATURE FUSION\nfeatures.py\n2h CGM window stats\n1h/3h steps, HR, sleep, time\n+ EHR vector (concat)", bold=False, fs=8.5)
arrow(ax, 3.3, 5.2, 4.2, 4.6); arrow(ax, 3.3, 3.2, 4.2, 3.7)
box(ax, 7.6, 3.0, 2.4, 2.2, "TWIN MODEL\nHistGradientBoosting\nP(glucose > 180 mg/dL\nwithin next 2 h)\npatient-level split", fs=8.5)
arrow(ax, 6.8, 4.1, 7.6, 4.1)
box(ax, 10.7, 4.2, 2.1, 1.6, "CLINICIAN DASHBOARD\nStreamlit: EHR card, CGM,\nrisk timeline, alerts", fs=8)
box(ax, 10.7, 2.2, 2.1, 1.5, "WHAT-IF ENGINE\nsteps / sleep sliders\n-> counterfactual risk", fs=8)
arrow(ax, 10.0, 4.4, 10.7, 4.9); arrow(ax, 10.0, 3.8, 10.7, 3.0)
box(ax, 4.2, 0.6, 5.8, 1.5, "EVALUATION (train.py)\nAblation: CGM only vs +wearable vs +EHR\nROC-AUC, PR-AUC, permutation importance", fc="white", fs=8.5)
arrow(ax, 8.8, 3.0, 8.0, 2.1)
ax.text(6.5, 0.15, "All data synthetic / anonymised: DPDP Act and HIPAA compliant by design. Not a medical device.",
        ha="center", fontsize=8, style="italic", color=D)
fig.savefig("docs/architecture.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote docs/architecture.pdf")
