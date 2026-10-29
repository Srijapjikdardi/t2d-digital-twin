"""Clinician dashboard, T2D Digital Twin v2.  Run:  streamlit run app.py"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
except ImportError:
    shap = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import LOW, THRESH, patient_features  # noqa: E402

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # always run from project root
st.set_page_config(page_title="T2D Digital Twin", layout="wide")

NICE = {
    "g_now": "Glucose now", "g_mean_2h": "Mean glucose (2 h)", "g_std_2h": "Glucose variability (2 h)",
    "g_min_2h": "Lowest glucose (2 h)", "g_max_2h": "Highest glucose (2 h)",
    "g_delta_15m": "Change in 15 min", "g_delta_30m": "Change in 30 min", "g_delta_2h": "Change in 2 h",
    "steps_1h": "Steps, last hour", "steps_3h": "Steps, last 3 h", "hr_now": "Heart rate",
    "hr_mean_1h": "Mean heart rate (1 h)", "sleep_prev_night": "Sleep last night (h)",
    "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)", "age": "Age", "sex_male": "Male sex",
    "bmi": "BMI", "diabetes_duration_yrs": "Diabetes duration (y)", "hba1c": "HbA1c",
    "fasting_glucose": "Fasting glucose", "family_history": "Family history",
    "tcf7l2_risk_alleles": "TCF7L2 risk alleles", "on_metformin": "On metformin",
    "on_secretagogue": "On sulfonylurea/insulin", "egfr": "eGFR", "sbp": "Systolic BP",
}


@st.cache_data
def load():
    ehr = pd.read_csv("data/ehr.csv")
    wear = pd.read_csv("data/wearable.csv", parse_dates=["timestamp"])
    sleep = pd.read_csv("data/sleep.csv")
    sleep["date"] = pd.to_datetime(sleep["date"]).dt.date
    return ehr, wear, sleep


@st.cache_resource
def load_bundle():
    return joblib.load("models/twin_model.joblib")


if not (os.path.exists("data/ehr.csv") and os.path.exists("models/twin_model.joblib")):
    st.error("Data or model files are missing. In the terminal run:  python run_all.py   then reload this page.")
    st.stop()


@st.cache_resource
def explainer(task):
    b = load_bundle()[task]
    f = b["features"]
    fn = lambda x: b["model"].predict_proba(pd.DataFrame(x, columns=f))[:, 1]  # noqa: E731
    return shap.Explainer(fn, b["background"])


ehr, wear, sleep = load()
try:
    bundle = load_bundle()
except Exception as e:  # noqa: BLE001  (usually a scikit-learn version mismatch)
    st.error(f"Could not load the saved model ({type(e).__name__}). Run  python src/train.py  to rebuild it on this computer, then reload.")
    st.stop()

st.title("Digital Twin: Type 2 Diabetes glycaemic-event forecaster")
st.caption("Proof of concept. 100% synthetic data. Not a medical device; not clinically validated. "
           "Forecasts BOTH high (>180 mg/dL) and low (<70 mg/dL) glucose 2 hours ahead.")

pid = st.sidebar.selectbox("Virtual patient", ehr.patient_id.tolist(), index=3)
row = ehr[ehr.patient_id == pid].iloc[0]
feats = patient_features(wear[wear.patient_id == pid], sleep[sleep.patient_id == pid], row)
valid = feats.dropna(subset=["g_delta_2h", "steps_3h"]).copy()
for t in ("high", "low"):
    b = bundle[t]
    valid[f"risk_{t}"] = b["model"].predict_proba(valid[b["features"]])[:, 1]

thr_high = st.sidebar.slider("High-glucose alert threshold", 0.2, 0.9, float(bundle["high"]["threshold"]), 0.05)
thr_low = st.sidebar.slider("Low-glucose alert threshold", 0.1, 0.9, float(bundle["low"]["threshold"]), 0.05)
days = sorted(valid.timestamp.dt.date.unique())
day = st.sidebar.selectbox("Day", days, index=3)
d = valid[valid.timestamp.dt.date == day].reset_index(drop=True)
slot = st.sidebar.slider("Time of day (now)", 0, len(d) - 1, min(47, len(d) - 1), format="slot %d (15-min)")
now = d.iloc[slot]

c1, c2 = st.columns([1, 3])
with c1:
    st.subheader(f"Patient #{pid}")
    st.markdown(
        f"**Age** {int(row.age)} · **{'M' if row.sex_male else 'F'}** · **BMI** {row.bmi}\n\n"
        f"**HbA1c** {row.hba1c}% · **Fasting glucose** {int(row.fasting_glucose)} mg/dL\n\n"
        f"**T2D duration** {int(row.diabetes_duration_yrs)} y · **eGFR** {int(row.egfr)}\n\n"
        f"**SBP** {int(row.sbp)} · **Metformin** {'yes' if row.on_metformin else 'no'}\n\n"
        f"**Sulfonylurea/insulin** {'yes' if row.on_secretagogue else 'no'}\n\n"
        f"**Family history** {'yes' if row.family_history else 'no'} · "
        f"**TCF7L2 alleles** {int(row.tcf7l2_risk_alleles)}")
    st.metric("Glucose now", f"{now.g_now:.0f} mg/dL")
    st.metric("Sleep last night", f"{now.sleep_prev_night:.1f} h")
    st.metric("Steps last hour", f"{int(now.steps_1h)}")

with c2:
    rh, rl = float(now.risk_high), float(now.risk_low)
    a, b_ = st.columns(2)
    with a:
        if now.g_now > THRESH:
            st.error(f"HIGH now ({now.g_now:.0f} mg/dL).")
        elif rh >= thr_high:
            st.warning(f"HIGH alert: {rh:.0%} chance of >{THRESH:.0f} mg/dL within 2 h.")
        else:
            st.success(f"High-glucose risk low: {rh:.0%}.")
    with b_:
        if now.g_now < LOW:
            st.error(f"LOW now ({now.g_now:.0f} mg/dL). Treat per clinical protocol.")
        elif rl >= thr_low:
            st.warning(f"LOW alert: {rl:.0%} chance of <{LOW:.0f} mg/dL within 2 h.")
        else:
            st.success(f"Low-glucose risk low: {rl:.0%}.")

    ch = d.set_index("timestamp")[["g_now"]].rename(columns={"g_now": "Glucose (mg/dL)"})
    ch["High threshold (180)"] = THRESH
    ch["Low threshold (70)"] = LOW
    st.line_chart(ch, height=260, color=["#3FA796", "#F2A488", "#6FB3E0"])
    rk = d.set_index("timestamp")[["risk_high", "risk_low"]].rename(
        columns={"risk_high": "Predicted high risk", "risk_low": "Predicted low risk"})
    st.line_chart(rk, height=160, color=["#F2A488", "#6FB3E0"])

# ---------------- SHAP: why this alert ----------------
st.subheader("Why this risk? (SHAP explanation)")
st.caption("Each bar shows how much a feature pushed the predicted risk up (coral) or down (teal) "
           "for this patient at this moment. It explains the model, not biology: it does not prove cause.")
if shap is None:
    st.info("SHAP is not installed on this computer (pip install shap). The rest of the dashboard works without it.")
which = st.radio("Explain", ["high", "low"], horizontal=True,
                 format_func=lambda t: "High-glucose risk (>180)" if t == "high" else "Low-glucose risk (<70)")
if shap is not None:
    bt = bundle[which]
    x = now[bt["features"]].astype(float)
    ex = explainer(which)(pd.DataFrame([x.values], columns=bt["features"]), silent=True)
    contrib = pd.Series(ex.values[0], index=bt["features"])
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([f"{NICE.get(k, k)} = {x[k]:.4g}" for k in top.index], top.values * 100,
            color=["#F2A488" if v > 0 else "#3FA796" for v in top.values])
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Change in risk (percentage points)")
    ax.set_title(f"Base risk {float(ex.base_values[0]):.0%}  ->  this moment {float(contrib.sum() + ex.base_values[0]):.0%}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    st.pyplot(fig)

# ---------------- what-if ----------------
st.subheader("What-if simulation (counterfactual on the twin)")
st.caption("Change behaviour at this moment; the twin re-scores both risks.")
w1, w2 = st.columns(2)
steps_new = w1.slider("Steps in the last hour", 0, 3000, int(now.steps_1h), 100)
sleep_new = w2.slider("Sleep last night (h)", 3.0, 10.0, float(now.sleep_prev_night), 0.5)
m = st.columns(4)
for i, (t, base) in enumerate((("high", rh), ("low", rl))):
    bt = bundle[t]
    xx = now[bt["features"]].copy().astype(float)
    xx["steps_1h"] = steps_new
    xx["steps_3h"] = now.steps_3h + (steps_new - now.steps_1h)
    xx["sleep_prev_night"] = sleep_new
    new = float(bt["model"].predict_proba(pd.DataFrame([xx]))[0, 1])
    m[2 * i].metric(f"{t.capitalize()} risk now", f"{base:.0%}")
    m[2 * i + 1].metric(f"{t.capitalize()} risk what-if", f"{new:.0%}", f"{(new - base) * 100:+.1f} pts",
                        delta_color="inverse")

with st.expander("Model performance (held-out virtual patients)"):
    if os.path.exists("models/metrics.json"):
        st.json(json.load(open("models/metrics.json")))
    for img in ("docs/results.png", "docs/shap_global.png"):
        if os.path.exists(img):
            st.image(img)
