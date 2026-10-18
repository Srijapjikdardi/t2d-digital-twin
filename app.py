"""Clinician dashboard for the T2D Digital Twin PoC.   Run:  streamlit run app.py"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import EHR_COLS, THRESH, patient_features  # noqa: E402

st.set_page_config(page_title="T2D Digital Twin", layout="wide")


@st.cache_data
def load():
    ehr = pd.read_csv("data/ehr.csv")
    wear = pd.read_csv("data/wearable.csv", parse_dates=["timestamp"])
    sleep = pd.read_csv("data/sleep.csv")
    sleep["date"] = pd.to_datetime(sleep["date"]).dt.date
    return ehr, wear, sleep


@st.cache_resource
def load_model():
    return joblib.load("models/twin_model.joblib")


ehr, wear, sleep = load()
bundle = load_model()
model, cols = bundle["model"], bundle["features"]

st.title("Digital Twin: Type 2 Diabetes glycaemic-event forecaster")
st.caption(
    "Proof of concept. 100% synthetic data. Not a medical device; not clinically validated."
)

pid = st.sidebar.selectbox("Virtual patient", ehr.patient_id.tolist(), index=3)
row = ehr[ehr.patient_id == pid].iloc[0]
w = wear[wear.patient_id == pid]
feats = patient_features(w, sleep[sleep.patient_id == pid], row)
valid = feats.dropna(subset=["g_delta_2h", "steps_3h"]).copy()
valid["risk"] = model.predict_proba(valid[cols])[:, 1]

alert_thr = st.sidebar.slider("Alert threshold (risk of >180 mg/dL in 2h)", 0.2, 0.9, 0.5, 0.05)

days = sorted(valid.timestamp.dt.date.unique())
day = st.sidebar.selectbox("Day", days, index=3)
d = valid[valid.timestamp.dt.date == day].reset_index(drop=True)
slot = st.sidebar.slider("Time of day (now)", 0, len(d) - 1, min(36, len(d) - 1),
                         format="slot %d (15-min)")
now = d.iloc[slot]

c1, c2 = st.columns([1, 3])
with c1:
    st.subheader(f"Patient #{pid}")
    st.markdown(
        f"**Age** {int(row.age)} · **{'M' if row.sex_male else 'F'}** · **BMI** {row.bmi}\n\n"
        f"**HbA1c** {row.hba1c}% · **Fasting glucose** {int(row.fasting_glucose)} mg/dL\n\n"
        f"**T2D duration** {int(row.diabetes_duration_yrs)} y · **eGFR** {int(row.egfr)}\n\n"
        f"**SBP** {int(row.sbp)} · **Metformin** {'yes' if row.on_metformin else 'no'}\n\n"
        f"**Family history** {'yes' if row.family_history else 'no'} · "
        f"**TCF7L2 risk alleles** {int(row.tcf7l2_risk_alleles)}"
    )
    st.metric("Glucose now", f"{now.g_now:.0f} mg/dL")
    st.metric("Sleep last night", f"{now.sleep_prev_night:.1f} h")
    st.metric("Steps last hour", f"{int(now.steps_1h)}")

with c2:
    risk = float(now.risk)
    if now.g_now > THRESH:
        st.error(f"Already hyperglycaemic ({now.g_now:.0f} mg/dL). Model forecasts onset only.")
    elif risk >= alert_thr:
        st.warning(f"ALERT: {risk:.0%} probability glucose exceeds {THRESH:.0f} mg/dL within 2 h.")
    else:
        st.success(f"Low near-term risk: {risk:.0%}.")

    chart = d.set_index("timestamp")[["g_now"]].rename(columns={"g_now": "Glucose (mg/dL)"})
    chart["Hyperglycaemia threshold"] = THRESH
    st.line_chart(chart, height=260, color=["#3FA796", "#F2A488"])
    rk = d.set_index("timestamp")[["risk"]].rename(columns={"risk": "Predicted 2h risk"})
    st.line_chart(rk, height=160, color=["#6FB3E0"])

st.subheader("What-if simulation (counterfactual on the twin)")
st.caption("Change behaviour at this moment; the twin re-scores the 2-hour risk.")
a, b = st.columns(2)
steps_new = a.slider("Steps in the last hour", 0, 3000, int(now.steps_1h), 100)
sleep_new = b.slider("Sleep last night (h)", 3.0, 10.0, float(now.sleep_prev_night), 0.5)
x = now[cols].copy().astype(float)
x["steps_1h"] = steps_new
x["steps_3h"] = now.steps_3h + (steps_new - now.steps_1h)
x["sleep_prev_night"] = sleep_new
new_risk = float(model.predict_proba(pd.DataFrame([x]))[0, 1])
m1, m2, m3 = st.columns(3)
m1.metric("Baseline risk", f"{risk:.0%}")
m2.metric("What-if risk", f"{new_risk:.0%}", f"{(new_risk - risk) * 100:+.1f} pts", delta_color="inverse")
m3.metric("Model", "HistGradientBoosting")

with st.expander("Model performance (held-out virtual patients)"):
    if os.path.exists("models/metrics.json"):
        m = json.load(open("models/metrics.json"))
        st.json(m)
    if os.path.exists("docs/results.png"):
        st.image("docs/results.png")
