"""Feature fusion: static EHR + dynamic wearable streams -> model input.

Prediction task: at time t (15-min grid), given glucose currently <= 180 mg/dL,
will glucose exceed 180 mg/dL at any point in the next 2 hours?
"""
import numpy as np
import pandas as pd

HORIZON = 8          # 8 x 15 min = 2 hours
THRESH = 180.0       # mg/dL hyperglycaemia threshold

EHR_COLS = [
    "age", "sex_male", "bmi", "diabetes_duration_yrs", "hba1c",
    "fasting_glucose", "family_history", "tcf7l2_risk_alleles",
    "on_metformin", "egfr", "sbp",
]
CGM_COLS = [
    "g_now", "g_mean_2h", "g_std_2h", "g_min_2h", "g_max_2h",
    "g_delta_15m", "g_delta_30m", "g_delta_2h",
]
WEAR_COLS = [
    "steps_1h", "steps_3h", "hr_now", "hr_mean_1h", "sleep_prev_night",
    "hour_sin", "hour_cos",
]
FEATURE_SETS = {
    "CGM only": CGM_COLS,
    "CGM + wearable": CGM_COLS + WEAR_COLS,
    "CGM + wearable + EHR (twin)": CGM_COLS + WEAR_COLS + EHR_COLS,
}


def patient_features(w: pd.DataFrame, sleep: pd.DataFrame, ehr_row: pd.Series) -> pd.DataFrame:
    """Vectorised features for one patient's full time series."""
    w = w.sort_values("timestamp").reset_index(drop=True).copy()
    g = w["glucose_mgdl"]
    f = pd.DataFrame({"timestamp": w["timestamp"], "patient_id": w["patient_id"]})
    f["g_now"] = g
    r = g.rolling(8, min_periods=8)
    f["g_mean_2h"] = r.mean()
    f["g_std_2h"] = r.std()
    f["g_min_2h"] = r.min()
    f["g_max_2h"] = r.max()
    f["g_delta_15m"] = g - g.shift(1)
    f["g_delta_30m"] = g - g.shift(2)
    f["g_delta_2h"] = g - g.shift(7)
    f["steps_1h"] = w["steps_15min"].rolling(4, min_periods=4).sum()
    f["steps_3h"] = w["steps_15min"].rolling(12, min_periods=12).sum()
    f["hr_now"] = w["hr_bpm"]
    f["hr_mean_1h"] = w["hr_bpm"].rolling(4, min_periods=4).mean()

    date = w["timestamp"].dt.date
    sl = sleep.set_index("date")["sleep_hours"]
    f["sleep_prev_night"] = date.map(sl).astype(float).values

    hour = w["timestamp"].dt.hour + w["timestamp"].dt.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    for c in EHR_COLS:
        f[c] = float(ehr_row[c])

    fut = np.stack([g.shift(-k).values for k in range(1, HORIZON + 1)], axis=1)
    f["future_max"] = np.nanmax(np.where(np.isnan(fut), -np.inf, fut), axis=1)
    f.loc[g.shift(-HORIZON).isna(), "future_max"] = np.nan
    f["label"] = (f["future_max"] > THRESH).astype(float)
    f.loc[f["future_max"].isna(), "label"] = np.nan
    return f


def build_dataset(ehr: pd.DataFrame, wear: pd.DataFrame, sleep: pd.DataFrame) -> pd.DataFrame:
    parts = []
    wear = wear.copy()
    wear["timestamp"] = pd.to_datetime(wear["timestamp"])
    sleep = sleep.copy()
    sleep["date"] = pd.to_datetime(sleep["date"]).dt.date
    for pid, w in wear.groupby("patient_id"):
        parts.append(
            patient_features(
                w, sleep[sleep.patient_id == pid], ehr.set_index("patient_id").loc[pid]
            )
        )
    return pd.concat(parts, ignore_index=True)
