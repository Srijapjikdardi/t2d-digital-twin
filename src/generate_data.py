"""Synthetic data generator for the T2D Digital Twin PoC.

Produces (all fully synthetic, no real patient data):
  data/ehr.csv       static EHR: demographics, labs, genetic marker, meds
  data/wearable.csv  15-min CGM glucose, heart rate, step count
  data/sleep.csv     nightly sleep hours

Physiology is a simple mechanistic simulator: meal-driven glucose excursions
(Indian-diet carb loads), scaled by a per-patient insulin-resistance factor
derived from the EHR (BMI, HbA1c, TCF7L2 risk alleles, metformin), reduced by
post-meal walking, and raised by poor sleep. The twin model must therefore
fuse EHR + wearable streams to predict excursions.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # always run from project root
import argparse
import os

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96  # 15-minute resolution


def make_ehr(n, rng):
    age = rng.integers(30, 76, n)
    sex = rng.integers(0, 2, n)  # 1 = male
    bmi = np.clip(rng.normal(27.5, 4.5, n), 18, 45)
    duration = rng.integers(0, 20, n)
    hba1c = np.clip(rng.normal(7.4, 1.0, n) + 0.04 * (bmi - 27), 5.8, 11.5)
    family_history = rng.binomial(1, 0.45, n)
    tcf7l2 = rng.binomial(2, 0.3, n)  # risk alleles (0-2), TCF7L2 rs7903146
    metformin = rng.binomial(1, 0.7, n)
    secretagogue = rng.binomial(1, 0.3, n)  # sulfonylurea / insulin: hypoglycaemia risk
    drug_strength = rng.uniform(0.5, 1.3, n)  # latent, NOT in EHR
    egfr = np.clip(rng.normal(95 - 0.5 * (age - 30), 12), 30, 130)
    sbp = np.clip(rng.normal(128 + 0.2 * (age - 50), 12), 95, 190)
    fasting = np.clip(0.7 * (28.7 * hba1c - 46.7) + rng.normal(0, 6, n), 80, 250)
    ehr = pd.DataFrame(
        dict(
            patient_id=np.arange(n),
            age=age,
            sex_male=sex,
            bmi=bmi.round(1),
            diabetes_duration_yrs=duration,
            hba1c=hba1c.round(1),
            fasting_glucose=fasting.round(0),
            family_history=family_history,
            tcf7l2_risk_alleles=tcf7l2,
            on_metformin=metformin,
            on_secretagogue=secretagogue,
            egfr=egfr.round(0),
            sbp=sbp.round(0),
        )
    )
    ehr.attrs['drug_strength'] = drug_strength
    return ehr


def insulin_resistance_multiplier(row, rng):
    ir = (
        0.04 * (row.bmi - 25)
        + 0.12 * (row.hba1c - 7.0)
        + 0.10 * row.tcf7l2_risk_alleles
        - 0.15 * row.on_metformin
        + rng.normal(0, 0.05)
    )
    return float(np.exp(ir))


def activity_profile():
    """Relative likelihood of steps over the 96 slots of a day."""
    hrs = np.arange(STEPS_PER_DAY) / 4.0
    w = np.full(STEPS_PER_DAY, 0.05)
    w[(hrs >= 6) & (hrs < 8)] = 1.5
    w[(hrs >= 9) & (hrs < 18)] = 0.7
    w[(hrs >= 18) & (hrs < 21)] = 1.2
    w[(hrs >= 21) | (hrs < 6)] = 0.03
    return w / w.sum()


def simulate_patient_full(row, days, rng, drug_strength=0.0):
    """Single-pass simulator that keeps meal timing for the excursion model."""
    mult = insulin_resistance_multiplier(row, rng)
    n = days * STEPS_PER_DAY
    sleep_hours = np.clip(rng.normal(6.5, 1.1, days), 4, 9)
    prof = activity_profile()
    resting_hr = 66 + 0.3 * (row.bmi - 27) + rng.normal(0, 3)
    base = row.fasting_glucose

    steps = np.zeros(n)
    meal_events = []  # (slot, carbs, walked)
    for d in range(days):
        daily = np.clip(rng.normal(5500, 2500), 500, 15000)
        s = rng.multinomial(int(daily), prof).astype(float)
        meals = [
            (8.0, 0.5, rng.normal(60, 15)),
            (13.0, 0.5, rng.normal(90, 20)),
            (20.5, 0.7, rng.normal(80, 20)),
        ]
        if rng.random() < 0.5:
            meals.append((17.0, 0.4, rng.normal(30, 10)))
        for mean_h, sd_h, carbs in meals:
            h = np.clip(rng.normal(mean_h, sd_h), 5, 22.5)
            slot = int(h * 4)
            carbs = max(carbs, 15)
            walked = rng.random() < 0.35
            skipped = rng.random() < 0.12  # meal skipped (drug dose still taken)
            if walked:
                for k in (slot + 1, slot + 2):
                    if k < STEPS_PER_DAY:
                        s[k] += 700
            meal_events.append((d * STEPS_PER_DAY + slot, carbs, walked, d, skipped))
        steps[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY] = s

    glucose = np.zeros(n)
    for d in range(days):
        day_base = base + 4.0 * max(0.0, 7.0 - sleep_hours[d])
        for i in range(STEPS_PER_DAY):
            hour = i / 4.0
            dawn = 10.0 * np.exp(-0.5 * ((hour - 6.0) / 1.0) ** 2)
            circ = 3.0 * np.sin(2 * np.pi * (hour - 9) / 24)
            glucose[d * STEPS_PER_DAY + i] = day_base + dawn + circ

    t = np.arange(0, 12) * 15.0  # 3h response window, minutes
    tau = 60.0
    curve = (t / tau) * np.exp(1 - t / tau)
    t2 = np.arange(0, 20) * 15.0  # 5 h window for delayed drug effect
    curve2 = (t2 / 120.0) * np.exp(1 - t2 / 120.0)
    for slot, carbs, walked, d, skipped in meal_events:
        sleep_pen = 1.0 + 0.10 * max(0.0, 6.5 - sleep_hours[d])
        eaten = 0.0 if skipped else carbs
        amp = 0.9 * eaten * mult * sleep_pen * (0.75 if walked else 1.0)
        for k, c in enumerate(curve):
            if slot + k < n:
                glucose[slot + k] += amp * c
        if drug_strength > 0:
            # sulfonylurea / insulin dose sized for the *planned* meal; acts later.
            # Walking after the dose adds extra glucose uptake (exercise).
            dip = 0.60 * carbs * drug_strength * (1.25 if walked else 1.0)
            for k, c in enumerate(curve2):
                if slot + k < n:
                    glucose[slot + k] -= dip * c

    # AR(1) sensor/physiology noise
    noise = np.zeros(n)
    e = rng.normal(0, 3.0, n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + e[i]
    glucose = np.clip(glucose + noise, 40, 400)  # CGM range ~40-400

    hr = resting_hr + 0.03 * steps + rng.normal(0, 3, n)
    hr = np.clip(hr, 45, 170)
    return glucose, steps, hr, sleep_hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    ehr = make_ehr(a.patients, rng)
    start = pd.Timestamp("2026-01-01 00:00")
    ts = start + pd.to_timedelta(np.arange(a.days * STEPS_PER_DAY) * 15, unit="m")

    wear, sleep = [], []
    for row in ehr.itertuples():
        ds = float(ehr.attrs['drug_strength'][row.patient_id]) if row.on_secretagogue else 0.0
        g, s, h, sl = simulate_patient_full(row, a.days, rng, ds)
        wear.append(
            pd.DataFrame(
                dict(
                    patient_id=row.patient_id,
                    timestamp=ts,
                    glucose_mgdl=g.round(1),
                    hr_bpm=h.round(0),
                    steps_15min=s.astype(int),
                )
            )
        )
        sleep.append(
            pd.DataFrame(
                dict(
                    patient_id=row.patient_id,
                    date=ts[::STEPS_PER_DAY].date,
                    sleep_hours=sl.round(2),
                )
            )
        )
    ehr.to_csv(f"{a.out}/ehr.csv", index=False)
    pd.concat(wear).to_csv(f"{a.out}/wearable.csv", index=False)
    pd.concat(sleep).to_csv(f"{a.out}/sleep.csv", index=False)
    print(f"wrote {a.patients} patients x {a.days} days to {a.out}/")


if __name__ == "__main__":
    main()
