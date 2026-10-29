# Clinical Safety and Regulatory Note: T2D Digital Twin (research prototype)

> **Status: research / education prototype. Not a medical device. Not clinically validated. Not for patient care.**
> Trained on synthetic data. All performance numbers below come from simulated patients.

## 1. Intended use (hypothetical, if this were ever developed further)
**Decision support for clinicians** managing adults with Type 2 Diabetes who wear a continuous glucose monitor (CGM). The software would **estimate the probability** that glucose will go above 180 mg/dL or below 70 mg/dL in the next 2 hours, show **why** (SHAP explanation), and let the clinician explore **lifestyle what-ifs** (steps, sleep).

**Not intended for:** autonomous treatment, insulin dosing, replacing CGM device alarms, diagnosis, Type 1 diabetes, children, pregnancy, hospital in-patients, or use by patients without clinical oversight.

## 2. Where it would sit as Software as a Medical Device (SaMD)
Software that is intended to monitor or inform clinical management of a disease on its own, without being part of a hardware device, is generally treated as **Software as a Medical Device**. The IMDRF SaMD framework sorts such software by (a) how significant the information is (treat/diagnose, drive, or *inform* clinical management) and (b) how serious the health situation is. For a tool that *informs or drives* management of diabetes (a serious condition), this would plausibly land in a **low-to-middle risk category (roughly IMDRF I to II)**, but the exact class depends on the final intended-use claims and must be decided by a regulatory assessment. In India the regulator is **CDSCO** (Medical Devices Rules, 2017). Other markets (EU MDR, US FDA) have their own rules.
*I am a student, not a regulatory professional; treat this section as orientation, not advice.*

## 3. Why this project does NOT claim compliance
"Compliant" is a formal, evidence-based status, not a description of good intentions. It would require, at minimum:
- a quality management system (**ISO 13485**) with documented design controls;
- a documented risk management file (**ISO 14971**);
- a software lifecycle process with verification and traceability (**IEC 62304**, **IEC 82304-1**);
- usability engineering with real users (**IEC 62366-1**);
- clinical evaluation on real patient data and, in India, regulatory approval/licensing (CDSCO);
- personal-data protections for real patients under the **DPDP Act, 2023** (and HIPAA/GDPR where applicable).

This repository has none of that evidence. Claiming compliance would be false and, in a real company, a serious red flag. What it *does* do: use **no real personal data** (so no personal-data exposure), follow a **risk-first mindset** (below), and document limits openly. *(Please verify standard names and editions against current official texts before citing them elsewhere.)*

## 4. Risk table (informal, ISO 14971-style thinking, not a formal risk file)
Severity: how bad if it happens. Likelihood: qualitative, for this prototype if misused clinically.

| # | Hazard | Cause | Possible harm | Severity | Likelihood | Mitigation in this project / needed before real use |
|---|---|---|---|---|---|---|
| 1 | **Missed LOW-glucose warning** (false negative) | Model recall is limited (0.63 at threshold 0.3 on simulated data); skipped meals and drug effects are hard to predict | Unrecognised hypoglycaemia: confusion, seizure, loss of consciousness | **High** | Medium | Lower default alert threshold for lows; never replace CGM device alarms; show current glucose and trend; real-data validation; clinician training |
| 2 | **Missed HIGH-glucose warning** | Recall 0.56 at threshold 0.5 on simulated data | Prolonged hyperglycaemia, delayed intervention | Medium | Medium | Adjustable threshold; displayed as one input only; real-data validation |
| 3 | **False alarms** → alert fatigue | Precision 0.73 (high) and 0.54 (low) on simulated data | Alerts ignored, including real ones | Medium | Medium | Threshold sliders; SHAP reasons so alerts can be judged; pilot study to measure alert burden |
| 4 | **Model trained on synthetic data** does not generalise | Simulator encodes assumed physiology (e.g. the drug-linked hypo effect is built in) | Wrong risk estimates in real patients | **High** | **High** | Prominent "synthetic, not validated" banner; real-data check (`src/eval_real.py`); no clinical use until prospective validation |
| 5 | **Population / device shift** | Real data differs (diet, BMI, ethnicity, CGM brand, age) | Systematically poor performance in some groups | High | Medium | Subgroup performance analysis; local Indian cohort validation; monitoring after deployment |
| 6 | **Wrong or missing input data** | Mistyped EHR values (e.g. medication flag), CGM gaps, compression lows, sensor lag | Misleading risk | Medium | Medium | Input range checks; require a complete 2 h CGM window; show data quality status |
| 7 | **What-if misread as medical advice** | Counterfactuals show model associations, not proven causal effects | Patient changes behaviour or medicines unsafely | Medium | Medium | What-if limited to steps and sleep; no medication what-ifs; disclaimer in UI |
| 8 | **Explanation misread** | SHAP shows what the model used, not biological cause | False confidence in the reason | Low-Medium | Medium | Wording in UI: "explains the model, not biology" |
| 9 | **Automation bias** | Clinician over-trusts a numeric risk | Clinical judgement displaced | Medium | Medium | Show uncertainty and raw glucose; decision stays with clinician; training |
| 10 | **Privacy breach** (future, with real data) | Real patient records stored or shared insecurely | Loss of confidentiality | High | Low-Medium | Use only de-identified data; consent; access control and encryption; DPDP Act 2023 review |
| 11 | **Software defects / stale model** | Bugs, silent model drift, version mix-ups | Wrong outputs unnoticed | Medium | Low-Medium | Automated tests, versioned models, drift monitoring, change control |

## 5. Known limitations (read before quoting any number)
- All headline metrics are on **simulated** patients from my own simulator. They show the pipeline works; they are **not** evidence of clinical accuracy.
- Hypoglycaemia in the simulator is driven by a sulfonylurea/insulin flag, skipped meals and post-meal walking. The model therefore finds that flag very informative (see SHAP plot); in real life the relationship is messier.
- The real-data script (`src/eval_real.py`) evaluates on ShanghaiT2DM using **CGM + available static fields only**; that dataset has no steps, heart rate or sleep, so the wearable benefit cannot be tested there.
- Alert thresholds (0.5 for high, 0.3 for low) are illustrative, not clinically derived.
- Only adults with T2D are represented; Type 1 diabetes, children and pregnancy are out of scope.

## 6. What would be needed next
1. Validate on real, de-identified CGM + EHR cohorts, including an Indian cohort; report calibration and subgroup results.
2. Prospective silent-mode study (alerts logged, not shown) before any clinical display.
3. Human-factors testing with clinicians; measure alert fatigue.
4. A proper risk-management file and quality system with a regulatory consultant.
