# T2D Digital Twin: 2-Hour Glycaemic Event Forecaster

Proof-of-concept Digital Twin for Type 2 Diabetes. Submission for the Happiest Health *Reimagining and Reforming Healthcare in India Summit 2026* (Bengaluru) student challenge.

## 1. Team details and college / incubator information
| Name | Role | College / Department | Email |
|---|---|---|---|
| Sri Japji Kaur Dardi (Team Leader) | Biomedical domain, data, docs | Thapar Institute of Engineering and Technology, Patiala (B.Tech, Biomedical) | srijapjikdardi@gmail.com / sdardi_btech23@thapar.edu |

Solo team. Incubator: N/A. Full-time student.

## 2. Project title
**T2D Digital Twin: fusing EHR and wearable streams to predict hyperglycaemia 2 hours ahead**

## 3. Problem statement and healthcare use case
India has one of the world's largest Type 2 Diabetes burdens, yet care is mostly reactive: glucose excursions between clinic visits go unnoticed. A Digital Twin (a continuously updated virtual replica of the patient) can forecast adverse events and let clinicians test interventions virtually.

**Use case:** For a T2D patient, predict whether glucose will exceed **180 mg/dL within the next 2 hours** (given it is currently at or below 180), alert the clinician, and simulate how behaviour changes (more post-meal steps, better sleep) alter that risk.

**Data-stream fusion (challenge requirement):**
- *Static / historical:* synthetic EHR: age, sex, BMI, diabetes duration, HbA1c, fasting glucose, family history, TCF7L2 risk alleles (genetic marker), metformin, eGFR, systolic BP.
- *Dynamic / real-time:* synthetic wearable / CGM time series at 15-min resolution: glucose, heart rate, steps, nightly sleep hours.

## 4. Technical stack, AI/ML model and framework
- Python 3, NumPy, pandas, scikit-learn, matplotlib, joblib, Streamlit
- **Model:** `HistGradientBoostingClassifier` on fused features
  - CGM window (last 2 h): current, mean, std, min, max, 15-min / 30-min / 2-h deltas
  - Wearable and time: steps 1 h / 3 h, heart rate, previous-night sleep, hour of day (sin/cos)
  - EHR: 11 static features, concatenated to every time step
- **Evaluation:** patient-level train/test split (75/25, no leakage); ablation CGM only, CGM + wearable, CGM + wearable + EHR; ROC-AUC, PR-AUC, permutation importance
- **Dashboard:** Streamlit clinician UI with EHR card, glucose trace, risk timeline, alert threshold, what-if sliders

### Data (sandbox rules)
Fully **synthetic**; no real patient data (DPDP Act / HIPAA safe). `src/generate_data.py` simulates 300 virtual patients x 14 days. Glucose = baseline from EHR + dawn/circadian effect + meal excursions (Indian-diet carbohydrate loads) scaled by an insulin-resistance factor (BMI, HbA1c, TCF7L2 alleles, metformin) - reduced by post-meal walking, raised by sleep debt - plus AR(1) noise. Synthea / MIMIC-IV were **not** used in this PoC; the pipeline is designed so those EHR sources and open CGM datasets can be mapped to the same column schema in future work (not yet tested).

### Results (held-out virtual patients; reproducible with seed 42)
| Feature set | ROC-AUC | PR-AUC |
|---|---|---|
| CGM only | 0.782 | 0.506 |
| CGM + wearable | 0.948 | 0.820 |
| **CGM + wearable + EHR (twin)** | **0.958** | **0.853** |

At threshold 0.5 the twin reaches precision 0.79 and recall 0.71 (positive rate 19.3%). Fusing both streams clearly beats CGM alone. See `docs/results.png`.

> **Limitations:** results are on synthetic data produced by an assumed physiological simulator, so they are an upper bound and **not clinical evidence**. Not a medical device. Next step is validation on real, de-identified open CGM datasets.

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py      # data/ehr.csv, wearable.csv, sleep.csv
python src/train.py              # models/twin_model.joblib, metrics.json, docs/results.png
streamlit run app.py             # clinician dashboard
python docs/make_docs.py         # regenerates architecture.pdf
```

## Repository layout
```
app.py                  Streamlit dashboard (doctor view + what-if)
src/generate_data.py    synthetic EHR + wearable simulator
src/features.py         stream fusion / feature engineering
src/train.py            training, ablation, metrics, plots
docs/architecture.pdf   architecture diagram
docs/presentation.pptx/.pdf project presentation (light theme)
docs/results.png        ablation ROC + feature importance
models/                 trained model + metrics.json
```

## 5. Demo video (minimum 20 minutes)
[PASTE UNLISTED YOUTUBE LINK or link to PowerPoint recording here]

## 6. Open-source license
MIT License. See [LICENSE](LICENSE).

## 7. Architecture diagram
[docs/architecture.pdf](docs/architecture.pdf)

## 8. Presentation
[docs/presentation.pptx](docs/presentation.pptx) and [PDF](docs/presentation.pdf)
