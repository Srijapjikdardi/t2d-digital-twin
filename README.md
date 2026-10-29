# T2D Digital Twin v2: forecasting HIGH and LOW glucose 2 hours ahead, with explanations

Proof-of-concept Digital Twin for Type 2 Diabetes. Submission for the Happiest Health *Reimagining and Reforming Healthcare in India Summit 2026* (Bengaluru) student challenge.

> **v2 adds:** hypoglycaemia (<70 mg/dL) forecasting, SHAP "why this alert" explanations, a clinical safety page ([SAFETY.md](SAFETY.md)), and a script to test on real open CGM data. The original submitted version is on branch `main` history.

## 1. Team details and college / incubator information
| Name | Role | College / Department | Email |
|---|---|---|---|
| Sri Japji Kaur Dardi (Team Leader) | Biomedical domain, data, ML, docs | Thapar Institute of Engineering and Technology, Patiala (B.Tech, Biomedical) | srijapjikdardi@gmail.com / sdardi_btech23@thapar.edu |

Solo team. Incubator: N/A. Full-time student.

## 2. Project title
**T2D Digital Twin: fusing EHR and wearable streams to forecast hyper- and hypoglycaemia 2 hours ahead**

## 3. Problem statement and healthcare use case
India has one of the world's largest Type 2 Diabetes burdens (about 101 million people, ICMR-INDIAB 2023), yet care is mostly reactive: glucose excursions between clinic visits go unnoticed. A Digital Twin (a continuously updated virtual replica of the patient) can forecast adverse events and let clinicians test interventions virtually.

**Use case:** For a T2D patient wearing a CGM, forecast within the next 2 hours:
- **HIGH:** glucose above **180 mg/dL** (given it is at or below 180 now);
- **LOW:** glucose below **70 mg/dL** (given it is at or above 70 now);

alert the clinician, **explain why** (SHAP), and simulate how behaviour (steps, sleep) changes the risk.

**Data-stream fusion (challenge requirement):**
- *Static / historical:* synthetic EHR: age, sex, BMI, diabetes duration, HbA1c, fasting glucose, family history, TCF7L2 risk alleles (genetic marker), metformin, sulfonylurea/insulin, eGFR, systolic BP.
- *Dynamic / real-time:* synthetic wearable / CGM time series at 15-min resolution: glucose, heart rate, steps, nightly sleep hours.

## 4. Technical stack, AI/ML model and framework
- Python 3, NumPy, pandas, scikit-learn, SHAP, matplotlib, joblib, Streamlit (openpyxl for the real-data script)
- **Models:** two `HistGradientBoostingClassifier`s (HIGH and LOW) on fused features
  - CGM window (last 2 h): current, mean, std, min, max, 15-min / 30-min / 2-h deltas
  - Wearable and time: steps 1 h / 3 h, heart rate, previous-night sleep, hour of day (sin/cos)
  - EHR: 12 static features concatenated to every time step
- **Explainability:** SHAP (model-agnostic permutation explainer): global plot `docs/shap_global.png` and a per-patient, per-moment "why" panel in the dashboard
- **Evaluation:** patient-level train/test split (75/25, no leakage); ablation CGM only, CGM + wearable, CGM + wearable + EHR; ROC-AUC, PR-AUC
- **Dashboard:** Streamlit clinician UI (EHR card, glucose trace, separate high/low alerts with thresholds, SHAP panel, what-if sliders)

### Data (sandbox rules)
Fully **synthetic**; no real patient data. `src/generate_data.py` simulates 300 virtual patients x 14 days. Glucose = baseline from EHR + dawn/circadian effect + meal excursions (Indian-diet carbohydrate loads) scaled by an insulin-resistance factor (BMI, HbA1c, TCF7L2 alleles, metformin), reduced by post-meal walking, raised by sleep debt, plus AR(1) noise. **Lows (v2):** patients on a sulfonylurea/insulin get a delayed glucose-lowering effect sized for the planned meal; skipped meals (12%) and post-meal walking then cause hypoglycaemia (about 2.3% of readings below 70 mg/dL, 29% of patients have at least one low).

### Results (held-out virtual patients; seed 42)
| Feature set | HIGH ROC-AUC | HIGH PR-AUC | LOW ROC-AUC | LOW PR-AUC |
|---|---|---|---|---|
| CGM only | 0.788 | 0.436 | 0.942 | 0.437 |
| CGM + wearable | 0.936 | 0.709 | 0.962 | 0.527 |
| **CGM + wearable + EHR (twin)** | **0.947** | **0.734** | **0.981** | **0.614** |

HIGH positive rate 13.5% (chance PR-AUC 0.14). LOW positive rate 2.9% (chance PR-AUC 0.02). At the illustrative alert thresholds (0.5 high, 0.3 low): HIGH precision 0.73 / recall 0.56; LOW precision 0.54 / recall 0.63. Fusing both streams beats CGM alone. Plots: `docs/results.png`, `docs/shap_global.png`.

> **Limitations:** results are on synthetic data from an assumed physiological simulator, so they are an upper bound and **not clinical evidence**. The low-glucose model leans heavily on the simulated drug flag (see SHAP plot). Not a medical device. See [SAFETY.md](SAFETY.md).

### Real-data check (optional, ShanghaiT2DM)
`src/eval_real.py` tests the pipeline on the public ShanghaiT2DM CGM dataset (Zhao et al., *Scientific Data*, 2023; Figshare collection https://doi.org/10.6084/m9.figshare.c.6310860). It has CGM and clinical summary data but **no steps, heart rate or sleep**, so only CGM + static fields are used. It reports (A) synthetic-trained model tested on real patients and (B) 5-fold patient-level CV on real data. Dataset license is not stated by the authors; check terms before redistributing (this repo does not include the data).
```bash
python src/eval_real.py "path/to/Shanghai_T2DM"     # writes results/real_data_results.json
```
Status: the script is tested for plumbing on a mock folder only; results on the real files will be added here after running it.

## Run it (Windows / macOS / Linux, VS Code or any terminal)
Python 3.10 to 3.12 recommended (3.13 also worked in testing). Run commands from the folder that contains `app.py`.
```bash
python -m venv .venv                      # private package folder
.venv\Scripts\Activate.ps1               # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt           # install packages (a few minutes)
python run_all.py                         # generate data + train models + plots (about 3-6 min)
python -m streamlit run app.py            # open the dashboard at http://localhost:8501
```
Optional: `python src/eval_real.py "path/to/Shanghai_T2DM"` (real-data check), `python docs/make_docs.py` (architecture PDF). The step-by-step explanation is in `PROJECT_GUIDE.md`, section 13.

## Repository layout
```
app.py                  Streamlit dashboard (alerts, SHAP, what-if)
run_all.py              one command: generate data + train models
SAFETY.md               intended use, SaMD note, risk table, limits
src/generate_data.py    synthetic EHR + wearable simulator (incl. hypoglycaemia)
src/features.py         stream fusion / feature engineering / labels
src/train.py            training, ablation, SHAP, metrics, plots
src/eval_real.py        test on real ShanghaiT2DM CGM data
docs/architecture.pdf   architecture diagram
docs/presentation.pptx/.pdf  project presentation
models/                 trained models + metrics.json
```

## 5. Demo video (minimum 20 minutes)
[PASTE UNLISTED YOUTUBE LINK HERE]

## 6. Open-source license
MIT License. See [LICENSE](LICENSE).

## 7. Architecture diagram
[docs/architecture.pdf](docs/architecture.pdf)

## 8. Presentation
[docs/presentation.pptx](docs/presentation.pptx) and [PDF](docs/presentation.pdf)
