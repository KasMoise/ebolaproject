# Ebola Bundibugyo 2026 (DRC) — Forecasting with Real Transfer Learning, a Spatio-Temporal GNN and Explainable AI

A single Jupyter notebook, `Ebola_BDBV2026.ipynb`, that downloads the official surveillance data of the 2026 *Bundibugyo ebolavirus* (BDBV) outbreak in the Democratic Republic of the Congo, and benchmarks **22 forecasting models** of weekly new cases per health zone: naive references, classical machine learning, deep sequence models, **transfer learning pre-trained on real past outbreaks**, and a **spatio-temporal GraphSAGE** network. Results are explained with SHAP and LIME, evaluated against persistence with bootstrap intervals, and exported as figures, a LaTeX table and a JSON report.

> **Status:** research code. The numbers below come from one run of the notebook on a short chronological test window (7 weeks). Read them together with the [limitations](#limitations).

---

## Contents

- [Key findings](#key-findings)
- [Data](#data)
- [Notebook structure](#notebook-structure)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Reproducibility, citation and license](#reproducibility-citation-and-license)

---

## Key findings

1. **No model is significantly better than persistence** ("next week = last week") on this test window. The best learned model, the transfer-learning LSTM (`TL-LSTM`), improves the RMSE by 5.5 % (95 % bootstrap interval of the skill: −2.2 % to +11.8 %; P(skill > 0) = 0.92).
2. **Pre-training on real outbreaks helps compared with training from scratch** with the same architecture and input: RMSE 5.79 ± 0.04 versus 6.70 ± 1.15 over 3 seeds, i.e. lower error and much higher stability. In the few-shot ablation the pre-trained model is at least as good as the scratch model for 5 of 6 data sizes.
3. **The graph helps:** the spatio-temporal GNN reaches an RMSE of 6.03 versus 6.45 for the same network without neighbour aggregation.
4. **The previous version of this work was inflated by target leakage** (features that contained the value to predict). This version builds every feature from information up to *t − 1* and verifies it automatically (see [Methods](#methods)).

---

## Data

### 2026 target — INRB-UMIE

Source: **[INRB-UMIE/BDBV2026-Data](https://github.com/INRB-UMIE/BDBV2026-Data)** (INSP situation reports compiled by INRB-UMIE). The notebook queries the GitHub Releases API, downloads the latest release whose tag contains `build`, extracts it to `donnees_extraites/` and loads eight long-format files (`zone | date | value`): national and per-zone cumulative confirmed cases, deaths, recoveries and suspected cases, plus new confirmed cases. If GitHub is unreachable, a previously extracted copy is used. The release tag is stored in the report.

Cleaning rules: null values are dropped (not replaced by 0), duplicated `(zone, date)` rows are merged, invalid zone names (`'nan'`, empty) are removed, and downward corrections of cumulative counts are reported. The sum of the zone-level counts is compared with the national series (a small difference exists in the source data and is printed).

An optional `drc_health_zones.geojson` (in the same release) gives health-zone polygons for the maps and for the geographic edges of the GNN.

### Historical outbreaks for transfer learning

Put the files in `historical_data/` (sub-folders allowed). All three sources were **real** in the reference run:

| Outbreak | File used | Content used | Series for pre-training |
|---|---|---|---|
| West Africa 2014-16 (Zaire) | HDX dataset `ebola-cases-2014` (downloaded automatically if no local file) | indicator *Cumulative number of confirmed Ebola cases*, per country; 15,251 cases after dropping a 2-case tail separated by a long silence | 3 countries + total |
| North Kivu / Ituri 2018-2020 (Zaire) | `dataonly_ebolaDRC.csv` (epi-week table) | **confirmed** cases only; 29 health zones, 78 weeks, 3,301 cases, peak 123 on 2019-04-28 | 14 zones + total |
| Uganda 2022 (Sudan virus) | `Surveillance_data_Ebola_outbreak.csv` | confirmed cases, 2022 segment only: 142 cases, peak 39 on 2022-10-30 (the file also contains the 2025 outbreak, 9 cases, which is detected as a separate outbreak and dropped) | 1 national series |

The loader recognises several layouts without configuration: aggregated counts (cumulative or incident, automatically detected), long *indicator | country | date | value* files, wide tables (one column per zone), **period-wide tables** (one column per reporting period such as `5 Aug 2018 - 12 Aug 2018`: periods of irregular length are spread over days, negative reclassifications are absorbed in the cumulative series), and individual line lists. If a file cannot be read, the notebook prints a diagnostic of its columns; if no file is found, it falls back to a **clearly flagged SIR simulation** (the model is then named `TL-LSTM*`).

---

## Notebook structure

| Module | Purpose |
|---|---|
| **0 · Configuration & environment** | imports with optional dependencies, `Config` dataclass, seeds, plotting style, PyTorch training utilities |
| **1 · Data acquisition & validation** | GitHub release download, robust parsing, data-quality report, zone-vs-national consistency check |
| **2 · Exploratory analysis, tests and maps** | national curves, top zones, ADF / Ljung-Box / Shapiro-Wilk, ACF/PACF, point map, static choropleth and interactive map (real coordinates only), zone statistics |
| **3 · Transfer learning from real outbreaks** | historical-file loaders, flagged SIR fallback, pre-training corpus, `EpidemicEncoder`, **leave-one-outbreak-out** transferability check |
| **4 · Feature engineering** | incidence table, 13 leakage-free features, automatic leakage test |
| **5 · Chronological split** | one train/test split, preprocessing fitted on the training period only |
| **6 · Unified metrics** | RMSE, MAE, MAPE, SMAPE, RMSLE, R², Theil U2, **skill versus persistence**, classification metrics for high-risk zone-weeks |
| **7 · Baselines** | naive references; Linear, Ridge, Random Forest, XGBoost, LightGBM — each in a raw and a persistence-anchored (`_logdelta`) version |
| **8 · SVR** | RBF and linear kernels (raw and `_logdelta`) |
| **9 · Deep learning** | per-zone sequences; LSTM and Transformer; **TL-LSTM** versus a from-scratch control (3 seeds); **few-shot ablation** |
| **10 · Spatio-temporal GNN** | graph construction, GraphSAGE network in plain PyTorch, "no-graph" control, next-period forecast per zone |
| **11 · Explainability** | exact TreeSHAP for the tree models, LIME on the best classical model (in raw feature units) |
| **12 · Final evaluation and export** | leaderboard with bootstrap skill intervals, dashboard, predicted vs actual, chronological cross-validation, residual analysis, probabilistic national forecast, LaTeX table, JSON report |

---

## Methods

**Target and resolution.** New confirmed cases per health zone and per week (weeks end on Sunday; the last incomplete week is dropped). `CFG.FREQ = 'D'` switches to daily data (transfer learning then requires weekly resolution and is skipped). The first week of a zone (its emergence) is not forecast.

**Leakage-free features (13).** `lag_1…4`, rolling mean and standard deviation over 2 and 4 weeks, `trend` (relative change between the last two weeks), `cum_prev` (cumulative cases up to the previous week), `age` (weeks since the zone's first case), `month`, ISO `week`. The same function builds the training table and the features of future weeks used in the forecast. `check_no_leakage` modifies the observation to predict and verifies that no feature changes; the notebook stops if it fails.

**Chronological split.** Train: weeks up to 2026-09-30. Test: weeks from 2026-10-01 to the last complete week (7 weeks, 2026-10-04 → 2026-11-15 in the reference run). A high-risk label is defined as more than the 75th percentile of positive training targets (18 cases/week in the reference run). Winsorisation (1st–99th percentile) and robust scaling are fitted on the training period only.

**Persistence-anchored models.** Every classical model is trained on the raw count and on the log growth `log1p(y) − log1p(lag_1)` (`_logdelta`, prediction = `expm1(log1p(lag_1) + growth)`). The LSTM, Transformer and GNN receive `log1p(lag_1)` as an anchor channel and predict a correction. Such models can match persistence by construction, so any gain is a genuine improvement. Predictions are protected against extrapolation: the log growth is clipped to the range seen in training (± 0.5) and counts are capped at 3 × the training maximum.

**Transfer learning.** A univariate LSTM (`EpidemicEncoder`, 2 layers, 32 units, output = last value + learned correction) is pre-trained on the historical series with a *local* normalisation: `log1p` of the window divided by the window maximum, so that the input space is identical in pre-training and fine-tuning and no future peak is used. Fine-tuning on 2026 has two stages (LSTM frozen, then all weights with a low learning rate). Controls: an identical network trained from scratch, a leave-one-outbreak-out zero-shot test, and a few-shot ablation (train on the first *K* weeks of 2026, evaluate on the next 4).

**Spatio-temporal GNN.** A GRU encodes the 4-week feature window of every zone; two dense GraphSAGE layers (mean aggregation, residual connections, layer norm) exchange information between neighbouring zones. Edges: geographic (contact between GeoJSON polygons, 74 nodes and 135 edges in the reference run; otherwise k = 3 nearest known centroids) plus similarity edges (correlated log-incidence curves) computed **on the training period only**.

**Evaluation.** All models are compared on the same test rows. Skill = 1 − RMSE / RMSE(persistence), with a 95 % bootstrap interval obtained by resampling whole test weeks (all zones of a week stay together). Chronological cross-validation uses an expanding window by date with fresh estimators and a fresh preprocessor in each fold.

**Probabilistic forecast.** Recursive multi-week forecast of the best classical model. Noise: out-of-fold Pearson residuals of the most recent cross-validation folds, drawn **by whole week** (to keep the correlation between zones) and applied as `y = max(ŷ + sqrt(ŷ + 1)·r, 0)`; 500 simulated paths; the national value is the sum over zones. No new zone can appear inside the horizon.

---

## Results

Reference run — weekly resolution, test window 2026-10-04 → 2026-11-15, persistence RMSE = 6.0852. Sorted by RMSE (top of the leaderboard; the full table of 22 models is in `outputs/reports/model_comparison.csv`).

| Model | Family | RMSE | R² | Skill vs persistence | 95 % interval | F1 (high-risk) |
|---|---|---:|---:|---:|---|---:|
| **TL-LSTM** | transfer | **5.750** | 0.857 | +0.055 | [−0.022, +0.118] | 0.621 |
| LightGBM_logdelta | classical | 5.894 | 0.850 | +0.031 | [−0.031, +0.106] | 0.651 |
| ST-GNN-GraphSAGE | graph | 6.030 | 0.843 | +0.009 | [−0.072, +0.084] | 0.667 |
| *Naive_lag1 (persistence)* | naive | *6.085* | *0.840* | *0* | — | *0.644* |
| *Naive_roll_mean_4* | naive | *6.091* | *0.840* | *−0.001* | — | *0.638* |
| SVR_Linear | classical | 6.187 | 0.835 | −0.017 | — | 0.675 |
| LSTM | deep | 6.249 | 0.831 | −0.027 | — | 0.642 |
| Transformer | deep | 6.396 | 0.823 | −0.051 | — | 0.629 |
| ST-GRU-NoGraph | graph | 6.452 | 0.820 | −0.060 | — | 0.613 |
| Ridge | classical | 6.493 | 0.818 | −0.067 | — | 0.700 |
| LSTM-Univariate-Scratch | deep | 7.906 | 0.730 | −0.299 | — | 0.648 |

Only 3 of 20 learned models have a positive point estimate and **none** has an interval entirely above 0.

**Transfer learning on the 2026 target (3 seeds):** transfer 5.792 ± 0.037, scratch 6.702 ± 1.147, persistence 6.085.

**Leave-one-outbreak-out (zero-shot, normalised log space, skill versus persistence):** North Kivu +0.032, West Africa +0.082, Uganda −0.158 (only 13 windows).

**Few-shot ablation (RMSE on the next 4 weeks; K = weeks of 2026 data used for fine-tuning):**

| K | persistence | from scratch | pre-trained |
|---:|---:|---:|---:|
| 2 | 8.80 | 12.25 | 9.40 |
| 4 | 14.30 | 13.34 | **12.17** |
| 6 | 17.99 | 20.13 | 20.02 |
| 8 | 18.87 | 17.61 | **17.08** |
| 12 | 19.22 | 16.32 | **16.15** |
| 16 | **7.20** | 8.72 | 7.99 |

**Chronological cross-validation (4 folds, mean RMSE):** Random Forest 14.06, SVR_Linear 14.12, XGBoost 14.21, Ridge 14.41, LightGBM_logdelta 14.52, persistence 14.96. The linear `_logdelta` variants and plain Linear Regression are unstable with small training windows (RMSE 24–29).

---

## Installation

The reference run used Python 3.12, PyTorch 2.11 (CPU) and pandas 3.0 on Windows. No GPU is needed. `torch_geometric` is **not** required.

```bash
conda create -n ebola python=3.12
conda activate ebola

pip install numpy pandas scipy scikit-learn matplotlib requests
pip install torch                                  # deep learning, transfer learning, GNN
pip install xgboost lightgbm shap lime statsmodels # optional: the notebook degrades gracefully without them
pip install geopandas shapely folium               # optional: polygon maps and geographic graph edges
pip install jupyterlab
```

Missing optional packages are reported in the first cell; the corresponding sections are skipped or replaced by scikit-learn equivalents (e.g. `GradientBoosting` instead of XGBoost).

---

## Usage

1. Open `Ebola_BDBV2026_complete_v3_3.ipynb` in Jupyter.
2. Put the historical files in `historical_data/`. If a file name is not recognised, set its path in **Module 0**:
   ```python
   "DRC_2018_Nord_Kivu": {..., "path": r"historical_data/hdx/dataonly_ebolaDRC.csv", ...}
   ```
   (entry `CFG.HISTORICAL_OUTBREAKS[name]['path']`; `columns`, `indicator_regex`, `keep_status` and `cumulative` can also be overridden there if the automatic detection is wrong).
3. Run all cells from top to bottom (`Kernel → Restart & Run All`). An internet connection is needed for the first data download.
4. Check the section status printed at the end (`ok` / `skipped` / `failed`) and the files in `outputs/`.

After a new data release, simply re-run the notebook: the test window extends to the last complete week.

---

## Configuration

Everything is in the `Config` dataclass (Module 0). Main parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `FREQ` | `'W'` | `'W'` weekly (recommended, required for transfer learning) or `'D'` daily |
| `TRAIN_END`, `TEST_START`, `TEST_END` | `2026-09-30`, `2026-10-01`, `None` | chronological split (`None` = last complete period) |
| `RISK_PCTILE` | 75 | high-risk threshold (percentile of positive training targets) |
| `VAL_STEPS` | 3 | last training weeks used for early stopping |
| `N_CV_FOLDS`, `N_BOOTSTRAP`, `N_SKILL_BOOT` | 4, 500, 2000 | cross-validation folds, forecast paths, skill-interval resamples |
| `RESID_LAST_FOLDS` | 2 | cross-validation folds whose residuals feed the forecast noise |
| `TL_SEQ_LEN`, `TL_SEEDS`, `TL_K_LIST` | 6, 3, (2, 4, 6, 8, 12, 16) | transfer-learning window, seeds, few-shot sizes |
| `GNN_WINDOW`, `GNN_SIM_THRESHOLD`, `GNN_SIM_TOPK` | 4, 0.6, 3 | GNN input window and similarity edges |
| `DL_EPOCHS`, `GNN_EPOCHS` | 300, 300 | maximum epochs (early stopping on a chronological validation block) |
| `FIG_DPI` | 600 | figure resolution (lower it to speed up the notebook) |
| `SEED` | 42 | global seed |

---

## Outputs

Created under `outputs/`:

```
outputs/
├── figures/
│   ├── 01_eda_national.png            02_top_zones.png              03_acf_pacf.png
│   ├── 04_historical_outbreaks.png    04b_tl_leave_one_outbreak_out.png
│   ├── 05_pretrain_loss.png           06_gnn_results.png
│   ├── 07_shap_comparison.png         08_lime_explanations.png
│   ├── 09_complete_dashboard.png      10_pred_vs_actual.png         10b_national_weekly_fit.png
│   ├── 11_residuals_<model>.png       12_probabilistic_forecast.png
│   ├── 13_geo_points_map.png          14_drc_choropleth_map.png
│   └── 16_tl_fewshot_ablation.png
├── 15_drc_interactive_map.html
└── reports/
    ├── complete_report.json           # configuration, provenance, rankings, CV, forecast, TL results, run status
    ├── model_comparison.csv           table_results.tex
    ├── zone_statistics.json           shap_importance.csv           lime_explanations.csv
```

Working folders: `donnees_ebola/` (downloaded archives), `donnees_extraites/` (extracted data) and `historical_data/`.

---

## Limitations

1. **Short test window.** Seven weekly periods: the ranking is indicative. Use the cross-validation table for stability and re-run when new data are released.
2. **Pooled metrics.** Errors are pooled over zone-weeks and dominated by the few large zones. Always read them next to the persistence reference and its bootstrap interval.
3. **Transfer learning.** Sources are Zaire-strain (West Africa, North Kivu) and Sudan-strain (Uganda) outbreaks, whereas 2026 is a Bundibugyo outbreak (no licensed vaccine, different context). The zero-shot transfer to Uganda is negative; the gain is demonstrated against a scratch model, not as a significant gain over persistence. Case definitions must match (confirmed cases are used throughout).
4. **Source data.** The sum of zone-level counts differs slightly from the national series; reporting delays and back-filling are not modelled; historical files are aggregated reports, not line lists, for West Africa and North Kivu. The approximate zone centroids used when no GeoJSON is available are not verified.
5. **Forecast.** Recursive forecast of one classical model with residual-based uncertainty; no new zone can appear and structural changes (interventions, population movement) are not modelled. Treat it as a scenario, not a prediction.
6. **Model selection.** The best model is chosen on the same test window that is used to report it. The persistence-anchored and transfer-learning designs were fixed before this comparison, but a confirmation on later data is needed.

---

## Reproducibility, citation and license

- Seeds are fixed (`SEED = 42`); data release, source files and package versions are written to `outputs/reports/complete_report.json`.
  
- **Data credits:** INRB-UMIE (*BDBV2026-Data*); WHO / HDX (West Africa 2014-16 cases); North Kivu 2018-2020 epi-week table; Uganda Ministry of Health surveillance data (Sudan virus 2022).
- **Author / contact:** MOISE KAMBALE KASAMBYA, University of the Philippines Diliman, mkkambale@up.edu.ph.
  
- **License and warranty:** The repository code is licensed under the terms in LICENSE. We do not claim ownership of or the right to license the third-party data or software tools used. Please pass forward any existing license/warranty/copyright information when redistributing.
- THE DATA AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. 
