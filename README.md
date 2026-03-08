# Cattle Drinking Behavior Anomaly Detection System

An ML-based early warning system for detecting health anomalies in feedlot cattle through automated analysis of RFID-monitored drinking behavior. Developed as a bachelor thesis project using data from the COWMAS Precision Livestock Farming (PLF) system.

---

## How to Use This Repository

Everything runs in your browser. **No software installation is needed.**

### 1. Browse the Code and Data

You are already on the right page. Click on any folder or file above to view it. Key locations:

- **`notebooks/`** — the 6 Jupyter notebooks with all analysis, code, and results
- **`src/`** — the Python source code for the detection system
- **`data/`** — raw and processed datasets
- **`models/`** — the trained machine learning model
- **`figures/`** — all 17 thesis figures (click any `.png` file to view it)
- **`app.py`** — the interactive dashboard code

### 2. Interactive Dashboard

Click the link below to open the live dashboard in your browser:

**[Open Dashboard](https://cattle-anomaly-detection-nzucgteqcdkeytsvlzafnp.streamlit.app)**

The dashboard lets you explore anomaly detection results, view per-animal behavioral timelines, and see which animals were flagged.

### 3. Run the Notebooks

The notebooks contain the full research pipeline — from raw data import to final results. You can view them directly on GitHub (just click the file), or run them interactively using Google Colab:

| # | Notebook | What it does | Run it |
|---|----------|-------------|:---:|
| 1 | Data Import and SQL | Loads raw RFID scanner data from the database | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/01_data_import_and_sql.ipynb) |
| 2 | Feature Engineering | Computes 4 daily behavioral metrics per animal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/02_feature_engineering.ipynb) |
| 3 | Exploratory Data Analysis | Visualizations and statistical analysis (Fig 1-7) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/03_eda_visualization.ipynb) |
| 4 | Anomaly Detection | Trains the autoencoder and isolation forest models (Fig 8-12) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/04_anomaly_detection.ipynb) |
| 5 | Validation and Results | Model evaluation, metrics, and discussion (Fig 13-17) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/05_validation_results.ipynb) |
| 6 | System Demo | Exports the model and demonstrates the full pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/06_system_demo.ipynb) |

**To run a notebook in Colab:**
1. Click the "Open in Colab" button next to the notebook
2. Sign in with a Google account if prompted
3. Click **Runtime** (top menu) then **Run all**
4. Wait for all cells to finish (a few minutes). You will see all outputs, tables, and figures generated live.

---

## Project Structure

```
├── notebooks/                # Research notebooks (01-06)
│   ├── 01_data_import_and_sql.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_validation_results.ipynb
│   └── 06_system_demo.ipynb
├── src/                      # Detection system source code
│   ├── preprocessing.py      # Raw data cleaning
│   ├── features.py           # Bout merging + daily metric computation
│   ├── detector.py           # Trained model loading + anomaly scoring
│   └── pipeline.py           # End-to-end pipeline
├── models/                   # Trained model artifacts
│   ├── autoencoder.keras     # TensorFlow autoencoder (4-3-2-3-4)
│   ├── scaler.pkl            # StandardScaler (fit on healthy training data)
│   ├── weights.json          # Extracted model weights for deployment
│   └── config.json           # Threshold and pipeline parameters
├── data/
│   ├── raw/                  # Original data (SQL dump, animal roster)
│   └── processed/            # Cleaned CSVs used by notebooks
├── figures/                  # All 17 thesis figures (fig1–fig17)
├── app.py                    # Streamlit interactive dashboard
├── detect.py                 # Command-line detection tool
└── requirements.txt          # Python dependencies
```

## System Architecture

```
Raw Scanner Data (CSV)
        |
        v
  Preprocessing          - timestamp conversion, session cleaning
        |
        v
  Feature Engineering    - bout merging (60s threshold), 4 daily metrics
        |
        v
  Anomaly Detection      - autoencoder reconstruction error scoring
        |
        v
  Results                - anomaly flags + scores per animal per day
```

## Features Computed

| Feature | Description |
|---------|-------------|
| `total_daily_duration` | Total seconds spent drinking per day |
| `visit_count` | Number of drinking bouts per day |
| `avg_visit_duration` | Mean duration of a single bout (seconds) |
| `max_absence_hours` | Longest gap between consecutive bouts (hours) |

## Model Details

- **Architecture**: Dense autoencoder (4 → 3 → 2 → 3 → 4), 48 parameters
- **Training**: Healthy animals only, first 70% of days (temporal split)
- **Threshold**: 95th percentile of training reconstruction errors
- **Anomaly score**: Mean Squared Error between input and reconstruction

## Data

- **Source**: COWMAS PLF system, RFID scanner at water drinkers
- **Period**: October 11 – November 7, 2024 (27 days, 2 partial days removed)
- **Animals**: 30 bulls in pen 121, feedlot in Kazakhstan
- **Known sick**: Animal 1149 (confirmed bloat), Animal 1129 (suspected pneumonia)
