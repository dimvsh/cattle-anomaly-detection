# Cattle Drinking Behavior Anomaly Detection System

An ML-based early warning system for detecting health anomalies in feedlot cattle through automated analysis of RFID-monitored drinking behavior. Developed as a bachelor thesis project using data from the COWMAS Precision Livestock Farming (PLF) system.

## Quick Start

**Interactive Dashboard:** [Open Streamlit App](https://dimvsh-cattle-anomaly-detection.streamlit.app) *(link will be active after Streamlit Cloud deployment)*

**Run Notebooks in Browser (no installation needed):**

| # | Notebook | Open in Colab |
|---|----------|:---:|
| 1 | Data Import and SQL Preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/01_data_import_and_sql.ipynb) |
| 2 | Feature Engineering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/02_feature_engineering.ipynb) |
| 3 | Exploratory Data Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/03_eda_visualization.ipynb) |
| 4 | Anomaly Detection Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/04_anomaly_detection.ipynb) |
| 5 | Validation and Results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/05_validation_results.ipynb) |
| 6 | System Demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dimvsh/cattle-anomaly-detection/blob/main/notebooks/06_system_demo.ipynb) |

> To run a notebook in Colab: click the badge, then **Runtime > Run all**. The first cell automatically clones the repo and installs dependencies.

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

## Local Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.10+, TensorFlow, scikit-learn, pandas, numpy, matplotlib, streamlit, joblib.

## Usage

### 1. Command-Line Interface

```bash
# Basic usage
python detect.py --input data/processed/scanner_data_clean.csv

# Save results to file
python detect.py --input data/processed/scanner_data_clean.csv --output results.csv
```

### 2. Web Dashboard

```bash
streamlit run app.py
```

The dashboard provides:
- Herd overview with anomaly counts
- Alert list ranking animals by anomaly frequency
- Per-animal detail with behavioral timelines and anomaly score plots
- Interactive data exploration

### 3. Python API

```python
from src.pipeline import run_pipeline

results, summary = run_pipeline('scanner_data.csv', model_dir='models/')

# results is a DataFrame with ae_score and ae_anomaly columns
flagged = results[results['ae_anomaly'] == 1]
```

## Project Structure

```
├── src/                      # System source code
│   ├── preprocessing.py      # Raw data cleaning
│   ├── features.py           # Bout merging + daily features
│   ├── detector.py           # Trained model loading + scoring
│   └── pipeline.py           # End-to-end pipeline
├── models/                   # Trained model artifacts
│   ├── autoencoder.keras     # TensorFlow autoencoder (4-3-2-3-4)
│   ├── scaler.pkl            # StandardScaler (fit on healthy data)
│   └── config.json           # Threshold and parameters
├── notebooks/                # Research & development notebooks (01-06)
├── data/
│   ├── raw/                  # Original data files
│   └── processed/            # Cleaned CSVs
├── figures/                  # Thesis-ready figures (fig1-fig17)
├── app.py                    # Streamlit dashboard
├── detect.py                 # CLI entry point
└── requirements.txt
```

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
