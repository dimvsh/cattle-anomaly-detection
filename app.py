"""
Streamlit dashboard for the Cattle Drinking Anomaly Detection System.

Run with:
    streamlit run app.py
"""

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
DEMO_DATA = os.path.join(PROJECT_DIR, 'data', 'processed', 'scanner_data_clean.csv')

st.set_page_config(page_title='Cattle Anomaly Detection', layout='wide')
st.title('Cattle Drinking Behavior Anomaly Detection System')
st.markdown('Upload RFID scanner data or use the demo dataset to detect anomalous drinking behavior.')

# --- Sidebar ---
st.sidebar.header('Data Input')
use_demo = st.sidebar.checkbox('Use demo dataset', value=True)

uploaded_file = None
if not use_demo:
    uploaded_file = st.sidebar.file_uploader('Upload scanner CSV', type=['csv'])

if not use_demo and uploaded_file is None:
    st.info('Upload a CSV file with scanner data, or check "Use demo dataset" to see the system in action.')
    st.stop()

# --- Run pipeline (cached in session state) ---
def get_results(data_path):
    cache_key = f'results_{data_path}'
    if cache_key not in st.session_state:
        from src.pipeline import run_pipeline
        results, summary = run_pipeline(data_path, model_dir=MODEL_DIR)
        st.session_state[cache_key] = (results, summary)
    return st.session_state[cache_key]

if use_demo:
    data_path = DEMO_DATA
else:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp.write(uploaded_file.read())
    tmp.close()
    data_path = tmp.name

try:
    with st.spinner('Running anomaly detection pipeline (first load takes ~10s for TensorFlow)...'):
        results, summary = get_results(data_path)
except Exception as e:
    st.error(f'Pipeline error: {e}')
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Load threshold from config
threshold = 2.32
try:
    with open(os.path.join(MODEL_DIR, 'config.json')) as f:
        threshold = json.load(f)['threshold']
except Exception:
    pass

# --- Overview Panel ---
st.header('Herd Overview')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Animals', summary['animals'])
col2.metric('Days', summary['days'])
col3.metric('Anomalies Flagged', summary['anomalies_flagged'])
col4.metric('Anomaly Rate', f"{summary['anomaly_rate']:.1%}")

# --- Alert List ---
st.header('Alert List — Animals Ranked by Anomaly Rate')
animal_summary = results.groupby('tag_short').agg(
    anomaly_days=('ae_anomaly', 'sum'),
    total_days=('ae_anomaly', 'count'),
    mean_score=('ae_score', 'mean'),
    max_score=('ae_score', 'max')
).reset_index()
animal_summary['anomaly_rate'] = animal_summary['anomaly_days'] / animal_summary['total_days']
animal_summary = animal_summary.sort_values('anomaly_days', ascending=False)

def highlight_anomaly(row):
    flag = row.get('anomaly_days', row.get('ae_anomaly', 0))
    if flag > 0:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

st.dataframe(
    animal_summary.style.apply(highlight_anomaly, axis=1).format({
        'anomaly_rate': '{:.0%}',
        'mean_score': '{:.4f}',
        'max_score': '{:.4f}'
    }),
    use_container_width=True
)

# --- Per-Animal Detail ---
st.header('Per-Animal Detail')
selected_animal = st.selectbox(
    'Select animal:',
    sorted(results['tag_short'].unique()),
    index=0
)

animal_data = results[results['tag_short'] == selected_animal].sort_values('date')

feature_cols = ['total_daily_duration', 'visit_count', 'avg_visit_duration', 'max_absence_hours']
herd_stats = results.groupby('date')[feature_cols].agg(['mean', 'std'])

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
feature_labels = ['Total Duration (sec)', 'Visit Count', 'Avg Visit Duration (sec)', 'Max Absence (hours)']

for i, (col, label) in enumerate(zip(feature_cols, feature_labels)):
    ax = axes[i]
    herd_mean = herd_stats[(col, 'mean')]
    herd_std = herd_stats[(col, 'std')]

    ax.fill_between(herd_mean.index, herd_mean - herd_std, herd_mean + herd_std,
                     alpha=0.15, color='gray', label='Herd +/- 1 std')
    ax.plot(herd_mean.index, herd_mean.values, 'k--', alpha=0.5, linewidth=1, label='Herd mean')
    ax.plot(animal_data['date'], animal_data[col], '-o', color='#e74c3c',
            markersize=4, linewidth=1.5, label=f'Animal {selected_animal}')

    anom = animal_data[animal_data['ae_anomaly'] == 1]
    if len(anom) > 0:
        ax.scatter(anom['date'], anom[col], s=120, facecolors='none',
                   edgecolors='red', linewidths=2, zorder=5, label='Anomaly')

    ax.set_title(label, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=45)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

fig.suptitle(f'Animal {selected_animal} — Behavioral Features vs Herd', fontsize=13)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Anomaly score timeline
st.subheader(f'Animal {selected_animal} — Anomaly Score Timeline')
fig2, ax = plt.subplots(figsize=(12, 4))
ax.plot(animal_data['date'], animal_data['ae_score'], '-o', color='steelblue',
        markersize=5, linewidth=1.5)
ax.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.2f})')
anom = animal_data[animal_data['ae_anomaly'] == 1]
if len(anom) > 0:
    ax.scatter(anom['date'], anom['ae_score'], color='red', s=80, zorder=5, label='Flagged')
ax.set_ylabel('Reconstruction Error (MSE)')
ax.set_title(f'Autoencoder Anomaly Score — Animal {selected_animal}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# Raw data table
st.subheader(f'Animal {selected_animal} — Daily Data')
display_cols = ['date', 'total_daily_duration', 'visit_count', 'avg_visit_duration',
                'max_absence_hours', 'ae_score', 'ae_anomaly']
st.dataframe(
    animal_data[display_cols].style.apply(highlight_anomaly, axis=1).format({
        'total_daily_duration': '{:.1f}',
        'visit_count': '{:.0f}',
        'avg_visit_duration': '{:.1f}',
        'max_absence_hours': '{:.1f}',
        'ae_score': '{:.4f}'
    }),
    use_container_width=True
)

# --- Anomaly Day Table ---
st.header('All Flagged Anomaly Days')
flagged = results[results['ae_anomaly'] == 1][display_cols + ['tag_short']].sort_values('ae_score', ascending=False)
if len(flagged) > 0:
    st.dataframe(flagged, use_container_width=True)
else:
    st.write('No anomalies detected.')

# --- Thesis Figures Gallery ---
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
if os.path.isdir(FIGURES_DIR):
    figure_files = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
    if figure_files:
        st.header('Research Figures')

        figure_descriptions = {
            'fig1_daily_duration_timeline.png': 'Fig 1 — Daily Drinking Duration Timeline (all animals, sick highlighted)',
            'fig2_visit_count_distribution.png': 'Fig 2 — Visit Count Distribution (healthy vs sick)',
            'fig3_hourly_pattern.png': 'Fig 3 — Hourly Drinking Activity Pattern',
            'fig4_heatmap_duration.png': 'Fig 4 — Heatmap: Animals × Days (total duration)',
            'fig5_sick_vs_herd_metrics.png': 'Fig 5 — Sick Animals vs Herd Average (all 4 metrics)',
            'fig6_boxplots_per_animal.png': 'Fig 6 — Box Plots per Animal per Feature',
            'fig7_correlation_matrix.png': 'Fig 7 — Feature Correlation Matrix',
            'fig8_autoencoder_training.png': 'Fig 8 — Autoencoder Training Loss',
            'fig9_anomaly_score_distributions.png': 'Fig 9 — Anomaly Score Distributions',
            'fig10_sick_animals_anomaly_timeline.png': 'Fig 10 — Sick Animals Anomaly Timeline',
            'fig11_anomaly_frequency_per_animal.png': 'Fig 11 — Anomaly Frequency per Animal',
            'fig12_ae_vs_if_scatter.png': 'Fig 12 — Autoencoder vs Isolation Forest Comparison',
            'fig13_confusion_matrices.png': 'Fig 13 — Confusion Matrices',
            'fig14_threshold_sensitivity.png': 'Fig 14 — Threshold Sensitivity (Precision-Recall Trade-off)',
            'fig15_timeline_1129.png': 'Fig 15a — Behavioral Timeline: Animal 1129 (suspected pneumonia)',
            'fig15_timeline_1149.png': 'Fig 15b — Behavioral Timeline: Animal 1149 (confirmed bloat)',
            'fig16_fp_feature_distributions.png': 'Fig 16 — False Positive Feature Distributions',
            'fig17_animal_mean_scores.png': 'Fig 17 — Per-Animal Mean Anomaly Scores',
        }

        # Sort numerically by figure number
        import re
        def fig_sort_key(fname):
            m = re.match(r'fig(\d+)', fname)
            return int(m.group(1)) if m else 999

        figure_files = sorted(figure_files, key=fig_sort_key)

        # Group into sections using underscore to prevent fig1 matching fig10
        eda_figs = [f for f in figure_files if any(f.startswith(p) for p in ['fig1_', 'fig2_', 'fig3_', 'fig4_', 'fig5_', 'fig6_', 'fig7_'])]
        model_figs = [f for f in figure_files if any(f.startswith(p) for p in ['fig8_', 'fig9_', 'fig10_', 'fig11_', 'fig12_'])]
        validation_figs = [f for f in figure_files if any(f.startswith(p) for p in ['fig13_', 'fig14_', 'fig15_', 'fig16_', 'fig17_'])]

        if eda_figs:
            st.subheader('Exploratory Data Analysis (Fig 1–7)')
            for fname in eda_figs:
                caption = figure_descriptions.get(fname, fname)
                st.image(os.path.join(FIGURES_DIR, fname), caption=caption, use_container_width=True)

        if model_figs:
            st.subheader('Anomaly Detection Models (Fig 8–12)')
            for fname in model_figs:
                caption = figure_descriptions.get(fname, fname)
                st.image(os.path.join(FIGURES_DIR, fname), caption=caption, use_container_width=True)

        if validation_figs:
            st.subheader('Validation & Results (Fig 13–17)')
            for fname in validation_figs:
                caption = figure_descriptions.get(fname, fname)
                st.image(os.path.join(FIGURES_DIR, fname), caption=caption, use_container_width=True)
