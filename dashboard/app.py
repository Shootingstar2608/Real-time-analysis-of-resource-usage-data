"""
Real-time Dashboard - Streamlit App
Visualizes streaming results: anomalies, metrics, forecasts.

Run: streamlit run dashboard/app.py
"""

import os
import sys
import json
import time
import glob
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DASHBOARD_PORT, DASHBOARD_REFRESH_INTERVAL

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Resource Usage Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DATA LOADING
# ============================================
@st.cache_data(ttl=5)
def load_ml_results():
    """Load ML inference results from Parquet output"""
    parquet_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "ml_results"
    )
    
    if not os.path.exists(parquet_dir):
        return None
    
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    if not parquet_files:
        return None
    
    dfs = [pd.read_parquet(f) for f in parquet_files[-10:]]  # last 10 files
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.sort_values('timestamp')


@st.cache_data(ttl=60)
def load_processed_data():
    """Load raw Azure data for historical view"""
    import gzip, csv
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw"
    )
    
    cpu_files = sorted(glob.glob(os.path.join(data_dir, "vm_cpu_readings-*.csv.gz")))
    if not cpu_files:
        return None
    
    # Load first file, sample for performance
    records = []
    max_records = 100_000
    
    # Load vmtable for enrichment
    vm_lookup = {}
    vmtable_path = os.path.join(data_dir, "vmtable.csv.gz")
    if os.path.exists(vmtable_path):
        with gzip.open(vmtable_path, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 11:
                    vm_id = row[0].strip()
                    try:
                        vm_lookup[vm_id] = {
                            'vm_category': row[8].strip() or 'Unknown',
                            'vm_core_count': int(row[9]) if row[9].strip() else 0,
                            'vm_memory_gb': int(row[10]) if row[10].strip() else 0,
                        }
                    except (ValueError, IndexError):
                        pass
    
    with gzip.open(cpu_files[0], 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                vm_id = row[1].strip()
                min_cpu = float(row[2])
                max_cpu = float(row[3])
                avg_cpu = float(row[4])
                vm_info = vm_lookup.get(vm_id, {})
                records.append({
                    'timestamp': float(row[0]),
                    'vm_id': vm_id,
                    'min_cpu': min_cpu,
                    'max_cpu': max_cpu,
                    'avg_cpu': avg_cpu,
                    'cpu_range': max_cpu - min_cpu,
                    'vm_category': vm_info.get('vm_category', 'Unknown'),
                    'vm_core_count': vm_info.get('vm_core_count', 0),
                    'vm_memory_gb': vm_info.get('vm_memory_gb', 0),
                })
                if len(records) >= max_records:
                    break
            except (ValueError, IndexError):
                continue
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    return df


@st.cache_data(ttl=60)
def load_training_plots():
    """Load saved plot images from ML training"""
    plot_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "plots"
    )
    
    plots = {}
    for name in ["isolation_forest_analysis.png", "lstm_analysis.png"]:
        path = os.path.join(plot_dir, name)
        if os.path.exists(path):
            plots[name] = path
    return plots


# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("⚙️ Controls")
refresh_rate = st.sidebar.slider(
    "Refresh interval (sec)", 1, 30, DASHBOARD_REFRESH_INTERVAL
)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

page = st.sidebar.radio(
    "📄 Navigation",
    ["🏠 Overview", "🔍 Anomaly Detection", "📈 Forecasting", 
     "📊 VM Analytics", "🤖 ML Model Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🏗️ Architecture
```
Data → Kafka → Spark → ML → Dashboard
```
""")

# ============================================
# PAGE: OVERVIEW
# ============================================
if page == "🏠 Overview":
    st.title("📊 Real-time Resource Usage Monitor")
    st.markdown("**Azure VM Traces | Kafka + Spark + ML/DL Pipeline**")
    
    df = load_processed_data()
    ml_df = load_ml_results()
    
    if df is not None:
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total VMs", f"{df['vm_id'].nunique():,}")
        with col2:
            st.metric("Avg CPU %", f"{df['avg_cpu'].mean():.1f}%")
        with col3:
            st.metric("Max CPU %", f"{df['max_cpu'].mean():.1f}%")
        with col4:
            high_cpu = (df['avg_cpu'] > 80).mean() * 100
            st.metric("High CPU Rate", f"{high_cpu:.1f}%")
        with col5:
            st.metric("Total Records", f"{len(df):,}")
        
        st.markdown("---")
        
        # CPU distribution over time
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("CPU Usage Distribution")
            sample = df.sample(min(5000, len(df)))
            fig = px.scatter(
                sample, x='timestamp', y='avg_cpu', color='vm_category',
                opacity=0.4, title="CPU Usage by VM Category",
                labels={'avg_cpu': 'Average CPU %', 'timestamp': 'Time (s)'}
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                         annotation_text="High CPU Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("CPU Range (Volatility)")
            fig = px.scatter(
                sample, x='timestamp', y='cpu_range', color='vm_category',
                opacity=0.4, title="CPU Volatility by VM Category",
                labels={'cpu_range': 'CPU Range (Max-Min)', 'timestamp': 'Time (s)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # VM Category Distribution
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("VM Category Distribution")
            cat_counts = df['vm_category'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                        title="VM Categories")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("CPU Usage by Category")
            cat_stats = df.groupby('vm_category').agg({
                'avg_cpu': 'mean',
                'max_cpu': 'mean',
                'cpu_range': 'mean',
            }).round(2)
            fig = px.bar(cat_stats, barmode='group',
                        title="Avg CPU Metrics by Category")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data found. Download Azure VM Traces first.")
        st.code("bash data/download_data.sh")


# ============================================
# PAGE: ANOMALY DETECTION
# ============================================
elif page == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection")
    
    df = load_processed_data()
    ml_df = load_ml_results()
    
    data = ml_df if ml_df is not None else df
    
    if data is not None and 'is_anomaly' in data.columns:
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        anomalies = data[data['is_anomaly'] == True]
        
        with col1:
            st.metric("🚨 Total Anomalies", f"{len(anomalies):,}")
        with col2:
            st.metric("Anomaly Rate", 
                      f"{len(anomalies)/len(data)*100:.2f}%")
        with col3:
            if 'severity' in anomalies.columns:
                critical = (anomalies['severity'] == 'CRITICAL').sum()
                st.metric("🔴 Critical", f"{critical:,}")
        
        st.markdown("---")
        
        # Anomaly scatter plot: CPU Avg vs CPU Range
        st.subheader("CPU Avg vs CPU Range: Normal vs Anomaly")
        sample = data.sample(min(10000, len(data)))
        fig = px.scatter(
            sample, x='avg_cpu', y='cpu_range' if 'cpu_range' in sample.columns else 'max_cpu',
            color=sample['is_anomaly'].map({True: '🔴 Anomaly', False: '🟢 Normal'}),
            opacity=0.5,
            color_discrete_map={'🔴 Anomaly': 'red', '🟢 Normal': 'blue'},
            title="Anomaly Detection: CPU Avg vs CPU Range"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly timeline
        if 'timestamp' in data.columns:
            st.subheader("Anomaly Timeline")
            data_sorted = data.sort_values('timestamp')
            
            # Resample anomaly count
            data_sorted = data_sorted.set_index('timestamp')
            anomaly_ts = data_sorted['is_anomaly'].resample('1h').sum()
            total_ts = data_sorted['is_anomaly'].resample('1h').count()
            rate_ts = (anomaly_ts / total_ts * 100).fillna(0)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=anomaly_ts.index, y=anomaly_ts.values,
                       name="Anomaly Count", marker_color="coral",
                       opacity=0.7),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=rate_ts.index, y=rate_ts.values,
                          name="Anomaly Rate %", line=dict(color='red', width=2)),
                secondary_y=True
            )
            fig.update_layout(title="Anomalies Over Time (Hourly)")
            fig.update_yaxes(title_text="Count", secondary_y=False)
            fig.update_yaxes(title_text="Rate (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            data_sorted = data_sorted.reset_index()
        
        # Anomaly details table
        st.subheader("Recent Anomalies")
        if not anomalies.empty:
            display_cols = ['timestamp', 'vm_id', 'vm_category', 'avg_cpu', 
                          'max_cpu', 'min_cpu', 'cpu_range']
            if 'anomaly_score' in anomalies.columns:
                display_cols.append('anomaly_score')
            if 'severity' in anomalies.columns:
                display_cols.append('severity')
            
            available_cols = [c for c in display_cols if c in anomalies.columns]
            st.dataframe(
                anomalies[available_cols].tail(50).sort_values(
                    'anomaly_score' if 'anomaly_score' in anomalies.columns else 'avg_cpu'
                ),
                use_container_width=True
            )
    else:
        st.info("Run ML pipeline first to see anomaly detection results")


# ============================================
# PAGE: FORECASTING
# ============================================
elif page == "📈 Forecasting":
    st.title("📈 Resource Usage Forecasting")
    
    df = load_processed_data()
    
    if df is not None:
        # VM selector
        vm_ids = sorted(df['vm_id'].unique())
        selected_vm = st.selectbox("Select VM", vm_ids[:20])
        
        vm_data = df[df['vm_id'] == selected_vm].sort_values('timestamp')
        
        # CPU time series for selected VM
        st.subheader(f"CPU Usage: {selected_vm[:20]}...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vm_data['timestamp'], y=vm_data['avg_cpu'],
            mode='lines', name='Avg CPU',
            line=dict(color='blue', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=vm_data['timestamp'], y=vm_data['max_cpu'],
            mode='lines', name='Max CPU',
            line=dict(color='red', width=0.5), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=vm_data['timestamp'], y=vm_data['min_cpu'],
            mode='lines', name='Min CPU',
            line=dict(color='green', width=0.5), opacity=0.5
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                     annotation_text="Warning")
        fig.update_layout(
            yaxis_title="CPU %",
            xaxis_title="Time (s)",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # CPU Range (Volatility) time series
        if 'cpu_range' in vm_data.columns:
            st.subheader(f"CPU Volatility: {selected_vm[:20]}...")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vm_data['timestamp'], y=vm_data['cpu_range'],
                mode='lines', name='CPU Range (Max-Min)',
                line=dict(color='purple', width=1)
            ))
            fig.update_layout(yaxis_title="CPU Range %", xaxis_title="Time (s)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = ['min_cpu', 'max_cpu', 'avg_cpu', 'cpu_range',
                       'vm_core_count', 'vm_memory_gb']
        available = [c for c in numeric_cols if c in vm_data.columns]
        corr = vm_data[available].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                       title="Metric Correlations")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available")


# ============================================
# PAGE: VM ANALYTICS
# ============================================
elif page == "📊 VM Analytics":
    st.title("📊 VM Analytics")
    
    df = load_processed_data()
    
    if df is not None:
        # CPU distribution by VM category
        st.subheader("CPU Distribution by Category")
        fig = px.box(df.sample(min(20000, len(df))), 
                     x='vm_category', y='avg_cpu',
                     color='vm_category',
                     title="CPU Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # CPU Range distribution
        st.subheader("CPU Volatility by Category")
        fig = px.box(df.sample(min(20000, len(df))),
                     x='vm_category', y='cpu_range',
                     color='vm_category',
                     title="CPU Range (Max-Min) Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top CPU consumers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Top 10 CPU Consumers")
            top_cpu = df.groupby('vm_id')['avg_cpu'].mean() \
                .sort_values(ascending=False).head(10)
            # Shorten VM IDs for display
            display_ids = [f"{vid[:12]}..." for vid in top_cpu.index]
            fig = px.bar(x=top_cpu.values, y=display_ids,
                        orientation='h', title="Highest Avg CPU VMs",
                        labels={'x': 'Avg CPU %', 'y': 'VM ID'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔥 Top 10 Most Volatile")
            top_range = df.groupby('vm_id')['cpu_range'].mean() \
                .sort_values(ascending=False).head(10)
            display_ids = [f"{vid[:12]}..." for vid in top_range.index]
            fig = px.bar(x=top_range.values, y=display_ids,
                        orientation='h', title="Highest CPU Range VMs",
                        labels={'x': 'Avg CPU Range %', 'y': 'VM ID'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Core count vs CPU usage
        st.subheader("CPU Usage vs Core Count")
        core_stats = df.groupby('vm_core_count')['avg_cpu'].mean().reset_index()
        fig = px.bar(core_stats, x='vm_core_count', y='avg_cpu',
                    title="Average CPU by Core Count",
                    labels={'vm_core_count': 'Core Count', 'avg_cpu': 'Avg CPU %'})
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: ML MODEL RESULTS
# ============================================
elif page == "🤖 ML Model Results":
    st.title("🤖 ML Model Results")
    
    plots = load_training_plots()
    
    if plots:
        for name, path in plots.items():
            st.subheader(name.replace("_", " ").replace(".png", "").title())
            st.image(path, use_container_width=True)
    else:
        st.info("No training plots found. Train models first:")
        st.code("""
# Train Isolation Forest
python ml_models/isolation_forest_model.py

# Train LSTM models  
python ml_models/lstm_models.py
""")
    
    # Model comparison
    st.subheader("Model Comparison")
    comparison_data = {
        'Model': ['Isolation Forest', 'LSTM Autoencoder', 'LSTM Forecaster'],
        'Type': ['Anomaly Detection', 'Anomaly Detection', 'Forecasting'],
        'Approach': ['Unsupervised (point)', 'Unsupervised (sequence)', 'Supervised'],
        'Strength': [
            'Fast, no temporal context needed',
            'Captures temporal patterns, sequence anomalies',
            'Predicts future usage for capacity planning'
        ],
        'Weakness': [
            'Misses gradual trends',
            'Slower, needs sequence of data points',
            'Needs retraining for new patterns'
        ],
    }
    st.table(pd.DataFrame(comparison_data))


# ============================================
# AUTO-REFRESH
# ============================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
