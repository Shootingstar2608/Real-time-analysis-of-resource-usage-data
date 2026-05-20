"""
Real-time Dashboard - Streamlit App
Visualizes streaming results: anomalies, metrics (CPU, RAM, Disk).

Run: streamlit run dashboard/app.py
"""

import os
import sys
import json
import time
import glob
import tarfile
import csv
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import TextIOWrapper

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
    
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not parquet_files:
        return None
    
    # Load last 20 batch files
    dfs = [pd.read_parquet(f) for f in parquet_files[-20:]]
    df = pd.concat(dfs, ignore_index=True)
    return df


@st.cache_data(ttl=120)
def load_historical_data(max_records=200_000):
    """Load raw Alibaba data from .tar.gz for historical view"""
    tar_gz_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "alibaba", "machine_usage.tar.gz"
    )
    
    if not os.path.exists(tar_gz_path):
        return None
    
    records = []
    
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            
            text_stream = TextIOWrapper(f, encoding='utf-8')
            reader = csv.reader(text_stream)
            
            for row in reader:
                if len(row) < 9:
                    continue
                try:
                    records.append({
                        'machine_id': row[0].strip(),
                        'timestamp': float(row[1]) if row[1] else 0.0,
                        'cpu_util_percent': float(row[2]) if row[2] else 0.0,
                        'mem_util_percent': float(row[3]) if row[3] else 0.0,
                        'disk_io_percent': float(row[8]) if row[8] else 0.0,
                    })
                    if len(records) >= max_records:
                        break
                except (ValueError, IndexError):
                    continue
            
            text_stream.close()
            if len(records) >= max_records:
                break
    
    if not records:
        return None
    
    return pd.DataFrame(records)


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
    ["🏠 Overview", "🔍 Anomaly Detection", "📊 Machine Analytics", "🤖 ML Model Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🏗️ Architecture
```
Alibaba Data (.tar.gz)
  → Kafka Producer
    → Kafka Broker
      → Spark Streaming
        → Isolation Forest
          → Dashboard
```
""")

# ============================================
# PAGE: OVERVIEW
# ============================================
if page == "🏠 Overview":
    st.title("📊 Real-time Resource Usage Monitor")
    st.markdown("**Alibaba Cluster Trace V2018 | Kafka + Spark + ML Pipeline**")
    
    df = load_historical_data()
    ml_df = load_ml_results()
    
    if df is not None:
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Machines", f"{df['machine_id'].nunique():,}")
        with col2:
            st.metric("Avg CPU %", f"{df['cpu_util_percent'].mean():.1f}%")
        with col3:
            st.metric("Avg RAM %", f"{df['mem_util_percent'].mean():.1f}%")
        with col4:
            st.metric("Avg Disk IO %", f"{df['disk_io_percent'].mean():.1f}%")
        with col5:
            st.metric("Total Records", f"{len(df):,}")
        
        st.markdown("---")
        
        # Multi-metric distribution
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Resource Usage Distribution")
            sample = df.sample(min(5000, len(df)))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=sample['cpu_util_percent'], name='CPU %',
                                        opacity=0.7, marker_color='#636EFA'))
            fig.add_trace(go.Histogram(x=sample['mem_util_percent'], name='RAM %',
                                        opacity=0.7, marker_color='#EF553B'))
            fig.add_trace(go.Histogram(x=sample['disk_io_percent'], name='Disk IO %',
                                        opacity=0.7, marker_color='#00CC96'))
            fig.update_layout(barmode='overlay', title="CPU / RAM / Disk Distribution",
                            xaxis_title="Usage %", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("CPU vs RAM Correlation")
            fig = px.scatter(
                sample, x='cpu_util_percent', y='mem_util_percent',
                color='disk_io_percent', opacity=0.4,
                title="CPU vs RAM (color = Disk IO)",
                labels={'cpu_util_percent': 'CPU %', 'mem_util_percent': 'RAM %',
                        'disk_io_percent': 'Disk IO %'},
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("Resource Usage Over Time (sampled)")
        # Group by timestamp buckets for time series
        ts_sample = df.sample(min(20000, len(df))).sort_values('timestamp')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_sample['timestamp'], y=ts_sample['cpu_util_percent'],
                                mode='markers', name='CPU %', opacity=0.3, marker=dict(size=2)))
        fig.add_trace(go.Scatter(x=ts_sample['timestamp'], y=ts_sample['mem_util_percent'],
                                mode='markers', name='RAM %', opacity=0.3, marker=dict(size=2)))
        fig.add_trace(go.Scatter(x=ts_sample['timestamp'], y=ts_sample['disk_io_percent'],
                                mode='markers', name='Disk IO %', opacity=0.3, marker=dict(size=2)))
        fig.add_hline(y=90, line_dash="dash", line_color="red",
                     annotation_text="High Threshold (90%)")
        fig.update_layout(xaxis_title="Timestamp (s)", yaxis_title="Usage %",
                         title="Multi-Metric Usage Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # ML results summary
        if ml_df is not None and 'is_anomaly_ai' in ml_df.columns:
            st.markdown("---")
            st.subheader("🤖 AI Anomaly Summary (Latest Batches)")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Records Analyzed", f"{len(ml_df):,}")
            with col_b:
                anomaly_count = ml_df['is_anomaly_ai'].sum()
                st.metric("AI Anomalies", f"{int(anomaly_count):,}")
            with col_c:
                rate = anomaly_count / len(ml_df) * 100 if len(ml_df) > 0 else 0
                st.metric("Anomaly Rate", f"{rate:.2f}%")
    else:
        st.warning("⚠️ No data found. Download Alibaba dataset first.")
        st.code("wget -c http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -P data/alibaba/")


# ============================================
# PAGE: ANOMALY DETECTION
# ============================================
elif page == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection (AI + Rule-based)")
    
    ml_df = load_ml_results()
    
    if ml_df is not None and 'is_anomaly_ai' in ml_df.columns:
        # Anomaly summary
        col1, col2, col3, col4 = st.columns(4)
        anomalies = ml_df[ml_df['is_anomaly_ai'] == True]

        with col1:
            st.metric("🚨 IF Anomalies", f"{int(ml_df.get('is_anomaly_if', ml_df['is_anomaly_ai']).sum()):,}")
        with col2:
            n_lstm = int(ml_df['is_anomaly_lstm'].sum()) if 'is_anomaly_lstm' in ml_df.columns else 'N/A'
            st.metric("🔵 LSTM Anomalies", f"{n_lstm:,}" if isinstance(n_lstm, int) else n_lstm)
        with col3:
            n_ens = int(ml_df['is_anomaly_ensemble'].sum()) if 'is_anomaly_ensemble' in ml_df.columns else 'N/A'
            st.metric("⚡ Ensemble Anomalies", f"{n_ens:,}" if isinstance(n_ens, int) else n_ens)
        with col4:
            total = len(ml_df)
            n_any = int(ml_df['is_anomaly_ensemble'].sum()) if 'is_anomaly_ensemble' in ml_df.columns else int(ml_df['is_anomaly_ai'].sum())
            rate = n_any / total * 100 if total > 0 else 0
            st.metric("Anomaly Rate", f"{rate:.2f}%")
        
        st.markdown("---")
        
        # 3D Scatter: CPU vs RAM vs Disk with anomaly coloring
        st.subheader("Multi-Metric Anomaly View")
        sample = ml_df.sample(min(5000, len(ml_df)))
        fig = px.scatter_3d(
            sample, 
            x='cpu_util_percent', y='mem_util_percent', z='disk_io_percent',
            color=sample['is_anomaly_ai'].map({True: '🔴 Anomaly', False: '🟢 Normal'}),
            opacity=0.5,
            color_discrete_map={'🔴 Anomaly': 'red', '🟢 Normal': 'blue'},
            title="CPU vs RAM vs Disk (3D Anomaly View)",
            labels={'cpu_util_percent': 'CPU %', 'mem_util_percent': 'RAM %',
                    'disk_io_percent': 'Disk IO %'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly score distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Anomaly Score Distribution")
            fig = px.histogram(ml_df, x='if_anomaly_score', nbins=50,
                             color=ml_df['is_anomaly_ai'].map({True: 'Anomaly', False: 'Normal'}),
                             title="Isolation Forest Anomaly Scores",
                             color_discrete_map={'Anomaly': 'red', 'Normal': 'steelblue'})
            fig.add_vline(x=-0.05, line_dash="dash", line_color="red",
                         annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("Anomalies by Resource Type")
            if not anomalies.empty:
                high_cpu = (anomalies['cpu_util_percent'] > 90).sum()
                high_mem = (anomalies['mem_util_percent'] > 95).sum()
                high_disk = (anomalies['disk_io_percent'] > 90).sum()
                fig = px.bar(
                    x=['High CPU (>90%)', 'High RAM (>95%)', 'High Disk (>90%)'],
                    y=[high_cpu, high_mem, high_disk],
                    color=['CPU', 'RAM', 'Disk'],
                    title="Anomaly Breakdown by Resource"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies table
        st.subheader("🔥 Top Anomalous Machines")
        if not anomalies.empty:
            display_cols = [
                'machine_id', 'cpu_util_percent', 'mem_util_percent',
                'disk_io_percent', 'if_anomaly_score'
            ]
            # Add LSTM/ensemble columns when available
            for c in ['lstm_recon_error', 'is_anomaly_lstm', 'is_anomaly_ensemble']:
                if c in anomalies.columns:
                    display_cols.append(c)
            top_anomalies = anomalies.nsmallest(20, 'if_anomaly_score')[
                display_cols
            ].reset_index(drop=True)
            st.dataframe(top_anomalies, use_container_width=True)
    else:
        st.info("⏳ No ML results yet. Start the pipeline first:")
        st.code("""
# 1. Start Spark Consumer
python3.10 spark_processing/streaming_consumer.py

# 2. Start Kafka Producer  
python3.10 kafka_stream/real_data_producer.py --max-records 100000
""")


# ============================================
# PAGE: MACHINE ANALYTICS
# ============================================
elif page == "📊 Machine Analytics":
    st.title("📊 Machine Analytics")
    
    df = load_historical_data()
    
    if df is not None:
        # Per-machine stats
        machine_stats = df.groupby('machine_id').agg({
            'cpu_util_percent': 'mean',
            'mem_util_percent': 'mean',
            'disk_io_percent': 'mean',
        }).reset_index()
        machine_stats.columns = ['machine_id', 'avg_cpu', 'avg_mem', 'avg_disk']
        
        # Box plots
        st.subheader("Resource Usage Distribution (per-record)")
        col1, col2, col3 = st.columns(3)
        sample = df.sample(min(10000, len(df)))
        
        with col1:
            fig = px.box(sample, y='cpu_util_percent', title="CPU Usage %")
            fig.update_traces(marker_color='#636EFA')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(sample, y='mem_util_percent', title="RAM Usage %")
            fig.update_traces(marker_color='#EF553B')
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = px.box(sample, y='disk_io_percent', title="Disk IO %")
            fig.update_traces(marker_color='#00CC96')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top consumers
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🔥 Top 10 CPU Consumers")
            top_cpu = machine_stats.nlargest(10, 'avg_cpu')
            fig = px.bar(top_cpu, x='avg_cpu', y='machine_id',
                        orientation='h', title="Highest Avg CPU",
                        labels={'avg_cpu': 'Avg CPU %', 'machine_id': 'Machine'})
            fig.update_traces(marker_color='#636EFA')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("🔥 Top 10 RAM Consumers")
            top_mem = machine_stats.nlargest(10, 'avg_mem')
            fig = px.bar(top_mem, x='avg_mem', y='machine_id',
                        orientation='h', title="Highest Avg RAM",
                        labels={'avg_mem': 'Avg RAM %', 'machine_id': 'Machine'})
            fig.update_traces(marker_color='#EF553B')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = ['cpu_util_percent', 'mem_util_percent', 'disk_io_percent']
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.3f', color_continuous_scale='RdBu_r',
                       title="Metric Correlations", 
                       x=['CPU', 'RAM', 'Disk'], y=['CPU', 'RAM', 'Disk'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Per-machine selector
        st.markdown("---")
        st.subheader("🔎 Machine Deep Dive")
        machine_ids = sorted(df['machine_id'].unique())
        selected = st.selectbox("Select Machine", machine_ids[:50])
        
        m_data = df[df['machine_id'] == selected].sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m_data['timestamp'], y=m_data['cpu_util_percent'],
                                mode='lines', name='CPU %'))
        fig.add_trace(go.Scatter(x=m_data['timestamp'], y=m_data['mem_util_percent'],
                                mode='lines', name='RAM %'))
        fig.add_trace(go.Scatter(x=m_data['timestamp'], y=m_data['disk_io_percent'],
                                mode='lines', name='Disk IO %'))
        fig.update_layout(title=f"Resource Usage: {selected}",
                         xaxis_title="Timestamp (s)", yaxis_title="Usage %")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available")


# ============================================
# PAGE: ML MODEL RESULTS
# ============================================
elif page == "🤖 ML Model Results":
    st.title("🤖 ML Model Results & Explainable AI (XAI)")
    
    ml_df = load_ml_results()
    
    if ml_df is not None:
        st.subheader("Latest Batch Results")
        st.dataframe(ml_df.tail(50), use_container_width=True)
    else:
        st.info("No ML results yet. Run the pipeline first.")
    
    # Explainable AI (XAI) insights
    st.markdown("---")
    st.subheader("🧠 BiLSTM Attention Weights (XAI)")
    attention_plot_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "plots", "attention_weights.png"
    )
    if os.path.exists(attention_plot_path):
        st.image(attention_plot_path, caption="Temporal Attention Weights from BiLSTM", use_container_width=True)
        st.info("💡 **Giải thích (Interpretability):** Biểu đồ Heatmap (Attention Weights) cho thấy khoảng thời gian (Time Steps) nào có trọng số chú ý cao nhất, tức là ảnh hưởng nhiều nhất tới quyết định phát hiện bất thường của mô hình BiLSTM.")
    else:
        st.info("Chưa có biểu đồ Attention Weights. Hãy chạy huấn luyện mô hình `lstm_models.py` để sinh biểu đồ này.")
    
    # Model info
    st.subheader("Model Information")
    model_info = {
        'Model': ['Isolation Forest', 'BiLSTM + Attention Autoencoder', 'Ensemble (IF + BiLSTM)'],
        'Type': ['Point Anomaly (Unsupervised)', 'Sequence Anomaly (Semi-supervised)', 'Union vote'],
        'Features': ['cpu_util_percent, mem_util_percent, disk_io_percent'] * 3,
        'Config / Performance': [
            '5% contamination | threshold = -0.05',
            'seq_len=30 | threshold=p95 recon error | BiLSTM-Attention',
            'Anomaly if IF=True OR LSTM=True',
        ],
        'Dataset': ['Alibaba Cluster Trace V2018'] * 3,
    }
    st.table(pd.DataFrame(model_info))


# ============================================
# AUTO-REFRESH
# ============================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
