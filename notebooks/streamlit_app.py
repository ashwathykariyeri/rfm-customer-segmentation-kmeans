"""
streamlit_app.py
-----------------
Interactive dashboard for the RFM + K-Means customer segmentation project.

Run locally:
    streamlit run streamlit_app.py

Deploy for free:
    Push this file + pipeline.py + requirements.txt to a GitHub repo,
    then deploy on Streamlit Community Cloud (share.streamlit.io) -
    no server management required.
"""

import io

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import (
    validate_schema,
    run_full_pipeline,
    REQUIRED_COLUMNS,
)

st.set_page_config(
    page_title="RFM Customer Segmentation Dashboard",
    page_icon="\U0001F4CA",
    layout="wide",
)

# ---------------------------------------------------------------- Header --
st.title("\U0001F4CA RFM Customer Segmentation Dashboard")
st.caption(
    "MSc Data Science \u2014 UEAS  |  Ashwathy, Adarsh Verma, Dheeraj K B  "
    "|  Supervised by Dr. Shan Faiz"
)
st.markdown(
    "Upload a transaction CSV (`InvoiceNo, StockCode, Description, Quantity, "
    "InvoiceDate, UnitPrice, CustomerID, Country`) to segment customers using "
    "RFM analysis and K-Means clustering \u2014 live, in the browser."
)

# ---------------------------------------------------------------- Sidebar --
st.sidebar.header("\u2699\ufe0f Controls")

uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled sample dataset instead", value=uploaded_file is None)

k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tip:** Use the Elbow and Silhouette charts on the *Model Selection* tab "
    "to justify your choice of k before reading the segments."
)

# ---------------------------------------------------------------- Load data --
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file, encoding="ISO-8859-1")


raw_df = None
if uploaded_file is not None and not use_sample:
    raw_df = load_csv(uploaded_file)
elif use_sample:
    raw_df = load_csv("sample_online_retail.csv")

if raw_df is None:
    st.info("\u2b06\ufe0f Upload a CSV in the sidebar, or tick 'Use bundled sample dataset' to explore the demo.")
    st.stop()

missing = validate_schema(raw_df)
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

# ---------------------------------------------------------------- Run pipeline --
with st.spinner("Cleaning data, engineering RFM features, and clustering..."):
    result = run_full_pipeline(raw_df, k)

rfm = result["rfm"]
elbow_df = result["elbow_df"]
sil_df = result["sil_df"]
cluster_summary = result["cluster_summary"]

# ---------------------------------------------------------------- KPI row --
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw transaction rows", f"{len(raw_df):,}")
c2.metric("Cleaned rows used", f"{len(result['cleaned']):,}")
c3.metric("Unique customers", f"{rfm['CustomerID'].nunique():,}")
c4.metric("Segments (k)", k)

tab1, tab2, tab3 = st.tabs(["\U0001F4C8 Model Selection", "\U0001F9E9 Segment Profiles", "\U0001F50D Customer Explorer"])

# ---------------------------------------------------------------- Tab 1 --
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
        ax.axvline(k, color="crimson", linestyle="--", alpha=0.7, label=f"Selected k={k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia (WCSS)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Silhouette Score")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.plot(sil_df["k"], sil_df["silhouette"], marker="o", color="teal")
        if k in sil_df["k"].values:
            ax2.axvline(k, color="crimson", linestyle="--", alpha=0.7, label=f"Selected k={k}")
        ax2.set_xlabel("Number of clusters (k)")
        ax2.set_ylabel("Silhouette Coefficient")
        ax2.legend()
        st.pyplot(fig2)

    best_k_sil = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    st.success(
        f"Highest Silhouette Score at k={best_k_sil} "
        f"({sil_df['silhouette'].max():.3f}). Currently viewing k={k}."
    )

# ---------------------------------------------------------------- Tab 2 --
with tab2:
    st.subheader("Cluster-level RFM Summary")
    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("Normalised Cluster Heatmap")
    heat_data = cluster_summary.set_index("Segment")[["Recency", "Frequency", "Monetary"]]
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)
    fig3, ax3 = plt.subplots(figsize=(6, 0.6 * len(heat_norm) + 1.5))
    sns.heatmap(heat_norm, annot=heat_data.round(1), fmt="", cmap="YlGnBu", ax=ax3, cbar=True)
    st.pyplot(fig3)

    st.subheader("Segment Size Distribution")
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.barh(seg_counts["Segment"], seg_counts["Customers"], color="#12897A")
    ax4.set_xlabel("Number of customers")
    st.pyplot(fig4)

# ---------------------------------------------------------------- Tab 3 --
with tab3:
    st.subheader("Filterable Customer-Segment Table")
    segment_options = sorted(rfm["Segment"].unique().tolist())
    chosen_segments = st.multiselect("Filter by segment", segment_options, default=segment_options)

    filtered = rfm[rfm["Segment"].isin(chosen_segments)].sort_values("Monetary", ascending=False)
    st.dataframe(
        filtered[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Segment"]],
        use_container_width=True,
        height=420,
    )

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        "\u2b07\ufe0f Download segmented customers as CSV",
        data=csv_buffer.getvalue(),
        file_name="customer_segments.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Pipeline: clean_transactions \u2192 compute_rfm \u2192 scale_rfm \u2192 "
    "K-Means (Elbow + Silhouette validated) \u2192 label_segments. "
    "See pipeline.py for the underlying functions."
)
