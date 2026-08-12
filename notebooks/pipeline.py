"""
pipeline.py
-----------
Reusable RFM + K-Means customer segmentation pipeline.

This module wraps the group's original notebook logic (data cleaning ->
RFM feature engineering -> scaling -> K-Means -> evaluation) as plain,
testable Python functions so they can be called from the Streamlit
dashboard (streamlit_app.py) instead of living in notebook cells.

No business data is hard-coded here; every function accepts a DataFrame
and returns a DataFrame / dict, so the same pipeline works for the
original Online Retail export or any similarly-structured transaction
CSV a user uploads.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of missing required columns (empty list = valid)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw invoice-level transaction DataFrame.

    Steps (matching the project's original notebook methodology):
      1. Drop rows with a missing CustomerID (RFM is customer-level).
      2. Drop rows with non-positive Quantity (returns / cancellations).
      3. Drop rows with non-positive UnitPrice (data entry errors).
      4. Parse InvoiceDate to datetime.
      5. Derive TotalAmount = Quantity * UnitPrice.
    """
    out = df.copy()
    out = out.dropna(subset=["CustomerID"])
    out = out[out["Quantity"] > 0]
    out = out[out["UnitPrice"] > 0]
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out = out.dropna(subset=["InvoiceDate"])
    out["TotalAmount"] = out["Quantity"] * out["UnitPrice"]
    return out


def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Aggregate cleaned transactions to one row per CustomerID with
    Recency (days), Frequency (unique invoices), and Monetary (total spend).
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()

    return rfm


def scale_rfm(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Z-score standardise the three RFM columns for distance-based clustering."""
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    return X, scaler


def elbow_scores(X: np.ndarray, k_range: range = range(1, 11)) -> pd.DataFrame:
    """Compute inertia (WCSS) for each k in k_range -> Elbow Method data."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        rows.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def silhouette_scores(X: np.ndarray, k_range: range = range(2, 11)) -> pd.DataFrame:
    """Compute the Silhouette Coefficient for each k in k_range."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """Fit a final K-Means model with the chosen k and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Rank clusters by a composite value score and assign human-readable
    segment names, generalising to any k (not hard-coded to k=4).

    Value score = high Frequency & Monetary, low Recency.
    """
    summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    summary["ValueScore"] = (
        summary["Frequency"].rank(ascending=True)
        + summary["Monetary"].rank(ascending=True)
        + summary["Recency"].rank(ascending=False)
    )
    ranked = summary.sort_values("ValueScore", ascending=False).index.tolist()

    base_names = ["Champions", "Loyal Customers", "Regular / At-Risk Customers",
                  "Lost / Inactive Customers"]
    # Generalise label pool for k != 4
    if len(ranked) <= len(base_names):
        names = base_names[:len(ranked)]
    else:
        names = base_names + [f"Segment {i+1}" for i in range(len(ranked) - len(base_names))]

    label_map = {cluster_id: names[i] for i, cluster_id in enumerate(ranked)}
    rfm = rfm.copy()
    rfm["Segment"] = rfm["Cluster"].map(label_map)
    return rfm


def run_full_pipeline(raw_df: pd.DataFrame, k: int) -> dict:
    """
    Convenience wrapper that runs the entire pipeline end-to-end and
    returns everything the dashboard needs to render in one call.
    """
    cleaned = clean_transactions(raw_df)
    rfm = compute_rfm(cleaned)
    X, scaler = scale_rfm(rfm)

    elbow_df = elbow_scores(X)
    sil_df = silhouette_scores(X)

    labels, model = fit_kmeans(X, k)
    rfm["Cluster"] = labels
    rfm = label_segments(rfm)

    cluster_summary = rfm.groupby(["Cluster", "Segment"])[["Recency", "Frequency", "Monetary"]] \
        .mean().round(1).reset_index()

    return {
        "cleaned": cleaned,
        "rfm": rfm,
        "elbow_df": elbow_df,
        "sil_df": sil_df,
        "cluster_summary": cluster_summary,
        "model": model,
        "scaler": scaler,
    }
