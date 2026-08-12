# RFM Customer Segmentation using K-Means Clustering

## Project Overview
This project focuses on customer segmentation using RFM analysis and K-Means clustering. RFM stands for Recency, Frequency, and Monetary value. These metrics are used to understand customer purchasing behavior.

## Problem Statement
Businesses often treat all customers similarly, even though customer behavior differs. This project aims to group customers based on transaction behavior to support targeted marketing and customer retention.

## Objectives
- Analyze customer transaction data
- Create an RFM table
- Apply feature scaling
- Use Elbow Method and Silhouette Score to choose optimal clusters
- Apply K-Means clustering
- Interpret customer segments

## Dataset
The project uses the Online Retail dataset from the UCI Machine Learning Repository / Kaggle.

Dataset link:
https://archive.ics.uci.edu/dataset/352/online+retail

## Methodology
1. Data collection
2. Data cleaning
3. RFM feature creation
4. Feature scaling
5. Elbow Method
6. Silhouette Score
7. K-Means clustering
8. Customer segment interpretation

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Files
- `pipeline.py` — reusable data-cleaning, RFM engineering, scaling, K-Means,
  and evaluation functions (no notebook cells).
- `streamlit_app.py` — the interactive dashboard (3 tabs: Model Selection,
  Segment Profiles, Customer Explorer).
- `requirements.txt` — Python dependencies.
- `sample_online_retail.csv` — a synthetic, schema-matched demo dataset
  (800 customers, ~26.7k transaction rows) so you can try the dashboard
  immediately without needing the original Online Retail export.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).
Tick "Use bundled sample dataset" in the sidebar to try it instantly, or
upload your own transaction CSV with columns:
`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`.

## Deploy for free
1. Push this folder to a GitHub repository.
2. Go to share.streamlit.io, sign in, and point it at the repo /
   `streamlit_app.py`.
3. It redeploys automatically on every push — no server management.

## Notes
- To use your team's real, cleaned Online Retail data, just upload that CSV
  instead of the bundled sample — the pipeline works identically.
- `pipeline.py` functions are independently testable/importable, e.g.:
  `from pipeline import run_full_pipeline`.
