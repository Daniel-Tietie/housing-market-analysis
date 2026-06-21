# Housing Market Analysis

An end-to-end data analytics and machine learning project built on the Ames Housing dataset. The pipeline covers data ingestion, SQL-based storage and querying, exploratory analysis, predictive modeling, and an interactive dashboard.

## Data source and methodology

**Dataset:** Ames Housing, compiled by Dean De Cock (2011). 2,919 residential property sales in Ames, Iowa from 2006 to 2010, with 79 explanatory features covering lot characteristics, construction quality, interior area, and sale conditions.

**Source:** OpenML dataset ID 42165, fetched programmatically via scikit-learn's `fetch_openml`. No manual download required — run `python src/ingest.py` to reproduce from scratch.

**Why Ames, not a Canadian dataset:** CREA and CMHC publish aggregate price indices, not individual property records. Without individual-level data, meaningful ML modeling (feature importance, train/test evaluation) is not possible. Ames is an established academic benchmark dataset suitable for demonstrating the full analytics pipeline.

**Storage:** SQLite local database (`data/housing.db`) with two tables: `raw_housing` (original data, unmodified) and `clean_housing` (analysis-ready, produced by `src/clean.py`).

## Key findings

- Neighborhood is the largest price determinant: average prices range from $105k (MeadowV) to $335k (NridgHt) — a 3x spread within the same city.
- Overall quality rating (1–10) has the strongest single-feature correlation with price (r = 0.80). Each quality step above 6 adds roughly $40–50k to the average sale price, accelerating sharply at 9–10.
- Sale volume dropped significantly in 2010 following the 2008 financial crisis, but average prices in Ames held relatively stable (less than 5% decline), suggesting the local market was insulated from the national correction.
- The spring buying season is pronounced: May and June account for over 30% of annual transactions, but average prices do not meaningfully differ by month — timing affects volume, not price.
- Gradient Boosting outperformed linear regression with a test RMSE of approximately $25,000 and R² of 0.89 vs. the baseline's RMSE of ~$40,000 and R² of 0.77. The model is reliable for homes in the $100k–$350k range; predictions above $400k are systematically low due to limited training examples.

## Dashboard

The Power BI dashboard (`dashboard/housing_dashboard.pbix`) provides four views:

1. **Overview** — total listings, average and median price, price range by year
2. **Geographic** — average price by neighborhood (bar chart ranked by price)
3. **Trends** — sale price and volume over time, seasonal patterns by month
4. **Risk/Volatility** — price standard deviation and coefficient of variation by neighborhood

Screenshots are in `dashboard/screenshots/`.

## Tech stack

- Python 3.11, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- SQLite (via Python's built-in `sqlite3`)
- Power BI Desktop (dashboard)
- Streamlit (optional live app)

## Repository structure

```
housing-market-analysis/
  data/
    raw/              original CSV (reproduced by ingest.py)
    processed/        cleaned CSV (reproduced by clean.py)
  sql/
    schema.sql        table definitions
    queries.sql       analytical SQL queries used in the EDA notebook
  notebooks/
    01_data_cleaning.ipynb
    02_exploratory_analysis.ipynb
    03_modeling.ipynb
  src/
    ingest.py         fetch from OpenML, load into SQLite
    clean.py          cleaning and feature engineering
    features.py       feature matrix preparation for ML
    model.py          model training and evaluation
  dashboard/
    housing_dashboard.pbix
    screenshots/
  streamlit_app/
    app.py
  requirements.txt
  .gitignore
```

## How to run locally

```bash
git clone https://github.com/Daniel-Tietie/housing-market-analysis.git
cd housing-market-analysis

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Fetch data and build the database
python src/ingest.py
python src/clean.py

# Run the notebooks in order (requires Jupyter)
jupyter notebook notebooks/

# Or launch the Streamlit app
streamlit run streamlit_app/app.py
```

The `data/housing.db` SQLite file is created automatically by `ingest.py`. All data can be reproduced from scratch; no manual downloads are required.
