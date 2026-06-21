"""
Streamlit app for the Ames Housing Market Analysis.
Filters by neighborhood and year sold; shows key price charts interactively.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "data" / "housing.db"

st.set_page_config(page_title="Housing Market Analysis", layout="wide")

PALETTE = ["#2b6cb0", "#4299e1", "#90cdf4", "#bee3f8"]


@st.cache_data
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM clean_housing", conn)
    conn.close()
    return df


df_all = load_data()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

neighborhoods = sorted(df_all["neighborhood"].unique())
selected_nbhd = st.sidebar.multiselect(
    "Neighborhood",
    options=neighborhoods,
    default=neighborhoods,
)

years = sorted(df_all["yr_sold"].unique())
yr_min, yr_max = int(min(years)), int(max(years))
selected_years = st.sidebar.slider("Year sold", yr_min, yr_max, (yr_min, yr_max))

df = df_all[
    (df_all["neighborhood"].isin(selected_nbhd))
    & (df_all["yr_sold"].between(*selected_years))
].copy()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Ames Housing Market Analysis")
st.caption(
    "Ames, Iowa residential sales 2006–2010. "
    "Data: De Cock (2011) / OpenML dataset 42165."
)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ── Key metrics ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Listings", f"{len(df):,}")
col2.metric("Avg price", f"${df['sale_price'].mean():,.0f}")
col3.metric("Median price", f"${df['sale_price'].median():,.0f}")
col4.metric("Avg living area", f"{df['gr_liv_area'].mean():,.0f} sq ft")

st.divider()

# ── Row 1: price by neighborhood and price distribution ───────────────────────
c1, c2 = st.columns([3, 2])

with c1:
    nbhd_agg = (
        df.groupby("neighborhood", as_index=False)["sale_price"]
        .agg(avg_price="mean", count="count")
        .sort_values("avg_price", ascending=True)
    )
    fig = px.bar(
        nbhd_agg,
        x="avg_price",
        y="neighborhood",
        orientation="h",
        labels={"avg_price": "Average price ($)", "neighborhood": ""},
        title="Average sale price by neighborhood",
        color="avg_price",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.histogram(
        df,
        x="sale_price",
        nbins=40,
        labels={"sale_price": "Sale price ($)", "count": "Count"},
        title="Price distribution",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: trend and quality ──────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    trend = (
        df.groupby("yr_sold", as_index=False)["sale_price"]
        .agg(avg_price="mean", count="count")
    )
    fig3 = px.line(
        trend,
        x="yr_sold",
        y="avg_price",
        markers=True,
        labels={"yr_sold": "Year", "avg_price": "Average price ($)"},
        title="Average price by year sold",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    qual_agg = (
        df.groupby("overall_qual", as_index=False)["sale_price"].mean()
    )
    fig4 = px.bar(
        qual_agg,
        x="overall_qual",
        y="sale_price",
        labels={"overall_qual": "Overall quality (1–10)", "sale_price": "Average price ($)"},
        title="Average price by quality rating",
        color="sale_price",
        color_continuous_scale="Blues",
    )
    fig4.update_layout(coloraxis_showscale=False, height=350)
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: scatter ────────────────────────────────────────────────────────────
st.subheader("Price vs. living area")
fig5 = px.scatter(
    df,
    x="gr_liv_area",
    y="sale_price",
    color="overall_qual",
    hover_data=["neighborhood", "year_built", "yr_sold"],
    labels={
        "gr_liv_area": "Above-grade living area (sq ft)",
        "sale_price": "Sale price ($)",
        "overall_qual": "Quality",
    },
    title="Sale price vs. living area, colored by overall quality",
    color_continuous_scale="Blues",
    opacity=0.6,
)
fig5.update_layout(height=420)
st.plotly_chart(fig5, use_container_width=True)
