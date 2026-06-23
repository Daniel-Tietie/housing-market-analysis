"""
Housing Market Analysis Dashboard
Four pages: Overview, Geographic, Trends, Risk/Volatility
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "data" / "housing.db"

# ── Design constants ──────────────────────────────────────────────────────────
PRIMARY   = "#2b6cb0"
ACCENT    = "#4299e1"
LIGHT     = "#bee3f8"
GRAY      = "#718096"
BG        = "#f7fafc"
FONT      = "Inter, Segoe UI, sans-serif"

PLOTLY_LAYOUT = dict(
    font_family=FONT,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=12, r=12, t=40, b=12),
    coloraxis_colorbar=dict(thickness=12),
)

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

st.set_page_config(
    page_title="Housing Market Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    html, body, [class*="css"] {{ font-family: {FONT}; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; }}
    /* Let vertical scroll gestures pass through chart elements on touch
       devices instead of being captured as a chart zoom/pan/select. */
    .js-plotly-plot, .plotly, .main-svg {{ touch-action: pan-y !important; }}
    .metric-card {{
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .metric-label {{ color: {GRAY}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }}
    .metric-value {{ color: #1a202c; font-size: 1.6rem; font-weight: 600; }}
    .metric-delta {{ font-size: 0.78rem; margin-top: 0.2rem; }}
    .section-header {{ color: #2d3748; font-size: 1rem; font-weight: 600; margin: 1rem 0 0.5rem; }}
    div[data-testid="stMetric"] label {{ font-size: 0.78rem !important; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM clean_housing", conn)
    conn.close()
    df["month_label"] = df["mo_sold"].map(MONTH_LABELS)
    df["price_per_sqft"] = df["sale_price"] / df["gr_liv_area"].replace(0, np.nan)
    return df


df_all = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Housing Market Analysis")
    st.caption("Ames, Iowa — 2006 to 2010")
    st.divider()

    page = st.radio(
        "Page",
        ["Overview", "Geographic", "Trends", "Risk & Volatility"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Filters**")

    years = sorted(df_all["yr_sold"].unique())
    yr_range = st.slider(
        "Year sold",
        int(min(years)), int(max(years)),
        (int(min(years)), int(max(years))),
    )

    bldg_types = sorted(df_all["bldg_type"].dropna().unique())
    sel_bldg = st.multiselect("Building type", bldg_types, default=bldg_types)

    qual_range = st.slider("Overall quality (1–10)", 1, 10, (1, 10))

    st.divider()
    st.caption("Data: De Cock (2011) / OpenML 42165")

df = df_all[
    df_all["yr_sold"].between(*yr_range)
    & df_all["bldg_type"].isin(sel_bldg)
    & df_all["overall_qual"].between(*qual_range)
].copy()

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()


# ── Shared helpers ────────────────────────────────────────────────────────────
def fmt_price(v): return f"${v:,.0f}"
def fmt_delta(curr, prev):
    if prev == 0:
        return ""
    pct = (curr - prev) / prev * 100
    sign = "+" if pct >= 0 else ""
    color = "#276749" if pct >= 0 else "#c53030"
    return f'<span style="color:{color}">{sign}{pct:.1f}% vs prior year</span>'

def card(label, value, delta_html=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta_html}</div>
    </div>
    """, unsafe_allow_html=True)

def clean_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f0f4f8", linecolor="#e2e8f0")
    return fig


def show_chart(fig):
    """Render a Plotly figure with touch-drag interactions disabled, so
    accidental finger contact while scrolling on mobile doesn't trigger
    zoom/pan/select on the chart."""
    fig.update_layout(dragmode=False)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False},
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## Overview")
    st.caption(f"Showing **{len(df):,}** listings · {yr_range[0]}–{yr_range[1]}")

    # Key metrics row
    avg_price   = df["sale_price"].mean()
    med_price   = df["sale_price"].median()
    total       = len(df)
    avg_sqft    = df["gr_liv_area"].mean()
    avg_ppsf    = df["price_per_sqft"].mean()
    avg_qual    = df["overall_qual"].mean()

    # Year-over-year delta for avg price (last two years in filter)
    yr_trend = df.groupby("yr_sold")["sale_price"].mean()
    if len(yr_trend) >= 2:
        last_yr, prev_yr = yr_trend.iloc[-1], yr_trend.iloc[-2]
        delta_html = fmt_delta(last_yr, prev_yr)
    else:
        delta_html = ""

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: card("Total listings",    f"{total:,}")
    with c2: card("Average price",     fmt_price(avg_price), delta_html)
    with c3: card("Median price",      fmt_price(med_price))
    with c4: card("Avg living area",   f"{avg_sqft:,.0f} sq ft")
    with c5: card("Avg price / sq ft", fmt_price(avg_ppsf))
    with c6: card("Avg quality",       f"{avg_qual:.1f} / 10")

    st.divider()

    # Price distribution + cumulative
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Sale price distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="sale_price", nbins=45,
            labels={"sale_price": "Sale price ($)", "count": "Listings"},
            color_discrete_sequence=[PRIMARY],
        )
        fig.add_vline(x=avg_price, line_dash="dash", line_color=GRAY,
                      annotation_text=f"Mean {fmt_price(avg_price)}", annotation_position="top right")
        fig.add_vline(x=med_price, line_dash="dot", line_color=ACCENT,
                      annotation_text=f"Median {fmt_price(med_price)}", annotation_position="top left")
        show_chart(clean_fig(fig))

    with col2:
        st.markdown('<div class="section-header">Price by overall quality</div>', unsafe_allow_html=True)
        qual_agg = df.groupby("overall_qual").agg(
            avg_price=("sale_price", "mean"),
            count=("sale_price", "count")
        ).reset_index()
        fig2 = px.bar(
            qual_agg, x="overall_qual", y="avg_price",
            text=qual_agg["avg_price"].apply(lambda v: f"${v/1000:.0f}k"),
            labels={"overall_qual": "Overall quality (1–10)", "avg_price": "Average price ($)"},
            color="avg_price", color_continuous_scale=[[0, LIGHT], [1, PRIMARY]],
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False)
        show_chart(clean_fig(fig2))

    # Price by year + building type breakdown
    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown('<div class="section-header">Average price by year sold</div>', unsafe_allow_html=True)
        yr_agg = df.groupby("yr_sold").agg(
            avg_price=("sale_price", "mean"),
            count=("sale_price", "count"),
        ).reset_index()
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(
            x=yr_agg["yr_sold"], y=yr_agg["count"],
            name="Listings", marker_color=LIGHT, opacity=0.7,
        ), secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=yr_agg["yr_sold"], y=yr_agg["avg_price"],
            name="Avg price", mode="lines+markers",
            line=dict(color=PRIMARY, width=2.5), marker=dict(size=7),
        ), secondary_y=True)
        fig3.update_yaxes(title_text="Listing volume", secondary_y=False, showgrid=False)
        fig3.update_yaxes(title_text="Average price ($)", secondary_y=True, gridcolor="#f0f4f8")
        fig3.update_xaxes(tickvals=yr_agg["yr_sold"], showgrid=False)
        fig3.update_layout(
            legend=dict(orientation="h", y=1.12),
            **PLOTLY_LAYOUT,
        )
        show_chart(fig3)

    with col4:
        st.markdown('<div class="section-header">Listings by building type</div>', unsafe_allow_html=True)
        bt_agg = df.groupby("bldg_type").agg(
            count=("sale_price", "count"),
            avg_price=("sale_price", "mean"),
        ).reset_index().sort_values("count", ascending=False)
        fig4 = px.pie(
            bt_agg, values="count", names="bldg_type",
            color_discrete_sequence=[PRIMARY, ACCENT, LIGHT, "#2c5282", "#63b3ed"],
            hole=0.45,
        )
        fig4.update_traces(textinfo="percent+label", textfont_size=11)
        fig4.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        show_chart(fig4)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GEOGRAPHIC (Neighborhood)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Geographic":
    st.markdown("## Geographic — Price by Neighborhood")
    st.caption(f"{len(df):,} listings · {yr_range[0]}–{yr_range[1]}")

    nbhd_agg = (
        df.groupby("neighborhood")
        .agg(
            listings=("sale_price", "count"),
            avg_price=("sale_price", "mean"),
            median_price=("sale_price", "median"),
            avg_sqft=("gr_liv_area", "mean"),
            avg_ppsf=("price_per_sqft", "mean"),
            avg_qual=("overall_qual", "mean"),
        )
        .reset_index()
        .sort_values("avg_price", ascending=False)
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Average sale price by neighborhood</div>', unsafe_allow_html=True)
        fig = px.bar(
            nbhd_agg.sort_values("avg_price", ascending=True),
            x="avg_price", y="neighborhood", orientation="h",
            text=nbhd_agg.sort_values("avg_price", ascending=True)["avg_price"].apply(lambda v: f"${v/1000:.0f}k"),
            labels={"avg_price": "Average price ($)", "neighborhood": ""},
            color="avg_price",
            color_continuous_scale=[[0, LIGHT], [1, PRIMARY]],
        )
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(coloraxis_showscale=False, height=620, **PLOTLY_LAYOUT)
        show_chart(fig)

    with col2:
        st.markdown('<div class="section-header">Price range by neighborhood (top 15)</div>', unsafe_allow_html=True)
        top15 = nbhd_agg.head(15)
        box_df = df[df["neighborhood"].isin(top15["neighborhood"])]
        fig2 = px.box(
            box_df.sort_values("neighborhood"),
            x="neighborhood", y="sale_price",
            labels={"neighborhood": "", "sale_price": "Sale price ($)"},
            color_discrete_sequence=[PRIMARY],
        )
        fig2.update_layout(
            height=620,
            xaxis_tickangle=-45,
            **PLOTLY_LAYOUT,
        )
        show_chart(fig2)

    st.divider()

    # Neighborhood comparison table + scatter
    col3, col4 = st.columns([2, 3])

    with col3:
        st.markdown('<div class="section-header">Neighborhood summary</div>', unsafe_allow_html=True)
        tbl = nbhd_agg.copy()
        tbl["avg_price"]   = tbl["avg_price"].apply(fmt_price)
        tbl["median_price"] = tbl["median_price"].apply(fmt_price)
        tbl["avg_ppsf"]    = tbl["avg_ppsf"].apply(lambda v: f"${v:.0f}")
        tbl["avg_qual"]    = tbl["avg_qual"].apply(lambda v: f"{v:.1f}")
        tbl = tbl.rename(columns={
            "neighborhood": "Neighborhood",
            "listings": "Listings",
            "avg_price": "Avg price",
            "median_price": "Median",
            "avg_ppsf": "$/sq ft",
            "avg_qual": "Avg quality",
        }).drop(columns=["avg_sqft"])
        st.dataframe(tbl, use_container_width=True, height=420, hide_index=True)

    with col4:
        st.markdown('<div class="section-header">Quality vs. price by neighborhood</div>', unsafe_allow_html=True)
        fig3 = px.scatter(
            nbhd_agg,
            x="avg_qual", y="avg_price",
            size="listings", text="neighborhood",
            color="avg_price",
            color_continuous_scale=[[0, LIGHT], [1, PRIMARY]],
            labels={
                "avg_qual": "Average quality rating",
                "avg_price": "Average sale price ($)",
                "listings": "Listing count",
            },
            size_max=40,
        )
        fig3.update_traces(textposition="top center", textfont_size=9)
        fig3.update_layout(coloraxis_showscale=False, height=420, **PLOTLY_LAYOUT)
        show_chart(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trends":
    st.markdown("## Trends")
    st.caption(f"{len(df):,} listings · {yr_range[0]}–{yr_range[1]}")

    # Year-over-year trend
    st.markdown('<div class="section-header">Price and volume by year</div>', unsafe_allow_html=True)
    yr_agg = df.groupby("yr_sold").agg(
        count=("sale_price", "count"),
        avg_price=("sale_price", "mean"),
        median_price=("sale_price", "median"),
        avg_ppsf=("price_per_sqft", "mean"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=yr_agg["yr_sold"], y=yr_agg["count"],
        name="Listings", marker_color=LIGHT, opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=yr_agg["yr_sold"], y=yr_agg["avg_price"],
        name="Avg price", mode="lines+markers",
        line=dict(color=PRIMARY, width=2.5), marker=dict(size=8),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=yr_agg["yr_sold"], y=yr_agg["median_price"],
        name="Median price", mode="lines+markers",
        line=dict(color=ACCENT, width=2, dash="dot"), marker=dict(size=7),
    ), secondary_y=True)
    fig.update_yaxes(title_text="Listing volume", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Price ($)", secondary_y=True, gridcolor="#f0f4f8")
    fig.update_xaxes(tickvals=yr_agg["yr_sold"], showgrid=False)
    fig.update_layout(legend=dict(orientation="h", y=1.12), **PLOTLY_LAYOUT)
    show_chart(fig)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Seasonal patterns — month sold</div>', unsafe_allow_html=True)
        mo_agg = df.groupby("mo_sold").agg(
            count=("sale_price", "count"),
            avg_price=("sale_price", "mean"),
        ).reset_index()
        mo_agg["month"] = mo_agg["mo_sold"].map(MONTH_LABELS)

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(
            x=mo_agg["month"], y=mo_agg["count"],
            name="Listings", marker_color=LIGHT, opacity=0.8,
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=mo_agg["month"], y=mo_agg["avg_price"],
            name="Avg price", mode="lines+markers",
            line=dict(color=PRIMARY, width=2.5), marker=dict(size=7),
        ), secondary_y=True)
        fig2.update_yaxes(title_text="Listings", secondary_y=False, showgrid=False)
        fig2.update_yaxes(title_text="Avg price ($)", secondary_y=True, gridcolor="#f0f4f8")
        fig2.update_xaxes(showgrid=False)
        fig2.update_layout(legend=dict(orientation="h", y=1.12), **PLOTLY_LAYOUT)
        show_chart(fig2)

    with col2:
        st.markdown('<div class="section-header">Price per sq ft by year</div>', unsafe_allow_html=True)
        fig3 = px.line(
            yr_agg, x="yr_sold", y="avg_ppsf",
            markers=True,
            labels={"yr_sold": "Year", "avg_ppsf": "Avg price per sq ft ($)"},
            color_discrete_sequence=[PRIMARY],
        )
        fig3.update_traces(line_width=2.5, marker_size=8)
        fig3.update_xaxes(tickvals=yr_agg["yr_sold"])
        show_chart(clean_fig(fig3))

    st.divider()

    # Decade built analysis
    st.markdown('<div class="section-header">Price by decade built</div>', unsafe_allow_html=True)
    df["decade"] = (df["year_built"] // 10) * 10
    decade_agg = df.groupby("decade").agg(
        count=("sale_price", "count"),
        avg_price=("sale_price", "mean"),
        avg_ppsf=("price_per_sqft", "mean"),
    ).reset_index()

    col3, col4 = st.columns(2)
    with col3:
        fig4 = px.bar(
            decade_agg, x="decade", y="avg_price",
            text=decade_agg["avg_price"].apply(lambda v: f"${v/1000:.0f}k"),
            labels={"decade": "Decade built", "avg_price": "Average sale price ($)"},
            color="avg_price", color_continuous_scale=[[0, LIGHT], [1, PRIMARY]],
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        show_chart(fig4)

    with col4:
        fig5 = px.bar(
            decade_agg, x="decade", y="avg_ppsf",
            text=decade_agg["avg_ppsf"].apply(lambda v: f"${v:.0f}"),
            labels={"decade": "Decade built", "avg_ppsf": "Avg price per sq ft ($)"},
            color="avg_ppsf", color_continuous_scale=[[0, LIGHT], [1, PRIMARY]],
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        show_chart(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK & VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk & Volatility":
    st.markdown("## Risk & Volatility")
    st.caption(
        "Volatility measured as standard deviation and coefficient of variation (CV = StdDev / Mean) "
        "per neighborhood. High CV = wide price spread relative to the average."
    )
    st.caption(f"{len(df):,} listings · {yr_range[0]}–{yr_range[1]}")

    risk_agg = (
        df.groupby("neighborhood")
        .agg(
            count=("sale_price", "count"),
            avg_price=("sale_price", "mean"),
            std_price=("sale_price", "std"),
            min_price=("sale_price", "min"),
            max_price=("sale_price", "max"),
            avg_ppsf=("price_per_sqft", "mean"),
        )
        .reset_index()
    )
    risk_agg["cv"] = risk_agg["std_price"] / risk_agg["avg_price"]
    risk_agg["price_range"] = risk_agg["max_price"] - risk_agg["min_price"]
    risk_agg = risk_agg[risk_agg["count"] >= 5].sort_values("cv", ascending=False)

    # Headline metrics
    most_volatile  = risk_agg.iloc[0]
    least_volatile = risk_agg.iloc[-1]
    highest_price  = risk_agg.sort_values("avg_price", ascending=False).iloc[0]
    lowest_price   = risk_agg.sort_values("avg_price").iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Most volatile neighborhood",  most_volatile["neighborhood"],
                  f'CV = {most_volatile["cv"]:.2f}')
    with c2: card("Least volatile neighborhood", least_volatile["neighborhood"],
                  f'CV = {least_volatile["cv"]:.2f}')
    with c3: card("Highest avg price",           highest_price["neighborhood"],
                  fmt_price(highest_price["avg_price"]))
    with c4: card("Lowest avg price",            lowest_price["neighborhood"],
                  fmt_price(lowest_price["avg_price"]))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Price volatility by neighborhood (CV)</div>', unsafe_allow_html=True)
        fig = px.bar(
            risk_agg.sort_values("cv", ascending=True),
            x="cv", y="neighborhood", orientation="h",
            text=risk_agg.sort_values("cv", ascending=True)["cv"].apply(lambda v: f"{v:.2f}"),
            labels={"cv": "Coefficient of variation", "neighborhood": ""},
            color="cv",
            color_continuous_scale=[[0, LIGHT], [0.5, ACCENT], [1, "#c53030"]],
        )
        fig.update_traces(textposition="outside", textfont_size=9)
        fig.update_layout(coloraxis_showscale=False, height=600, **PLOTLY_LAYOUT)
        show_chart(fig)

    with col2:
        st.markdown('<div class="section-header">Price std dev vs. avg price</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            risk_agg,
            x="avg_price", y="std_price",
            size="count", text="neighborhood",
            color="cv",
            color_continuous_scale=[[0, LIGHT], [0.5, ACCENT], [1, "#c53030"]],
            labels={
                "avg_price": "Average price ($)",
                "std_price": "Std deviation ($)",
                "count": "Listings",
                "cv": "CV",
            },
            size_max=35,
        )
        fig2.update_traces(textposition="top center", textfont_size=8)
        fig2.update_layout(height=600, **PLOTLY_LAYOUT)
        fig2.update_layout(coloraxis_colorbar=dict(title="CV", thickness=12))
        show_chart(fig2)

    st.divider()

    # Price range waterfall per neighborhood
    st.markdown('<div class="section-header">Min / max price range by neighborhood</div>', unsafe_allow_html=True)
    range_df = risk_agg.sort_values("avg_price", ascending=False).head(20)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Min price",
        x=range_df["neighborhood"],
        y=range_df["min_price"],
        marker_color=LIGHT,
    ))
    fig3.add_trace(go.Bar(
        name="Price range (min → max)",
        x=range_df["neighborhood"],
        y=range_df["price_range"],
        marker_color=PRIMARY,
        opacity=0.75,
    ))
    fig3.update_layout(
        barmode="stack",
        xaxis_tickangle=-40,
        yaxis_title="Sale price ($)",
        legend=dict(orientation="h", y=1.1),
        **PLOTLY_LAYOUT,
    )
    show_chart(clean_fig(fig3))

    # Full risk table
    st.markdown('<div class="section-header">Full risk table</div>', unsafe_allow_html=True)
    tbl = risk_agg.copy().sort_values("cv", ascending=False)
    tbl["avg_price"]   = tbl["avg_price"].apply(fmt_price)
    tbl["std_price"]   = tbl["std_price"].apply(fmt_price)
    tbl["min_price"]   = tbl["min_price"].apply(fmt_price)
    tbl["max_price"]   = tbl["max_price"].apply(fmt_price)
    tbl["price_range"] = tbl["price_range"].apply(fmt_price)
    tbl["cv"]          = tbl["cv"].apply(lambda v: f"{v:.3f}")
    tbl = tbl.rename(columns={
        "neighborhood": "Neighborhood", "count": "Listings",
        "avg_price": "Avg price", "std_price": "Std dev",
        "min_price": "Min", "max_price": "Max",
        "price_range": "Range", "cv": "CV", "avg_ppsf": "$/sq ft",
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
