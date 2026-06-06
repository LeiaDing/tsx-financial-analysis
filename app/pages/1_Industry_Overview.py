"""
1_Industry_Overview.py — 行业概览页
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Industry Overview", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PANEL = BASE_DIR / "data" / "processed" / "panel_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(PANEL)
    df["revenue_bn"] = df["revenue"] / 1e9
    df["net_income_bn"] = df["net_income"] / 1e9
    df["net_margin_pct"] = df["net_margin"] * 100
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 TSX Macro Lens")
    st.divider()
    sectors = sorted(df["sector"].dropna().unique())
    selected_sectors = st.multiselect("Select sectors", sectors, default=sectors)
    year_range = st.slider(
        "Year range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
    )
    st.divider()
    st.caption(f"{len(selected_sectors)} of {len(sectors)} sectors · {year_range[0]}–{year_range[1]}")

filtered = df[
    df["sector"].isin(selected_sectors) &
    df["year"].between(year_range[0], year_range[1])
]

st.title("🏭 Industry Overview")
st.caption(f"{filtered['company'].nunique()} companies · {filtered['year'].nunique()} years · {len(selected_sectors)} sectors")

# ── Chart 1: Revenue trend ─────────────────────────────────────────────────────
st.subheader("Revenue by Sector (CAD Billions)")

rev_agg = (
    filtered.groupby(["sector", "year"])["revenue_bn"]
    .sum()
    .reset_index()
)
fig1 = px.line(
    rev_agg, x="year", y="revenue_bn", color="sector",
    markers=True,
    labels={"revenue_bn": "Revenue (CAD B)", "year": "Year", "sector": "Sector"},
)
fig1.update_layout(hovermode="x unified", legend_title="Sector", height=380)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Aggregate revenue per sector. Energy dominates in absolute size; Tech companies show steeper per-company growth rates.")

# ── Chart 2: Net margin distribution ─────────────────────────────────────────
st.subheader("Net Margin Distribution by Sector")

fig2 = px.box(
    filtered.dropna(subset=["net_margin_pct"]),
    x="sector", y="net_margin_pct", color="sector",
    points="all",
    hover_data=["company", "year"],
    labels={"net_margin_pct": "Net Margin (%)", "sector": "Sector"},
)
fig2.update_layout(showlegend=False, xaxis_tickangle=-30, height=400)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Each dot is one company-year. Wide boxes indicate high margin variability within the sector. Hover to identify outliers.")

# ── Chart 3: Sentiment vs GDP ─────────────────────────────────────────────────
st.subheader("MD&A Sentiment Score vs GDP Growth")

sent_df = filtered.dropna(subset=["sentiment_score", "gdp_yoy"])
if sent_df.empty:
    st.info("No sentiment + macro overlap in the selected filters.")
else:
    fig3 = px.scatter(
        sent_df,
        x="sentiment_score", y="gdp_yoy",
        color="sector", symbol="sector",
        hover_data=["company", "year"],
        trendline="ols",
        labels={
            "sentiment_score": "MD&A Sentiment Score (pos − neg)",
            "gdp_yoy": "Canada GDP YoY Growth (%)",
        },
    )
    fig3.update_layout(legend_title="Sector", height=420)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Each point = one company in one year. OLS trend line fitted per sector. "
        "A positive slope suggests management language becomes more optimistic when GDP grows. "
        "GDP from FRED (Statistics Canada · NGDPRSAXDCCAQ)."
    )

# ── Chart 4: Sentiment trend ──────────────────────────────────────────────────
st.subheader("Average MD&A Sentiment by Sector")

sent_trend = (
    filtered.dropna(subset=["sentiment_score"])
    .groupby(["sector", "year"])["sentiment_score"]
    .mean()
    .reset_index()
)
if sent_trend.empty:
    st.info("No sentiment data in the selected filters.")
else:
    fig4 = px.line(
        sent_trend, x="year", y="sentiment_score", color="sector",
        markers=True,
        labels={
            "sentiment_score": "Avg Sentiment Score",
            "year": "Year",
            "sector": "Sector",
        },
    )
    fig4.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Neutral")
    fig4.update_layout(hovermode="x unified", height=380)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Sector-average FinBERT sentiment (positive − negative probability). Above zero = net positive tone in annual disclosures.")

# ── Chart 5: GDP / CPI Macro Overlay ─────────────────────────────────────────
st.subheader("Canada Macro Context: GDP & CPI Growth")

macro_df = (
    df[["year", "gdp_yoy", "cpi_yoy_annual"]]
    .drop_duplicates()
    .sort_values("year")
    .query("@year_range[0] <= year <= @year_range[1]")
)

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=macro_df["year"], y=macro_df["gdp_yoy"],
    name="GDP YoY (%)", marker_color="steelblue", opacity=0.7,
))
fig5.add_trace(go.Scatter(
    x=macro_df["year"], y=macro_df["cpi_yoy_annual"],
    name="CPI YoY (%)", mode="lines+markers",
    line=dict(color="tomato", width=2),
    yaxis="y2",
))
fig5.update_layout(
    yaxis=dict(title="GDP YoY (%)", titlefont=dict(color="steelblue")),
    yaxis2=dict(
        title="CPI YoY (%)", titlefont=dict(color="tomato"),
        overlaying="y", side="right",
    ),
    legend=dict(x=0.01, y=0.99),
    hovermode="x unified",
    height=380,
)
st.plotly_chart(fig5, use_container_width=True)
st.caption("Blue bars: Canada real GDP growth YoY. Red line: CPI inflation YoY. Source: FRED (NGDPRSAXDCCAQ + CPALCY01CAA661N).")

# ── Chart 6: Financial Health Scorecard ──────────────────────────────────────
st.subheader("Industry Financial Health Score")

score_df = filtered.copy()
score_df["net_margin_rank"] = score_df.groupby("year")["net_margin"].rank(pct=True)
score_df["capex_rank"] = score_df.groupby("year")["capex_to_revenue"].rank(pct=True)
score_df["sent_rank"] = score_df.groupby("year")["sentiment_score"].rank(pct=True).fillna(0.5)
score_df["health_score"] = (
    score_df["net_margin_rank"] * 40
    + score_df["capex_rank"] * 20
    + score_df["sent_rank"] * 40
)

health_agg = score_df.groupby(["sector", "year"])["health_score"].mean().reset_index()
pivot = health_agg.pivot(index="sector", columns="year", values="health_score")

fig6 = px.imshow(
    pivot,
    color_continuous_scale="RdYlGn",
    labels=dict(color="Score (0–100)"),
    text_auto=".0f",
    aspect="auto",
)
fig6.update_layout(height=350)
st.plotly_chart(fig6, use_container_width=True)
st.caption(
    "Composite score (0–100): 40% net margin percentile + 20% capex intensity percentile + 40% MD&A sentiment percentile. "
    "Scores are relative within each year's cross-section. Green = stronger; red = weaker."
)
