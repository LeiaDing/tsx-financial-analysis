"""
3_Sentiment_Dashboard.py — MD&A 情绪仪表板
图表：行业情绪对比 / 情绪分布 / 词云
"""

import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="MD&A Sentiment", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PANEL = BASE_DIR / "data" / "processed" / "panel_data.csv"
MDA_DIR = BASE_DIR / "data" / "processed" / "mda_texts"

EXTRA_STOPS = {
    "company", "year", "fiscal", "annual", "report", "financial",
    "million", "billion", "quarter", "period", "ended", "december",
    "march", "june", "september", "management", "discussion", "analysis",
    "including", "also", "may", "will", "used", "based", "provided",
    "total", "increase", "decrease", "compared", "results", "operations",
}


@st.cache_data
def load_data():
    return pd.read_csv(PANEL)


@st.cache_data
def load_all_mda_text(sector: str, df: pd.DataFrame) -> str:
    if sector == "All":
        companies = df["company"].dropna().unique()
    else:
        companies = df[df["sector"] == sector]["company"].dropna().unique()

    texts = []
    for company in companies:
        slug = re.sub(r"[^\w]+", "_", company).strip("_")
        matched = None
        for f in MDA_DIR.glob("*.txt"):
            if f.stem.lower() == slug.lower():
                matched = f
                break
        if matched is None:
            first_word = slug.split("_")[0].lower()
            for f in MDA_DIR.glob("*.txt"):
                if f.stem.lower().startswith(first_word):
                    matched = f
                    break
        if matched:
            texts.append(matched.read_text(errors="ignore"))
    return " ".join(texts)


df = load_data()
sent_df = df.dropna(subset=["sentiment_score"])

st.title("💬 MD&A Sentiment Dashboard")
st.caption(f"FinBERT sentiment scores from {sent_df['company'].nunique()} companies · {len(sent_df)} data points")

# ── Sidebar ───────────────────────────────────────────────────────────────────
sectors = sorted(df["sector"].dropna().unique())
with st.sidebar:
    st.markdown("### 📊 TSX Macro Lens")
    st.divider()
    sel_sectors = st.multiselect("Filter sectors", sectors, default=sectors)

filtered = sent_df[sent_df["sector"].isin(sel_sectors)]

# ── Chart 1: 行业平均情绪对比（横向条形）───────────────────────────────────────
st.subheader("Average Sentiment Score by Sector")

sector_avg = (
    filtered.groupby("sector")["sentiment_score"]
    .mean()
    .reset_index()
    .sort_values("sentiment_score")
)
fig1 = px.bar(
    sector_avg,
    x="sentiment_score", y="sector",
    orientation="h",
    color="sentiment_score",
    color_continuous_scale="RdYlGn",
    labels={"sentiment_score": "Avg Sentiment (pos − neg)", "sector": ""},
)
fig1.add_vline(x=0, line_dash="dot", line_color="gray")
fig1.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=20, b=20))
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: 情绪分 pos/neg/neu 堆叠条形（按公司）─────────────────────────────
st.subheader("Positive / Neutral / Negative Breakdown by Company")

stack_df = (
    filtered.dropna(subset=["sentiment_pos", "sentiment_neg", "sentiment_neu"])
    .groupby("company")[["sentiment_pos", "sentiment_neg", "sentiment_neu"]]
    .mean()
    .reset_index()
    .sort_values("sentiment_pos", ascending=False)
)

fig2 = px.bar(
    stack_df.melt(id_vars="company", var_name="Sentiment", value_name="Score"),
    x="company", y="Score", color="Sentiment",
    color_discrete_map={
        "sentiment_pos": "#2ca02c",
        "sentiment_neg": "#d62728",
        "sentiment_neu": "#aec7e8",
    },
    labels={"company": "", "Score": "Proportion", "Sentiment": ""},
    barmode="stack",
)
fig2.update_layout(
    height=380,
    xaxis_tickangle=-40,
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=40, b=20),
)
fig2.for_each_trace(lambda t: t.update(
    name=t.name.replace("sentiment_pos", "Positive")
               .replace("sentiment_neg", "Negative")
               .replace("sentiment_neu", "Neutral")
))
st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: 词云 ─────────────────────────────────────────────────────────────
st.subheader("MD&A Word Cloud")

wc_sector = st.selectbox("Sector for word cloud", ["All"] + sectors)
text = load_all_mda_text(wc_sector, df)

if text.strip():
    stops = STOPWORDS | EXTRA_STOPS
    wc = WordCloud(
        width=1200, height=500,
        background_color="white",
        stopwords=stops,
        max_words=150,
        colormap="tab10",
        collocations=False,
    ).generate(text)

    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    st.image(buf.getvalue(), use_container_width=True)
    st.caption(f"Top 150 words from MD&A texts · sector: {wc_sector}")
else:
    st.info("No MD&A text available for the selected sector.")
