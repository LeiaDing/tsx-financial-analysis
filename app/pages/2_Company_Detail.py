"""
2_Company_Detail.py — 公司详情页
"""

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Company Detail", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PANEL = BASE_DIR / "data" / "processed" / "panel_data.csv"
MDA_DIR = BASE_DIR / "data" / "processed" / "mda_texts"


@st.cache_data
def load_data():
    return pd.read_csv(PANEL)


def find_mda_file(company: str) -> Path | None:
    slug = re.sub(r"[^\w]+", "_", company).strip("_")
    for f in MDA_DIR.glob("*.txt"):
        if f.stem.lower() == slug.lower():
            return f
    first_word = slug.split("_")[0].lower()
    for f in MDA_DIR.glob("*.txt"):
        if f.stem.lower().startswith(first_word):
            return f
    return None


def read_mda_excerpt(company: str, n_chars: int = 2000) -> str:
    f = find_mda_file(company)
    if f is None:
        return "_MD&A text not available for this company._"
    text = f.read_text(errors="ignore")
    lines = [l for l in text.splitlines() if len(l.strip()) > 40]
    excerpt = "\n\n".join(lines[:30])[:n_chars]
    return excerpt + "…"


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 TSX Macro Lens")
    st.divider()
    sectors = sorted(df["sector"].dropna().unique())
    sel_sector = st.selectbox("Sector", ["All"] + sectors)

    if sel_sector == "All":
        companies = sorted(df["company"].dropna().unique())
    else:
        companies = sorted(df[df["sector"] == sel_sector]["company"].dropna().unique())

    sel_company = st.selectbox("Company", companies)

# ── Header ────────────────────────────────────────────────────────────────────
co = df[df["company"] == sel_company].sort_values("year")

if co.empty:
    st.warning("No data for this company.")
    st.stop()

ticker = co["ticker"].iloc[0]
sector = co["sector"].iloc[0]

st.title("🏢 Company Detail")
st.subheader(f"{sel_company}  `{ticker}`")
st.caption(f"Sector: {sector}  ·  {int(co['year'].min())}–{int(co['year'].max())}  ·  {len(co)} year(s) of data")

st.divider()

# ── Financial charts (2×2) ────────────────────────────────────────────────────
st.markdown("#### Financial Performance")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Revenue (CAD B)",
        "Net Income (CAD B)",
        "Net Margin (%)",
        "CapEx (CAD B)",
    ),
    vertical_spacing=0.18,
    horizontal_spacing=0.1,
)

years = co["year"].tolist()

fig.add_trace(go.Scatter(
    x=years, y=co["revenue"] / 1e9, mode="lines+markers",
    line=dict(color="#1f77b4", width=2), marker=dict(size=7),
    showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(
    x=years, y=co["net_income"] / 1e9, mode="lines+markers",
    line=dict(color="#2ca02c", width=2), marker=dict(size=7),
    showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(
    x=years, y=co["net_margin"] * 100, mode="lines+markers",
    line=dict(color="#ff7f0e", width=2), marker=dict(size=7),
    showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(
    x=years, y=co["capex"].abs() / 1e9, mode="lines+markers",
    line=dict(color="#d62728", width=2), marker=dict(size=7),
    showlegend=False), row=2, col=2)

fig.update_layout(height=460, margin=dict(t=50, b=20))
fig.update_xaxes(tickformat="d")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Sentiment score ───────────────────────────────────────────────────────────
st.markdown("#### MD&A Sentiment (FinBERT)")

sent_row = co.dropna(subset=["sentiment_score"])
if not sent_row.empty:
    score = sent_row["sentiment_score"].iloc[0]
    pos   = sent_row["sentiment_pos"].iloc[0]
    neg   = sent_row["sentiment_neg"].iloc[0]
    neu   = sent_row["sentiment_neu"].iloc[0]

    # Sector average for delta comparison
    sector_avg = df[df["sector"] == sector]["sentiment_score"].mean()
    delta_val  = score - sector_avg

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Sentiment Score",
        f"{score:.3f}",
        delta=f"{delta_val:+.3f} vs sector avg",
        delta_color="normal",
        help="FinBERT positive − negative probability. Positive = optimistic tone.",
    )
    c2.metric("Positive", f"{pos:.1%}")
    c3.metric("Negative", f"{neg:.1%}")
    c4.metric("Neutral",  f"{neu:.1%}")

    st.caption(f"Sector average sentiment: {sector_avg:.3f}  ·  Scored on first 20 MD&A chunks")
else:
    st.info("Sentiment score not available for this company.")

st.divider()

# ── MD&A excerpt ──────────────────────────────────────────────────────────────
st.markdown("#### MD&A Key Excerpt")
st.caption("Source: SEDAR+ annual report · first ~2,000 characters of MD&A body text")

excerpt = read_mda_excerpt(sel_company)
st.markdown(
    f'<div style="background:#f8f9fa; padding:20px; border-radius:8px; '
    f'border-left:4px solid #1f77b4; font-size:0.87rem; line-height:1.7; '
    f'max-height:380px; overflow-y:auto; font-family:Georgia,serif;">'
    f'{excerpt.replace(chr(10), "<br>")}</div>',
    unsafe_allow_html=True,
)

st.divider()
st.markdown(
    "💡 **Want to ask questions about this company's MD&A?** "
    "Head to the **🔍 RAG Q&A** page.",
    help="RAG Q&A uses vector search + LLM to answer natural language questions."
)
