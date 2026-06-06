"""
Home.py — Streamlit 入口页
"""

import streamlit as st

st.set_page_config(
    page_title="TSX Macro Lens",
    page_icon="📊",
    layout="wide",
)

st.title("📊 TSX Macro Lens")
st.subheader("Do corporate disclosures predict Canada's macroeconomic trends?")

# ── Key metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("TSX Companies", "50", help="Annual financials from yfinance")
c2.metric("Sectors", "11", help="GICS sector classification")
c3.metric("MD&A Documents", "36", help="English annual reports from SEDAR+")
c4.metric("Years Covered", "2022 – 2024", help="Financial + macro data")

st.divider()

# ── Description ───────────────────────────────────────────────────────────────
st.markdown("""
This dashboard investigates whether **MD&A language and financial metrics** from TSX-listed
companies contain leading signals for Canadian GDP and CPI — combining NLP sentiment analysis,
vector search, and interactive data exploration.
""")

# ── Page guide ────────────────────────────────────────────────────────────────
st.markdown("### Pages")

col1, col2 = st.columns(2)

with col1:
    st.info("**🏭 Industry Overview**\nRevenue trends, net margin distributions, MD&A sentiment vs GDP growth by sector.")
    st.info("**💬 MD&A Sentiment**\nFinBERT sentiment scores across industries — bar charts, breakdowns, and word clouds.")

with col2:
    st.info("**🏢 Company Detail**\nPer-company financial trends + FinBERT sentiment scores + MD&A key excerpts.")
    st.success("**🔍 RAG Q&A**\nAsk questions in natural language — retrieves relevant MD&A passages and answers with an LLM.")

st.divider()

# ── Data sources & stack ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Data Sources")
    st.markdown("""
- **Financials**: yfinance — 50 TSX companies, 2022–2024
- **MD&A text**: SEDAR+ — 36 English annual reports
- **Macro**: FRED API — Canada GDP (quarterly) + CPI (monthly), 2018–2024
""")

with col_b:
    st.markdown("#### Tech Stack")
    st.markdown("""
- **NLP**: HuggingFace FinBERT (sentiment) · sentence-transformers (embeddings)
- **Retrieval**: numpy cosine similarity search over FAISS-style vector index
- **LLM**: Groq API (llama-3.3-70b) via RAG pipeline
- **App**: Streamlit · Plotly
""")
