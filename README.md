# TSX Macro Lens — Do Corporate Disclosures Predict Canada's Macro Trends?

> **Live Demo:** [tsx-financial-analysis.streamlit.app](https://tsx-financial-analysis.streamlit.app)

An end-to-end data pipeline and interactive dashboard that combines structured financial data, unstructured MD&A text, and macroeconomic indicators to explore whether TSX-listed companies' disclosures contain leading signals for GDP growth and CPI.

---

## Research Question

> *Do the language patterns in corporate MD&A filings — sentiment, forward-looking statements, sector-specific risk disclosures — move ahead of measurable shifts in GDP and inflation?*

This project operationalizes that question into a full analytics stack: data ingestion → NLP → vector embeddings → RAG Q&A → interactive dashboard.

---

## What's Inside

### 4 Interactive Dashboard Pages

| Page | What It Shows |
|------|--------------|
| **Industry Overview** | Revenue & margin trends by sector, GDP/CPI dual-axis overlay, financial health heatmap |
| **Company Detail** | Per-company financial KPIs over time, FinBERT sentiment score, MD&A excerpts |
| **Sentiment Dashboard** | Cross-sector sentiment bar chart, word clouds from actual filings |
| **RAG Q&A** | Ask natural-language questions about any company's MD&A; powered by Groq (Llama 3.3 70B) with macro-data tool-calling and multi-turn conversation memory |

### Dataset

- **50 TSX-listed companies** across **8 GICS sectors** (Energy, Financials, Technology, Consumer, Industrials, Materials, Real Estate, …)
- **36 English MD&A filings** parsed from SEDAR+ PDFs
- **Structured financials** (2022–2024): revenue, net income, capex, margins via `yfinance`
- **Macro indicators**: Canada GDP growth + CPI via FRED API
- **Panel dataset**: 196 rows × 18 columns (`data/processed/panel_data.csv`)

---

## Tech Stack

```
Data Ingestion     yfinance · pdfminer.six · FRED API
NLP / Embeddings   HuggingFace all-MiniLM-L6-v2 · FinBERT sentiment scoring
Vector Search      NumPy cosine similarity (8,802 chunks × 384 dims)
LLM / RAG          Groq API · Llama-3.3-70B · tool execution layer
Memory             SQLite (long-term) + sliding window (short-term)
Frontend           Streamlit · Plotly · WordCloud
```

---

## Architecture

```
SEDAR+ PDFs → pdfminer → data/processed/mda_texts/
yfinance    → annual_financials.csv
FRED API    → macro indicators
                  ↓
         panel_data.csv (196 × 18)
                  ↓
    HuggingFace embeddings (8,802 chunks)
                  ↓
    Streamlit Dashboard (4 pages)
         └── RAG Q&A: Groq + Llama-3.3-70B
              ├── Vector retrieval (MD&A context)
              ├── Tool: macro indicator lookup
              └── SQLite conversation memory
```

---

## Exploratory Observations

*Computed directly from the panel data — explore them yourself in the dashboard:*

- **Tone diverges from fundamentals.** Real Estate filings carry the most positive MD&A sentiment of any sector (mean FinBERT score +0.58), yet the sector posts the *lowest* median net margin in 2024 (−9.8%). Management tone is not a proxy for financial health — which is exactly why reading both signals together matters.
- **Energy talks cautiously, earns strongly.** Energy MD&As score near-neutral in sentiment (+0.04) while delivering top-quartile margins (11% median net margin in 2024).
- **Financial Services pairs confident language with the strongest profitability** (15.7% median net margin) — the only sector where tone and fundamentals clearly agree.

### Limitations (read before over-interpreting)

With 36 filings concentrated in fiscal 2021–2023 and annual reporting frequency, this dataset **cannot establish lead–lag relationships** between disclosure language and macro outcomes — the time dimension is too short and the sample too small for that inference. The project is built as a *methodology demonstration*: the pipeline scales directly to longer filing histories and quarterly frequency, which is the natural next step.

---

## Local Setup

```bash
git clone https://github.com/LeiaDing/tsx-financial-analysis.git
cd tsx-financial-analysis
pip install -r requirements.txt

# Set your Groq API key (RAG Q&A page)
export GROQ_API_KEY=gsk_...

streamlit run app/Home.py
```

> **Note:** Processed data (`panel_data.csv`, embeddings, parsed MD&A texts) is included in the repo, so the dashboard runs out of the box. Raw SEDAR+ PDFs (~105 MB) are excluded.

---

## Project Structure

```
tsx-financial-analysis/
├── app/
│   ├── Home.py                          # Landing page
│   └── pages/
│       ├── 1_Industry_Overview.py
│       ├── 2_Company_Detail.py
│       ├── 3_Sentiment_Dashboard.py
│       └── 4_RAG_QA.py
├── src/
│   ├── 1_fetch_financial_data.py        # yfinance data pull
│   ├── 3_parse_mda_pdfs.py              # pdfminer extraction
│   ├── 4_fetch_macro_data_fred.py       # FRED API
│   ├── 5_clean_financials.py            # pandas cleaning
│   ├── 6_embed_mda_texts.py             # HuggingFace embeddings
│   ├── 7_tool_executor.py               # macro tool calls
│   ├── 8_memory.py                      # SQLite + sliding window
│   ├── 9_rag_query.py                   # RAG engine
│   ├── 10_extract_sentiment.py          # FinBERT scoring
│   └── 11_build_panel_data.py           # merge all sources
├── data/processed/
│   ├── panel_data.csv                   # 196 × 18 panel
│   ├── embeddings/                      # 8,802 × 384 vectors + metadata
│   └── mda_texts/                       # 36 parsed filings
└── requirements.txt
```

---

## About

Built as a portfolio project demonstrating full-stack data engineering + NLP + LLM integration, targeting applied AI/data roles in financial services.

**Author:** Leia Ding · [LinkedIn](https://linkedin.com/in/leia-ding) · leiasxufe@gmail.com
