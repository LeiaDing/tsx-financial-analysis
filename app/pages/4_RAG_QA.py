"""
4_RAG_QA.py — RAG 问答页
用户用自然语言提问 → MD&A 向量检索 → Groq LLM 回答
"""

import importlib
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

_te = importlib.import_module("7_tool_executor")
ToolResult     = _te.ToolResult
search_mda     = _te.search_mda
get_financials  = _te.get_financials
get_macro      = _te.get_macro

load_dotenv(BASE_DIR / ".env")

MODEL = "llama-3.3-70b-versatile"
MAX_MDA_CHARS = 3000

SYSTEM_PROMPT = (
    "You are a professional Canadian TSX-listed company analyst. "
    "You have expertise in combining MD&A disclosures, financial data, "
    "and macroeconomic context for comprehensive analysis.\n"
    "Guidelines:\n"
    "1. Prioritize citing the provided source data — do not fabricate numbers\n"
    "2. If data is insufficient, explicitly state what is missing\n"
    "3. Keep conclusions concise and use bullet points\n"
    "4. You may answer in the same language the user writes in (English or Chinese)"
)

EXAMPLES = [
    "What were Shopify's key risk factors in 2023?",
    "How did the macroeconomic environment affect energy sector strategies?",
    "What is Barrick Gold's approach to managing gold price volatility?",
    "Describe Constellation Software's acquisition strategy.",
    "Shopify 2023年的主要增长驱动力是什么？",
]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, original length {len(text)} chars]"


@st.cache_resource
def _get_client() -> OpenAI | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def retrieve_context(query: str) -> tuple[str, list[str]]:
    """Run RAG tools and return (context_text, source_labels)."""
    outputs, sources = [], []

    mda: ToolResult = search_mda(query, top_k=4)
    if mda.status == "ok":
        outputs.append("[MD&A Relevant Passages]\n" + _truncate(mda.content, MAX_MDA_CHARS))
        sources.append("MD&A vector search")

    macro_kw = ["macro", "gdp", "cpi", "inflation", "宏观", "通胀", "经济"]
    if any(kw in query.lower() for kw in macro_kw):
        mac: ToolResult = get_macro()
        if mac.status == "ok":
            outputs.append("[Canada Macro Data (last 12 periods)]\n" + mac.content)
            sources.append("Macro data")

    return "\n\n---\n\n".join(outputs), sources


def call_llm(history: list[dict], user_msg_with_context: str) -> str:
    client = _get_client()
    if client is None:
        return (
            "⚠️ **GROQ_API_KEY not found.**\n\n"
            "Add to `.env` in the project root:\n"
            "```\nGROQ_API_KEY=gsk_...\n```"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg_with_context})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("🔍 RAG Q&A — TSX Annual Report Assistant")
st.caption(f"MD&A vector search  ·  Powered by Groq / {MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.subheader("Example questions")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    if os.getenv("GROQ_API_KEY"):
        st.success("API key loaded ✓")
    else:
        st.warning("No GROQ_API_KEY found")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))
        if msg.get("context"):
            with st.expander("View retrieved passages"):
                preview = msg["context"]
                st.text(preview[:2000] + ("…" if len(preview) > 2000 else ""))

# Handle input
prompt = st.chat_input("Ask anything about TSX companies…")
if st.session_state.get("pending_question"):
    prompt = st.session_state.pop("pending_question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching MD&A and generating answer…"):
            context, sources = retrieve_context(prompt)

            user_msg = (
                f"{context}\n\n---\nUser question: {prompt}"
                if context
                else prompt
            )

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            answer = call_llm(history, user_msg)

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))
        if context:
            with st.expander("View retrieved passages"):
                st.text(context[:2000] + ("…" if len(context) > 2000 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "context": context,
    })
