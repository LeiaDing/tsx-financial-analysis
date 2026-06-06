"""
tool_executor.py

对 RAG 管道里的每个工具做分级失败处理：
  第一类：文件/索引不存在   → 明确提示缺什么
  第二类：空结果/相似度太低 → 不重试，告知换关键词
  第三类：内容质量不足      → 过滤，不把垃圾传给 LLM
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
EMB_PATH  = BASE_DIR / "data/processed/embeddings/embeddings.npy"
META_PATH = BASE_DIR / "data/processed/embeddings/chunks_metadata.json"
FIN_PATH  = BASE_DIR / "data/processed/annual_financials.csv"
MACRO_PATH = BASE_DIR / "data/raw/macro/macro_combined.csv"

# 模型在模块级别只加载一次，避免重复初始化
_MODEL: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


@dataclass
class ToolResult:
    content: str
    status: str             # "ok" / "skip"
    error_type: Optional[str] = None


# ── 工具1：MD&A 向量检索 ────────────────────────────────────────────────────

def search_mda(query: str, top_k: int = 5) -> ToolResult:
    """检索与问题最相关的 MD&A 段落，带三类失败处理"""

    # 第一类：索引文件不存在（还没跑 embed_mda_texts.py）
    if not EMB_PATH.exists() or not META_PATH.exists():
        return ToolResult(
            content="[向量索引不存在，请先运行: python src/embed_mda_texts.py]",
            status="skip",
            error_type="missing_embeddings"
        )

    embeddings = np.load(EMB_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    query_emb = _get_model().encode([query])[0]
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)
    similarities = normalized @ query_norm
    top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # 多取一些，过滤后再截断

    # 第二类：最高相似度太低，说明没有相关内容
    if similarities[top_indices[0]] < 0.25:
        return ToolResult(
            content="[未找到与问题相关的 MD&A 内容，请尝试换一种表述或指定具体公司名称]",
            status="skip",
            error_type="low_similarity"
        )

    # 第三类：过滤垃圾内容（表格噪声、过短碎片）
    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        chunk = metadata[int(idx)]
        if _is_garbage(chunk["text"]):
            continue
        company = chunk["company"]
        seg_num = chunk["chunk_index"] + 1
        excerpt = chunk["text"][:600]
        results.append(f"【{company} — 段落{seg_num}】\n{excerpt}")

    if not results:
        return ToolResult(
            content="[检索到的段落内容质量不足（疑似表格噪声），请换关键词重试]",
            status="skip",
            error_type="garbage_content"
        )

    return ToolResult(content="\n\n---\n\n".join(results), status="ok")


# ── 工具2：财务数据查询 ──────────────────────────────────────────────────────

def get_financials(company_keyword: str) -> ToolResult:
    """按公司名称或 ticker 查询年度财务数据"""

    if not FIN_PATH.exists():
        return ToolResult(
            content="[财务数据文件不存在，请检查 data/processed/annual_financials.csv]",
            status="skip",
            error_type="missing_file"
        )

    df = pd.read_csv(FIN_PATH)
    keyword = company_keyword.lower()

    # 优先按 ticker 匹配，再按全行文本模糊匹配
    if "ticker" in df.columns:
        mask = df["ticker"].str.lower().str.contains(keyword, na=False)
    else:
        mask = pd.Series([False] * len(df))

    if not mask.any():
        mask = df.apply(lambda r: keyword in str(r).lower(), axis=1)

    if not mask.any():
        return ToolResult(
            content=f"[未找到 '{company_keyword}' 的财务数据，请确认公司名称或 Ticker 符号]",
            status="skip",
            error_type="not_found"
        )

    rows = df[mask].to_string(index=False)
    return ToolResult(content=rows, status="ok")


# ── 工具3：宏观经济数据 ──────────────────────────────────────────────────────

def get_macro() -> ToolResult:
    """读取加拿大宏观经济数据（GDP + CPI），返回最近 12 期"""

    if not MACRO_PATH.exists():
        return ToolResult(
            content="[宏观数据文件不存在，请检查 data/raw/macro/macro_combined.csv]",
            status="skip",
            error_type="missing_file"
        )

    df = pd.read_csv(MACRO_PATH).tail(12)
    return ToolResult(content=df.to_string(index=False), status="ok")


# ── 内容质量检测 ─────────────────────────────────────────────────────────────

def _is_garbage(text: str) -> bool:
    """判断文本是否为低质量内容（MD&A 里的表格噪声、过短碎片）"""
    if len(text.strip()) < 100:
        return True
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return True
    # 孤立数字行占比超过 50% → 表格噪声
    numeric_lines = sum(
        1 for l in lines
        if l.replace(".", "").replace(",", "").replace("-", "").replace(" ", "").isdigit()
    )
    return numeric_lines / len(lines) > 0.5
