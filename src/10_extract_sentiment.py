"""
10_extract_sentiment.py

用 FinBERT（ProsusAI/finbert）对每家公司 MD&A 前20个文本块打情绪分。

流程：
    1. 读取 chunks_metadata.json，按公司分组，取前20块
    2. 从 MD&A 文本提取财年（正则）
    3. FinBERT 打分：positive / negative / neutral 概率
    4. 每家公司取均值，输出 mda_sentiment.csv

输出字段：
    ticker, company, year, sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score

运行方式：
    python src/10_extract_sentiment.py          # 全部36家
    python src/10_extract_sentiment.py --test   # 只跑前3家，验证环境
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import pipeline

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "data" / "processed" / "embeddings" / "chunks_metadata.json"
TEXT_DIR      = BASE_DIR / "data" / "processed" / "mda_texts"
FINANCIALS    = BASE_DIR / "data" / "processed" / "annual_financials.csv"
OUTPUT_PATH   = BASE_DIR / "data" / "processed" / "mda_sentiment.csv"

CHUNKS_PER_COMPANY = 20
MAX_TOKEN_LENGTH   = 512  # FinBERT 上限


def build_ticker_map():
    """从 annual_financials.csv 构建 company_name → ticker 映射（模糊匹配，忽略末尾标点和大小写）。"""
    df = pd.read_csv(FINANCIALS)
    raw = dict(zip(df["company"], df["ticker"]))
    # 同时建一份标准化版本（去末尾标点、小写）用于回退匹配
    normalized = {k.rstrip(".,"): v for k, v in raw.items()}
    return raw, normalized


def lookup_ticker(company: str, raw: dict, normalized: dict) -> str:
    if company in raw:
        return raw[company]
    return normalized.get(company.rstrip(".,"), "")


def extract_year(text: str) -> int | None:
    """从 MD&A 文本提取财年（取最早出现的四位年份，1990-2030范围内）。"""
    text_clean = re.sub(r"\s+", " ", text)
    matches = re.findall(r"\b((?:19|20)\d{2})\b", text_clean)
    years = [int(y) for y in matches if 1990 <= int(y) <= 2030]
    # 财年通常在文档开头附近提到，取众数
    if not years:
        return None
    return max(set(years), key=years.count)


def chunk_text(text: str, max_len: int = MAX_TOKEN_LENGTH) -> list[str]:
    """把长文本按字符数截断，避免超出 FinBERT token 限制（粗略估计：4字符≈1token）。"""
    char_limit = max_len * 4
    if len(text) <= char_limit:
        return [text]
    return [text[i:i + char_limit] for i in range(0, len(text), char_limit)]


def main(test_mode: bool = False):
    print("加载 chunks_metadata.json ...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # 按公司分组，保留原始顺序，取前 CHUNKS_PER_COMPANY 块
    company_chunks: dict[str, list[str]] = {}
    for item in metadata:
        name = item["company"]
        if name not in company_chunks:
            company_chunks[name] = []
        if len(company_chunks[name]) < CHUNKS_PER_COMPANY:
            company_chunks[name].append(item["text"])

    companies = list(company_chunks.keys())
    if test_mode:
        companies = companies[:3]
        print(f"[TEST] 只处理前3家：{companies}")

    ticker_raw, ticker_norm = build_ticker_map()

    print("加载 FinBERT 模型（首次运行会下载，约400MB）...")
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,          # 返回全部三类概率
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        device=-1,           # CPU
    )

    records = []
    for i, company in enumerate(companies):
        chunks = company_chunks[company]
        print(f"[{i+1}/{len(companies)}] {company} — {len(chunks)} 块")

        # 从第一块提取财年
        year = extract_year(chunks[0])

        # 批量推理
        all_pos, all_neg, all_neu = [], [], []
        for chunk in chunks:
            try:
                result = finbert(chunk[:MAX_TOKEN_LENGTH * 4])[0]
                scores = {r["label"]: r["score"] for r in result}
                all_pos.append(scores.get("positive", 0))
                all_neg.append(scores.get("negative", 0))
                all_neu.append(scores.get("neutral", 0))
            except Exception as e:
                print(f"  跳过一块（{e}）")

        if not all_pos:
            continue

        pos = sum(all_pos) / len(all_pos)
        neg = sum(all_neg) / len(all_neg)
        neu = sum(all_neu) / len(all_neu)

        records.append({
            "ticker":          lookup_ticker(company, ticker_raw, ticker_norm),
            "company":         company,
            "year":            year,
            "sentiment_pos":   round(pos, 4),
            "sentiment_neg":   round(neg, 4),
            "sentiment_neu":   round(neu, 4),
            "sentiment_score": round(pos - neg, 4),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n完成！共 {len(df)} 家公司，结果保存到：{OUTPUT_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="只处理前3家公司")
    args = parser.parse_args()
    main(test_mode=args.test)
