"""
11_build_panel_data.py

合并三张表 → panel_data.csv:
    annual_financials.csv  (ticker / company / sector / year / 财务指标)
  + mda_sentiment.csv      (ticker / company / year / sentiment_score 等)
  + macro_combined.csv     (季度 GDP + CPI → 按年聚合)

合并逻辑：
    - 以 annual_financials 为主表（49家公司 × 3年）
    - 左连接 mda_sentiment（按 company 模糊匹配 + year）
    - 左连接 macro_annual（按 year）

运行方式：
    python src/11_build_panel_data.py
"""

from pathlib import Path

import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
FINANCIALS    = BASE_DIR / "data" / "processed" / "annual_financials.csv"
SENTIMENT     = BASE_DIR / "data" / "processed" / "mda_sentiment.csv"
MACRO         = BASE_DIR / "data" / "raw" / "macro" / "macro_combined.csv"
OUTPUT_PATH   = BASE_DIR / "data" / "processed" / "panel_data.csv"


def normalize_name(s: str) -> str:
    """去末尾标点、多余空格，转小写，用于模糊匹配。"""
    return str(s).rstrip(".,").strip().lower()


def aggregate_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """季度数据 → 年度数据：GDP取Q4值（年末存量），CPI取全年均值。"""
    macro = macro.copy()
    macro["year"] = macro["quarter"].str[:4].astype(int)
    macro["is_q4"] = macro["quarter"].str.endswith("Q4")

    gdp_annual = (
        macro[macro["is_q4"]][["year", "gdp_level", "gdp_yoy"]]
        .rename(columns={"gdp_level": "gdp_level_q4"})
    )
    cpi_annual = (
        macro.groupby("year")["cpi_yoy"]
        .mean()
        .reset_index()
        .rename(columns={"cpi_yoy": "cpi_yoy_annual"})
    )
    return gdp_annual.merge(cpi_annual, on="year", how="outer")


def main():
    fin = pd.read_csv(FINANCIALS)
    sent = pd.read_csv(SENTIMENT)
    macro = pd.read_csv(MACRO)

    # ── 宏观数据年度化 ────────────────────────────────────────────────────────
    macro_annual = aggregate_macro(macro)

    # ── 情绪分：标准化公司名，修正 Open Text 误识别年份 ───────────────────────
    sent["_name_key"] = sent["company"].apply(normalize_name)
    sent.loc[sent["company"] == "Open Text Corporation", "year"] = 2022

    fin["_name_key"] = fin["company"].apply(normalize_name)
    fin["year"] = fin["year"].astype(int)

    # 情绪分按 (标准化公司名, year) 合并；year 有 NaN 的行单独按公司名合并
    sent_with_year = sent.dropna(subset=["year"]).copy()
    sent_with_year["year"] = sent_with_year["year"].astype(int)

    sent_no_year = sent[sent["year"].isna()][["_name_key", "sentiment_pos",
                                               "sentiment_neg", "sentiment_neu",
                                               "sentiment_score"]]

    sentiment_cols = ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_score"]

    # 主合并：有年份的情绪分
    panel = fin.merge(
        sent_with_year[["_name_key", "year"] + sentiment_cols],
        on=["_name_key", "year"],
        how="left",
    )

    # 对仍然缺少情绪分的行，用无年份情绪分回填
    missing_mask = panel["sentiment_score"].isna()
    panel = panel.merge(
        sent_no_year.rename(columns={c: f"{c}_fallback" for c in sentiment_cols}),
        on="_name_key",
        how="left",
    )
    for col in sentiment_cols:
        panel.loc[missing_mask, col] = panel.loc[missing_mask, f"{col}_fallback"]
        panel.drop(columns=f"{col}_fallback", inplace=True)

    # ── 合并宏观数据 ──────────────────────────────────────────────────────────
    panel = panel.merge(macro_annual, on="year", how="left")

    # ── 清理，输出 ────────────────────────────────────────────────────────────
    panel.drop(columns=["_name_key", "is_q4"], inplace=True, errors="ignore")

    col_order = [
        "ticker", "company", "sector", "year",
        "revenue", "net_income", "gross_profit", "capex", "operating_cf",
        "net_margin", "capex_to_revenue",
        "sentiment_score", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "gdp_level_q4", "gdp_yoy", "cpi_yoy_annual",
    ]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel = panel.sort_values(["company", "year"]).reset_index(drop=True)

    panel.to_csv(OUTPUT_PATH, index=False)

    print(f"完成！panel_data.csv 已生成：{OUTPUT_PATH}")
    print(f"行数：{len(panel)}，列数：{len(panel.columns)}")
    print(f"\n情绪分覆盖率：{panel['sentiment_score'].notna().sum()}/{len(panel)} 行")
    print(f"宏观数据覆盖率：{panel['gdp_yoy'].notna().sum()}/{len(panel)} 行\n")
    print(panel.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
