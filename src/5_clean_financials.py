"""
Week 1, Day 3: 数据清洗
输入：data/raw/tsx_financials.json
输出：
  - data/processed/annual_financials.csv   年度宽表
  - data/processed/quarterly_revenue.csv   季度收入长表
"""

import json
import os
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/tsx_financials.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_year(ts_str: str) -> int:
    """从 '2024-10-31 00:00:00' 提取年份"""
    return pd.Timestamp(ts_str).year


def build_annual_df(data: list) -> pd.DataFrame:
    """
    构建年度宽表：每行 = 一家公司 × 一个年度
    列：ticker, company, sector, employees, year,
        revenue, net_income, gross_profit, capex, operating_cf,
        net_margin, capex_to_revenue
    """
    rows = []
    for rec in data:
        ticker = rec["ticker"]
        company = rec.get("company_name", "")
        sector = rec.get("sector", "")
        employees = rec.get("employees")

        # 收集所有出现的年份
        years = set()
        for field in ["annual_revenue", "annual_net_income", "annual_gross_profit",
                      "annual_capex", "annual_operating_cf"]:
            for ts in rec.get(field, {}).keys():
                years.add(parse_year(ts))

        for year in sorted(years):
            def get_val(field):
                d = rec.get(field, {})
                for ts, v in d.items():
                    if parse_year(ts) == year:
                        return v if pd.notna(v) else None
                return None

            revenue = get_val("annual_revenue")
            net_income = get_val("annual_net_income")
            gross_profit = get_val("annual_gross_profit")
            capex = get_val("annual_capex")
            operating_cf = get_val("annual_operating_cf")

            # 派生指标
            net_margin = (net_income / revenue) if revenue and net_income else None
            capex_to_rev = (abs(capex) / revenue) if revenue and capex else None

            rows.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "employees": employees,
                "year": year,
                "revenue": revenue,
                "net_income": net_income,
                "gross_profit": gross_profit,
                "capex": capex,
                "operating_cf": operating_cf,
                "net_margin": net_margin,
                "capex_to_revenue": capex_to_rev,
            })

    df = pd.DataFrame(rows)
    # 只保留有 revenue 的行（核心字段）
    df = df.dropna(subset=["revenue"])
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    return df


def build_quarterly_df(data: list) -> pd.DataFrame:
    """
    构建季度长表：每行 = 一家公司 × 一个季度
    列：ticker, company, sector, period, revenue, net_income, capex
    """
    rows = []
    for rec in data:
        ticker = rec["ticker"]
        company = rec.get("company_name", "")
        sector = rec.get("sector", "")

        # 收集所有季度时间点
        periods = set()
        for field in ["quarterly_revenue", "quarterly_net_income", "quarterly_capex"]:
            for ts in rec.get(field, {}).keys():
                periods.add(ts)

        for ts in periods:
            def get_q(field):
                v = rec.get(field, {}).get(ts)
                return v if v is not None and pd.notna(v) else None

            rows.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "period": pd.Timestamp(ts).date(),
                "revenue": get_q("quarterly_revenue"),
                "net_income": get_q("quarterly_net_income"),
                "capex": get_q("quarterly_capex"),
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["revenue"])
    df = df.sort_values(["ticker", "period"]).reset_index(drop=True)
    return df


def print_summary(annual: pd.DataFrame, quarterly: pd.DataFrame):
    print("=" * 50)
    print("【年度宽表】annual_financials.csv")
    print(f"  行数: {len(annual)}，列数: {len(annual.columns)}")
    print(f"  公司数: {annual['ticker'].nunique()}")
    print(f"  年份范围: {annual['year'].min()} - {annual['year'].max()}")
    print(f"  net_margin 有效率: {annual['net_margin'].notna().mean():.1%}")
    print(f"  capex_to_revenue 有效率: {annual['capex_to_revenue'].notna().mean():.1%}")
    print()

    print("【季度长表】quarterly_revenue.csv")
    print(f"  行数: {len(quarterly)}，列数: {len(quarterly.columns)}")
    print(f"  公司数: {quarterly['ticker'].nunique()}")
    print(f"  时间范围: {quarterly['period'].min()} ~ {quarterly['period'].max()}")
    print()

    print("【各行业公司数（年度表）】")
    print(annual.groupby("sector")["ticker"].nunique().sort_values(ascending=False).to_string())
    print()

    print("【净利润率 TOP5（各公司最新年度）】")
    latest = annual.sort_values("year").groupby("ticker").last().reset_index()
    latest = latest.dropna(subset=["net_margin"])
    top5 = latest.nlargest(5, "net_margin")[["ticker", "company", "sector", "year", "net_margin"]]
    top5 = top5.copy()
    top5["net_margin"] = top5["net_margin"].map("{:.1%}".format)
    print(top5.to_string(index=False))
    print("=" * 50)


def main():
    with open(RAW_PATH) as f:
        data = json.load(f)

    print(f"读取 {len(data)} 条原始记录...\n")

    annual = build_annual_df(data)
    quarterly = build_quarterly_df(data)

    annual.to_csv(os.path.join(OUT_DIR, "annual_financials.csv"), index=False)
    quarterly.to_csv(os.path.join(OUT_DIR, "quarterly_revenue.csv"), index=False)

    print_summary(annual, quarterly)
    print(f"文件已保存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
