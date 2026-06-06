"""
Week 1, Day 1-2: 拉取50家TSX上市公司财务结构化数据
数据字段：收入、净利润、资本支出(capex)、利润率、员工数
时间跨度：2018-2024 季度数据
"""

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime

# 50家TSX上市公司，覆盖6个行业
TSX_COMPANIES = {
    "金融": [
        "RY.TO",   # Royal Bank of Canada
        "TD.TO",   # TD Bank
        "BNS.TO",  # Bank of Nova Scotia
        "BMO.TO",  # Bank of Montreal
        "CM.TO",   # CIBC
        "SLF.TO",  # Sun Life Financial
        "MFC.TO",  # Manulife
        "GWO.TO",  # Great-West Lifeco
    ],
    "能源": [
        "CNQ.TO",  # Canadian Natural Resources
        "SU.TO",   # Suncor Energy
        "CVE.TO",  # Cenovus Energy
        "IMO.TO",  # Imperial Oil
        "TRP.TO",  # TC Energy
        "ENB.TO",  # Enbridge
        "PPL.TO",  # Pembina Pipeline
        "ARX.TO",  # ARC Resources
    ],
    "科技": [
        "SHOP.TO", # Shopify
        "CSU.TO",  # Constellation Software
        "OTEX.TO", # OpenText
        "BB.TO",   # BlackBerry
        "KXS.TO",  # Kinaxis
        "DCBO.TO", # Docebo
        "LSPD.TO", # Lightspeed Commerce
        "NVEI.TO", # Nuvei
    ],
    "零售/消费": [
        "L.TO",    # Loblaw Companies
        "MRU.TO",  # Metro Inc
        "EMP-A.TO",# Empire Company
        "ATD.TO",  # Alimentation Couche-Tard
        "DOL.TO",  # Dollarama
        "CTC-A.TO",# Canadian Tire
        "GIL.TO",  # Gildan Activewear
        "PBH.TO",  # Premium Brands Holdings
    ],
    "工业/运输": [
        "CNR.TO",  # Canadian National Railway
        "CP.TO",   # Canadian Pacific Kansas City
        "WSP.TO",  # WSP Global
        "STN.TO",  # Stantec
        "TIH.TO",  # Toromont Industries
        "WCN.TO",  # Waste Connections
        "BYD.TO",  # Boyd Group Services
        "CAE.TO",  # CAE Inc
    ],
    "房地产/材料": [
        "NTR.TO",  # Nutrien
        "ABX.TO",  # Barrick Gold
        "AGI.TO",  # Alamos Gold
        "FM.TO",   # First Quantum Minerals
        "CCO.TO",  # Cameco
        "IFC.TO",  # Intact Financial
        "BAM.TO",  # Brookfield Asset Management
        "BN.TO",   # Brookfield Corporation
        "AP-UN.TO",# Allied Properties REIT
        "REI-UN.TO",# RioCan REIT
    ],
}

def fetch_annual_financials(ticker_symbol: str) -> dict:
    """拉取单个公司的年度财务数据"""
    ticker = yf.Ticker(ticker_symbol)
    result = {"ticker": ticker_symbol, "error": None}

    try:
        info = ticker.info
        result["company_name"] = info.get("longName", "")
        result["sector"] = info.get("sector", "")
        result["employees"] = info.get("fullTimeEmployees")

        # 年度损益表
        income = ticker.financials  # columns = years
        if income is not None and not income.empty:
            result["annual_revenue"] = income.loc["Total Revenue"].to_dict() if "Total Revenue" in income.index else {}
            result["annual_net_income"] = income.loc["Net Income"].to_dict() if "Net Income" in income.index else {}
            result["annual_gross_profit"] = income.loc["Gross Profit"].to_dict() if "Gross Profit" in income.index else {}

        # 年度现金流（capex）
        cashflow = ticker.cashflow
        if cashflow is not None and not cashflow.empty:
            result["annual_capex"] = cashflow.loc["Capital Expenditure"].to_dict() if "Capital Expenditure" in cashflow.index else {}
            result["annual_operating_cf"] = cashflow.loc["Operating Cash Flow"].to_dict() if "Operating Cash Flow" in cashflow.index else {}

    except Exception as e:
        result["error"] = str(e)

    return result


def fetch_quarterly_financials(ticker_symbol: str) -> dict:
    """拉取单个公司的季度财务数据"""
    ticker = yf.Ticker(ticker_symbol)
    result = {"ticker": ticker_symbol, "error": None}

    try:
        # 季度损益表
        income_q = ticker.quarterly_financials
        if income_q is not None and not income_q.empty:
            result["quarterly_revenue"] = income_q.loc["Total Revenue"].to_dict() if "Total Revenue" in income_q.index else {}
            result["quarterly_net_income"] = income_q.loc["Net Income"].to_dict() if "Net Income" in income_q.index else {}

        # 季度现金流
        cashflow_q = ticker.quarterly_cashflow
        if cashflow_q is not None and not cashflow_q.empty:
            result["quarterly_capex"] = cashflow_q.loc["Capital Expenditure"].to_dict() if "Capital Expenditure" in cashflow_q.index else {}

    except Exception as e:
        result["error"] = str(e)

    return result


def serialize_dict(d: dict) -> dict:
    """把 Timestamp key 转成字符串，方便 JSON 存储"""
    if not isinstance(d, dict):
        return d
    return {str(k): v for k, v in d.items() if v is not None}


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "../data/raw")
    os.makedirs(output_dir, exist_ok=True)
；
    all_tickers = [t for tickers in TSX_COMPANIES.values() for t in tickers]
    print(f"共 {len(all_tickers)} 家公司，开始拉取...\n")

    results = []
    failed = []

    for i, ticker in enumerate(all_tickers, 1):
        print(f"[{i}/{len(all_tickers)}] {ticker} ...", end=" ")
        try:
            annual = fetch_annual_financials(ticker)
            quarterly = fetch_quarterly_financials(ticker)

            # 合并
            record = {**annual}
            record.update({k: v for k, v in quarterly.items() if k not in record or k == "ticker"})

            # 序列化 Timestamp keys
            for key in ["annual_revenue", "annual_net_income", "annual_gross_profit",
                        "annual_capex", "annual_operating_cf",
                        "quarterly_revenue", "quarterly_net_income", "quarterly_capex"]:
                if key in record:
                    record[key] = serialize_dict(record[key])

            results.append(record)

            if record.get("error"):
                print(f"⚠️  partial error: {record['error']}")
                failed.append(ticker)
            else:
                print("✅")

        except Exception as e:
            print(f"❌ {e}")
            failed.append(ticker)

    # 保存 JSON
    output_path = os.path.join(output_dir, "tsx_financials.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n完成！数据保存至 {output_path}")
    print(f"成功: {len(results) - len(failed)} 家 | 失败/部分失败: {len(failed)} 家")
    if failed:
        print(f"失败列表: {failed}")


if __name__ == "__main__":
    main()
