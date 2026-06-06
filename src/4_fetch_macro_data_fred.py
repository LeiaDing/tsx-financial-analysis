"""
fetch_macro_data_fred.py

从 FRED（美联储经济数据库）拉取加拿大宏观数据：
  - 季度实际GDP：NGDPRSAXDCCAQ（链式2012年价格，季节调整，百万加元）
  - 月度CPI：CPALTT01CAM659N（2015=100，月度，再聚合到季度均值）

FRED 直接返回 CSV，无需 API key，WSL 下无 SSL 问题。

时间范围：2018 Q1 — 2024 Q4
输出：data/raw/macro/gdp_quarterly.csv
      data/raw/macro/cpi_quarterly.csv
      data/raw/macro/macro_combined.csv

运行方式：
    python3 src/fetch_macro_data_fred.py
"""

import requests
import pandas as pd
from pathlib import Path
from io import StringIO

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "data" / "raw" / "macro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FRED CSV 直接下载地址（无需 API key）
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

GDP_SERIES = "NGDPRSAXDCCAQ"    # 加拿大季度实际GDP，链式2012年价格
CPI_SERIES = "CPALTT01CAM659N"  # 加拿大CPI同比增速，月度（%），已是YoY增速，无需再计算

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}


def fetch_fred_series(series_id: str, label: str) -> pd.DataFrame:
    """从 FRED 下载单个序列的 CSV，返回 DataFrame（date, value）"""
    url = FRED_CSV_URL.format(series_id=series_id)
    print(f"下载 {label} ({series_id}) ...", end=" ", flush=True)

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"])
    df.columns = ["date", "value"]
    df = df[df["value"] != "."]          # FRED 用 "." 表示缺失
    df["value"] = pd.to_numeric(df["value"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)

    print(f"✅  {len(df)} 条记录")
    return df


def process_gdp(df: pd.DataFrame) -> pd.DataFrame:
    """GDP 季度 DataFrame，计算同比/环比"""
    df = df.rename(columns={"value": "gdp_level"}).copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["gdp_yoy"] = df["gdp_level"].pct_change(4) * 100
    df["gdp_qoq"] = df["gdp_level"].pct_change(1) * 100
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    return df[["quarter", "date", "gdp_level", "gdp_yoy", "gdp_qoq"]]


def process_cpi(df: pd.DataFrame) -> pd.DataFrame:
    """CPI 月度 → 季度均值，计算同比"""
    df = df.rename(columns={"value": "cpi"}).copy()
    df = df.sort_values("date")

    df["quarter"] = df["date"].dt.to_period("Q")
    cpi_q = df.groupby("quarter")["cpi"].mean().reset_index()
    cpi_q.columns = ["quarter", "cpi_yoy"]   # 序列本身就是同比增速(%)
    cpi_q["quarter"] = cpi_q["quarter"].astype(str)

    return cpi_q


def main():
    gdp_raw = fetch_fred_series(GDP_SERIES, "加拿大实际GDP（季度）")
    cpi_raw = fetch_fred_series(CPI_SERIES, "加拿大CPI（月度）")

    print("\n处理 GDP ...", end=" ")
    gdp = process_gdp(gdp_raw)
    print(f"✅  {len(gdp)} 个季度")

    print("处理 CPI ...", end=" ")
    cpi = process_cpi(cpi_raw)
    print(f"✅  {len(cpi)} 个季度")

    combined = pd.merge(
        gdp[["quarter", "gdp_level", "gdp_yoy", "gdp_qoq"]],
        cpi[["quarter", "cpi_yoy"]],
        on="quarter", how="inner"
    )

    gdp.to_csv(OUT_DIR / "gdp_quarterly.csv", index=False)
    cpi.to_csv(OUT_DIR / "cpi_quarterly.csv", index=False)
    combined.to_csv(OUT_DIR / "macro_combined.csv", index=False)

    print(f"\n保存完成：{OUT_DIR}")
    print(f"\n宏观数据预览（最新5个季度）：")
    print(combined.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
