"""
fetch_macro_data.py

从 Statistics Canada WDS JSON API 拉取宏观数据：
  - GDP 实际季度水平（向量 v62305752）
  - CPI All-items Canada 月度（向量 v41690973）

WDS API 返回 JSON，体积远小于下载整张表，WSL 下不会超时。

时间范围：2018 Q1 — 2024 Q4
输出：data/raw/macro/gdp_quarterly.csv
      data/raw/macro/cpi_quarterly.csv
      data/raw/macro/macro_combined.csv

运行方式：
    python3 src/fetch_macro_data.py
"""

import requests
import pandas as pd
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "data" / "raw" / "macro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Statistics Canada WDS API — 按向量 ID 拉取最近 N 期数据
# v62305752：实际 GDP，季节调整，链式2012年价格，季度
# v41690973：CPI All-items，Canada，月度
WDS_URL    = "https://www150.statcan.gc.ca/t1/tbl1/en/getDataFromVectorsAndLatestNPeriods/{vectors}/{n_periods}"
GDP_VECTOR = "v62305752"
CPI_VECTOR = "v41690973"
N_PERIODS  = 60   # 60个季度 ≈ 15年，足够覆盖2018-2024

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}


def fetch_vector(vector: str, n_periods: int, label: str) -> list:
    """调用 WDS API 拉取单个向量数据，返回观测值列表"""
    url = WDS_URL.format(vectors=vector, n_periods=n_periods)
    print(f"下载 {label} ...", end=" ", flush=True)
    resp = requests.get(url, headers=HEADERS, timeout=60, verify=False)
    resp.raise_for_status()
    data = resp.json()

    # WDS 返回结构：[{"object": {"vectorDataPoint": [...]}}]
    obs = data[0]["object"]["vectorDataPoint"]
    print(f"✅  {len(obs)} 条记录")
    return obs


def process_gdp(obs: list) -> pd.DataFrame:
    """把 GDP 观测值列表转成季度 DataFrame，计算同比/环比"""
    rows = []
    for o in obs:
        ref = o.get("refPer", "")        # 格式 "2024-10-01"
        val = o.get("value")
        if ref and val is not None:
            rows.append({"date": pd.to_datetime(ref), "gdp_level": float(val)})

    df = pd.DataFrame(rows).sort_values("date")
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)

    df["gdp_yoy"] = df["gdp_level"].pct_change(4) * 100   # 同比
    df["gdp_qoq"] = df["gdp_level"].pct_change(1) * 100   # 环比
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    return df[["quarter", "date", "gdp_level", "gdp_yoy", "gdp_qoq"]]


def process_cpi(obs: list) -> pd.DataFrame:
    """把 CPI 月度观测值转成季度均值 DataFrame"""
    rows = []
    for o in obs:
        ref = o.get("refPer", "")
        val = o.get("value")
        if ref and val is not None:
            rows.append({"date": pd.to_datetime(ref), "cpi": float(val)})

    df = pd.DataFrame(rows).sort_values("date")
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

    # 月度 → 季度均值
    df["quarter"] = df["date"].dt.to_period("Q")
    cpi_q = df.groupby("quarter")["cpi"].mean().reset_index()
    cpi_q.columns = ["quarter", "cpi_avg"]
    cpi_q["quarter"] = cpi_q["quarter"].astype(str)
    cpi_q["cpi_yoy"] = cpi_q["cpi_avg"].pct_change(4) * 100

    return cpi_q


def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    gdp_obs = fetch_vector(GDP_VECTOR, N_PERIODS, f"GDP（{GDP_VECTOR}）")
    cpi_obs = fetch_vector(CPI_VECTOR, 180, f"CPI（{CPI_VECTOR}）")  # 月度取180期=15年

    print("\n处理 GDP ...", end=" ")
    gdp = process_gdp(gdp_obs)
    print(f"✅  {len(gdp)} 个季度")

    print("处理 CPI ...", end=" ")
    cpi = process_cpi(cpi_obs)
    print(f"✅  {len(cpi)} 个季度")

    combined = pd.merge(
        gdp[["quarter", "gdp_level", "gdp_yoy", "gdp_qoq"]],
        cpi[["quarter", "cpi_avg", "cpi_yoy"]],
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
