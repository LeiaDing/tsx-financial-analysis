"""
fetch_sedar_mda.py

从 SEDAR+ 下载 TSX 上市公司的 MD&A PDF（2018-2024 年度报告）
目标：先跑通 10 家公司，每家取最近一份年度 MD&A

运行方式：
    python src/fetch_sedar_mda.py
"""

import json
import os
import time
import requests
from pathlib import Path

# ── 路径配置 ────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent          # project/
DATA_DIR   = BASE_DIR / "data" / "raw" / "mda_pdfs"         # 下载到这里
INPUT_FILE = BASE_DIR / "data" / "raw" / "tsx_financials.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── SEDAR+ API ──────────────────────────────────────────────
# SEDAR+ 的搜索 API（公开端点，无需登录）
SEARCH_URL = "https://www.sedarplus.ca/csa-party/records/search.action"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project, contact: research@example.com)",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# 只抓这些年份的年度报告
TARGET_YEARS = list(range(2018, 2025))  # 2018-2024

# 先只处理前 10 家，跑通后改成 50
TEST_LIMIT = 10


# ── 函数：搜索公司在 SEDAR+ 的 MD&A 文件列表 ────────────────
def search_mda_filings(company_name: str, year: int) -> list:
    """
    在 SEDAR+ 搜索指定公司、指定年份的 MD&A 文件。
    返回：文件信息列表，每项包含文件名和下载 URL。
    """
    payload = {
        "category": "Annual filings",      # 年度报告类别
        "subCategory": "Annual report",
        "issuerName": company_name,
        "dateFrom": f"{year}-01-01",
        "dateTo":   f"{year}-12-31",
        "pageNumber": 1,
        "pageSize": 10,
    }

    try:
        resp = requests.post(SEARCH_URL, json=payload, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("documents", [])

    except requests.exceptions.HTTPError as e:
        print(f"    [HTTP错误] {company_name} {year}: {e}")
        return []
    except requests.exceptions.ConnectionError:
        print(f"    [连接失败] 无法访问 SEDAR+，检查网络")
        return []
    except Exception as e:
        print(f"    [未知错误] {company_name} {year}: {e}")
        return []


# ── 函数：下载单个 PDF ───────────────────────────────────────
def download_pdf(url: str, save_path: Path) -> bool:
    """
    下载 PDF 到指定路径。
    返回：True=成功，False=失败
    """
    if save_path.exists():
        print(f"    [已存在] 跳过 {save_path.name}")
        return True

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = save_path.stat().st_size // 1024
        print(f"    [下载成功] {save_path.name}  ({size_kb} KB)")
        return True

    except Exception as e:
        print(f"    [下载失败] {url}: {e}")
        if save_path.exists():
            save_path.unlink()  # 删除不完整文件
        return False


# ── 函数：把公司名清理成安全的文件夹名 ─────────────────────
def safe_dirname(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_").replace(",", "")[:50]


# ── 主流程 ──────────────────────────────────────────────────
def main():
    # 读取公司列表
    with open(INPUT_FILE) as f:
        companies = json.load(f)

    # 只取前 TEST_LIMIT 家
    companies = companies[:TEST_LIMIT]

    print(f"开始处理 {len(companies)} 家公司（测试模式）")
    print(f"下载目录：{DATA_DIR}\n")

    results = []  # 记录每家公司的下载结果

    for i, company in enumerate(companies, 1):
        ticker       = company["ticker"]
        company_name = company["company_name"]
        print(f"[{i}/{len(companies)}] {ticker}  {company_name}")

        company_dir = DATA_DIR / safe_dirname(company_name)
        company_dir.mkdir(exist_ok=True)

        downloaded = 0

        for year in TARGET_YEARS:
            print(f"  → 搜索 {year} 年 MD&A ...")
            filings = search_mda_filings(company_name, year)

            if not filings:
                print(f"  → {year}: 未找到文件")
                continue

            # 取第一个匹配的文件
            filing = filings[0]
            doc_url  = filing.get("url") or filing.get("downloadUrl", "")
            doc_name = filing.get("fileName") or f"{ticker}_{year}_MDA.pdf"

            if not doc_url:
                print(f"  → {year}: 找到条目但无下载链接")
                continue

            save_path = company_dir / doc_name
            success = download_pdf(doc_url, save_path)
            if success:
                downloaded += 1

            time.sleep(1)  # 礼貌性延迟，避免被限速

        results.append({
            "ticker": ticker,
            "company_name": company_name,
            "files_downloaded": downloaded,
        })
        print(f"  ✓ {company_name}：下载 {downloaded}/{len(TARGET_YEARS)} 个文件\n")

    # 打印汇总
    print("=" * 50)
    print("下载汇总：")
    total = sum(r["files_downloaded"] for r in results)
    for r in results:
        status = "✅" if r["files_downloaded"] > 0 else "❌"
        print(f"  {status} {r['ticker']:10s} {r['company_name']:40s} {r['files_downloaded']} 个文件")
    print(f"\n合计下载：{total} 个 PDF")
    print(f"保存位置：{DATA_DIR}")


if __name__ == "__main__":
    main()
