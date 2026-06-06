"""
parse_mda_pdfs.py

用 pdfminer.six 解析 SEDAR+ 下载的 MD&A 英文 PDF，提取纯文本。
输出：每家公司一个 .txt 文件，保存到 data/processed/mda_texts/
运行方式：
    python src/parse_mda_pdfs.py
"""

import os
import re
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
PDF_DIR     = BASE_DIR / "data" / "raw" / "mda_pdfs"
OUTPUT_DIR  = BASE_DIR / "data" / "processed" / "mda_texts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_company_name(folder_name: str) -> str:
    """
    从 SEDAR+ 的文件夹名提取公司名。
    格式：'000000131 Imperial Oil Limited / ...'
    → 'Imperial Oil Limited'
    """
    name = re.sub(r"^\d+\s+", "", folder_name)   # 去掉开头数字ID
    name = name.split(" / ")[0].strip()            # 取 '/' 前面的英文名
    name = re.sub(r"\s*\(formerly.*?\)", "", name).strip()  # 去掉 'formerly...'
    return name


def safe_filename(name: str) -> str:
    """把公司名转成安全的文件名"""
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")[:60]


def parse_pdf(pdf_path: Path) -> str | None:
    """用 pdfminer 提取 PDF 文本，返回字符串，失败返回 None"""
    try:
        # 用二进制方式打开，避免文件名含特殊字符时出问题
        with open(pdf_path, "rb") as f:
            text = extract_text(f)
        text = re.sub(r"\n{3,}", "\n\n", text)  # 压缩多余空行
        text = text.strip()
        return text if len(text) > 500 else None  # 少于500字符视为解析失败
    except PDFSyntaxError as e:
        print(f"    [PDF格式错误] {pdf_path.name}: {e}")
        return None
    except Exception as e:
        print(f"    [解析失败] {pdf_path.name}: {e}")
        return None


def main():
    company_dirs = sorted([d for d in PDF_DIR.iterdir() if d.is_dir()])
    print(f"共找到 {len(company_dirs)} 个公司文件夹\n")

    success, failed, skipped = [], [], []

    for company_dir in company_dirs:
        company_name = extract_company_name(company_dir.name)
        out_filename = safe_filename(company_name) + ".txt"
        out_path = OUTPUT_DIR / out_filename

        # 找英文 PDF（文件名含 english，扩展名可能带哈希后缀如 .pdf (abc123)）
        english_pdfs = [
            f for f in company_dir.rglob("*")
            if "english" in f.name.lower() and ".pdf" in f.name.lower() and f.is_file()
        ]

        if not english_pdfs:
            print(f"⚠️  {company_name}：未找到英文 PDF，跳过")
            skipped.append(company_name)
            continue

        pdf_path = english_pdfs[0]
        print(f"解析 {company_name} ...", end=" ", flush=True)

        text = parse_pdf(pdf_path)
        if text:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {company_name}\n")
                f.write(f"# 来源：{pdf_path.name}\n\n")
                f.write(text)

            word_count = len(text.split())
            print(f"✅  ({word_count:,} 词) → {out_filename}")
            success.append(company_name)
        else:
            print(f"❌  文本为空或太短")
            failed.append(company_name)

    print(f"\n{'='*50}")
    print(f"成功：{len(success)} 家 | 失败：{len(failed)} 家 | 跳过（无英文PDF）：{len(skipped)} 家")
    print(f"输出目录：{OUTPUT_DIR}")

    if failed:
        print(f"\n失败列表：{failed}")
    if skipped:
        print(f"跳过列表：{skipped}")


if __name__ == "__main__":
    main()
