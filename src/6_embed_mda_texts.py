"""
embed_mda_texts.py

用 HuggingFace sentence-transformers (all-MiniLM-L6-v2) 对 MD&A 文本做 embedding。
替代 OpenAI embedding，完全本地运行，无需 API key，零成本。

流程：
    1. 读取 data/processed/mda_texts/*.txt
    2. 每篇文本切成 ~1000字符的 chunk（overlap 100）
    3. 批量生成 384维向量
    4. 输出：
       - data/processed/embeddings/embeddings.npy   (shape: N_chunks × 384)
       - data/processed/embeddings/chunks_metadata.json

运行方式：
    python src/embed_mda_texts.py              # 处理全部36家公司
    python src/embed_mda_texts.py --test       # 只处理第一家，验证环境
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
TEXT_DIR    = BASE_DIR / "data" / "processed" / "mda_texts"
OUTPUT_DIR  = BASE_DIR / "data" / "processed" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 超参数 ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"   # 384维，专为语义相似度优化，本地CPU可跑
CHUNK_SIZE   = 1000                  # 每块约1-2段话
CHUNK_OVERLAP = 100                  # 相邻块重叠，避免截断句子丢上下文
BATCH_SIZE   = 32                    # 每批送入模型的chunk数，太大吃内存


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    把长文本切成固定大小的块，相邻块有 overlap 字符的重叠。
    例：chunk_size=1000, overlap=100
    块0: [0:1000], 块1: [900:1900], 块2: [1800:2800] ...
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 50:   # 跳过过短的碎片（比如文件末尾空白）
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main(test_mode: bool = False):
    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[错误] {TEXT_DIR} 下没有找到 .txt 文件")
        return

    if test_mode:
        txt_files = txt_files[:1]
        print(f"[TEST MODE] 只处理第一个文件：{txt_files[0].name}\n")
    else:
        print(f"共找到 {len(txt_files)} 个公司文本文件\n")

    # 1. 加载模型（首次运行会自动从 HuggingFace 下载，约80MB）
    print(f"加载模型：{MODEL_NAME} ...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"模型加载完成（{time.time()-t0:.1f}s），embedding维度：{model.get_sentence_embedding_dimension()}\n")

    # 2. 切分文本，收集所有 chunk
    all_chunks = []      # 存文本内容
    all_metadata = []    # 存每块的元信息

    for txt_file in txt_files:
        company = txt_file.stem.replace("_", " ")  # 文件名转回公司名
        text = txt_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "chunk_id":    len(all_metadata),   # 全局唯一ID，对应 embeddings.npy 的行号
                "company":     company,
                "source_file": txt_file.name,
                "chunk_index": i,                   # 该公司内的第几块
                "total_chunks": len(chunks),        # 该公司总块数
                "char_start":  i * (CHUNK_SIZE - CHUNK_OVERLAP),
                "text":        chunk,
            })

        print(f"  {company}: {len(text):,} 字符 → {len(chunks)} 块")

    print(f"\n全部 chunk 数：{len(all_chunks)}")

    # 3. 批量生成 embedding
    print(f"\n开始 embedding（batch_size={BATCH_SIZE}）...")
    t1 = time.time()
    embeddings = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t1
    print(f"Embedding 完成！耗时 {elapsed:.1f}s，矩阵形状：{embeddings.shape}")

    # 4. 保存结果
    emb_path  = OUTPUT_DIR / "embeddings.npy"
    meta_path = OUTPUT_DIR / "chunks_metadata.json"

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n已保存：")
    print(f"  {emb_path}   ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  {meta_path}  ({len(all_metadata)} 条记录)")

    # 5. 快速验证：打印第一个 chunk 的向量头几个值
    print(f"\n[验证] 第一个 chunk 向量（前5维）：{embeddings[0, :5].round(4)}")
    print(f"[验证] 向量范数（应接近1.0）：{np.linalg.norm(embeddings[0]):.4f}")
    print("\n完成 ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="只处理第一个文件验证环境")
    args = parser.parse_args()
    main(test_mode=args.test)
