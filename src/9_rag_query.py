"""
rag_query.py

整合 tool_executor + memory 的 RAG 问答引擎。

流程：
    用户输入
        ↓
    检索长期记忆（相关背景）
        ↓
    调用工具（MD&A / 财务 / 宏观）
        ↓
    拼接 Prompt → 调用 LLM
        ↓
    更新短期记忆 + 保存重要结论到长期记忆

运行方式：
    python src/rag_query.py                    # 交互模式
    python src/rag_query.py --query "TSX银行板块2023营收趋势"
"""

import argparse
import os
import re
from pathlib import Path

import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
_te = importlib.import_module("7_tool_executor")
ToolResult, get_financials, get_macro, search_mda = _te.ToolResult, _te.get_financials, _te.get_macro, _te.search_mda
_mm = importlib.import_module("8_memory")
LongTermMemory, ShortTermMemory = _mm.LongTermMemory, _mm.ShortTermMemory

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# 如果没有 OpenAI key，打印 prompt 但不调用 API（方便本地演示）
_USE_LLM = _OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))

MAX_CONTEXT_CHARS = 6000   # 传给 LLM 的工具结果最大字符数


def _truncate(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [已截断，原始长度 {len(text)} 字符]"


def _extract_companies(query: str) -> list[str]:
    """从查询里粗提取可能的公司名/Ticker（大写字母串 + 常见缩写）"""
    tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
    return tickers


def run_query(
    query: str,
    short_mem: ShortTermMemory,
    long_mem: LongTermMemory,
    verbose: bool = True,
) -> str:
    """
    执行一次 RAG 问答，返回 LLM 回答（或拼好的 prompt 字符串）。
    """

    # ── 1. 检索长期记忆 ──────────────────────────────────────────────────────
    memory_records = long_mem.search(query, top_k=2)
    memory_context = ""
    if memory_records:
        snippets = [f"- {r.content}" for r in memory_records]
        memory_context = "【历史记忆】\n" + "\n".join(snippets) + "\n\n"

    # ── 2. 调用工具 ───────────────────────────────────────────────────────────
    tool_outputs: list[str] = []

    # 工具A：MD&A 语义检索
    mda_result: ToolResult = search_mda(query, top_k=4)
    if mda_result.status == "ok":
        tool_outputs.append("【MD&A 相关段落】\n" + _truncate(mda_result.content, 3000))
    elif verbose:
        print(f"[工具跳过 - MD&A] {mda_result.content}")

    # 工具B：财务数据（从查询里提取公司）
    companies = _extract_companies(query)
    for company in companies[:2]:    # 最多查2家，避免 context 爆炸
        fin_result: ToolResult = get_financials(company)
        if fin_result.status == "ok":
            tool_outputs.append(f"【{company} 年度财务数据】\n" + _truncate(fin_result.content, 1500))
        elif verbose:
            print(f"[工具跳过 - 财务/{company}] {fin_result.content}")

    # 工具C：宏观数据（问题里含宏观关键词时触发）
    macro_keywords = ["宏观", "gdp", "cpi", "通胀", "经济", "macro", "inflation"]
    if any(kw in query.lower() for kw in macro_keywords):
        macro_result: ToolResult = get_macro()
        if macro_result.status == "ok":
            tool_outputs.append("【加拿大宏观经济数据（近12期）】\n" + macro_result.content)
        elif verbose:
            print(f"[工具跳过 - 宏观] {macro_result.content}")

    if not tool_outputs:
        no_data_msg = "所有工具均无可用数据。请先运行 python src/embed_mda_texts.py 生成向量索引。"
        if verbose:
            print(f"\n[RAG] {no_data_msg}")
        return no_data_msg

    tool_context = "\n\n".join(tool_outputs)

    # ── 3. 拼接对话历史 ──────────────────────────────────────────────────────
    history = short_mem.get_context()

    # ── 4. 构建 Prompt ────────────────────────────────────────────────────────
    system_prompt = (
        "你是一位专业的加拿大 TSX 上市公司分析师，"
        "擅长结合公司 MD&A 披露、财务数据和宏观经济背景进行综合分析。\n"
        "回答时请：\n"
        "1. 优先引用提供的原文数据，不要凭空捏造数字\n"
        "2. 如果数据不足，明确说明缺少什么\n"
        "3. 结论简洁，分点列出"
    )

    user_message = f"""{memory_context}{tool_context}

---
用户问题：{query}"""

    if verbose:
        print(f"\n[RAG] 工具结果已收集（{len(tool_context)} 字符），准备调用 LLM...")

    # ── 5. 调用 LLM ──────────────────────────────────────────────────────────
    if _USE_LLM:
        client = OpenAI()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
    else:
        # 无 API key：把完整 prompt 打印出来，方便演示
        answer = (
            "[LLM 未配置] 以下是发送给模型的完整 Prompt，可手动粘贴到 ChatGPT 验证：\n\n"
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{user_message}"
        )

    # ── 6. 更新记忆 ───────────────────────────────────────────────────────────
    short_mem.add("user", query)
    short_mem.add("assistant", answer[:500])   # 只存前500字，节省窗口空间

    # 将有价值的结论保存到长期记忆（简单启发式：回答超过200字才值得保存）
    if _USE_LLM and len(answer) > 200:
        summary = f"Q: {query[:80]} → {answer[:200]}"
        long_mem.add(summary, importance=6.0, ttl_hours=24 * 30)   # 保存30天

    return answer


def interactive_loop():
    short_mem = ShortTermMemory(window_size=6)
    long_mem   = LongTermMemory()
    long_mem.cleanup_expired()

    print("=" * 60)
    print("TSX 财报分析助手（输入 'exit' 退出，'clear' 清空对话）")
    print("=" * 60)

    while True:
        try:
            query = input("\n你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "clear":
            short_mem.clear()
            print("[对话历史已清空]")
            continue

        answer = run_query(query, short_mem, long_mem)
        print(f"\n助手：{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="单次查询（不进入交互模式）")
    args = parser.parse_args()

    if args.query:
        short_mem = ShortTermMemory()
        long_mem   = LongTermMemory()
        print(run_query(args.query, short_mem, long_mem))
    else:
        interactive_loop()
