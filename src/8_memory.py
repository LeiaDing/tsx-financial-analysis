"""
memory.py

Agent 记忆模块：短期记忆（滑动窗口）+ 长期记忆（SQLite 向量存储）

短期记忆：对话上下文，最近 N 轮，直接拼进 prompt
长期记忆：用户偏好、重要结论，按混合评分检索
  hybrid_score = 0.4×语义相似度 + 0.3×时效性衰减 + 0.3×重要性
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "processed" / "memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_MODEL: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# ── 短期记忆 ──────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str      # "user" / "assistant" / "tool"
    content: str


class ShortTermMemory:
    """滑动窗口对话历史，最多保留最近 window_size 轮"""

    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self._messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        # 超出窗口时，从头删除（保留最新的）
        if len(self._messages) > self.window_size * 2:
            self._messages = self._messages[-(self.window_size * 2):]

    def get_context(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        self._messages.clear()


# ── 长期记忆 ──────────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    id: int
    content: str
    importance: float    # 1–10
    created_at: float    # unix timestamp
    ttl: float           # 0=永不过期，否则为过期时间戳


class LongTermMemory:
    """
    SQLite 长期记忆，支持语义搜索 + 冲突去重。

    冲突规则：新记忆与已有记忆余弦相似度 > 0.85 → UPDATE，否则 INSERT
    """

    SIMILARITY_UPDATE_THRESHOLD = 0.85

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    content    TEXT    NOT NULL,
                    embedding  BLOB    NOT NULL,
                    importance REAL    DEFAULT 5.0,
                    created_at REAL    NOT NULL,
                    ttl        REAL    DEFAULT 0,
                    is_deleted INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def _encode(self, text: str) -> np.ndarray:
        emb = _get_model().encode([text])[0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _load_all_active(self, conn: sqlite3.Connection):
        """加载所有未删除、未过期的记忆"""
        now = time.time()
        rows = conn.execute("""
            SELECT id, content, embedding, importance, created_at, ttl
            FROM memories
            WHERE is_deleted = 0
              AND (ttl = 0 OR ttl > ?)
        """, (now,)).fetchall()
        return rows

    def add(self, content: str, importance: float = 5.0, ttl_hours: float = 0) -> int:
        """
        添加记忆。若与已有记忆高度相似则更新，否则新建。
        返回记录 ID。
        """
        new_emb = self._encode(content)
        ttl = (time.time() + ttl_hours * 3600) if ttl_hours > 0 else 0

        with sqlite3.connect(self.db_path) as conn:
            rows = self._load_all_active(conn)

            # 检查是否与已有记忆冲突
            for row_id, row_content, emb_blob, row_imp, row_created, row_ttl in rows:
                old_emb = np.frombuffer(emb_blob, dtype=np.float32)
                sim = float(new_emb @ old_emb)
                if sim >= self.SIMILARITY_UPDATE_THRESHOLD:
                    # UPDATE：保留旧 ID，更新内容和 embedding
                    conn.execute("""
                        UPDATE memories
                        SET content=?, embedding=?, importance=?, created_at=?
                        WHERE id=?
                    """, (content, new_emb.astype(np.float32).tobytes(),
                          importance, time.time(), row_id))
                    conn.commit()
                    return row_id

            # INSERT
            cur = conn.execute("""
                INSERT INTO memories (content, embedding, importance, created_at, ttl)
                VALUES (?, ?, ?, ?, ?)
            """, (content, new_emb.astype(np.float32).tobytes(),
                  importance, time.time(), ttl))
            conn.commit()
            return cur.lastrowid

    def search(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        """
        混合评分检索：
          score = 0.4×语义相似度 + 0.3×时效性(0.995^小时) + 0.3×重要性/10
        """
        query_emb = self._encode(query)
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            rows = self._load_all_active(conn)

        if not rows:
            return []

        scored = []
        for row_id, content, emb_blob, importance, created_at, ttl in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            semantic  = float(query_emb @ emb)
            hours_ago = (now - created_at) / 3600
            recency   = 0.995 ** hours_ago
            imp_norm  = min(importance, 10.0) / 10.0
            score = 0.4 * semantic + 0.3 * recency + 0.3 * imp_norm
            scored.append((score, MemoryRecord(
                id=row_id, content=content, importance=importance,
                created_at=created_at, ttl=ttl
            )))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def cleanup_expired(self) -> int:
        """软删除已过期记忆，返回删除数量"""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                UPDATE memories SET is_deleted=1
                WHERE ttl > 0 AND ttl < ? AND is_deleted=0
            """, (now,))
            conn.commit()
            return cur.rowcount

    def delete(self, memory_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE memories SET is_deleted=1 WHERE id=?", (memory_id,))
            conn.commit()
