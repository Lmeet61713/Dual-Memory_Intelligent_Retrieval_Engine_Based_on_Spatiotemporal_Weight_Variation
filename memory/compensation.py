"""
补偿缓冲队列 —— 基于 SQLite 的短期记忆补偿机制

当空间记忆检索返回 Top-5 候选时，Top-1 返回给 Agent，Top 2-5 存入补偿队列。
用户否定当前结果时，从队列头部弹出下一个候选，而非重新检索。
"""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from typing import Optional

import config

_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（每次新建，线程安全）"""
    conn = sqlite3.connect(config.COMPENSATION_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库和表（幂等操作）"""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compensation_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    buffer_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    item_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_buffer_id 
                ON compensation_queue(buffer_id)
            """)
            conn.commit()
        finally:
            conn.close()


def create_buffer(candidates: list[dict]) -> str:
    """
    将候选列表写入补偿队列。
    
    Args:
        candidates: Top 2-5 候选列表，每个元素为 search_spatial_memory 返回的 dict
                   格式: {'item': {...}, 'score': float, 'tone': str, 'likelihood': float, 'prior': float}
    
    Returns:
        唯一的 buffer_id (UUID 字符串)
    """
    buffer_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    with _lock:
        conn = _get_conn()
        try:
            for position, candidate in enumerate(candidates, start=1):
                item_json = json.dumps(candidate, ensure_ascii=False)
                conn.execute(
                    "INSERT INTO compensation_queue (buffer_id, position, item_json, created_at) VALUES (?, ?, ?, ?)",
                    (buffer_id, position, item_json, now)
                )
            conn.commit()
        finally:
            conn.close()
    
    return buffer_id


def pop_next(buffer_id: str) -> Optional[dict]:
    """
    弹出队列中 position 最小的候选并返回。
    
    Args:
        buffer_id: 补偿队列标识
    
    Returns:
        候选 dict（与 create_buffer 输入格式一致），队列为空时返回 None
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id, item_json FROM compensation_queue WHERE buffer_id = ? ORDER BY position ASC LIMIT 1",
                (buffer_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            # 删除该记录
            conn.execute("DELETE FROM compensation_queue WHERE id = ?", (row["id"],))
            conn.commit()
            
            return json.loads(row["item_json"])
        finally:
            conn.close()


def clear_buffer(buffer_id: str):
    """删除指定 buffer 的所有记录"""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM compensation_queue WHERE buffer_id = ?", (buffer_id,))
            conn.commit()
        finally:
            conn.close()


def get_remaining_count(buffer_id: str) -> int:
    """返回指定 buffer 的剩余候选数量"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM compensation_queue WHERE buffer_id = ?",
            (buffer_id,)
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def cleanup_expired_buffers():
    """清理过期的补偿队列（超过 COMPENSATION_BUFFER_TTL_HOURS 的缓冲）"""
    expiry_time = (datetime.now() - timedelta(hours=config.COMPENSATION_BUFFER_TTL_HOURS)).isoformat()
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM compensation_queue WHERE created_at < ?", (expiry_time,))
            conn.commit()
        finally:
            conn.close()


# 模块加载时自动初始化数据库
init_db()
