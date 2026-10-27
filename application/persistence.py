# application/persistence.py
"""
运行期"记忆落盘"开关。

- 开启（默认）：对话产生的记忆写穿到 data/ 下的 JSON / NPY 文件。
- 关闭：只更新进程内缓存，不写文件；重新开启时把缓存中尚未落盘的数据一次性刷盘。
- 进程重启后开关恢复为"开启"，仅存在于内存中的会话会丢失。

注意：该开关只控制"对话记忆"（会话消息、话题摘要、摘要向量、用户画像）。
空间物品库（spatial_memory.json）、反馈审计（feedback/*.jsonl）和补偿队列
（compensation.db）属于运行数据/审计数据，不受此开关影响。
"""
import threading

_lock = threading.Lock()
_enabled = True


def is_enabled() -> bool:
    """当前是否允许把对话记忆写入文件。"""
    with _lock:
        return _enabled


def set_enabled(value: bool) -> None:
    """设置开关状态（只改标志，刷盘由调用方负责）。"""
    global _enabled
    with _lock:
        _enabled = bool(value)
