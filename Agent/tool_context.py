# memory/tool_context.py
"""
工具上下文管理器 —— 使用 contextvars 在异步/多线程环境中安全传递请求级上下文
"""
import contextvars
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolContext:
    session_id: str
    all_messages: list
    retrieval_triggered: bool = False   # 供工具内部使用，标识是否触发了检索


# 定义一个 contextvar，默认值为 None
_tool_context_var: contextvars.ContextVar = contextvars.ContextVar('tool_context', default=None)


def set_current_context(ctx: ToolContext):
    """设置当前请求的上下文（可在任何异步/线程环境中继承）"""
    _tool_context_var.set(ctx)


def get_current_context() -> Optional[ToolContext]:
    """获取当前请求的上下文"""
    return _tool_context_var.get()


def clear_current_context():
    """重置上下文（通常不必手动清理，请求结束会自动丢弃）"""
    _tool_context_var.set(None)