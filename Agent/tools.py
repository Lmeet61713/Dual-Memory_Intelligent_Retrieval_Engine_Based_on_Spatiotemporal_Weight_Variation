
import time
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from memory.spatial_memory import run_spatial_tool
from memory.semantic_memory import run_semantic_tool
from Agent.tool_context import get_current_context
# ---------- 参数模型：空间检索 ----------
class SpatialSearchInput(BaseModel):
    queries: List[str] = Field(
        description="""用户想要查找的物品名称或详细描述，例如 "水杯"、"带黑色杯盖的保温杯"。
                请尽量从用户原话中提取最核心的物品名，只传入一个 query。
                实在模糊提取不到有效名称再生成一个类似物品名称。
                """
    )


# ---------- 工具 1：查找物品位置（带计时） ----------
@tool(args_schema=SpatialSearchInput)
def search_item_location(queries: List[str]) -> str:
    """
    查找物品在家庭环境中的位置。
    输入：一个物品的名称或特征描述。
    输出：只返回最佳匹配结果，格式为自然语言描述。
    """
    # 只要执行记忆检索，就触发全局变量更替
    ctx = get_current_context()
    if ctx:
        ctx.retrieval_triggered = True
    if not queries:
        return "【环境记忆】没有提供任何查询条件。"

    lines = []
    for i, q in enumerate(queries, start=1):
        start_time = time.perf_counter()
        result = run_spatial_tool(q)          # 核心检索
        elapsed = time.perf_counter() - start_time

        # 终端打印耗时（用户不可见）
        print(f"[检索耗时] 查询 '{q}' 完成，耗时 {elapsed:.4f} 秒")

        # 构建返回内容（不含耗时）
        if result is None:
            lines.append(f"[{i}] {q}：未找到相关物品。")
        else:
            lines.append(f"[{i}] {q}：{result}")

    return "\n".join(lines)


# ---------- 工具 2：搜索对话记忆（不变） ----------
class MemorySearch(BaseModel):
    query: str = Field(description="当用户询问与空间物品无关的问题时、当用户询问还记不记的之前聊过的内容（涉及学习、工作等具有回忆倾向的话题调用）")
@tool(args_schema=MemorySearch)
def search_conversation_memory(query: str) -> str:
    """
    搜索对话历史中的话题记忆。
    当用户回忆或询问之前聊过的事情、想继续之前的话题时调用。
    输入：用户想回忆的话题关键词，如 "昨天说的那个餐厅"、"上次聊到的方案"
    输出：相关的历史对话摘要和原始片段
    """
    # 只要执行记忆检索，就触发全局变量更替
    ctx = get_current_context()
    if ctx:
        ctx.retrieval_triggered = True
    if ctx is None:
        return "【对话记忆】无法获取当前会话上下文。"
    result = run_semantic_tool(query, ctx.session_id, ctx.all_messages)
    if result is None:
        return "【对话记忆】未找到相关历史话题。"
    return result


# ---------- 工具列表 ----------
ALL_TOOLS = [
    search_item_location,
    search_conversation_memory,
]