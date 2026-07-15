
import threading
import time
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from memory.spatial_memory import run_spatial_tool, search_spatial_memory, format_spatial_result
from memory.semantic_memory import run_semantic_tool
from memory.compensation import create_buffer, clear_buffer, pop_next, get_remaining_count
from Agent.tool_context import get_current_context
# ---------- 参数模型：空间检索 ----------
class SpatialSearchInput(BaseModel):
    queries: List[str] = Field(
        description="""
        用户想要查找的物品名称或详细描述，例如 "水杯"、"带黑色杯盖的保温杯"。
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
        results = search_spatial_memory(q)          # 核心检索（Top-5）
        elapsed = time.perf_counter() - start_time

        # 终端打印耗时（用户不可见）
        print(f"[检索耗时] 查询 '{q}' 完成，耗时 {elapsed:.4f} 秒")

        # 构建返回内容（不含耗时）
        if results is None or len(results) == 0:
            lines.append(f"[{i}] {q}：未找到相关物品。")
        else:
            # 格式化 Top-1 结果用于显示
            best = results[0]
            # 存储 Top-1 物品信息到上下文，供反馈闭环使用
            if ctx and len(results) > 0:
                ctx.last_top1_item = results[0]['item']
            result_text = format_spatial_result(best)
            lines.append(f"[{i}] {q}：{result_text}")

            # 如果有更多候选，存入补偿队列
            if len(results) > 1 and ctx:
                # 清除之前的补偿缓冲
                if ctx.compensation_buffer_id:
                    clear_buffer(ctx.compensation_buffer_id)
                    print(f"[补偿机制] 已清除旧缓冲 {ctx.compensation_buffer_id}")
                # 创建新的补偿缓冲
                ctx.compensation_buffer_id = create_buffer(results[1:])
                print(f"[补偿机制] 已创建缓冲 {ctx.compensation_buffer_id}，候选数: {len(results) - 1}")

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


# ---------- 参数模型：用户纠正反馈 ----------
class CorrectionFeedbackInput(BaseModel):
    reason: str = Field(
        default="",
        description="用户否定的原因，如 'item_wrong'（物品不对）、'location_wrong'（位置不对）、'memory_wrong'（记忆不对）。可选，留空表示通用否定。"
    )


# ---------- 工具 3：用户纠正反馈 ----------
@tool(args_schema=CorrectionFeedbackInput)
def user_correction_feedback(reason: str = "") -> str:
    """
    当用户否定当前检索结果时调用此工具，从补偿队列中弹出下一个候选物品。
    
    调用时机：
    - 用户说"不对"、"不是这个"、"位置不对"等否定当前结果时
    - 用户纠正物品名称或位置时
    
    输入：否定的原因（可选），如 "item_wrong"、"location_wrong"
    输出：下一个候选物品的信息，或提示候选已耗尽
    """
    ctx = get_current_context()
    if ctx is None:
        return "【补偿机制】无法获取当前会话上下文，请重新搜索物品。"
    
    buffer_id = ctx.compensation_buffer_id
    if not buffer_id:
        return "【补偿机制】当前没有备选物品。请调用 search_item_location 重新搜索，或请用户提供更多线索。"
    
    # 记录反馈原因（日志用）
    if reason:
        print(f"[补偿机制] 用户否定，原因: {reason}，buffer_id: {buffer_id}")

    # ====== 闭环反馈：对 Top-1 物品执行权重惩罚 ======
    last_top1_item = ctx.last_top1_item
    if last_top1_item:
        feedback_record = {
            "type": "location_wrong",
            "session_id": ctx.session_id,
            "payload": {
                "item_id": last_top1_item.get('name', 'unknown'),
                "confidence": 0.9,
            }
        }

        def _feedback_thread():
            from feedback.dispatcher import dispatch_correction
            try:
                dispatch_correction(feedback_record)
                print(f"[闭环反馈] 已对 {last_top1_item.get('name')} 提交位置错误修正")
            except Exception as e:
                print(f"[闭环反馈] 修正失败: {e}")

        threading.Thread(target=_feedback_thread, daemon=True).start()
        ctx.last_top1_item = None  # 清除，避免重复修正

    # 弹出下一候选
    next_candidate = pop_next(buffer_id)
    if next_candidate is None:
        # 队列耗尽
        ctx.compensation_buffer_id = None
        return (
            "【补偿机制】备选物品已全部耗尽。\n"
            "请用户提供更多线索（如物品的颜色、形状、附近有什么），"
            "然后调用 search_item_location 重新搜索。"
        )
    
    remaining = get_remaining_count(buffer_id)
    result_text = format_spatial_result(next_candidate)
    
    if remaining == 0:
        ctx.compensation_buffer_id = None
    
    print(f"[补偿机制] 弹出候选: {next_candidate['item']['name']}，剩余 {remaining} 个")
    return f"{result_text}\n（备选剩余 {remaining} 个）"


# ---------- 工具列表 ----------
ALL_TOOLS = [
    search_item_location,
    search_conversation_memory,
    user_correction_feedback,
]