# memory/semantic_memory.py
"""
语义记忆检索 — 基于 BGE 向量的动态话题分段 + 摘要 + 召回

区间管理：
  每一轮用户消息与上一轮做向量余弦相似度
  → >= 0.75：话题延续，区间保持开放
  → < 0.75：区间闭合 → LLM 摘要 → 向量存储 → 开启新区间

检索：
  用户触发 → 当前问题向量与所有摘要向量做余弦匹配
  → 取 top-k → 拼接摘要 + 原始对话原文 → 返回 LLM 上下文
"""
import json
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from application.data_exporter import DataExporter
import config
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==================== BGE 模型（单例，全模块复用） ====================
_model = None


def get_model():
    """获取 BGE 模型单例"""
    global _model
    if _model is None:
        model_path = config.BGE_MODEL_PATH
        if model_path and os.path.exists(model_path):
            _model = SentenceTransformer(model_path)
        else:
            _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        print(f"[SemanticMemory] BGE 模型已加载")
    return _model


# ==================== LLM 摘要调用（同步，阻塞） ====================
def _generate_segment_abstract(
    messages_slice: list[dict],
    start_turn: int,
    end_turn: int
) -> dict:
    """
    调用 deepseek-chat 对区间对话进行摘要

    参数:
        messages_slice: 该区间的原始消息列表 [{"role":"user","content":"..."}, ...]
        start_turn/end_turn: 起止轮次

    返回:
        {
            "user_abstract": str,
            "ai_abstract": str,
            "importance": float
        }
    """

    # 构建对话原文
    dialog_lines = []
    for msg in messages_slice:
        role_label = "用户" if msg["role"] == "user" else "AI"
        dialog_lines.append(f"{role_label}: {msg['content']}")
    dialog_text = "\n".join(dialog_lines)

    prompt = f"""你是一个对话摘要助手。请根据以下对话片段，分别总结用户和AI的内容。

    对话片段（第 {start_turn} 轮 ~ 第 {end_turn} 轮）：
    {dialog_text}
    
    请输出 JSON：
    {{
      "user_abstract": "用户主要咨询了...，与...相关",
      "ai_abstract": "AI主要回复了...，与...相关",
      "importance": 0.6
    }}
    
    要求：
    1. user_abstract 总结用户提问的主题、需求、关心的内容
    2. ai_abstract 总结AI回复的核心内容
    3. importance 为信息重要性 0.3~0.9：纯寒暄 0.3，一般聊天 0.5~0.6，重要指令/关键信息 0.7~0.9
    4. 只输出 JSON，不要额外内容"""

    try:
        client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个精准的对话摘要助手，只输出指定 JSON 格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        print(f"[SemanticMemory] LLM 摘要生成完毕 (第{start_turn}-{end_turn}轮)")
        return {
            "user_abstract": result.get("user_abstract", ""),
            "ai_abstract": result.get("ai_abstract", ""),
            "importance": min(max(result.get("importance", 0.6), 0.3), 0.9)
        }
    except Exception as e:
        print(f"[SemanticMemory] LLM 摘要失败: {e}，使用后备摘要")
        # 列表推导式，取对应对象的输出内容前几十个字符
        user_msgs = [m["content"][:60] for m in messages_slice if m["role"] == "user"]
        ai_msgs = [m["content"][:100] for m in messages_slice if m["role"] == "assistant"]
        return {
            # 将区间里截取过的消息选前五段拼接起来
            "user_abstract": "；".join(user_msgs[:5]) if user_msgs else "无用户消息",
            "ai_abstract": "；".join(ai_msgs[:5]) if ai_msgs else "无AI回复",
            "importance": 0.5
        }


# ==================== 区间管理（在 web_ui 的回复流程中调用） ====================
def check_and_close_segment(
    session_id: str,
    user_msg_current: str,          # 当前轮用户消息
    current_turn: int,              # 当前轮次（从1开始）
    segment_start: int,             # 当前区间起始轮次
    interval_vec: Optional[np.ndarray],  # 当前区间累积归一化向量，None 表示新区间或无累积
    all_messages: list[dict]
) -> tuple[bool, int, np.ndarray]:
    """
    检查是否需要闭合区间（基于区间中心向量相似度）
    返回: (closed, new_segment_start, new_interval_vec)
    """
    model = get_model()
    curr_vec = model.encode(user_msg_current, normalize_embeddings=True)

    # 新区间或区间内尚无消息 → 直接当作新区间起点，不闭合
    if interval_vec is None or segment_start >= current_turn:
        return (False, segment_start, curr_vec)

    # 已有区间，计算相似度
    sim = float(np.dot(interval_vec, curr_vec))
    print(f"[SemanticMemory] 区间[{segment_start},{current_turn-1}] vs 轮次{current_turn} 相似度={sim:.4f}")

    if sim >= config.SEMANTIC_THRESHOLD:
        alpha = 0.4
        new_vec = (1 - alpha) * interval_vec + alpha * curr_vec
        new_vec = new_vec / np.linalg.norm(new_vec)

        # 重新归一化以保证单位长度
        new_vec = new_vec / np.linalg.norm(new_vec)
        return (False, segment_start, new_vec)

    # ====== 话题转变，闭合区间 ======
    end_turn = current_turn - 1
    print(f"[SemanticMemory] 话题转变！闭合区间 [{segment_start}, {end_turn}]")

    # 切片原文（假设消息列表严格交替且完整）
    start_idx = 2 * (segment_start - 1)
    end_idx = 2 * end_turn
    if end_idx > len(all_messages):
        end_idx = len(all_messages)
    slice_msgs = all_messages[start_idx:end_idx]

    if slice_msgs:
        abstract_data = _generate_segment_abstract(slice_msgs, segment_start, end_turn)
        combined_text = abstract_data["user_abstract"] + " " + abstract_data["ai_abstract"]
        vec = model.encode(combined_text, normalize_embeddings=True)

        record = {
            "user_abstract": abstract_data["user_abstract"],
            "ai_abstract": abstract_data["ai_abstract"],
            "start_turn": segment_start,
            "end_turn": end_turn,
            "importance": abstract_data["importance"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vector_path": ""   # 保留字段但暂时不使用
        }

        # 保存向量并获取路径
        vec_path = DataExporter.add_vector_and_get_path(session_id, segment_start, end_turn, vec)
        record["vector_path"] = vec_path  # ← 不再是空字符串

        # 保存摘要（此时 record 已包含 vector_path）
        DataExporter.add_segment_to_abstract(session_id, segment_start, end_turn, record)

        print(f"[SemanticMemory] ✅ 摘要已保存: abstract/{session_id}/{segment_start}_{end_turn}.json")
    else:
        print("[SemanticMemory] ⚠️ 切片为空，跳过摘要生成")

    # 新区间从当前轮开始，区间向量即为当前消息向量
    return (True, current_turn, curr_vec)

def semantic_retrieve(
    user_query: str,
    session_id: str,
    all_messages: list[dict],
    top_k: int = None
) -> str:
    start_time = time.time()

    if top_k is None:
        top_k = 1  # todo测试阶段固定为1

    # 1. 加载所有摘要
    abstracts = DataExporter.load_all_segments(session_id)
    if not abstracts:
        return "【对话记忆】当前没有任何历史话题摘要。"

    # 2. 加载所有向量
    vectors = DataExporter.load_all_vectors(session_id)

    # 3. 编码用户查询
    model = get_model()
    query_vec = model.encode(user_query, normalize_embeddings=True)

    print(f"[语义记忆检索] 查询：\"{user_query}\"")
    print(f"共加载 {len(abstracts)} 个摘要区间，开始计算得分...\n")

    # 4. 计算每个摘要的得分
    scored = []
    for idx, ab in enumerate(abstracts, start=1):
        key = (ab["start_turn"], ab["end_turn"])
        vec = vectors.get(key)
        if vec is None:
            continue
        sim = float(np.dot(query_vec, vec))
        if sim >= config.RETRIEVAL_MIN_SIMILARITY:
            importance = ab.get("importance", 0.5)
            final_score = 0.7 * sim + 0.3 * importance
            scored.append((final_score, ab))
            # 打印每个候选摘要的详细信息
            print(f"--- 候选 {idx}: 区间 [{ab['start_turn']}-{ab['end_turn']}] ---")
            print(f"  用户摘要: {ab.get('user_abstract', '')[:50]}...")
            print(f"  向量相似度: {sim:.4f}")
            print(f"  重要性 (importance): {importance:.2f}")
            print(f"  综合得分 (0.7*sim + 0.3*imp): {final_score:.4f}")
        else:
            # 仅作记录，不打印过多
            pass

    if not scored:
        print("[语义记忆检索] 无符合条件的摘要。")
        elapsed = time.time() - start_time
        print(f"总耗时: {elapsed:.2f} 秒\n")
        return "【对话记忆】未找到与您问题相关的历史话题。"

    # 按综合得分降序排序，取 top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    # 5. 打印最佳结果
    best_score, best_ab = top[0]
    print(f"\n===== 最佳记忆区间 =====")
    print(f"区间: 第 {best_ab['start_turn']} - {best_ab['end_turn']} 轮")
    print(f"用户摘要: {best_ab.get('user_abstract', '')}")
    print(f"AI摘要: {best_ab.get('ai_abstract', '')}")
    print(f"综合得分: {best_score:.4f}")

    # 6. 拼接上下文（保持原有格式）
    lines = ["【对话记忆 —— 相关历史话题】"]
    for rank, (score, ab) in enumerate(top, 1):
        lines.append(f"\n### 话题{rank} (第{ab['start_turn']}-{ab['end_turn']}轮，相关度:{score:.2f})")
        lines.append(f"用户话题: {ab['user_abstract']}")
        lines.append(f"AI回复概要: {ab['ai_abstract']}")

        # 原始对话片段
        start_idx = 2 * (ab["start_turn"] - 1)
        end_idx = min(2 * ab["end_turn"], len(all_messages))
        slice_msgs = all_messages[start_idx:end_idx]
        if slice_msgs:
            lines.append("\n原始对话片段:")
            for msg in slice_msgs[:5]:  # 只取5条，减少上下文长度
                role_label = "用户" if msg["role"] == "user" else "AI"
                content = msg["content"][:100]
                lines.append(f"  {role_label}: {content}")
            if len(slice_msgs) > 5:
                lines.append(f"  ... (共{len(slice_msgs)}条消息)")

    lines.append("\n请根据以上记忆片段回答用户的问题，优先引用其中的关键信息。")

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒\n")
    return "\n".join(lines)


# ==================== Tool 入口（供 web_ui 调用） ====================
def run_semantic_tool(user_query: str, session_id: str, all_messages: list[dict] = None) -> str:
    """
    语义记忆 Tool 入口
    参数:
        user_query: 用户输入
        session_id: 当前会话 ID
        all_messages: 完整消息列表（从 data_exporter 加载）

    返回:
        可直接注入 LLM 的上下文字符串
    """
    print("\n🔍 检测到用户有回忆意图，启动语义记忆检索...\n")
    if all_messages is None:
        session = DataExporter.load_session(session_id)
        if session:
            all_messages = session.get("messages", [])
        else:
            all_messages = []

    return semantic_retrieve(user_query, session_id, all_messages)