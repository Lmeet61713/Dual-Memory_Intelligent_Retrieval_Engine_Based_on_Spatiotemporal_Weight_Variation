# application/web_ui.py
"""
FastAPI 后端 —— 会话独立文件 + 流式聊天 + 语义记忆区间管理
（已升级为 LangChain Agent 决策）
支持 WebSocket 端到端移动端交互
"""
import numpy as np
from datetime import datetime
import sys
import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from decision.personal.Peronal_manager import PersonalManager
from memory.semantic_memory import check_and_close_segment
import uvicorn
import config
from application.data_exporter import DataExporter
from Agent.agent import get_agent
from Agent.tool_context import ToolContext, set_current_context, clear_current_context
from langchain_core.messages import HumanMessage, AIMessage
from decision.personal.item_habit_analyzer import get_habit_manager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ========================================
# FastAPI 应用
# ========================================
app = FastAPI(title="Homer Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# ========================================
# 首页
# ========================================
@app.get("/")
async def index():
    """首页"""
    index_path = os.path.join(config.STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Homer API is running."})

@app.get("/mobile")
async def mobile_page():
    """移动端页面"""
    mobile_path = os.path.join(config.STATIC_DIR, "mobile.html")
    if os.path.exists(mobile_path):
        return FileResponse(mobile_path)
    return JSONResponse({"message": "Mobile page not found. Please create mobile.html in static/."})

# ========================================
# 会话列表
# ========================================
@app.get("/api/sessions")
async def list_sessions():
    return DataExporter.list_sessions()

# ========================================
# 加载单个会话
# ========================================
@app.get("/api/sessions/{name:path}")
async def get_session(name: str):
    data = DataExporter.load_session(name)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return data

# ========================================
# 删除会话
# ========================================
@app.delete("/api/sessions/{name:path}")
async def delete_session(name: str):
    DataExporter.delete_session(name)
    return {"ok": True}


def _compute_interval_vector(segment_start: int, current_turn: int, all_messages: list[dict]):
    """从消息列表中提取 [segment_start, current_turn-1] 轮的所有用户消息，计算平均向量"""
    if segment_start >= current_turn:
        return None
    from memory.semantic_memory import get_model
    model = get_model()
    vecs = []
    turn = 0
    for msg in all_messages:
        if msg["role"] != "user":
            continue
        turn += 1
        if turn < segment_start:
            continue
        if turn > current_turn - 1:
            break
        vec = model.encode(msg["content"], normalize_embeddings=True)
        vecs.append(vec)

    if not vecs:
        return None
    avg = np.mean(vecs, axis=0)
    avg = avg / np.linalg.norm(avg)
    return avg


# application/web_ui.py

# ==================== 聊天预处理（提取复用，供 HTTP 和 WebSocket 共用） ====================
def _prepare_chat_context(session_id: str, user_message: str) -> dict:
    """
    聊天预处理：加载会话、记录用户消息、话题区间管理、设置工具上下文、构建消息列表。
    返回一个包含所有预处理结果的字典，供后续 Agent 流式生成使用。
    """
    # 1. 加载/创建会话，记录用户消息
    session = DataExporter.load_session(session_id)
    if session is None:
        session = DataExporter.create_session(session_id)

    DataExporter.append_message(session_id, "user", user_message)
    session = DataExporter.load_session(session_id)
    all_messages = session.get("messages", [])

    # 2. 计算 current_turn
    current_turn = session.get("count_session", 0) + 1

    # 3. 话题区间管理
    segment_start = session.get("segment_start", current_turn)
    interval_vec_data = session.get("interval_vec")

    if interval_vec_data is not None:
        interval_vec = np.array(interval_vec_data)
    else:
        interval_vec = _compute_interval_vector(
            segment_start, current_turn, all_messages
        )

    closed, new_segment_start, new_interval_vec = check_and_close_segment(
        session_id=session_id,
        user_msg_current=user_message,
        current_turn=current_turn,
        segment_start=segment_start,
        interval_vec=interval_vec,
        all_messages=all_messages,
    )

    # 持久化区间状态
    session["segment_start"] = new_segment_start
    session["interval_vec"] = new_interval_vec.tolist() if new_interval_vec is not None else None
    DataExporter.save_session(session_id, session)

    # 4. 设置工具上下文
    set_current_context(ToolContext(
        session_id=session_id,
        all_messages=all_messages,
    ))

    # 5. 构建 LangChain 消息列表
    recent = all_messages[-10:] if len(all_messages) > 20 else all_messages
    lc_messages = []
    for msg in recent:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # 6. 动态系统提示
    persona_mgr = PersonalManager(session_id=session_id)
    persona_summary = persona_mgr.update_and_get_summary(all_messages, current_turn)

    habit_mgr = get_habit_manager()
    habit_section = habit_mgr.get_formatted_for_prompt()

    dynamic_system = config.AGENT_SYSTEM_PROMPT + "\n\n" + persona_summary
    if habit_section:
        dynamic_system += "\n\n" + habit_section

    return {
        "session_id": session_id,
        "all_messages": all_messages,
        "current_turn": current_turn,
        "lc_messages": lc_messages,
        "dynamic_system": dynamic_system,
    }


def _run_agent_stream(agent, lc_messages: list, session_id: str):
    """
    Agent 流式生成器（同步），逐 chunk yield 内容字符串。
    完成后自动保存 assistant 回复到会话文件并清理上下文。
    """
    full_response = ""
    try:
        for chunk, _ in agent.stream(
            {"messages": lc_messages},
            stream_mode="messages"
        ):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
    finally:
        clear_current_context()
        if full_response:
            DataExporter.append_message(session_id, "assistant", full_response)


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "default")
    user_message = body.get("message", "")

    if not user_message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    ctx = _prepare_chat_context(session_id, user_message)
    agent = get_agent(ctx["dynamic_system"])

    async def generate():
        full_response = ""
        try:
            for chunk_content in _run_agent_stream(agent, ctx["lc_messages"], session_id):
                full_response += chunk_content
                yield f"data: {json.dumps({'c': chunk_content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'err': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ==================== 全局 Matcher 引用（不变） ====================
_matcher = None

def set_matcher(m):
    global _matcher
    _matcher = m

@app.get("/api/camera/status")
async def camera_status():
    if _matcher is None:
        return {"running": False}
    return {"running": _matcher.is_running}

@app.post("/api/camera/start")
async def camera_start():
    if _matcher is None:
        raise HTTPException(status_code=500, detail="Matcher 未初始化")
    _matcher.start()
    return {"running": _matcher.is_running}

@app.post("/api/camera/stop")
async def camera_stop():
    if _matcher is None:
        raise HTTPException(status_code=500, detail="Matcher 未初始化")
    _matcher.stop()
    return {"running": _matcher.is_running}

# ==================== 提醒接口 ====================
@app.get("/api/camera/reminders")
async def get_reminders():
    if _matcher is None:
        return {"reminders": []}
    reminders = _matcher.drain_reminders()
    return {"reminders": reminders}


def _get_persona_mgr(session_id: str) -> PersonalManager:
    """获取或创建画像管理器实例"""
    return PersonalManager(session_id=session_id)

@app.get("/api/persona/{session_id}")
async def get_persona(session_id: str):
    """获取指定会话的用户画像"""
    mgr = _get_persona_mgr(session_id)
    persona = mgr.persona
    # 按置信度降序排列每个维度
    sorted_persona = {}
    for dim, features in persona.items():
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        sorted_persona[dim] = [{"feature": k, "confidence": v} for k, v in sorted_features]
    return {
        "session_id": session_id,
        "persona": sorted_persona,
        "total_features": sum(len(v) for v in persona.values()),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.delete("/api/persona/{session_id}/{dimension}/{feature}")
async def delete_persona_feature(session_id: str, dimension: str, feature: str):
    """删除画像中某个特征"""
    # URL 中 feature 可能被截断，需要手动拼接
    mgr = _get_persona_mgr(session_id)
    if dimension in mgr.persona and feature in mgr.persona[dimension]:
        del mgr.persona[dimension][feature]
        mgr._save()
        return {"ok": True, "feature": feature}
    return {"ok": False, "reason": "特征不存在"}

# ========================================
# WebSocket 端点 —— 移动端端到端交互
# ========================================
@app.websocket("/ws/mobile")
async def mobile_websocket(websocket: WebSocket):
    """
    移动端 WebSocket 端点。
    消息协议（JSON）：
      客户端 → 服务端:
        {"type":"get_sessions"}
        {"type":"select_session","session_id":"..."}
        {"type":"send_message","session_id":"...","message":"..."}
        {"type":"new_session"}
      服务端 → 客户端:
        {"type":"session_list","data":[...]}
        {"type":"history","session_id":"...","messages":[...]}
        {"type":"reply_start","session_id":"..."}
        {"type":"reply_chunk","content":"..."}
        {"type":"reply_done","session_id":"..."}
        {"type":"error","message":"..."}
    """
    await websocket.accept()
    current_session_id: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "无效的 JSON 格式"}, ensure_ascii=False))
                continue

            msg_type = msg.get("type", "")

            # --- 获取会话列表 ---
            if msg_type == "get_sessions":
                sessions = DataExporter.list_sessions()
                session_data = []
                for sid in sessions:
                    s = DataExporter.load_session(sid)
                    count = len(s.get("messages", [])) if s else 0
                    session_data.append({"id": sid, "message_count": count})
                await websocket.send_text(json.dumps({
                    "type": "session_list",
                    "data": session_data
                }, ensure_ascii=False))

            # --- 新建会话 ---
            elif msg_type == "new_session":
                new_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                DataExporter.create_session(new_id)
                current_session_id = new_id
                await websocket.send_text(json.dumps({
                    "type": "new_session_created",
                    "session_id": new_id
                }, ensure_ascii=False))
                # 推送更新后的会话列表
                sessions = DataExporter.list_sessions()
                session_data = []
                for sid in sessions:
                    s = DataExporter.load_session(sid)
                    count = len(s.get("messages", [])) if s else 0
                    session_data.append({"id": sid, "message_count": count})
                await websocket.send_text(json.dumps({
                    "type": "session_list",
                    "data": session_data
                }, ensure_ascii=False))

            # --- 选择会话，全量加载历史 ---
            elif msg_type == "select_session":
                session_id = msg.get("session_id", "")
                if not session_id:
                    await websocket.send_text(json.dumps({"type": "error", "message": "缺少 session_id"}, ensure_ascii=False))
                    continue
                session = DataExporter.load_session(session_id)
                if session is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "会话不存在"}, ensure_ascii=False))
                    continue
                current_session_id = session_id
                messages = session.get("messages", [])
                await websocket.send_text(json.dumps({
                    "type": "history",
                    "session_id": session_id,
                    "messages": messages
                }, ensure_ascii=False))

            # --- 发送消息，触发 Agent 流式回复 ---
            elif msg_type == "send_message":
                session_id = msg.get("session_id", current_session_id or "default")
                user_text = msg.get("message", "")
                if not user_text:
                    await websocket.send_text(json.dumps({"type": "error", "message": "消息不能为空"}, ensure_ascii=False))
                    continue

                # 确保会话存在
                if not DataExporter.load_session(session_id):
                    DataExporter.create_session(session_id)
                current_session_id = session_id

                # 预处理（包含话题区间管理）
                ctx = _prepare_chat_context(session_id, user_text)
                agent = get_agent(ctx["dynamic_system"])

                # 通知客户端回复开始
                await websocket.send_text(json.dumps({
                    "type": "reply_start",
                    "session_id": session_id
                }, ensure_ascii=False))

                # 使用线程+队列将同步 Agent 流式生成桥接到异步 WebSocket
                import queue
                import threading
                chunk_queue: queue.Queue = queue.Queue()
                stream_done = threading.Event()

                def _stream_to_queue():
                    try:
                        for chunk_content in _run_agent_stream(agent, ctx["lc_messages"], session_id):
                            chunk_queue.put(("chunk", chunk_content))
                        chunk_queue.put(("done", None))
                    except Exception as e:
                        chunk_queue.put(("error", str(e)))
                    finally:
                        stream_done.set()

                thread = threading.Thread(target=_stream_to_queue, daemon=True)
                thread.start()

                try:
                    while not stream_done.is_set() or not chunk_queue.empty():
                        try:
                            item = chunk_queue.get(timeout=0.05)
                            kind, value = item
                            if kind == "chunk":
                                await websocket.send_text(json.dumps({
                                    "type": "reply_chunk",
                                    "content": value,
                                    "session_id": session_id
                                }, ensure_ascii=False))
                            elif kind == "error":
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": value
                                }, ensure_ascii=False))
                                break
                            elif kind == "done":
                                break
                        except queue.Empty:
                            continue

                    thread.join(timeout=30)

                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Agent 执行异常: {str(e)}"
                    }, ensure_ascii=False))

                # 通知客户端回复完成
                await websocket.send_text(json.dumps({
                    "type": "reply_done",
                    "session_id": session_id
                }, ensure_ascii=False))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"未知消息类型: {msg_type}"
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        print(f"[WebSocket] 移动端断开连接 (session: {current_session_id})")
    except Exception as e:
        print(f"[WebSocket] 异常: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

# ========================================
# 启动函数
# ========================================
def start_server():
    uvicorn.run(
        app,
        host=config.WEB_SERVER_HOST,
        port=config.WEB_SERVER_PORT,
        log_level="info",
        access_log=False        # 关闭访问日志
                )