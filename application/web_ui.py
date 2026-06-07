# application/web_ui.py
"""
FastAPI 后端 —— 会话独立文件 + 流式聊天 + 语义记忆区间管理
（已升级为 LangChain Agent 决策）
"""
import numpy as np
import asyncio
import sys
import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from memory.semantic_memory import check_and_close_segment
import uvicorn
import config
from application.data_exporter import DataExporter
from memory.agent import get_agent
from memory.tool_context import ToolContext, set_current_context, clear_current_context
from langchain_core.messages import HumanMessage, AIMessage
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
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()             # await的作用是获取请求体
    session_id = body.get("session_id", "default")
    user_message = body.get("message", "")

    if not user_message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # ========== 1. 加载/创建会话，记录用户消息 ==========
    session = DataExporter.load_session(session_id)
    if session is None:
        session = DataExporter.create_session(session_id)

    DataExporter.append_message(session_id, "user", user_message)
    session = DataExporter.load_session(session_id)
    all_messages = session.get("messages", [])

    # ========== 2. 计算 current_turn（关键修复！） ==========
    # count_session 只在 AI 回复后 +1，所以本轮编号需要 +1
    current_turn = session.get("count_session", 0) + 1

    # ========== 3. 话题区间管理（每轮都要执行） ==========
    segment_start = session.get("segment_start", current_turn)
    interval_vec_data = session.get("interval_vec")

    if interval_vec_data is not None:
        interval_vec = np.array(interval_vec_data)
    else:
        # 重启后没有向量缓存，从历史消息重新计算
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

    # ========== 4. 设置工具上下文（供 Agent 工具使用） ==========
    set_current_context(ToolContext(
        session_id=session_id,
        all_messages=all_messages,
    ))

    # ========== 5. 构建 LangChain 消息列表 ==========
    recent = all_messages[-10:] if len(all_messages) > 20 else all_messages
    lc_messages = []
    for msg in recent:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # ========== 6. Agent 流式生成 ==========
    agent = get_agent()

    async def generate():
        full_response = ""
        try:
            for chunk, _ in agent.stream(
                {"messages": lc_messages},
                stream_mode="messages"
            ):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'c': chunk.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'err': str(e)})}\n\n"
        finally:
            clear_current_context()
            if full_response:
                DataExporter.append_message(session_id, "assistant", full_response)

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