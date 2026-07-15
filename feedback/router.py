# feedback/router.py
"""
Module A — 反馈接口 FastAPI Router
POST /feedback 接收三类反馈，写入 JSONL，调用 dispatcher
"""
from __future__ import annotations
import json
import os
import time
import uuid
import threading
from fastapi import APIRouter, HTTPException
from feedback.schemas import FeedbackRequest, FeedbackEvent
from feedback.dispatcher import dispatch_correction
from feedback.metrics import record_metric
import config

router = APIRouter(prefix="/feedback", tags=["feedback"])

# JSONL 写入锁（线程安全）
_jsonl_lock = threading.Lock()


def _get_events_file_path(day: str | None = None) -> str:
    """获取当天的事件 JSONL 文件路径"""
    if day is None:
        day = time.strftime("%Y-%m-%d")
    events_dir = os.path.join(config.BASE_DIR, "data", "feedback", "events")
    os.makedirs(events_dir, exist_ok=True)
    return os.path.join(events_dir, f"{day}.jsonl")


def _generate_event_id() -> str:
    """生成全局唯一 event_id: fb_YYYYMMDD_xxxx"""
    short_uid = uuid.uuid4().hex[:6]
    return f"fb_{time.strftime('%Y%m%d')}_{short_uid}"


@router.post("")
async def submit_feedback(req: FeedbackRequest):
    """
    接收用户对 AI 回答的反馈。

    支持三类反馈:
    - location_wrong: 物品位置不正确
    - memory_wrong: 对话记忆被否认
    - persona_wrong: 人格判断被否认

    返回: {"ok": true, "event_id": "fb_..."}
    """
    start_time = time.time()

    # 1. 生成 event_id
    event_id = _generate_event_id()

    # 2. 构造完整事件
    try:
        event = FeedbackEvent.from_request(req, event_id)
    except ValueError as e:
        record_metric(req.type, (time.time() - start_time) * 1000, False)
        raise HTTPException(status_code=422, detail=f"payload 校验失败: {str(e)}")
    except Exception as e:
        record_metric(req.type, (time.time() - start_time) * 1000, False)
        raise HTTPException(status_code=500, detail=f"事件构造失败: {str(e)}")

    # 3. 持久化到 JSONL
    event_dict = event.model_dump(mode="json")
    try:
        with _jsonl_lock:
            filepath = _get_events_file_path()
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
    except OSError as e:
        record_metric(req.type, (time.time() - start_time) * 1000, False)
        raise HTTPException(status_code=500, detail=f"JSONL 写入失败: {str(e)}")

    # 4. 调用 dispatcher 执行权重修正（Module B）
    record = {
        "event_id": event_id,
        "type": req.type,
        "session_id": req.session_id,
        "timestamp": event.timestamp,
        "payload": req.payload,
    }
    dispatch_ok = dispatch_correction(record)

    record_metric(req.type, (time.time() - start_time) * 1000, dispatch_ok)
    return {"ok": True, "event_id": event_id}


@router.get("/stats")
async def get_stats(hours: int = 24):
    """
    获取反馈指标聚合数据。
    可选参数 hours: 统计最近多少小时的数据（默认 24）
    """
    from feedback.metrics import get_stats as metrics_get_stats
    return metrics_get_stats(hours)


@router.get("/graph")
async def get_graph(hours: int = 24):
    """
    获取共生记忆图谱数据（Module D 后端接口）。
    返回: {"nodes": [...], "links": [...]}
    """
    from collections import Counter
    nodes_set = set()
    links_counter = Counter()

    now = time.time()
    cutoff = now - hours * 3600

    events_dir = os.path.join(config.BASE_DIR, "data", "feedback", "events")
    if not os.path.exists(events_dir):
        return {"nodes": [], "links": []}

    for filename in sorted(os.listdir(events_dir)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(events_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("timestamp", 0) < cutoff:
                        continue
                    t = evt.get("type", "")
                    p = evt.get("payload", {})
                    session_id = evt.get("session_id", "unknown")
                    nodes_set.add(session_id)

                    if t == "location_wrong":
                        item = p.get("item_id", "unknown")
                        nodes_set.add(item)
                        links_counter[(session_id, item)] += 1
                    elif t == "persona_wrong":
                        dim = p.get("dimension", "unknown")
                        nodes_set.add(dim)
                        links_counter[(session_id, dim)] += 1
                    elif t == "memory_wrong":
                        mid = p.get("memory_id", "unknown")
                        nodes_set.add(mid)
                        links_counter[(session_id, mid)] += 1
        except Exception as e:
            print(f"[graph] 读取 {filepath} 出错: {e}")

    nodes = [
        {"id": n, "name": n, "category": _node_category(n)}
        for n in nodes_set
    ]
    links = [
        {"source": s, "target": t, "value": v}
        for (s, t), v in links_counter.items()
    ]
    return {"nodes": nodes, "links": links}


def _node_category(node_id: str) -> int:
    """0=session, 1=item, 2=dimension, 3=memory"""
    if len(node_id) >= 10 and "_" in node_id:
        return 0  # session_id
    if node_id in {"personality_traits", "interests", "values", "communication_style", "habits"}:
        return 2  # persona dimension
    return 1  # item or memory
