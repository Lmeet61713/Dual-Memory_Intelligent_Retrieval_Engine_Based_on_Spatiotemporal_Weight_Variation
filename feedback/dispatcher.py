# feedback/dispatcher.py
"""
Module B — 反馈分发器（轻量 stub，供 Module A 调用）
完整实现: feedback/correctors.py + bayesian_update
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional
import config


# ====== 贝叶斯更新原语 ======
def bayesian_update(
    confidence: float,
    evidence_conf: float,
    direction: int,
    alpha: float = 0.5,
) -> float:
    """
    贝叶斯权重修正原语。

    原理: 将当前置信度视为先验，用户反馈视为证据。
    根据方向 (+1 加强, -1 削弱) 和证据强度 (evidence_conf × alpha) 更新后验。

    direction = +1: new = confidence + (1 - confidence) × evidence_conf × alpha
    direction = -1: new = confidence × (1 - evidence_conf × alpha)
    """
    evidence_strength = evidence_conf * alpha
    if direction >= 0:
        new_conf = confidence + (1.0 - confidence) * evidence_strength
    else:
        new_conf = confidence * (1.0 - evidence_strength)
    return max(0.01, min(0.99, new_conf))


# ====== dispatcher ======
def dispatch_correction(record: dict) -> bool:
    """
    分发反馈修正请求到对应的 corrector。
    返回 True 表示分发成功（包括 corrector 返回正确），
    False 表示处理过程中出现错误。
    """
    t = record.get("type", "")
    p = record.get("payload", {})
    try:
        if t == "location_wrong":
            _correct_location(
                p.get("item_id", ""),
                p.get("correct_location", ""),
                p.get("confidence", 0.9),
            )
        elif t == "memory_wrong":
            _correct_memory(
                p.get("memory_id", ""),
                p.get("reason", ""),
                p.get("confidence", 0.85),
            )
        elif t == "persona_wrong":
            _correct_persona(
                p.get("dimension", ""),
                p.get("disagree_value", ""),
                p.get("suggest_value"),
                p.get("confidence", 0.9),
                p.get("session_id", ""),
            )
        else:
            print(f"[dispatcher] 未知反馈类型: {t}")
        return True
    except Exception as e:
        _log_error("dispatch_fail", record, e)
        return False


# ====== 各 corrector ======
def _compute_adaptive_alpha(item_id: str) -> float:
    """根据反馈历史自适应计算贝叶斯修正的 alpha 参数"""
    spatial_path = config.SPATIAL_MEMORY_FILE
    if not os.path.exists(spatial_path):
        return getattr(config, 'FEEDBACK_BASE_ALPHA', 0.5)

    with open(spatial_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        if item.get("item_id", "") == item_id or item.get("name", "") == item_id:
            correction_history = item.get("correction_history", [])
            now = time.time()
            cutoff = now - 86400
            negation_count = sum(
                1 for h in correction_history
                if h.get("source") == "user" and h.get("delta", 0) < 0
                and h.get("ts", 0) >= cutoff
            )
            base_alpha = getattr(config, 'FEEDBACK_BASE_ALPHA', 0.5)
            alpha_step = getattr(config, 'FEEDBACK_ALPHA_STEP', 0.1)
            max_alpha = getattr(config, 'FEEDBACK_MAX_ALPHA', 0.9)
            return min(base_alpha + alpha_step * negation_count, max_alpha)

    return getattr(config, 'FEEDBACK_BASE_ALPHA', 0.5)


def _correct_location(item_id: str, correct_location: str, user_conf: float):
    """修正空间记忆中的物品位置和置信度"""
    spatial_path = config.SPATIAL_MEMORY_FILE
    if not os.path.exists(spatial_path):
        print(f"[corrector] spatial_memory.json 不存在，无法修正 {item_id}")
        return

    with open(spatial_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        if item.get("item_id", "") == item_id or item.get("name", "") == item_id:
            old_conf = item.get("confidence", item.get("weight", 0.5))
            alpha = _compute_adaptive_alpha(item_id)
            new_conf = bayesian_update(old_conf, user_conf, direction=-1, alpha=alpha)

            item["confidence"] = new_conf
            item["last_corrected_at"] = time.time()

            # correction_history
            hist = item.setdefault("correction_history", [])
            hist.append({
                "ts": time.time(),
                "delta": round(new_conf - old_conf, 4),
                "source": "user",
            })
            if len(hist) > 20:
                item["correction_history"] = hist[-20:]

            # 可选：更新位置
            if correct_location:
                item["corrected_location"] = correct_location

            # unstable_perception 标记
            _check_unstable(item)

            print(f"[corrector] 物品 {item_id} 置信度: {old_conf:.4f} → {new_conf:.4f}")

            with open(spatial_path, "w", encoding="utf-8") as fw:
                json.dump(items, fw, ensure_ascii=False, indent=2)

            # 同步调用 ConfidenceManager.apply_feedback_penalty
            from decision.confidence_manager import ConfidenceManager
            cm = ConfidenceManager()
            cm.apply_feedback_penalty(item_id, round(new_conf - old_conf, 4), "fb_auto_location", "location_wrong")

            return

    # 未在 spatial_memory.json 中找到，回退到 ConfidenceManager
    from decision.confidence_manager import ConfidenceManager
    cm = ConfidenceManager()
    cm.apply_feedback_penalty(item_id, -0.05, "fb_auto_location", "location_wrong")
    print(f"[corrector] 未在 spatial_memory.json 找到物品 {item_id}，已回退到 confidence_state.json 修正")


def _correct_memory(memory_id: str, reason: str, user_conf: float):
    """修正语义记忆 —— 在最近 5 个 session 的 abstracts 中查找并降权"""
    session_dir = config.MEMORY_DIR
    try:
        files = [f for f in os.listdir(session_dir) if f.endswith(".json")]
    except FileNotFoundError:
        print(f"[corrector] MEMORY_DIR 不存在: {session_dir}")
        return

    # 按修改时间排序，最近优先，取前 5
    files.sort(key=lambda f: os.path.getmtime(os.path.join(session_dir, f)), reverse=True)
    recent_files = files[:5]

    found = False
    for filename in recent_files:
        filepath = os.path.join(session_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception:
            continue

        abstracts = session.get("abstracts")
        if not abstracts or not isinstance(abstracts, list):
            continue

        for ab in abstracts:
            ab_id = ab.get("id") or ab.get("content", "")
            if str(memory_id) in str(ab_id):
                old_importance = ab.get("importance", 0.5)
                ab["importance"] = old_importance * 0.5
                ab["negation"] = time.time()
                found = True
                print(f"[corrector] 记忆 {memory_id} importance: {old_importance:.4f} → {ab['importance']:.4f}")
                break

        if found:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
            return

    if not found:
        error_path = os.path.join(config.FEEDBACK_ERRORS_DIR, f"{time.strftime('%Y-%m-%d')}.jsonl")
        os.makedirs(os.path.dirname(error_path), exist_ok=True)
        with open(error_path, "a", encoding="utf-8") as f:
            json.dump({
                "tag": "memory_not_found",
                "memory_id": memory_id,
                "timestamp": time.time(),
            }, f, ensure_ascii=False)
            f.write("\n")
        print(f"[corrector] 记忆未找到（已检查 {len(recent_files)} 个最近 session）: {memory_id}")


def _correct_persona(dimension: str, disagree_value: str, suggest_value: Optional[str], user_conf: float, session_id: str = ""):
    """修正用户画像 —— 优先使用 session_id 精确定位，否则回退到最近 3 个 session"""
    from decision.personal.Peronal_manager import PersonalManager

    def _apply_to_manager(mgr):
        if dimension not in mgr.persona:
            return False
        person = mgr.persona[dimension]
        if disagree_value not in person:
            return False
        old_conf = person[disagree_value]
        new_conf = bayesian_update(old_conf, user_conf, direction=-1, alpha=0.6)
        person[disagree_value] = new_conf
        print(f"[corrector] 画像 {dimension}.{disagree_value}: {old_conf:.4f} → {new_conf:.4f}")

        if suggest_value:
            person[suggest_value] = bayesian_update(
                person.get(suggest_value, 0.55),
                user_conf,
                direction=+1,
                alpha=0.4,
            )
            print(f"[corrector] 画像 {dimension}.{suggest_value}: → {person[suggest_value]:.4f}")
        mgr._save()
        return True

    # 优先：使用 session_id 精确查找
    if session_id:
        session_path = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
        if os.path.exists(session_path):
            mgr = PersonalManager(session_id=session_id)
            if _apply_to_manager(mgr):
                return

    # 回退：遍历最近 3 个修改过的 session 文件
    sessions_dir = config.MEMORY_DIR
    try:
        session_files = sorted(
            [f for f in os.listdir(sessions_dir) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(sessions_dir, f)),
            reverse=True,
        )
    except FileNotFoundError:
        print(f"[corrector] MEMORY_DIR 不存在: {sessions_dir}")
        return

    for sf in session_files[:3]:
        sid = sf.replace(".json", "")
        mgr = PersonalManager(session_id=sid)
        if _apply_to_manager(mgr):
            return

    print(f"[corrector] 未找到画像维度匹配: {dimension}.{disagree_value}")


def _check_unstable(item: dict):
    """检查 24 小时内是否被修正 ≥ 2 次，标记 unstable_perception"""
    now = time.time()
    hist = item.get("correction_history", [])
    recent = [h for h in hist if now - h.get("ts", 0) < 86400]
    flags = set(item.get("flags", []))
    if len(recent) >= 2:
        flags.add("unstable_perception")
    else:
        flags.discard("unstable_perception")
    item["flags"] = list(flags)


def _log_error(tag: str, record: dict, exc: Exception):
    """错误日志"""
    error_dir = os.path.join(config.BASE_DIR, "data", "feedback", "errors")
    os.makedirs(error_dir, exist_ok=True)
    day = time.strftime("%Y-%m-%d")
    err_path = os.path.join(error_dir, f"{day}.jsonl")
    with open(err_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "tag": tag,
            "record": record,
            "error": str(exc),
            "timestamp": time.time(),
        }, ensure_ascii=False) + "\n")


# ====== CLI replay 工具 ======
if __name__ == "__main__":
    """python -m feedback.dispatcher --replay <jsonl_path>"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=str, help="回放 JSONL 文件中的反馈事件")
    args = ap.parse_args()
    if args.replay:
        success = 0
        fail = 0
        with open(args.replay, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if dispatch_correction(record):
                    success += 1
                else:
                    fail += 1
        print(f"回放完成: 成功 {success}, 失败 {fail}")
