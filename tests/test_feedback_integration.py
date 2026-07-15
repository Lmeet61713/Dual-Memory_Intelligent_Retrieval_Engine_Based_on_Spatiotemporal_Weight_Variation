# tests/test_feedback_integration.py
"""Module A HTTP API 集成测试 — 对运行中的服务器发送真实请求"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
import concurrent.futures

BASE = "http://127.0.0.1:8501"


def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    results = []

    print("=" * 60)
    print("  Module A HTTP API 集成测试")
    print("=" * 60)
    print()

    # Test 1: location_wrong
    print("[TEST 1] POST /feedback (location_wrong)")
    status, data = post_json(f"{BASE}/feedback", {
        "type": "location_wrong",
        "session_id": "integration_test_session",
        "payload": {"item_id": "test_cup", "correct_location": "厨房桌子", "confidence": 0.95},
    })
    print(f"  Status: {status}")
    print(f"  Response: {json.dumps(data, ensure_ascii=False)}")
    assert status == 200, f"Expected 200, got {status}"
    assert data.get("ok") is True
    assert data.get("event_id", "").startswith("fb_")
    results.append(("location_wrong", "PASS"))
    print("  PASS")

    # Test 2: memory_wrong
    print("[TEST 2] POST /feedback (memory_wrong)")
    status, data = post_json(f"{BASE}/feedback", {
        "type": "memory_wrong",
        "session_id": "integration_test_session",
        "payload": {"memory_id": "mem_test_001", "reason": "用户说记错了", "confidence": 0.85},
    })
    print(f"  Status: {status}")
    assert status == 200
    assert data.get("ok") is True
    results.append(("memory_wrong", "PASS"))
    print("  PASS")

    # Test 3: persona_wrong
    print("[TEST 3] POST /feedback (persona_wrong)")
    status, data = post_json(f"{BASE}/feedback", {
        "type": "persona_wrong",
        "session_id": "integration_test_session",
        "payload": {"dimension": "interests", "disagree_value": "喜欢古典乐", "suggest_value": "喜欢爵士乐", "confidence": 0.88},
    })
    print(f"  Status: {status}")
    assert status == 200
    assert data.get("ok") is True
    results.append(("persona_wrong", "PASS"))
    print("  PASS")

    # Test 4: 缺字段 -> 422
    print("[TEST 4] 缺字段请求 -> 应返回 422")
    status, data = post_json(f"{BASE}/feedback", {"type": "location_wrong"})
    print(f"  Status: {status}")
    if status == 422:
        results.append(("missing_field_422", "PASS"))
        print("  PASS (422 returned)")
    else:
        results.append(("missing_field_422", f"FAIL (got {status})"))
        print("  FAIL")
        assert False, f"Expected 422, got {status}"

    # Test 5: 无效反馈类型 -> 422
    print("[TEST 5] 无效反馈类型 -> 应返回 422")
    status, data = post_json(f"{BASE}/feedback", {
        "type": "invalid_type",
        "session_id": "test",
        "payload": {},
    })
    print(f"  Status: {status}")
    assert status == 422
    results.append(("invalid_type_422", "PASS"))
    print("  PASS")

    # Test 6: 置信度越界 -> 422
    print("[TEST 6] 置信度越界 (2.0) -> 应返回 422")
    status, data = post_json(f"{BASE}/feedback", {
        "type": "location_wrong",
        "session_id": "test",
        "payload": {"item_id": "cup", "confidence": 2.0},
    })
    print(f"  Status: {status}")
    assert status == 422
    results.append(("confidence_bounds_422", "PASS"))
    print("  PASS")

    # Test 7: JSONL 文件完整性
    print("[TEST 7] 验证 JSONL 文件已生成且格式正确")
    today = time.strftime("%Y-%m-%d")
    jsonl_path = os.path.join("data", "feedback", "events", f"{today}.jsonl")
    assert os.path.exists(jsonl_path), f"JSONL file not found: {jsonl_path}"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  文件路径: {jsonl_path}")
    print(f"  行数: {len(lines)}")
    assert len(lines) >= 3, f"Expected >=3 events, got {len(lines)}"
    for i, line in enumerate(lines[:5]):
        evt = json.loads(line)
        assert "event_id" in evt, f"Line {i+1} missing event_id"
        assert "type" in evt, f"Line {i+1} missing type"
        assert "session_id" in evt, f"Line {i+1} missing session_id"
        assert "timestamp" in evt, f"Line {i+1} missing timestamp"
        assert "payload" in evt, f"Line {i+1} missing payload"
    results.append(("jsonl_integrity", "PASS"))
    print("  PASS")

    # Test 8: GET /feedback/graph
    print("[TEST 8] GET /feedback/graph")
    req = urllib.request.Request(f"{BASE}/feedback/graph")
    resp = urllib.request.urlopen(req)
    graph = json.loads(resp.read())
    print(f"  Status: {resp.status}")
    print(f"  Nodes: {len(graph.get('nodes', []))}, Links: {len(graph.get('links', []))}")
    assert resp.status == 200
    assert "nodes" in graph
    assert "links" in graph
    results.append(("graph_endpoint", "PASS"))
    print("  PASS")

    # Test 9: 并发写入
    print("[TEST 9] 并发写入测试 (5 requests)")

    def send_feedback(i):
        return post_json(f"{BASE}/feedback", {
            "type": "memory_wrong",
            "session_id": f"concurrent_test_{i}",
            "payload": {"memory_id": f"concurrent_mem_{i}", "confidence": 0.8},
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_feedback, i) for i in range(5)]
        all_ok = all(f.result()[1].get("ok") for f in futures)

    assert all_ok, "Concurrent writes failed"
    results.append(("concurrent_writes", "PASS"))
    print("  PASS")

    # Test 10: 10 条反馈写入
    print("[TEST 10] 写入 10 条反馈（触发元学习阈值）")
    for i in range(7):
        status, data = post_json(f"{BASE}/feedback", {
            "type": "memory_wrong",
            "session_id": "batch_test",
            "payload": {"memory_id": f"batch_mem_{i}", "confidence": 0.8},
        })
        assert status == 200
    results.append(("batch_10_events", "PASS"))
    print("  PASS")

    # Summary
    print()
    print("=" * 60)
    print("  测试汇总")
    print("=" * 60)
    passed = sum(1 for _, r in results if "PASS" in r)
    failed = sum(1 for _, r in results if "FAIL" in r)
    for name, res in results:
        icon = "PASS" if "PASS" in res else "FAIL"
        print(f"  [{icon}] {name}")
    print(f"\n  总计: {passed} 通过, {failed} 失败, {len(results)} 项")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
