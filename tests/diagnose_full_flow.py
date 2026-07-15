"""完整模拟生产环境流式生成流程，测试 _run_agent_stream 是否产生内容"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.web_ui import _run_agent_stream, get_agent, _prepare_chat_context
from application.data_exporter import DataExporter
import config

# 创建测试会话
session_id = "test-flow-001"
DataExporter.create_session(session_id)

print("=" * 60)
print("测试完整流程: _prepare_chat_context → get_agent → _run_agent_stream")
print("=" * 60)

# Step 1: _prepare_chat_context
print("\n[Step 1] 调用 _prepare_chat_context...")
try:
    ctx = _prepare_chat_context(session_id, "你好，请简单打个招呼")
    print(f"  ✅ ctx keys: {list(ctx.keys())}")
    print(f"  ✅ lc_messages 数量: {len(ctx['lc_messages'])}")
    for m in ctx['lc_messages']:
        print(f"     - {type(m).__name__}: {repr(m.content)[:80]}")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 2: get_agent
print("\n[Step 2] 调用 get_agent()...")
try:
    agent = get_agent()
    print(f"  ✅ agent type: {type(agent)}")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 3: _run_agent_stream
print("\n[Step 3] 调用 _run_agent_stream() — 等待 LLM 响应...")
count = 0
full_response = ""
try:
    for text in _run_agent_stream(agent, ctx["lc_messages"], session_id):
        count += 1
        full_response += text
        print(f"  chunk #{count}: {repr(text)[:100]}")
except Exception as e:
    print(f"  ❌ 异常: {e}")
    import traceback; traceback.print_exc()

print(f"\n[结果] 收到 {count} 个 chunk")
print(f"[结果] full_response ({len(full_response)} 字符): {full_response[:300]}")

# 验证 session 是否保存了回复
print("\n[验证] 检查 session 文件...")
sess = DataExporter.load_session(session_id)
msgs = sess.get("messages", [])
for i, m in enumerate(msgs):
    content = str(m.get("content", ""))[:100]
    print(f"  [{i}] {m['role']}: {content}")
