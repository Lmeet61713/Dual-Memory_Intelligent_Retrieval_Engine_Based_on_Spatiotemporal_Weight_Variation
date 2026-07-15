"""测试 _run_agent_stream 三种路径的兜底保存行为"""
import json, os, sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import application.data_exporter as de
from application.web_ui import _run_agent_stream

# 模拟 Agent
class FakeChunk:
    def __init__(self, content=""):
        self.content = content

class FakeAgent:
    def __init__(self, scenario="normal"):
        self.scenario = scenario

    def stream(self, _input, stream_mode="messages"):
        if self.scenario == "normal":
            yield FakeChunk(content="Hello"), None
        elif self.scenario == "empty":
            yield FakeChunk(content=""), None
            yield FakeChunk(content=None), None
            yield FakeChunk(content=""), None
        elif self.scenario == "exception":
            raise RuntimeError("模拟 LLM 异常")
        elif self.scenario == "partial":
            yield FakeChunk(content="前半段"), None
            raise RuntimeError("中途异常")

def test_scenario(name, agent):
    print(f"\n{'='*50}")
    print(f"测试场景: {name}")
    print(f"{'='*50}")

    saved = []
    orig = de.DataExporter.append_message
    de.DataExporter.append_message = lambda sid, role, content: saved.append((role, content))

    try:
        chunks = list(_run_agent_stream(agent, [], f"test-{name}"))
        print(f"  生成 chunk 数: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  chunk[{i}]: {str(c)[:60]}")
        print(f"  保存消息数: {len(saved)}")
        for role, content in saved:
            print(f"  saved: {role} -> {str(content)[:60]}")
        if not saved:
            print("  ⚠️ 未保存任何消息！")
        elif saved[0][0] != "assistant":
            print(f"  ⚠️ 保存的不是 assistant！")
        else:
            print("  ✅ assistant 消息已保存")
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        import traceback; traceback.print_exc()
    finally:
        de.DataExporter.append_message = orig

test_scenario("正常有内容", FakeAgent("normal"))
test_scenario("空响应(全空chunk)", FakeAgent("empty"))
test_scenario("异常抛出", FakeAgent("exception"))
test_scenario("部分响应后异常", FakeAgent("partial"))
