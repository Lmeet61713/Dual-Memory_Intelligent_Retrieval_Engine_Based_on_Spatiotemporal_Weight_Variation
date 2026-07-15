"""诊断 agent.stream() 的实际返回结构和内容"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.agent import get_agent
from langchain_core.messages import HumanMessage

agent = get_agent()

# 测试1: subgraphs=False (修改后的方式)
print("=" * 60)
print("测试1: subgraphs=False")
print("=" * 60)
count = 0
for item in agent.stream(
    {"messages": [HumanMessage(content="你好，请简单打招呼")]},
    stream_mode="messages",
    subgraphs=False
):
    count += 1
    print(f"\n--- 第 {count} 个 yield ---")
    print(f"  type(item) = {type(item)}")
    print(f"  len(item)  = {len(item) if hasattr(item, '__len__') else 'N/A'}")
    if hasattr(item, '__getitem__'):
        for i, elem in enumerate(item):
            print(f"  item[{i}] type = {type(elem).__name__}")
            if hasattr(elem, 'content'):
                c = elem.content
                print(f"  item[{i}].content type = {type(c).__name__}")
                print(f"  item[{i}].content = {repr(c)[:200]}")
            elif hasattr(elem, '__class__'):
                print(f"  item[{i}] = {repr(elem)[:200]}")
            else:
                print(f"  item[{i}] = {repr(elem)[:200]}")

print(f"\n总计: {count} 个 yield")

# 测试2: subgraphs=True (默认，修改前的方式)
print("\n" + "=" * 60)
print("测试2: subgraphs=True (默认)")
print("=" * 60)
count2 = 0
for item in agent.stream(
    {"messages": [HumanMessage(content="你好，请简单打招呼")]},
    stream_mode="messages",
    subgraphs=True
):
    count2 += 1
    print(f"\n--- 第 {count2} 个 yield ---")
    print(f"  type(item) = {type(item)}")
    print(f"  len(item)  = {len(item) if hasattr(item, '__len__') else 'N/A'}")
    if hasattr(item, '__getitem__'):
        for i, elem in enumerate(item):
            print(f"  item[{i}] type = {type(elem).__name__}")
            if hasattr(elem, 'content'):
                c = elem.content
                print(f"  item[{i}].content type = {type(c).__name__}")
                print(f"  item[{i}].content = {repr(c)[:200]}")
            elif hasattr(elem, '__class__'):
                print(f"  item[{i}] = {repr(elem)[:200]}")
            else:
                print(f"  item[{i}] = {repr(elem)[:200]}")

print(f"\n总计: {count2} 个 yield")
