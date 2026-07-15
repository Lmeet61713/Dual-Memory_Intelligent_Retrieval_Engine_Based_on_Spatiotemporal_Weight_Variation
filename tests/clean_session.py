"""清理会话文件中由于 content 空响应导致的连续 user 消息污染"""
import json
import os

filepath = r"e:\mmmmmmmmmmmmmmmm\MyAgent\data\memory\2026-07-12T03_17_28_302Z.json"
with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

messages = data.get("messages", [])
print(f"原始消息数: {len(messages)}")

# 规则：如果存在连续 user 消息（中间没有 assistant），说明 assistant content 丢失
# 删除多余的连续 user 消息，只保留第一个
cleaned = []
i = 0
removed = 0
while i < len(messages):
    msg = messages[i]
    if msg["role"] == "user":
        # 收集连续的 user 消息
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "user":
            print(f"  发现连续 user 消息，移除: [{j}] '{messages[j]['content'][:30]}...'")
            j += 1
            removed += 1
        # 只保留第一个 user 消息
        cleaned.append(msg)
        i = j
    else:
        cleaned.append(msg)
        i += 1

print(f"清理后消息数: {len(cleaned)} (移除了 {removed} 条连续 user 消息)")

# 同时修复 segment_start 和 count_session
data["messages"] = cleaned
data["count_session"] = sum(1 for m in cleaned if m["role"] == "assistant")
data["segment_start"] = data["count_session"]

# 备份原文件
backup_path = filepath.replace(".json", "_backup.json")
with open(backup_path, "w", encoding="utf-8") as f:
    json.dump(json.load(open(filepath, "r", encoding="utf-8")), f, ensure_ascii=False, indent=2)
print(f"已备份原文件到: {backup_path}")

# 保存清理后的文件
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"已保存清理后文件: {filepath}")

# 打印最终消息
print("\n最终消息列表:")
for idx, m in enumerate(cleaned):
    content_preview = m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]
    print(f"  [{idx}] {m['role']}: {content_preview}")
