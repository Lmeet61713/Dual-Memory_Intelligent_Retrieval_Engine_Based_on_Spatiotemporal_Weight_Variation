"""验证 JSONL 内容格式和 feedback 目录结构"""
import json
import os
import time

today = time.strftime("%Y-%m-%d")
jsonl_path = os.path.join("data", "feedback", "events", f"{today}.jsonl")
print("JSONL 内容验证:")
print("=" * 60)
i = 0
with open(jsonl_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        evt = json.loads(line)
        print(f"#{i+1}: id={evt['event_id']}, type={evt['type']}, "
              f"session={evt['session_id'][:20]}, ts={evt['timestamp']:.0f}")
        for field in ["event_id", "type", "session_id", "timestamp", "payload"]:
            assert field in evt, f"Missing {field} in event #{i+1}"

print(f"\n共 {i+1} 条记录，格式全部正确")
print("=" * 60)

print("\n目录结构:")
for root, dirs, files in os.walk("data/feedback"):
    level = root.replace("data/feedback", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = "  " * (level + 1)
    for f in sorted(files):
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        print(f"{subindent}{f} ({size} bytes)")
