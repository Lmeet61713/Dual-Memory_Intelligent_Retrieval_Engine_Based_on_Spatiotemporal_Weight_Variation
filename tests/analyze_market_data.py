"""修正版：深度统计分析 xlsx 问卷原始数据"""
from collections import Counter
from openpyxl import load_workbook

xlsx_path = r"e:\mmmmmmmmmmmmmmmm\MyAgent\市场调研\359574280_按文本_智能物品记忆与定位系统需求调查问卷_97_97-1.xlsx"
wb = load_workbook(xlsx_path, data_only=True)
ws = wb[wb.sheetnames[0]]

# Read data
rows = []
for r in range(2, ws.max_row + 1):
    row = {}
    for c in range(1, ws.max_column + 1):
        val = ws.cell(row=r, column=c).value
        key = str(ws.cell(row=1, column=c).value or "")
        row[key] = str(val).strip() if val is not None else ""
    rows.append(row)

# Key mapping from actual headers
K_BH = "序号"
K_TIME = "提交答卷时间"
K_DUR = "所用时间"
K_SRC = "来源"
K_SRC_D = "来源详情"
K_IP = "来自IP"
K_ID = "1、您的身份是"
K_AGE = "2、您的年龄段是"
K_GENDER = "3、您的性别是"
K_ENV = "4、您的居住环境是"
K_FREQ = "5、您是否经常找不到物品"
K_ITEMS = "6、您经常找不到的物品类型是"
K_EMOTION = "7、您找不到物品的感受是"
K_ACCEPT = "8、如果有一个系统能自动记录物品位置，您是否愿意使用"
K_FUNCS = "9、您希望系统具备哪些功能"
K_PLATFORM = "10、您更倾向于哪种交互方式"
K_CAM = "11、您是否愿意接受摄像头用于识别物品"
K_PRIV = "12、您担心的问题是"
K_RESULT = "13、当系统不确定物品时，您希望他怎么做"
K_REMIND = "14、您是否希望他主动提醒（如药物，贵重物品）"
K_SCENE = "15、您认为系统最适合投入的场景是"
# Fix: read the actual header text from column 22 to handle special quotes
K_CHECK = str(ws.cell(row=1, column=22).value or "")
K_SUGGEST = "17、您觉得产品还需要有什么功能"

# Filter valid: Q16 attention check must be "从不" (per docx)
valid = [r for r in rows if r.get(K_CHECK, "") == "从不"]
invalid = [r for r in rows if r.get(K_CHECK, "") != "从不"]
print(f"=== 有效样本筛选 (Q16 == '从不') ===")
print(f"  有效: {len(valid)}, 无效: {len(invalid)}")
print(f"  排除理由: {Counter(r.get(K_CHECK, '') for r in invalid)}")
print()

# Use valid samples
rows = valid
total = len(rows)

# ===== 用户画像 =====
print("=" * 60)
print(f"一、用户画像分析 (n={total})")
print("=" * 60)

iden = Counter(r.get(K_ID, "") for r in rows)
print("\n【身份】")
for k, v in iden.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

age = Counter(r.get(K_AGE, "") for r in rows)
print("\n【年龄段】")
for k, v in age.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

gender = Counter(r.get(K_GENDER, "") for r in rows)
print("\n【性别】")
for k, v in gender.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

env = Counter(r.get(K_ENV, "") for r in rows)
print("\n【居住环境】")
for k, v in env.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

# ===== 找物行为 =====
print("\n" + "=" * 60)
print("二、找物行为分析")
print("=" * 60)

freq = Counter(r.get(K_FREQ, "") for r in rows)
print("\n【找物频率】")
for k, v in freq.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

items = Counter()
for r in rows:
    for item in r.get(K_ITEMS, "").split("┋"):
        item = item.strip()
        if item:
            items[item] += 1
print("\n【高频丢失物品（多选）】")
for k, v in items.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

emotion = Counter(r.get(K_EMOTION, "") for r in rows)
print("\n【情绪影响】")
for k, v in emotion.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

# Cross: freq x emotion
print("\n【找物频率 × 情绪 交叉表】")
print(f"  {'频率':<8s} {'情绪':<12s} {'人数':>5s} {'行占比':>8s}")
for f in ["经常", "偶尔", "很少"]:
    fv = freq.get(f, 0)
    if fv == 0: continue
    for e in emotion:
        cnt = sum(1 for r in rows if r.get(K_FREQ, "") == f and r.get(K_EMOTION, "") == e)
        if cnt > 0:
            print(f"  {f:<8s} {e:<12s} {cnt:>5d} {cnt/fv*100:>7.1f}%")

# ===== 采纳意愿 =====
print("\n" + "=" * 60)
print("三、采纳意愿与功能需求")
print("=" * 60)

accept = Counter(r.get(K_ACCEPT, "") for r in rows)
print("\n【采纳意愿】")
for k, v in accept.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

funcs = Counter()
for r in rows:
    for item in r.get(K_FUNCS, "").split("┋"):
        item = item.strip()
        if item:
            funcs[item] += 1
print("\n【功能需求（多选）】")
for k, v in funcs.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

platform = Counter(r.get(K_PLATFORM, "") for r in rows)
print("\n【交互平台偏好】")
for k, v in platform.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

# Cross: age x platform
print("\n【年龄 × 平台 交叉】")
for a in age:
    for p in platform:
        cnt = sum(1 for r in rows if r.get(K_AGE, "") == a and r.get(K_PLATFORM, "") == p)
        if cnt > 0:
            print(f"  {a}: {p} = {cnt}")

result_p = Counter(r.get(K_RESULT, "") for r in rows)
print("\n【结果呈现偏好】")
for k, v in result_p.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

# ===== 摄像头与隐私 =====
print("\n" + "=" * 60)
print("四、摄像头接受度与隐私分析")
print("=" * 60)

cam = Counter(r.get(K_CAM, "") for r in rows)
print("\n【摄像头接受度】")
for k, v in cam.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

priv = Counter()
for r in rows:
    for item in r.get(K_PRIV, "").split("┋"):
        item = item.strip()
        if item:
            priv[item] += 1
print("\n【隐私担忧（多选）】")
for k, v in priv.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

# Cross: cam x priv
print("\n【摄像头态度 × 隐私担忧】")
for c in cam:
    for p in priv:
        cnt = sum(1 for r in rows if r.get(K_CAM, "") == c and p in r.get(K_PRIV, ""))
        if cnt > 0:
            print(f"  [{c}] 担忧 {p}: {cnt}")

# ===== 场景分析 =====
print("\n" + "=" * 60)
print("五、应用场景分析")
print("=" * 60)

scenes = Counter()
for r in rows:
    for item in r.get(K_SCENE, "").split("┋"):
        item = item.strip()
        if item:
            scenes[item] += 1
print("\n【适用场景（多选）】")
for k, v in scenes.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

remind = Counter(r.get(K_REMIND, "") for r in rows)
print("\n【主动提醒需求】")
for k, v in remind.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

# Cross: iden x scene
print("\n【身份 × 场景 交叉】")
for i in iden:
    for s in scenes:
        cnt = sum(1 for r in rows if r.get(K_ID, "") == i and s in r.get(K_SCENE, ""))
        if cnt > 0:
            print(f"  {i}: {s} = {cnt}")

# ===== 竞品分析 =====
print("\n" + "=" * 60)
print("六、用户建议汇总")
print("=" * 60)
for r in rows:
    sug = r.get(K_SUGGEST, "").strip()
    if sug and sug != "(空)" and sug != "无":
        print(f"  [{r.get(K_ID, '')}, {r.get(K_AGE, '')}, {r.get(K_GENDER, '')}] {sug}")

# ===== 高频用户专项分析 =====
print("\n" + "=" * 60)
print("七、高频失物用户专项分析（找物频率='经常'的子群）")
print("=" * 60)
high_freq = [r for r in rows if r.get(K_FREQ, "") == "经常"]
hf_total = len(high_freq)
print(f"  高频人群: {hf_total} ({hf_total/total*100:.1f}%)")

hf_funcs = Counter()
for r in high_freq:
    for item in r.get(K_FUNCS, "").split("┋"):
        item = item.strip()
        if item:
            hf_funcs[item] += 1
print("  功能需求:")
for k, v in hf_funcs.most_common():
    print(f"    {k}: {v} ({v/hf_total*100:.1f}%)")

hf_scenes = Counter()
for r in high_freq:
    for item in r.get(K_SCENE, "").split("┋"):
        item = item.strip()
        if item:
            hf_scenes[item] += 1
print("  应用场景:")
for k, v in hf_scenes.most_common():
    print(f"    {k}: {v} ({v/hf_total*100:.1f}%)")

wb.close()
