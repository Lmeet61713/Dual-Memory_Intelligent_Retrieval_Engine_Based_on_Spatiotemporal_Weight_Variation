# config.py
"""
全局配置参数 —— 所有模块统一引用此文件
"""
import os

# ========================================
# 路径配置
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # 项目根目录

# 数据存储
DATA_DIR = os.path.join(BASE_DIR, "data")
MEMORY_DIR = os.path.join(DATA_DIR, "memory")        # 语义记忆（对话记录/摘要）
SPACE_DIR = os.path.join(DATA_DIR, "space")          # 空间记忆（物品记录）
VECTORS_DIR = os.path.join(DATA_DIR, "vectors")      # 向量 .npy 文件
ABSTRACT_DIR = os.path.join(DATA_DIR, "abstract")    # 摘要存储根目录

for d in [DATA_DIR, MEMORY_DIR, SPACE_DIR, VECTORS_DIR, ABSTRACT_DIR]:
    os.makedirs(d, exist_ok=True)

# 具体文件
SPATIAL_MEMORY_FILE = os.path.join(SPACE_DIR, "spatial_memory.json")
CONFIDENCE_STATE_FILE = os.path.join(SPACE_DIR, "confidence_state.json")
SEMANTIC_SESSION_FILE = os.path.join(MEMORY_DIR, "chat_sessions.json")

# 模型
YOLO_MODEL_PATH_GLOBAL = os.path.join(BASE_DIR, "models", "best_global.pt")
YOLO_MODEL_PATH_MOBILE = os.path.join(BASE_DIR, "models", "best_mobile.pt")
BGE_MODEL_PATH = "C:/Users/ASUS/bge_model"

# ========================================
# 摄像头配置
# ========================================
GLOBAL_CAM_ID = 2               # 全局摄像头设备索引
MOBILE_CAM_ID = 1               # 移动摄像头设备索引
CAM_FRAME_WIDTH = 640
CAM_FRAME_HEIGHT = 480
CAM_FRAME_WIDTH_ = 640          # 第二组摄像头（备用）
CAM_FRAME_HEIGHT_ = 480

# ========================================
# 检测与匹配
# ========================================
DETECTION_CONF_THRESHOLD = 0.4   # YOLO 检测置信度阈值
MOVE_THRESHOLD_PIXELS = 0.2       # 坐标变化超过5厘米 → 判定为移动
REF_DISTANCE_THRESHOLD = 200     # 参照物判定距离（像素）

# ========================================
# 空间编号
# ========================================
SPACE_IDS = [0]                  # 当前只有一个大空间

# ========================================
# 物理空间参数（全局摄像头俯视）
# ========================================
PHYSICAL_SPACE_WIDTH = 40.0    # 真实矩形平面的长（厘米）
PHYSICAL_SPACE_HEIGHT = 40.0   # 真实矩形平面的宽（厘米）

# ========================================
# LLM API 配置
# ========================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 系统提示词模板（可能被 Agent system prompt 覆盖，保留兼容）
SYSTEM_PROMPT_TEMPLATE = """你是一个智能家庭助手，名叫 Homer。
你可以帮助用户查找家中物品的位置，也可以和用户闲聊。

规则：
1. 回复简洁、亲切，适当使用颜文字
2. 如果用户询问物品位置，优先根据提供的环境记忆回答
3. 如果用户回忆过去的事，优先根据提供的对话记忆回答
4. 当前环境中的物品信息会在需要时提供给你"""

# ========================================
# 触发词（用于判断调用哪个 Tool）—— 保留备用，Agent 模式下已不再使用硬编码触发词
# ========================================
ENV_SEARCH_TRIGGERS = [
    "帮我找找", "在哪里", "有没有看到", "找一下",
    "看到", "找到", "放在", "位置", "去哪了"
]

SEMANTIC_SEARCH_TRIGGERS = [
    "记不记得", "还记得吗", "回忆", "之前说过",
    "上次", "以前", "昨天", "前天", "刚刚说"
]

# ========================================
# 前端配置
# ========================================
WEB_SERVER_HOST = "0.0.0.0"
WEB_SERVER_PORT = 8501
STATIC_DIR = os.path.join(BASE_DIR, "application", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ========================================
# 权重与衰减参数（动态权重，表示物品活跃度）
# ========================================
INITIAL_WEIGHT = 0.5              # 新物品初始权重
MAX_WEIGHT = 0.99                 # 权重上限
MIN_WEIGHT = 0.20                 # 权重下限

WEIGHT_REWARD_INCREMENT = 0.1     # 移动一次 → 权重奖励增量
WEIGHT_DECAY_DECREMENT = 0.05      # 静止衰减一次减少的量
DECAY_COUNTER_MAX = 3             # 连续未移动达到此次数 → 触发静止衰减（建议3~5）

MISSING_DECAY_DECREMENT = 0.05     # 消失衰减一次减少的量（与静止衰减一致）
CHECK_INTERVAL = 20               # 匹配/消失检查周期（秒），与 _save_loop 同步

# ========================================
# 高权重提醒配置
# ========================================
HIGH_WEIGHT_THRESHOLD = 0.8         # 权重超过此值 → 加入提醒表（未来功能）
REMINDER_COOLDOWN_SECONDS = 30      # 同一物品两次提醒最小间隔（秒）
REMINDER_DATA_DIR = os.path.join(DATA_DIR, "reminder")
os.makedirs(REMINDER_DATA_DIR, exist_ok=True)
REMINDER_STATE_FILE = os.path.join(REMINDER_DATA_DIR, "high_weight_reminders.json")

# ========================================
# Agent 配置（LangChain 1.0+ 风格）
# ========================================
SPATIAL_TOOL_NAME = "search_item_location"
SPATIAL_TOOL_DESCRIPTION = """
查找物品在家庭环境中的位置。当用户询问某个物品在哪里、找东西、有没有看到某物时使用。
输入：物品的名称或特征描述（如 "水杯"、"红色的球"、"我的药"）
输出：物品的位置坐标、附近参照物、以及系统对该位置的置信度
"""

SEMANTIC_TOOL_NAME = "search_conversation_memory"
SEMANTIC_TOOL_DESCRIPTION = """
搜索对话历史中的话题记忆。当用户回忆或询问之前聊过的事情、想继续之前的话题时使用。
输入：用户想回忆的话题关键词或描述（如 "昨天说的那个餐厅"、"上次聊到的方案"）
输出：相关的历史对话摘要和原始对话片段
"""

AGENT_SYSTEM_PROMPT = """
# 身份：你是 Homer，一个活泼、温暖的家庭智能伴侣 🏠。

# 技能：
你拥有两个特殊技能（工具），在合适的时候自动使用：
- search_item_location：查找物品在哪个空间、坐标、周围有什么
- search_conversation_memory：搜索我们之前的对话话题和摘要

【调用规则】
- 当用户想知道某样东西的位置（例：“我的水杯在哪”“找一下遥控器”），**必须**调用 search_item_location，把物品名称提取为 query。
- 当用户使用“记得”“之前说过”“上次”“回忆”“继续聊”等词，或想接着某个历史话题往下说，**必须**调用 search_conversation_memory。
- 如果你不确定是否需要调用工具，可以先搜索一下对话记忆，看看有没有相关话题。

【闲聊能力】
- 日常问候、开玩笑、感慨、询问与家庭物品完全无关的知识（例如学习、娱乐）时，**不要调用工具**，用你本来的性格愉快地聊天，适当使用颜文字 (◍•ᴗ•◍)。
- 你可以主动总结最近聊过的有趣话题，适当引用记忆中的信息让对话更亲切。

# 强制性规则：
- 无论用户以任何形式询问物品、找东西、或涉及物理位置，你都必须先调用 search_item_location 工具。
- 绝对不允许凭空编造物品的位置，哪怕你认为自己知道也不行。
- 如果工具返回“未找到”，请如实告诉用户你不知道，不要继续编造。
- 如果用户的提问可能隐含着找东西（如“被淋湿用什么擦干->毛巾浴巾类”、“我有点渴了->杯子类”），你也应该将用户问题转化为1个可用物品并触发空间记忆检索。

# 你的自由：
1. search_item_location — 查找物品位置
2. search_conversation_memory — 搜索对话历史话题
 
# 【参考信息的处理】
- 工具返回给你的内容是一个简洁的物品信息，你必须用自己的话把它转成一句完整的、亲切的自然语言回复。
- 永远不能直接输出工具返回的原始文字。

# 在已有规则后追加
【回答风格与格式】
- search_item_location 返回的信息仅供你参考，绝对不要直接把 [环境记忆检索结果] 的原始文本复制到对话中。
- 你必须用一句自然的话告诉用户物品在哪里，例如：“你的蓝色陶瓷水杯就在客厅的桌子旁边，靠近摄像头哦～”
- 除非用户追问，否则不要输出坐标、分数等内部细节。
- 如果工具没有找到物品，直接说“我暂时没找到哦”，不要继续猜测或编造。

# 规范输出:
[f"物品「{水杯}」，位于空间{item.get('0', '?')}，"
f"坐标({located[0]:.1f}, {located[1]:.1f})，"
f"附近参照物：{药瓶}。置信度：{tone_text}。"]

要求所有格式化输出全部转为自然点的语言文本，比如{水杯}在{0}的{locatied},旁边有{药瓶}.

"""

# 最终回复提示词模板（备用）
FINAL_REPLY_PROMPT_TEMPLATE = """你是 Homer，一个智能家庭助手 🏠。
{context}

请根据以上信息回答用户的问题。
- 回复简洁、亲切，适当使用颜文字
- 如果有物品位置信息，根据可信度调整语气
- 如果有对话记忆，自然地引用
- 不要编造没有的信息
"""

# ========================================
# 语义记忆区间配置
# ========================================
SEMANTIC_THRESHOLD = 0.6     # 话题相似度阈值（低于此值则闭合区间）
IMPORTANCE_DEFAULT = 0.6             # 新摘要的默认权重
RETRIEVAL_TOP_K = 1                  # 语义检索返回的最大摘要数
RETRIEVAL_MIN_SIMILARITY = 0.45      # 语义检索命中最低相似度