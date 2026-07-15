# config.py
"""
全局配置参数 —— 所有模块统一引用此文件
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

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
PERSONA_DIR = os.path.join(DATA_DIR, "persona")
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback")
FEEDBACK_EVENTS_DIR = os.path.join(FEEDBACK_DIR, "events")
FEEDBACK_META_DIR = os.path.join(FEEDBACK_DIR, "meta")
FEEDBACK_ERRORS_DIR = os.path.join(FEEDBACK_DIR, "errors")
for d in [DATA_DIR, MEMORY_DIR, SPACE_DIR, VECTORS_DIR, ABSTRACT_DIR, PERSONA_DIR,
          FEEDBACK_DIR, FEEDBACK_EVENTS_DIR, FEEDBACK_META_DIR, FEEDBACK_ERRORS_DIR]:
    os.makedirs(d, exist_ok=True)

# 具体文件
SPATIAL_MEMORY_FILE = os.path.join(SPACE_DIR, "spatial_memory.json")
CONFIDENCE_STATE_FILE = os.path.join(SPACE_DIR, "confidence_state.json")
ITEM_TRAJECTORIES_FILE = os.path.join(SPACE_DIR, "item_trajectories.json")
ITEM_HABITS_SUMMARY_FILE = os.path.join(SPACE_DIR, "item_habits_summary.json")
SEMANTIC_SESSION_FILE = os.path.join(MEMORY_DIR, "chat_sessions.json")
# ITEM_HABIT_DATA_FILE = os.path.join(DATA_DIR, "item_habit_data.json")

# 模型
YOLO_MODEL_PATH_GLOBAL = os.path.join(BASE_DIR, "models", "best_global.pt")
YOLO_MODEL_PATH_MOBILE = os.path.join(BASE_DIR, "models", "best_mobile.pt")
BGE_MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "models--BAAI--bge-small-zh-v1.5",
    "snapshots", "7999e1d3359715c523056ef9478215996d62a620"
)

# ========================================
# 摄像头配置
# ========================================
GLOBAL_CAM_ID = 0               # 全局摄像头设备索引
MOBILE_CAM_ID = 0               # 移动摄像头设备索引
CAM_FRAME_WIDTH = 640
CAM_FRAME_HEIGHT = 480
CAM_FRAME_WIDTH_ = 640          # 第二组摄像头（备用）
CAM_FRAME_HEIGHT_ = 480

# ========================================
# 检测与匹配
# ========================================
DETECTION_CONF_THRESHOLD = 0.4   # YOLO 检测置信度阈值
MOVE_THRESHOLD_PIXELS = 1        # 移动距离阈值（像素）
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
# 从 .env 文件读取 API Key，请在项目根目录的 .env 文件中配置
# 获取地址: https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 启动时检查 API Key 是否已配置
if DEEPSEEK_API_KEY == "" or "PLACEHOLDER" in DEEPSEEK_API_KEY.upper() or "PLEASE-REPLACE" in DEEPSEEK_API_KEY:
    print("=" * 55)
    print("  ❌ 错误: DeepSeek API Key 未设置!")
    print("  📝 请编辑项目根目录的 .env 文件")
    print("     将 DEEPSEEK_API_KEY 设置为你的 Key")
    print("  🔗 https://platform.deepseek.com/api_keys")
    print("=" * 55)
    import sys
    sys.exit(1)

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
WEIGHT_DECAY_DECREMENT = 0.05     # 静止衰减 / 消失衰减 基础衰减量
DECAY_COUNTER_MAX = 3             # 连续未移动达到此次数 → 触发静止衰减（建议3~5）

# 遗忘曲线衰减参数（将固定-0.05替换为曲线衰减因子）
FORGET_HALF_LIFE = 120            # 遗忘半衰期（秒），CHECK_INTERVAL ≈ 此值的 0.17x，衰减因子约 0.89
                                   # 每 20s 衰减到原来的 ≈89%，越接近下限越慢

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
你拥有三个特殊技能（工具），在合适的时候自动使用：
- search_item_location：查找物品在哪个空间、坐标、周围有什么
- search_conversation_memory：搜索我们之前的对话话题和摘要
- user_correction_feedback：当用户否定当前结果时，从备选列表中获取下一个可能的物品

【调用规则】
- 当用户想知道某样东西的位置（例：“我的水杯在哪”“找一下遥控器”），**必须**调用 search_item_location，把物品名称提取为 query。
- 当用户使用“记得”“之前说过”“上次”“回忆”“继续聊”等词，或想接着某个历史话题往下说，**必须**调用 search_conversation_memory。
- 如果你不确定是否需要调用工具，可以先搜索一下对话记忆，看看有没有相关话题。

【闲聊能力】
- 日常问候、开玩笑、感慨、询问与家庭物品完全无关的知识（例如学习、娱乐）时，**不要调用工具**，用你本来的性格愉快地聊天，适当使用颜文字 (◍•ᴗ•◍)。
- 你可以主动总结最近聊过的有趣话题，适当引用记忆中的信息让对话更亲切。

【补偿机制 —— 用户否定处理】
- 当用户对你给出的物品位置结果表示否定（如"不对"、"不是这个"、"搞错了"、"位置不对"），**必须优先调用 user_correction_feedback** 获取下一个备选物品，而不是重新调用 search_item_location。
- 只有当 user_correction_feedback 返回"备选物品已全部耗尽"或类似提示时，才引导用户提供更多线索（颜色、形状、大小、附近有什么等），然后调用 search_item_location 重新搜索。
- 如果用户直接在否定时提供了新的明确线索（如"不是维生素，是那个白色的胃药"），你应该同时传入 reason 参数给 user_correction_feedback，让系统记录纠正原因。

# 强制性规则：
- 无论用户以任何形式询问物品、找东西、或涉及物理位置，你都必须先调用 search_item_location 工具。
- 绝对不允许凭空编造物品的位置，哪怕你认为自己知道也不行。
- 如果工具返回“未找到”，请如实告诉用户你不知道，不要继续编造。
- 如果用户的提问可能隐含着找东西（如“被淋湿用什么擦干->毛巾浴巾类”、“我有点渴了->杯子类”），你也应该将用户问题转化为1个可用物品并触发空间记忆检索。

# 你的自由：
1. search_item_location — 查找物品位置
2. search_conversation_memory — 搜索对话历史话题
3. user_correction_feedback — 用户否定时获取备选候选
 
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

# 新增衰减参数（可选）
PERSONA_DECAY_INTERVAL = 2
PERSONA_REWARD_INCREMENT = 0.03
PERSONA_DECAY_DECREMENT = 0.05

# ========================================
# 物品使用习惯分析配置
# ========================================
ITEM_HABIT_ANALYSIS_INTERVAL = 300         # 分析周期（秒），需双摄运行
HABIT_NAME_MATCH_THRESHOLD = 0.65          # habit 候选名称匹配最低相似度，低于此值则使用 LLM 原始 query

# ========================================
# 补偿机制配置
# ========================================
COMPENSATION_DB_PATH = os.path.join(DATA_DIR, "compensation.db")
COMPENSATION_TOP_K = 5                # 空间检索返回候选数量
COMPENSATION_BUFFER_TTL_HOURS = 1     # 补偿队列过期时间（小时），超期自动清理

# ========================================
# 反馈系统优化配置
# ========================================
FEEDBACK_METRICS_DIR = os.path.join(FEEDBACK_DIR, "metrics")
os.makedirs(FEEDBACK_METRICS_DIR, exist_ok=True)

FEEDBACK_BASE_ALPHA = 0.5           # 贝叶斯修正基础 alpha（保守值，避免误伤）
FEEDBACK_ALPHA_STEP = 0.1           # 每次连续否定 alpha 递增量
FEEDBACK_MAX_ALPHA = 0.9            # alpha 上限（激进衰减上限）
HIGH_DISPUTE_THRESHOLD = 3          # 24h 内连续否定次数触发 high_dispute 标记
FEEDBACK_HISTORY_MAX = 50           # feedback_history 数组最大保留条数
