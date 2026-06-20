# decision/confidence_manager.py
"""
权重管理器（步进式 + 遗忘曲线衰减）
- 接收 matcher 传来的观测结果
- 移动奖励 / 静止衰减 / 消失衰减 三线管理
- 衰减量由艾宾浩斯遗忘曲线计算，替代固定 -0.05
- 持久化到 data/space/，并同步空间记忆
"""
import json
import os
from datetime import datetime
from typing import Optional
import config
import time
from decision.high_weight_reminder import ReminderManager
from decision.personal.item_habit_analyzer import get_habit_manager
from memory.semantic_memory import get_model   # 共享单例，无额外开销


def _decay_amount(current_weight: float) -> float:
    """
    遗忘曲线衰减量：基于当前权重计算本次 CHECK_INTERVAL 周期的衰减幅度。

    生物学原理：记忆越牢固（权重越高），遗忘越慢；
    记忆越弱（接近下限），衰减越平缓（接近 0）。
    相当于对 CHECK_INTERVAL 秒应用指数衰减后的减少量。
    """
    factor = 0.5 ** (config.CHECK_INTERVAL / config.FORGET_HALF_LIFE)  # 剩余比例
    new_w = current_weight * factor
    return max(current_weight - new_w, 0.0)


def _boost_amount(current_weight: float) -> float:
    """
    移动奖励增长量：接近上限时打折，模拟边际收益递减。
    """
    if current_weight >= config.MAX_WEIGHT:
        return 0.0
    room = config.MAX_WEIGHT - current_weight
    if current_weight >= 0.85:
        # 打折：越接近上限增幅越小
        return config.WEIGHT_REWARD_INCREMENT * (room / (config.MAX_WEIGHT - 0.85))
    return config.WEIGHT_REWARD_INCREMENT


class ConfidenceManager:
    """
    权重管理

    规则：
    - 物品被双摄观测到，坐标变化 > MOVE_THRESHOLD_PIXELS → 移动奖励，步长重置 0
    - 物品被观测到，坐标未变化 → 步长+1；步长达到 DECAY_COUNTER_MAX → 静止衰减，步长重置 0
    - 物品超过 CHECK_INTERVAL 秒未被任何观测到 → 消失衰减（每 20s 检查一次）
    - 新物品初始权重 = INITIAL_WEIGHT (0.5)，步长 0
    - 衰减量使用遗忘曲线计算，非固定 -0.05
    """

    def __init__(self):
        self._items: dict = {}               # key: (class_name, space_id) → item dict
        self._last_forget_check = None       # 保留字段
        self.reminder = ReminderManager()
        self._load()

    # ==================== 持久化 ====================
    def _load(self):
        if os.path.exists(config.CONFIDENCE_STATE_FILE):
            with open(config.CONFIDENCE_STATE_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self._items = {}
            for k, v in raw.items():
                parts = k.split("|")
                if len(parts) == 2:
                    key = (parts[0], int(parts[1]))
                    if 'weight' not in v:
                        v['weight'] = config.INITIAL_WEIGHT
                    if 'located' not in v and 'coordinate' in v:
                        v['located'] = v['coordinate']
                    if 'last_seen' not in v:
                        v['last_seen'] = None
                    elif isinstance(v['last_seen'], str):
                        v['last_seen'] = None
                    self._items[key] = v
            print(f"[ConfidenceManager] 已加载 {len(self._items)} 条物品状态")
        else:
            self._items = {}

    def _save(self):
        """
        保存物品状态
        """
        serializable = {}
        for k, v in self._items.items():
            key_str = f"{k[0]}|{k[1]}"
            serializable[key_str] = v
        with open(config.CONFIDENCE_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ==================== 核心逻辑 ====================
    def process_observation(
        self,
        class_name: str,
        located: list,
        yolo_confidence: float = 0.5,
        features: str = "",
        space_id: int = 0,
        references: list = None
    ):
        """
        处理一次双摄确认的观测
        """
        if references is None:
            references = []

        key = (class_name, space_id)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now_ts = time.time()
        new_coord = tuple(located)

        if key in self._items:
            old_item = self._items[key]
            old_located = tuple(old_item.get('located', [0, 0]))
            dist = ((new_coord[0] - old_located[0])**2 + (new_coord[1] - old_located[1])**2)**0.5

            # 更新坐标和时间
            self._items[key]['located'] = list(new_coord)
            self._items[key]['timestamp'] = now_str
            self._items[key]['last_seen'] = now_ts
            self._items[key]['references'] = references
            self._items[key]['confidence'] = yolo_confidence
            if features:
                self._items[key]['features'] = features

            # 移动判断
            if dist > config.MOVE_THRESHOLD_PIXELS:
                self._apply_reward(key, reason="移动")
            else:
                self._apply_static_increment(key)
        else:
            # 新物品
            self._items[key] = {
                'name': class_name,
                'space_id': space_id,
                'located': list(new_coord),
                'timestamp': now_str,
                'last_seen': now_ts,
                'references': references,
                'features': features,
                'confidence': yolo_confidence,
                'weight': config.INITIAL_WEIGHT,
                'step': 0
            }
            print(f"[ConfidenceManager] 新物品: {class_name} (空间{space_id}), 初始权重={config.INITIAL_WEIGHT}")
            print("======================================================================================================")
        # 观测后保存
        self._save()
        self._sync_spatial_memory()

        # ★ 记录物品历史轨迹（供习惯分析）
        habit_mgr = get_habit_manager()
        habit_mgr.record_observation(class_name, space_id, list(new_coord))

    # ==================== 奖励/衰减子函数 ====================
    def _apply_reward(self, key, reason="移动"):
        old_w = self._items[key].get('weight', config.INITIAL_WEIGHT)
        boost = _boost_amount(old_w)
        new_w = min(old_w + boost, config.MAX_WEIGHT)
        self._items[key]['weight'] = round(new_w, 2)
        self._items[key]['step'] = 0
        print(f"[ConfidenceManager] 奖励({reason}): {key[0]} ({old_w:.2f} → {new_w:.2f})")
        self._check_reminder_threshold(key[0], new_w)

    def _apply_decay(self, key, reason="静止"):
        old_w = self._items[key].get('weight', config.INITIAL_WEIGHT)
        decay = _decay_amount(old_w)
        new_w = max(old_w - decay, config.MIN_WEIGHT)
        self._items[key]['weight'] = round(new_w, 2)
        self._items[key]['step'] = 0
        print(f"[ConfidenceManager] 衰减({reason}): {key[0]} ({old_w:.2f} → {new_w:.2f}, 衰减量={decay:.3f})")
        self._check_reminder_threshold(key[0], new_w)

    def _apply_static_increment(self, key):
        """静止未移动，步长递增，达阈值触发静止衰减"""
        self._items[key]['step'] = self._items[key].get('step', 0) + 1
        if self._items[key]['step'] >= config.DECAY_COUNTER_MAX:
            self._apply_decay(key, reason="静止")

    # ==================== 消失衰减 ====================
    def check_missing(self, now_ts: float):
        """
        检查所有物品，若 last_seen 距今超过 CHECK_INTERVAL 则触发消失衰减。
        不更新 last_seen，直到物品被再次观测到。
        """
        for key, item in self._items.items():
            last = item.get('last_seen')
            if last is None:
                continue
            elapsed = now_ts - last
            if elapsed >= config.CHECK_INTERVAL:
                old_w = item['weight']
                decay = _decay_amount(old_w)
                new_w = max(old_w - decay, config.MIN_WEIGHT)
                item['weight'] = round(new_w, 2)
                item['step'] = 0
                print(f"[消失衰减] {item['name']}: 权重 {old_w:.2f} → {new_w:.2f} (距上次观测 {elapsed:.0f}s, 衰减量={decay:.3f})")

        self._save()
        self._sync_spatial_memory()

    # ==================== 提醒阈值检查 ====================
    def _check_reminder_threshold(self, name: str, weight: float):
        if weight >= config.HIGH_WEIGHT_THRESHOLD:
            self.reminder.add(name)
        else:
            self.reminder.remove(name)

    # ==================== 空间记忆同步 ====================
    def _sync_spatial_memory(self):
        """输出到 spatial_memory.json，同时更新每个物品的语义向量"""
        model = get_model()

        items_list = []
        for key, item in self._items.items():
            name = item['name']
            features = item.get('features', '')
            refs = item.get('references', [])
            refs_text = '，'.join(refs) if refs else ''

            name_vec = model.encode(name, normalize_embeddings=True).tolist() if name else []
            features_vec = model.encode(features, normalize_embeddings=True).tolist() if features else []
            refs_vec = model.encode(refs_text, normalize_embeddings=True).tolist() if refs_text else []

            items_list.append({
                'name': name,
                'space_id': item['space_id'],
                'located': item.get('located', [0, 0]),
                'timestamp': item['timestamp'],
                'references': refs,
                'features': features,
                'confidence': item.get('confidence', 0),
                'weight': item.get('weight', 1.0),
                'step': item.get('step', 0),
                'name_vec': name_vec,
                'features_vec': features_vec,
                'refs_vec': refs_vec
            })

    # ==================== 查询接口 ====================
    def get_all_items(self) -> list:
        """获取所有物品信息"""
        return [
            {
                'name': item['name'],
                'space_id': item['space_id'],
                'located': item.get('located', []),
                'timestamp': item['timestamp'],
                'references': item.get('references', []),
                'features': item.get('features', ''),
                'confidence': item.get('confidence', 0),
                'weight': item.get('weight', 1.0),
                'step': item.get('step', 0),
                'last_seen': item.get('last_seen', 0)
            }
            for item in self._items.values()
        ]

    def get_item(self, class_name: str, space_id: int = 0) -> Optional[dict]:
        """获取物品信息，返回 None 表示不存在"""
        key = (class_name, space_id)
        item = self._items.get(key)
        if item:
            return {
                'name': item['name'],
                'space_id': item['space_id'],
                'located': item.get('located', []),
                'timestamp': item['timestamp'],
                'references': item.get('references', []),
                'features': item.get('features', ''),
                'confidence': item.get('confidence', 0),
                'weight': item.get('weight', 1.0),
                'step': item.get('step', 0),
                'last_seen': item.get('last_seen', 0)
            }
        return None



# ==================== 纯内存自测试（不生成任何文件） ====================
if __name__ == '__main__':

    ConfidenceManager._save = lambda self: print("修改函数方法为打印本句。")
    ConfidenceManager._sync_spatial_memory = lambda self: None
    get_habit_manager().__class__.record_observation = lambda self, *a, **kw: None


    def show_item(cm, name):
        item = cm.get_item(name)
        if item:
            print(f"  {item['name']}: 权重={item['weight']:.2f}  步数={item['step']}  坐标={item['located']}")
        else:
            print(f"  {name} 不存在")

    # ===== 测试1：移动奖励（含接近上限打折） =====
    print("=" * 50)
    print("测试1：移动奖励（坐标变化 > MOVE_THRESHOLD_PIXELS） + 接近上限打折")
    cm = ConfidenceManager()
    cm.process_observation("水杯", located=[100, 100], yolo_confidence=0.7)
    show_item(cm, "水杯")

    cm.process_observation("水杯", located=[105, 100])
    show_item(cm, "水杯")

    cm.process_observation("水杯", located=[105, 105])
    show_item(cm, "水杯")
    print()

    # ===== 测试2：静止衰减（曲线衰减替代固定 -0.05） =====
    print("=" * 50)
    print("测试2：静止衰减（连续静止，步数累积触发衰减，曲线计算替代固定 -0.05）")
    cm = ConfidenceManager()
    cm.process_observation("遥控器", located=[50, 50])
    show_item(cm, "遥控器")

    for i in range(1, config.DECAY_COUNTER_MAX + 2):
        cm.process_observation("遥控器", located=[50, 50])
        print(f"第{i}次静止观测: ", end="")
        show_item(cm, "遥控器")
    print()

    # ===== 测试3：消失衰减（曲线衰减） =====
    print("=" * 50)
    print("测试3：消失衰减（超过 CHECK_INTERVAL 未观测，曲线计算替代固定 -0.05）")
    cm = ConfidenceManager()
    cm.process_observation("书本", located=[200, 200])
    show_item(cm, "书本")

    cm._items[("书本", 0)]["last_seen"] = time.time() - (config.CHECK_INTERVAL + 5)
    cm.check_missing(time.time())
    print("触发消失衰减后: ", end="")
    show_item(cm, "书本")
    print()
