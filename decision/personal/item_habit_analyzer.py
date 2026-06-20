"""
物品使用习惯分析器（精简版）
- 轨迹文件 config.ITEM_TRAJECTORIES_FILE（不设上限，每次观测实时落盘）
- 摘要文件 config.ITEM_HABITS_SUMMARY_FILE（累加早/中/晚/深夜计数）
- 300 秒分析一次，分析后清空所有已参与分析的轨迹
- 线程安全：文件 I/O 通过 threading.Lock 保护
"""
import json
import os
import threading
from datetime import datetime
from typing import Optional

import config

# 时段名称映射
PERIOD_CN = {
    "morning": "早上",
    "afternoon": "下午",
    "night": "晚上",
    "late_night": "深夜"
}


def _get_hour_period(hour: int) -> str:
    """将小时 (0-23) 映射到时段键"""
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "night"
    else:
        return "late_night"


class ItemHabitManager:
    """
    物品习惯管理器（存储 + 分析 一体化）
    外部只需调用 record_observation / run_analysis / get_formatted_for_prompt
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.traj_path = config.ITEM_TRAJECTORIES_FILE
        self.habit_path = config.ITEM_HABITS_SUMMARY_FILE
        self._trajectories: dict[str, list] = {}
        self._load_trajectories()

    # ---------- 轨迹文件读写 ----------
    def _load_trajectories(self):
        """加载轨迹文件（冷启动时文件不存在则初始化为空）"""
        if os.path.exists(self.traj_path):
            try:
                with open(self.traj_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._trajectories = data.get("items", {})
            except Exception as e:
                print(f"[ItemHabit] 轨迹文件损坏，将重新创建: {e}")
                self._trajectories = {}
        else:
            self._trajectories = {}

    def _save_trajectories(self):
        """将内存轨迹字典写入磁盘"""
        os.makedirs(os.path.dirname(self.traj_path), exist_ok=True)
        with open(self.traj_path, 'w', encoding='utf-8') as f:
            json.dump({
                "items": self._trajectories,
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)

    # ---------- 摘要文件读写 ----------
    def _load_habits(self) -> dict:
        """加载摘要文件，返回 items 字典；文件不存在或损坏则返回空字典"""
        if os.path.exists(self.habit_path):
            try:
                with open(self.habit_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get("items", {})
            except Exception as e:
                print(f"[ItemHabit] 摘要文件损坏，将重新创建: {e}")
                return {}
        return {}

    def _save_habits(self, items: dict):
        """保存摘要字典到文件"""
        os.makedirs(os.path.dirname(self.habit_path), exist_ok=True)
        with open(self.habit_path, 'w', encoding='utf-8') as f:
            json.dump({
                "items": items,
                "last_analysis": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)

    # ---------- 公有接口 ----------
    def record_observation(self, name: str, space_id: int, coord: list, ts: str = None):
        """
        记录一次物品观测（由 ConfidenceManager 调用，线程安全）
        """
        with self.lock:
            if ts is None:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if name not in self._trajectories:
                self._trajectories[name] = []

            records = self._trajectories[name]

            # 去重：时间间隔 < CHECK_INTERVAL 且位移 < MOVE_THRESHOLD_PIXELS 则跳过
            if records:
                last = records[-1]
                try:
                    last_ts = datetime.strptime(last["ts"], "%Y-%m-%d %H:%M:%S")
                    curr_ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    dt = abs((curr_ts - last_ts).total_seconds())
                    dx = last["loc"][0] - coord[0]
                    dy = last["loc"][1] - coord[1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dt < config.CHECK_INTERVAL and dist < config.MOVE_THRESHOLD_PIXELS:
                        return
                except (ValueError, KeyError):
                    pass

            records.append({
                "ts": ts,
                "space": space_id,
                "loc": [round(coord[0], 1), round(coord[1], 1)]
            })
            self._save_trajectories()

    def run_analysis(self) -> int:
        """
        分析所有当前轨迹，累加时段计数到摘要文件，分析后清空轨迹。
        返回本次更新的物品数量。
        """
        with self.lock:
            if not self._trajectories:
                return 0

            all_traj = dict(self._trajectories)
            self._trajectories.clear()
            self._save_trajectories()

            habits = self._load_habits()
            updated_count = 0

            for name, records in all_traj.items():
                period_counts = {"morning": 0, "afternoon": 0, "night": 0, "late_night": 0}
                for rec in records:
                    try:
                        h = datetime.strptime(rec["ts"], "%Y-%m-%d %H:%M:%S").hour
                        period = _get_hour_period(h)
                        period_counts[period] += 1
                    except (ValueError, KeyError):
                        continue

                if sum(period_counts.values()) == 0:
                    continue

                if name not in habits:
                    habits[name] = {
                        "period_counts": {"morning": 0, "afternoon": 0, "night": 0, "late_night": 0},
                        "last_analysis": ""
                    }

                for p in period_counts:
                    habits[name]["period_counts"][p] += period_counts[p]
                habits[name]["last_analysis"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated_count += 1

            if updated_count > 0:
                self._save_habits(habits)
                print(f"[ItemHabit] 分析完成：更新 {updated_count} 个物品的时段计数，轨迹已清空")

            return updated_count

    def get_formatted_for_prompt(self) -> str:
        """
        返回可注入 system prompt 的物品习惯摘要文本。
        每个物品只输出其最活跃的时段。
        """
        with self.lock:
            habits = self._load_habits()
            if not habits:
                return ""

            lines = ["## 用户物品使用习惯（长期观察所得）"]
            for name, info in habits.items():
                counts = info.get("period_counts", {})
                if not counts or sum(counts.values()) == 0:
                    continue
                max_period = max(counts, key=counts.get)
                cn_period = PERIOD_CN.get(max_period, max_period)
                total = sum(counts.values())
                lines.append(f"- {name} 通常在{cn_period}被使用（累计 {total} 次）")

            if len(lines) == 1:
                return ""
            lines.append("请根据以上习惯，在回答物品相关问题时灵活引用。")
            return "\n".join(lines)

    def get_habit_text(self, name: str = None) -> str:
        """根据物品名获取习惯文本（供查询）"""
        with self.lock:
            habits = self._load_habits()
            if not habits:
                return "暂无任何物品使用习惯记录。"
            if name:
                info = habits.get(name)
                if not info:
                    return f"暂无关于「{name}」的使用习惯记录。"
                items = [(name, info)]
            else:
                items = list(habits.items())

            lines = []
            for item_name, info in items:
                counts = info.get("period_counts", {})
                if not counts or sum(counts.values()) == 0:
                    continue
                max_period = max(counts, key=counts.get)
                cn_period = PERIOD_CN.get(max_period, max_period)
                total = sum(counts.values())
                lines.append(f"「{item_name}」常用时段：{cn_period}（累计观测 {total} 次）")
            return "\n".join(lines) if lines else "暂无有效习惯记录。"


# ==================== 全局单例 ====================
_habit_manager: Optional[ItemHabitManager] = None
_habit_manager_lock = threading.Lock()


def get_habit_manager() -> ItemHabitManager:
    """获取全局物品习惯管理器单例（线程安全）"""
    global _habit_manager
    if _habit_manager is None:
        with _habit_manager_lock:
            if _habit_manager is None:
                _habit_manager = ItemHabitManager()
    return _habit_manager