"""
高权重物品提醒管理器
- 维护一个简单的哈希表：name → 提醒文本模板
- 当物品权重大于阈值时自动加入，低于阈值时自动移除
- 每次触发提醒有冷却时间，防止刷屏
"""
import json
import os
from datetime import datetime, timedelta
import config


class ReminderManager:
    def __init__(self):
        self._reminders: dict = {}
        self._load()

    # ==================== 持久化 ====================
    def _load(self):
        if os.path.exists(config.REMINDER_STATE_FILE):
            with open(config.REMINDER_STATE_FILE, 'r', encoding='utf-8') as f:
                self._reminders = json.load(f)
            print(f"[ReminderManager] 已加载 {len(self._reminders)} 条高权重提醒")
        else:
            self._reminders = {}

    def _save(self):
        os.makedirs(os.path.dirname(config.REMINDER_STATE_FILE), exist_ok=True)
        with open(config.REMINDER_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._reminders, f, ensure_ascii=False, indent=2)

    # ==================== 增删 ====================
    def add(self, name: str):
        """将物品加入高权重提醒表（若已存在则忽略）"""
        if name not in self._reminders:
            self._reminders[name] = {
                "name": name,
                "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_triggered": None
            }
            self._save()
            print(f"[ReminderManager] 加入高权重提醒: {name}")

    def remove(self, name: str):
        """从高权重提醒表中移除物品"""
        if name in self._reminders:
            del self._reminders[name]
            self._save()
            print(f"[ReminderManager] 移除高权重提醒: {name}")

    # ==================== 触发检查 ====================
    def try_trigger(self, name: str) -> str | None:
        """
        尝试触发提醒。
        若物品在高权重表中，且距上次触发超过冷却时间，返回提醒文本并更新记录；否则返回 None。
        提醒文本固定为：f"这里有一个{name}"
        """
        if name not in self._reminders:
            return None

        entry = self._reminders[name]
        last_str = entry.get("last_triggered")
        if last_str:
            last_time = datetime.strptime(last_str, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - last_time < timedelta(seconds=config.REMINDER_COOLDOWN_SECONDS):
                return None  # 冷却中

        # 触发提醒
        entry["last_triggered"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        reminder_text = f"这里有一个{name}"
        print(f"[ReminderManager] 🛎️ 触发提醒: {reminder_text}")
        return reminder_text