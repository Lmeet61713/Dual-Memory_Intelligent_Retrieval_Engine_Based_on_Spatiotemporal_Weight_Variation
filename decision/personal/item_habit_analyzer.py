"""
物品使用习惯分析器
- ItemHistoryStore: 管理物品原始历史轨迹（name/location/time/space）
- ItemHabitAnalyzer: 周期性分析历史轨迹，生成使用习惯摘要
- 独立于性格画像，输出持久化到 data/persona/item_habits.json
"""
import json
import os
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher     # SequenceMatcher 的作用是找出两个序列的相似度
import config


# ==================== 原始历史轨迹存储 ====================
class ItemHistoryStore:
    """管理每个物品的时空历史记录，持久化到 data/item_histories.json"""

    def __init__(self):
        self._histories: dict[str, list] = {}   # { name: [ {ts, space, loc} ], name: [ {ts, space, loc} ]... ] }
        self._last_save_count = 0               #
        self._load()                            # 加载历史记录

    def _load(self):
        if os.path.exists(config.ITEM_HABIT_DATA_FILE):     # "item_histories.json"
            with open(config.ITEM_HABIT_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._histories = data.get("items", {})     # 取出 items 字段{ name: [ {ts, space, loc} }
            # 兼容旧字段
            for name, records in self._histories.items():   # records = [ {ts, space, loc} ]
                for r in records:
                    r.setdefault("space", 0)        # 兼容旧字段,将空间编号都设置为0
            print(f"[ItemHistory] 已加载 {len(self._histories)} 个物品的历史记录")
        else:
            self._histories = {}

    def _save(self):
        os.makedirs(os.path.dirname(config.ITEM_HABIT_DATA_FILE), exist_ok=True)
        with open(config.ITEM_HABIT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(
                {"items": self._histories, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},      # obj表示待写入的JSON对象
                      f,
                      ensure_ascii=False,
                      indent=2
                      )

    def append(self, name: str, space_id: int, coord: list, ts: str = None):
        """
        追加一条历史
        :param name: 物品名称
        :param space_id: 物品空间编号
        :param coord: 物品本次坐标
        :param ts: 上一次检测到物品的时间
        :return: None
        """

        if ts is None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if name not in self._histories:
            self._histories[name] = []      # 检测到新的物品，创建新物品的历史记录列表

        records = self._histories[name]     # 物品历史记录，存入列表

        # 只有该物品已有历史记录时才做去重检查；首次观测（records 为空列表）直接跳过此环节，无条件写入
        if records:
            last = records[-1]          # 取出上一次记录
            try:
                last_ts = datetime.strptime(last["ts"], "%Y-%m-%d %H:%M:%S")
                curr_ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                dt = abs((curr_ts - last_ts).total_seconds())


                dx = last["loc"][0] - coord[0]
                dy = last["loc"][1] - coord[1]
                dist = (dx ** 2 + dy ** 2) ** 0.5

                if dt < config.CHECK_INTERVAL and dist < config.MOVE_THRESHOLD_PIXELS:
                    return  # 跳过重复记录
            except (ValueError, KeyError):
                pass

        # 发生了位置变化意味着需生成新的历史记录
        records.append({
            "ts": ts,
            "space": space_id,
            "loc": [round(coord[0], 1), round(coord[1], 1)]
        })


        if len(records) > config.ITEM_HISTORY_MAX_PER_ITEM:                 # 历史记录超过最大限制时进行压缩
            trim_n = max(2, config.ITEM_HISTORY_MAX_PER_ITEM // 4)          # 最少压缩前两条，
            compressed = self._compress_old_records(records[:trim_n])       # 压缩最早前trim_n条记录
            self._histories[name] = [compressed] + records[trim_n:]         # 压缩完追加进去
            print(f"[ItemHistory] {name} 历史压缩: {len(records)} → {len(self._histories[name])} 条")

        # 每 3 次追加，写入一次磁盘文件，减少 IO
        self._last_save_count += 1
        if self._last_save_count >= 3:
            self._save()
            self._last_save_count = 0

    # todo  为什么这么聚合？
    def _compress_old_records(self, old_records: list) -> dict:
        """
        将最早前几条历史记录进行合并
        :param old_records:
        :return:
        """

        if not old_records:
            return {}
        lats = [r["loc"][0] for r in old_records if r.get("loc")]
        lons = [r["loc"][1] for r in old_records if r.get("loc")]
        spaces = list(set(r.get("space", 0) for r in old_records))
        avg_lat = round(sum(lats) / len(lats), 1) if lats else 0.0
        avg_lon = round(sum(lons) / len(lons), 1) if lons else 0.0
        return {
            "ts": old_records[0]["ts"],
            "ts_end": old_records[-1]["ts"],
            "space": spaces[0] if spaces else 0,
            "loc": [avg_lat, avg_lon],
            "count": len(old_records),
            "compressed": True
        }

    def force_save(self):
        self._save()

    def get_all(self) -> dict[str, list]:
        return self._histories

    def get(self, name: str) -> list:
        return self._histories.get(name, [])


# ==================== 分析器 ====================
class ItemHabitAnalyzer:
    """分析物品历史轨迹，生成使用习惯摘要"""

    def __init__(self, store: ItemHistoryStore):            # todo 类里面初始化类是什么用法？
        self.store = store      # 物品历史轨迹存储
        # 加载已有摘要
        self.summaries: list[dict] = self._load_summaries()

    def _load_summaries(self) -> list[dict]:
        if os.path.exists(config.ITEM_HABIT_ANALYSIS_RESULT_FILE):
            with open(config.ITEM_HABIT_ANALYSIS_RESULT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get("habits", [])
        return []

    def _save_summaries(self):
        os.makedirs(os.path.dirname(config.ITEM_HABIT_ANALYSIS_RESULT_FILE), exist_ok=True)
        with open(config.ITEM_HABIT_ANALYSIS_RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "habits": self.summaries,
                "last_analysis": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)

    def analyze(self) -> int:
        """
        执行一次完整分析。
        返回新增/更新的摘要数量。
        """
        histories = self.store.get_all()
        if not histories:
            return 0

        new_habits = []

        for name, records in histories.items():
            # 只分析有足够数据的物品（>= 3 条记录），todo后续转为全量分析
            raw_records = [r for r in records if not r.get("compressed")]   # 排除压缩节点
            if len(raw_records) < 3:
                continue

            habit_texts = []

            # 1. todo 频次分析待完善
            # records: [{"ts": "2023-07-01 12:00:00", "space": 1, "loc": [10.0, 20.0]}, ...]
            total = len(raw_records)
            if len(records) > 0:
                first_ts = records[0].get("ts", "")
                last_ts = records[-1].get("ts", "")
            else:
                first_ts = last_ts = ""
            if total >= 5:
                habit_texts.append(f"用户频繁使用「{name}」，累计观测{total}次")

            # 2. 空间聚类：将坐标网格化（10 像素一格），找 top 区域 todo有瑕疵
            grid_counts = defaultdict(int)      #  defaultdict 表示默认值为0的 dict
            for r in raw_records:
                loc = r.get("loc", [0, 0])
                gx = int(loc[0] / 10) * 10 if len(loc) >= 1 else 0
                gy = int(loc[1] / 10) * 10 if len(loc) >= 2 else 0
                grid_counts[(gx, gy)] += 1
            if grid_counts:
                top_grid = max(grid_counts, key=grid_counts.get)
                habit_texts.append(f"「{name}」常出现在坐标({top_grid[0]}, {top_grid[1]})附近区域")

            # 3. 时间模式
            hour_buckets = {"早晨(6-12)": 0, "下午(12-18)": 0, "晚上(18-24)": 0, "深夜(0-6)": 0}
            for r in raw_records:
                ts_str = r.get("ts", "")
                try:
                    h = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").hour
                    if 6 <= h < 12:
                        hour_buckets["早晨(6-12)"] += 1
                    elif 12 <= h < 18:
                        hour_buckets["下午(12-18)"] += 1
                    elif 18 <= h < 24:
                        hour_buckets["晚上(18-24)"] += 1
                    else:
                        hour_buckets["深夜(0-6)"] += 1
                except (ValueError, KeyError):
                    pass
            active_buckets = [k for k, v in hour_buckets.items() if v >= max(3, total * 0.2)]
            if active_buckets:
                habit_texts.append(f"「{name}」活跃时段集中在{'、'.join(active_buckets)}")

            # 合并多条 -> 一条摘要
            if habit_texts:
                combined = "；".join(habit_texts)
                new_habits.append({
                    "name": name,
                    "summary": combined,
                    "importance": min(0.5 + total * 0.02, 0.9),
                    "total_observations": total,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        # 合并 + 截断
        updated_count = self._merge_and_truncate(new_habits)
        self._save_summaries()
        return updated_count

    def _merge_and_truncate(self, new_habits: list) -> int:
        """将新摘要与已有摘要按相似度合并，并按需截断"""
        updated = 0
        for new_h in new_habits:
            merged = False
            for old_h in self.summaries:
                if old_h.get("name") != new_h.get("name"):
                    continue
                sim = SequenceMatcher(None, new_h["summary"], old_h["summary"]).ratio() # 计算相似度[0,1]
                if sim >= config.ITEM_HABIT_SIMILARITY_THRESHOLD:
                    # 合并：取更高 importance，更新摘要文本
                    if len(new_h["summary"]) > len(old_h["summary"]):
                        old_h["summary"] = new_h["summary"]
                    old_h["importance"] = max(old_h["importance"], new_h["importance"])
                    old_h["total_observations"] = new_h["total_observations"]
                    old_h["generated_at"] = new_h["generated_at"]
                    merged = True
                    updated += 1
                    break
            if not merged:
                self.summaries.append(new_h)
                updated += 1

        # 按 importance 降序排序，截断
        self.summaries.sort(key=lambda x: (x.get("importance", 0), x.get("generated_at", "")), reverse=True)
        if len(self.summaries) > config.ITEM_HABIT_MAX_SUMMARIES:
            removed = self.summaries[config.ITEM_HABIT_MAX_SUMMARIES:]
            self.summaries = self.summaries[:config.ITEM_HABIT_MAX_SUMMARIES]
            print(f"[ItemHabit] 截断移除 {len(removed)} 条低重要性习惯摘要")

        # todo 衰减：长期未更新的摘要降低 importance -------- 暂不考虑衰减
        now = datetime.now()
        for s in self.summaries:
            try:
                gen_ts = datetime.strptime(s.get("generated_at", ""), "%Y-%m-%d %H:%M:%S")
                days_since = (now - gen_ts).total_seconds() / 86400
                if days_since > 7:
                    decay = 0.03 * (days_since // 7)
                    s["importance"] = round(max(s["importance"] - decay, 0.2), 2)
            except (ValueError, KeyError):
                pass

        # todo 移除 importance <= 0.2 的摘要 -------- 移除方式更变
        before = len(self.summaries)
        self.summaries = [s for s in self.summaries if s.get("importance", 0) > 0.2]
        after = len(self.summaries)
        if before > after:
            print(f"[ItemHabit] 低重要性衰减移除 {before - after} 条摘要")

        return updated

    def get_all_summaries(self) -> list[dict]:
        return self.summaries

    def format_for_prompt(self) -> str:
        """将摘要格式化为可注入 system prompt 的文本"""
        if not self.summaries:
            return ""
        lines = ["## 用户物品使用习惯（长期观察所得）"]
        for s in self.summaries[:10]:  # 最多 10 条
            imp_star = "⭐" * min(3, int(s["importance"] * 3) + 1)
            lines.append(f"- {imp_star} {s['summary']}")
        lines.append("请根据以上使用习惯，在回答物品相关问题时灵活引用。")
        return "\n".join(lines)


# ==================== 统一管理门面 ====================
class ItemHabitManager:
    """物品习惯分析器的统一门面，供 Matcher 和 API 层使用"""

    def __init__(self):
        self.store = ItemHistoryStore()
        self.analyzer = ItemHabitAnalyzer(self.store)

    def record_observation(self, name: str, space_id: int, coord: list):
        """记录一次物品观测（从 confidence_manager 调用）"""
        self.store.append(name, space_id, coord)

    def run_analysis(self) -> int:
        """执行一次分析，返回更新的摘要数"""
        self.store.force_save()  # 分析前先保存最新历史
        return self.analyzer.analyze()

    def get_habit_text(self, name: str = None) -> str:
        """
        获取物品使用习惯文本。
        若指定 name，只返回该物品相关摘要；否则返回全部摘要。
        """
        summaries = self.analyzer.get_all_summaries()
        if name:
            summaries = [s for s in summaries if s.get("name") == name]
        if not summaries:
            return "暂无该物品的使用习惯记录。"

        lines = []
        for s in summaries:
            lines.append(f"- {s['summary']}（置信度: {s['importance']:.0%}）")
        return "\n".join(lines)

    def get_formatted_for_prompt(self) -> str:
        return self.analyzer.format_for_prompt()


# ==================== 全局单例 ====================
_habit_manager: ItemHabitManager | None = None


def get_habit_manager() -> ItemHabitManager:
    """获取全局物品习惯管理器单例"""
    global _habit_manager
    if _habit_manager is None:
        _habit_manager = ItemHabitManager()
    return _habit_manager