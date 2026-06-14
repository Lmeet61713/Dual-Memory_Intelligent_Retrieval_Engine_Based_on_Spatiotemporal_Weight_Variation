# memory/personal_manager.py
"""
用户性格画像管理器 —— 从 PersonalMind_AI 移植而来，适配 Homer 多会话架构
"""
import json
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict
from openai import OpenAI
import config
from memory.semantic_memory import get_model
from sentence_transformers import util
import numpy as np

# 五维画像结构
DIMS = ["personality_traits", "interests", "values", "communication_style", "habits"]

class PersonalManager:
    """每个用户 / 会话绑定一个实例，按 session_id 持久化"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.persona_file = os.path.join(config.PERSONA_DIR, f"{session_id}_persona.json")
        self.persona = self._load()
        # 衰减所需的状态
        self.base_persona = self._load_base()   # 锚点画像快照
        self.decay_step = self._load_counter()

    def _load(self) -> dict:
        if os.path.exists(self.persona_file):
            with open(self.persona_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("user_persona", self._empty_persona())
        return self._empty_persona()

    def _empty_persona(self):
        return {dim: {} for dim in DIMS}

    def _load_base(self) -> dict | None:
        base_file = os.path.join(config.PERSONA_DIR, f"{self.session_id}_base.json")
        if os.path.exists(base_file):
            with open(base_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _load_counter(self) -> int:
        counter_file = os.path.join(config.PERSONA_DIR, f"{self.session_id}_counter.json")
        if os.path.exists(counter_file):
            with open(counter_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("step", 0)
        return 0

    def _save(self):
        os.makedirs(config.PERSONA_DIR, exist_ok=True)
        data = {
            "user_persona": self.persona,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.persona_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_base(self):
        base_file = os.path.join(config.PERSONA_DIR, f"{self.session_id}_base.json")
        with open(base_file, 'w', encoding='utf-8') as f:
            json.dump(self.base_persona, f, ensure_ascii=False, indent=2)

    def _save_counter(self):
        counter_file = os.path.join(config.PERSONA_DIR, f"{self.session_id}_counter.json")
        with open(counter_file, 'w', encoding='utf-8') as f:
            json.dump({"step": self.decay_step}, f)

    # ==================== 画像提取（LLM） ====================
    def _extract_with_llm(self, messages: list, mode: str = "both") -> dict:
        """调用 deepseek-chat 提取画像和/或事实，返回 updated_persona"""
        # 只取用户最近10条消息
        user_msgs = [m for m in messages if m["role"] == "user"][-10:]
        if not user_msgs:
            return self.persona

        persona_str = json.dumps(self.persona, ensure_ascii=False, indent=2)
        prompt = f"""
你是一个精准的性格画像分析助手。根据用户最近的消息，更新五维性格画像。

## 当前画像：
{persona_str}

## 用户最近10条消息：
{chr(10).join([f"- {m['content'][:100]}" for m in user_msgs])}

## 任务：
1. 识别并更新画像中每个维度的特征及置信度（0.01~0.99，不能1.0）。
2. 新特征置信度控制在0.68~0.72，旧特征根据证据动态调整。
3. 合并相似特征（名称相似度>0.8），跨维度去重。
4. 每个维度最多保留15个特征，按置信度排序淘汰。
5. 只输出 JSON: {{"updated_persona": {{"personality_traits":…}}, "interests":…}}。
"""
        try:
            client = OpenAI(
                api_key=os.environ.get('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个心理分析师，只输出 JSON 格式的性格画像。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            raw = json.loads(response.choices[0].message.content)
            updated = raw.get("updated_persona", self.persona)
            # 确保所有维度存在
            for dim in DIMS:
                updated.setdefault(dim, {})
            return updated
        except Exception as e:
            print(f"[PersonalManager] 画像提取失败: {e}")
            return self.persona

    # ==================== 衰减与奖励（仿 PersonalMind_AI 逻辑） ====================
    def apply_decay(self, new_persona: dict) -> dict:
        """基于历史锚点 base 应用衰减/奖励，返回处理后的画像"""
        self.decay_step += 1
        should_trigger = (self.decay_step - 1) % getattr(config, 'PERSONA_DECAY_INTERVAL', 2) == 0

        if not should_trigger:
            # 不触发衰减，直接返回新画像，但暂存 base
            self._save_counter()
            return new_persona

        if self.base_persona is None:
            # 第一次触发：设定锚点
            self.base_persona = new_persona.copy()
            self._save_base()
            self._save_counter()
            return new_persona

        processed = {}
        reward_inc = getattr(config, 'PERSONA_REWARD_INCREMENT', 0.03)
        decay_dec = getattr(config, 'PERSONA_DECAY_DECREMENT', 0.05)
        stability_threshold = 0.05

        for dim in DIMS:
            processed[dim] = {}
            for feat, conf in new_persona.get(dim, {}).items():
                base_conf = self.base_persona.get(dim, {}).get(feat)
                new_conf = conf
                if base_conf is not None:
                    diff = conf - base_conf
                    if abs(diff) < stability_threshold:
                        new_conf = min(conf + reward_inc, 0.99)  # 稳定特征给予奖励
                    elif diff < -stability_threshold:
                        new_conf = max(conf - decay_dec, 0.01)   # 衰退衰减
                    else:
                        new_conf = min(conf + reward_inc * 0.5, 0.99)  # 增强奖励
                else:
                    new_conf = max(conf - 0.03, 0.01)  # 新特征轻微衰减，防噪
                if new_conf > 0.1:
                    processed[dim][feat] = round(new_conf, 2)

        # 更新锚点
        self.base_persona = processed.copy()
        self._save_base()
        self._save_counter()
        return processed

    # ==================== 对外接口：每轮调用 ====================
    def update_and_get_summary(self, messages: list, turn_count: int) -> str:
        """
        在每轮对话后（或按需）调用此方法：
        1. 判断是否需要更新画像（每5轮或手动触发）
        2. 执行 LLM 提取
        3. 应用衰减
        4. 返回画像的字符串摘要，用于注入系统提示词
        """
        # 这里按每5轮触发一次画像更新（也可从外部传入 need_update 参数）
        if turn_count % 5 == 0 and len(messages) > 0:
            raw = self._extract_with_llm(messages, mode="both")
            self.persona = self.apply_decay(raw)
            self._save()
            print(f"[PersonalManager] 第{turn_count}轮画像已更新，特征总数{sum(len(v) for v in self.persona.values())}")

        return self._build_summary_text()

    def _build_summary_text(self) -> str:
        """将画像转为简洁的自然语言描述，用于注入 Agent 系统提示"""
        if not any(self.persona.values()):
            return ""
        lines = ["## 用户画像（长期观察所得）"]
        cn_map = {
            "personality_traits": "性格特质",
            "interests": "兴趣爱好",
            "values": "价值观",
            "communication_style": "沟通风格",
            "habits": "生活习惯"
        }
        for dim in DIMS:
            features = self.persona.get(dim, {})
            # 只取置信度 > 0.5 的特征
            high = {k: v for k, v in features.items() if v > 0.5}
            if high:
                dim_cn = cn_map.get(dim, dim)
                features_str = "、".join([f"{k}({v:.0%})" for k, v in list(high.items())[:8]])
                lines.append(f"- {dim_cn}: {features_str}")
        lines.append("请根据以上画像调整你的语气和内容，让互动更贴心。")
        return "\n".join(lines)