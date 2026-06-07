# application/data_exporter.py
"""
数据层：单会话文件管理
- 每个会话存为独立的 JSON 文件，位于 config.MEMORY_DIR
- 文件名 = session_id + ".json"
"""
import json
import os
import glob     # glob用于匹配文件
from datetime import datetime
from typing import Optional
import config
import numpy as np


class DataExporter:

    # ==================== 会话列表 ====================
    @staticmethod
    def list_sessions() -> list:
        """返回所有会话 ID（按名倒序）"""
        files = glob.glob(os.path.join(config.MEMORY_DIR, "*.json"))
        sessions = []
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            sessions.append(name)
        sessions.sort(reverse=True)
        return sessions

    # ==================== 加载单个会话 ====================
    @staticmethod
    def load_session(session_id: str) -> Optional[dict]:
        """加载指定会话，不存在则返回 None"""
        filepath = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ==================== 保存会话 ====================
    @staticmethod
    def save_session(session_id: str, data: dict):
        """保存（覆盖）一个会话文件"""
        filepath = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================== 创建新会话 ====================
    @staticmethod
    def create_session(session_id: str) -> dict:
        """创建一个空会话并保存，返回会话数据"""
        data = {
            "session_id": session_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count_session": 0,
            "messages": []
        }
        DataExporter.save_session(session_id, data)
        return data

    # ==================== 追加消息 ====================
    @staticmethod
    def append_message(session_id: str, role: str, content: str):
        """
        向会话追加一条消息。
        - 如果是 assistant 回复，自动增加 count_session
        """
        data = DataExporter.load_session(session_id)
        if data is None:
            data = DataExporter.create_session(session_id)

        data["messages"].append({
            "role": role,
            "content": content
        })

        if role == "assistant":
            data["count_session"] = data.get("count_session", 0) + 1

        DataExporter.save_session(session_id, data)

    # ==================== 删除会话 ====================
    @staticmethod
    def delete_session(session_id: str):
        """删除会话文件"""
        filepath = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)


    #读取空间记忆
    @staticmethod
    def load_spatial_memories() -> list[dict]:
        """加载空间记忆文件"""
        if not os.path.exists(config.SPATIAL_MEMORY_FILE):
            return []
        with open(config.SPATIAL_MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_messages(session_id: str) -> list[dict]:
        """加载指定会话的 messages 列表（只返回 messages 字段）"""
        session = DataExporter.load_session(session_id)
        if session is None:
            return []
        return session.get("messages", [])


    # ==================== 语义记忆：摘要 ====================
    @staticmethod
    def get_abstract_file(session_id: str) -> str:
        """返回该 session 的唯一摘要文件路径"""
        return os.path.join(config.ABSTRACT_DIR, f"{session_id}.json")

    @staticmethod
    def add_segment_to_abstract(session_id: str, start_turn: int, end_turn: int, data: dict):
        """向 session 的摘要文件中追加一个区间"""
        filepath = DataExporter.get_abstract_file(session_id)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                wrapper = json.load(f)
        else:
            wrapper = {"session_id": session_id, "segments": []}
        wrapper["segments"].append(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_all_segments(session_id: str) -> list[dict]:
        """加载某 session 下所有摘要区间（按 start_turn 排序）"""
        filepath = DataExporter.get_abstract_file(session_id)
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)
        segs = wrapper.get("segments", [])
        segs.sort(key=lambda x: x.get('start_turn', 0))
        return segs

    # ==================== 语义记忆：向量 ====================
    @staticmethod
    def get_vectors_file(session_id: str) -> str:
        return os.path.join(config.VECTORS_DIR, f"{session_id}.npz")

    @staticmethod
    def add_vector_and_get_path(session_id: str, start_turn: int, end_turn: int, vector: 'np.ndarray') -> str:
        """保存向量为独立 .npy 文件，返回相对路径供摘要引用"""
        d = os.path.join(config.VECTORS_DIR, session_id)
        os.makedirs(d, exist_ok=True)
        filename = f"{start_turn}_{end_turn}.npy"
        filepath = os.path.join(d, filename)
        np.save(filepath, vector)
        return f"{session_id}/{filename}"

    @staticmethod
    def load_single_vector(vector_path: str) -> 'np.ndarray | None':
        """按路径加载单个向量文件"""
        filepath = os.path.join(config.VECTORS_DIR, vector_path)
        if os.path.exists(filepath):
            return np.load(filepath)
        return None

    @staticmethod
    def load_all_vectors(session_id: str) -> dict:
        """加载某 session 下所有向量 { (start_turn, end_turn): np.ndarray }"""
        d = os.path.join(config.VECTORS_DIR, session_id)
        if not os.path.exists(d):
            return {}
        vectors = {}
        for filename in os.listdir(d):
            if filename.endswith('.npy'):
                parts = filename.replace('.npy', '').split('_')
                if len(parts) == 2:
                    key = (int(parts[0]), int(parts[1]))
                    vectors[key] = np.load(os.path.join(d, filename))
        return vectors