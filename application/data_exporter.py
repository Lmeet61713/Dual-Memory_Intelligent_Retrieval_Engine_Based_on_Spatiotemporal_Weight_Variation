# application/data_exporter.py
"""
数据层：单会话文件管理
- 每个会话存为独立的 JSON 文件，位于 config.MEMORY_DIR
- 文件名 = session_id + ".json"

新增：运行期"记忆落盘"开关（application/persistence.py）
- 开关开启：写穿到文件，同时维护进程内缓存
- 开关关闭：只写进程内缓存；重新开启时由 flush_pending() 统一刷盘
"""
import copy
import glob
import json
import os
import threading
from datetime import datetime
from typing import Optional

import numpy as np

import config
from application import persistence


class DataExporter:

    _lock = threading.RLock()

    # 进程内缓存：开关关闭时这些缓存是唯一数据源
    _session_cache: dict[str, dict] = {}
    _session_dirty: set[str] = set()

    _abstract_cache: dict[str, dict] = {}
    _abstract_dirty: set[str] = set()

    # sid -> {(start_turn, end_turn): np.ndarray}
    _vector_cache: dict[str, dict] = {}
    _vector_dirty: set[str] = set()

    # ==================== 底层写文件 ====================
    @staticmethod
    def _write_json_atomic(filepath: str, data):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, filepath)

    @staticmethod
    def _write_session_file(session_id: str, data: dict):
        filepath = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
        DataExporter._write_json_atomic(filepath, data)

    @staticmethod
    def _write_abstract_file(session_id: str, data: dict):
        filepath = os.path.join(config.ABSTRACT_DIR, f"{session_id}.json")
        DataExporter._write_json_atomic(filepath, data)

    @staticmethod
    def _write_vector_file(session_id: str, start_turn: int, end_turn: int, vector):
        d = os.path.join(config.VECTORS_DIR, session_id)
        os.makedirs(d, exist_ok=True)
        filepath = os.path.join(d, f"{start_turn}_{end_turn}.npy")
        np.save(filepath, vector)

    # ==================== 会话列表 ====================
    @staticmethod
    def list_sessions() -> list:
        """返回所有会话 ID（磁盘 + 仅内存），按名倒序"""
        with DataExporter._lock:
            files = glob.glob(os.path.join(config.MEMORY_DIR, "*.json"))
            names = {os.path.splitext(os.path.basename(f))[0] for f in files}
            names.update(DataExporter._session_cache.keys())
            return sorted(names, reverse=True)

    # ==================== 加载单个会话 ====================
    @staticmethod
    def load_session(session_id: str) -> Optional[dict]:
        """加载指定会话；优先内存缓存，不存在则返回 None"""
        with DataExporter._lock:
            cached = DataExporter._session_cache.get(session_id)
            if cached is not None:
                return copy.deepcopy(cached)

            filepath = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
            if not os.path.exists(filepath):
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            DataExporter._session_cache[session_id] = copy.deepcopy(data)
            return copy.deepcopy(data)

    # ==================== 保存会话 ====================
    @staticmethod
    def save_session(session_id: str, data: dict):
        """保存（覆盖）一个会话：总是更新内存，按开关决定是否落盘"""
        with DataExporter._lock:
            DataExporter._session_cache[session_id] = copy.deepcopy(data)
            DataExporter._session_dirty.add(session_id)
            if persistence.is_enabled():
                DataExporter._write_session_file(session_id, DataExporter._session_cache[session_id])
                DataExporter._session_dirty.discard(session_id)

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
        return copy.deepcopy(data)

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
        """删除会话文件（以及其摘要/向量）；内存缓存同步清理"""
        with DataExporter._lock:
            DataExporter._session_cache.pop(session_id, None)
            DataExporter._session_dirty.discard(session_id)
            DataExporter._abstract_cache.pop(session_id, None)
            DataExporter._abstract_dirty.discard(session_id)
            DataExporter._vector_cache.pop(session_id, None)
            DataExporter._vector_dirty.discard(session_id)

            filepath = os.path.join(config.MEMORY_DIR, f"{session_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)

            abstract_path = os.path.join(config.ABSTRACT_DIR, f"{session_id}.json")
            if os.path.exists(abstract_path):
                os.remove(abstract_path)

            vector_dir = os.path.join(config.VECTORS_DIR, session_id)
            if os.path.isdir(vector_dir):
                for filename in os.listdir(vector_dir):
                    if filename.endswith(".npy"):
                        os.remove(os.path.join(vector_dir, filename))
                try:
                    os.rmdir(vector_dir)
                except OSError:
                    pass

    # ==================== 空间记忆（不受落盘开关控制） ====================
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
        with DataExporter._lock:
            wrapper = DataExporter._abstract_cache.get(session_id)
            if wrapper is None:
                filepath = DataExporter.get_abstract_file(session_id)
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        wrapper = json.load(f)
                else:
                    wrapper = {"session_id": session_id, "segments": []}
                wrapper = copy.deepcopy(wrapper)
                DataExporter._abstract_cache[session_id] = wrapper

            wrapper["segments"].append(copy.deepcopy(data))
            DataExporter._abstract_dirty.add(session_id)
            if persistence.is_enabled():
                DataExporter._write_abstract_file(session_id, wrapper)
                DataExporter._abstract_dirty.discard(session_id)

    @staticmethod
    def load_all_segments(session_id: str) -> list[dict]:
        """加载某 session 下所有摘要区间（按 start_turn 排序）"""
        with DataExporter._lock:
            wrapper = DataExporter._abstract_cache.get(session_id)
            if wrapper is None:
                filepath = DataExporter.get_abstract_file(session_id)
                if not os.path.exists(filepath):
                    return []
                with open(filepath, "r", encoding="utf-8") as f:
                    wrapper = json.load(f)
                wrapper = copy.deepcopy(wrapper)
                DataExporter._abstract_cache[session_id] = wrapper

            segs = copy.deepcopy(wrapper.get("segments", []))
            segs.sort(key=lambda x: x.get('start_turn', 0))
            return segs

    # ==================== 语义记忆：向量 ====================
    @staticmethod
    def get_vectors_file(session_id: str) -> str:
        return os.path.join(config.VECTORS_DIR, f"{session_id}.npz")

    @staticmethod
    def add_vector_and_get_path(session_id: str, start_turn: int, end_turn: int, vector: 'np.ndarray') -> str:
        """保存向量为独立 .npy 文件，返回相对路径供摘要引用（关闭落盘时仅存内存）"""
        with DataExporter._lock:
            DataExporter._vector_cache.setdefault(session_id, {})[(start_turn, end_turn)] = np.array(vector, copy=True)
            DataExporter._vector_dirty.add(session_id)
            if persistence.is_enabled():
                DataExporter._write_vector_file(session_id, start_turn, end_turn, vector)
                # 开启落盘时每次 add 都写穿，因此该 session 没有待刷盘向量
                DataExporter._vector_dirty.discard(session_id)
            return f"{session_id}/{start_turn}_{end_turn}.npy"

    @staticmethod
    def load_single_vector(vector_path: str) -> 'np.ndarray | None':
        """按路径加载单个向量文件（优先内存）"""
        parts = vector_path.replace("\\", "/").split("/")
        if len(parts) != 2 or not parts[1].endswith(".npy"):
            return None
        session_id, filename = parts[0], parts[1]
        turns = filename[:-4].split("_")
        if len(turns) != 2:
            return None
        try:
            key = (int(turns[0]), int(turns[1]))
        except ValueError:
            return None

        with DataExporter._lock:
            cached = DataExporter._vector_cache.get(session_id, {}).get(key)
            if cached is not None:
                return np.array(cached, copy=True)

        filepath = os.path.join(config.VECTORS_DIR, vector_path)
        if os.path.exists(filepath):
            return np.load(filepath)
        return None

    @staticmethod
    def load_all_vectors(session_id: str) -> dict:
        """加载某 session 下所有向量 { (start_turn, end_turn): np.ndarray }"""
        vectors = {}

        # 先读磁盘（可能包含历史数据）
        d = os.path.join(config.VECTORS_DIR, session_id)
        if os.path.exists(d):
            for filename in os.listdir(d):
                if filename.endswith('.npy'):
                    parts = filename.replace('.npy', '').split('_')
                    if len(parts) == 2:
                        try:
                            key = (int(parts[0]), int(parts[1]))
                        except ValueError:
                            continue
                        vectors[key] = np.load(os.path.join(d, filename))

        # 内存中的向量优先（关闭落盘时只有内存）
        with DataExporter._lock:
            for key, vec in DataExporter._vector_cache.get(session_id, {}).items():
                vectors[key] = np.array(vec, copy=True)

        return vectors

    # ==================== 落盘开关支持 ====================
    @staticmethod
    def persistence_stats() -> dict:
        with DataExporter._lock:
            return {
                "cached_sessions": len(DataExporter._session_cache),
                "dirty_sessions": len(DataExporter._session_dirty),
                "dirty_abstracts": len(DataExporter._abstract_dirty),
                "dirty_vector_sessions": len(DataExporter._vector_dirty),
            }

    @staticmethod
    def flush_pending() -> dict:
        """把内存中尚未落盘的会话/摘要/向量写入文件，返回写入数量"""
        counts = {"sessions": 0, "abstracts": 0, "vector_sessions": 0}
        with DataExporter._lock:
            for session_id in list(DataExporter._session_dirty):
                data = DataExporter._session_cache.get(session_id)
                if data is not None:
                    DataExporter._write_session_file(session_id, data)
                    counts["sessions"] += 1
                DataExporter._session_dirty.discard(session_id)

            for session_id in list(DataExporter._abstract_dirty):
                wrapper = DataExporter._abstract_cache.get(session_id)
                if wrapper is not None:
                    DataExporter._write_abstract_file(session_id, wrapper)
                    counts["abstracts"] += 1
                DataExporter._abstract_dirty.discard(session_id)

            for session_id in list(DataExporter._vector_dirty):
                vectors = DataExporter._vector_cache.get(session_id, {})
                for (start_turn, end_turn), vector in vectors.items():
                    DataExporter._write_vector_file(session_id, start_turn, end_turn, vector)
                if vectors:
                    counts["vector_sessions"] += 1
                DataExporter._vector_dirty.discard(session_id)

        return counts
