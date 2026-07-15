# feedback/schemas.py
"""
Module A — 反馈接口 Pydantic 数据模型
三类反馈: location_wrong / memory_wrong / persona_wrong
"""
from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


# ====== 位置错误 Payload ======
class LocationWrongPayload(BaseModel):
    item_id: str = Field(..., description="被错误定位的物品 ID")
    correct_location: Optional[str] = Field(None, description="用户提供的正确位置描述")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="用户的确信程度")


# ====== 记忆错误 Payload ======
class MemoryWrongPayload(BaseModel):
    memory_id: str = Field(..., description="被否认的记忆片段 ID")
    reason: Optional[str] = Field(None, description="用户否认的原因")
    confidence: float = Field(0.85, ge=0.0, le=1.0, description="用户的确信程度")


# ====== 人格判断错误 Payload ======
class PersonaWrongPayload(BaseModel):
    dimension: str = Field(..., description="被否认的画像维度，如 personality/interests/values/communication_style/habits")
    disagree_value: str = Field(..., description="用户不同意的具体特征描述")
    suggest_value: Optional[str] = Field(None, description="用户建议的替代特征")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="用户的确信程度")

    @field_validator("dimension")
    @classmethod
    def check_dimension(cls, v: str) -> str:
        allowed = {"personality_traits", "interests", "values", "communication_style", "habits"}
        if v not in allowed:
            raise ValueError(f"dimension 必须是以下之一: {allowed}")
        return v


# ====== 反馈类型判别 ======
FeedbackType = Literal["location_wrong", "memory_wrong", "persona_wrong"]

FeedbackPayload = Union[LocationWrongPayload, MemoryWrongPayload, PersonaWrongPayload]


# ====== 反馈请求体 ======
class FeedbackRequest(BaseModel):
    """
    统一反馈请求体。
    根据 type 字段自动选择对应的 payload 校验模型。
    """
    type: FeedbackType = Field(..., description="反馈类型")
    session_id: str = Field(..., min_length=1, description="会话 ID")
    payload: dict = Field(..., description="反馈数据载荷，结构取决于 type")

    @field_validator("payload")
    @classmethod
    def validate_payload_by_type(cls, v: dict, info) -> dict:
        """根据 type 字段动态校验 payload"""
        # 使用 model_post_init 更优雅，但 field_validator 也能工作
        return v  # 实际校验在 FeedbackEvent.to_event() 中完成


# ====== 反馈事件（落盘结构） ======
class FeedbackEvent(BaseModel):
    """写入 JSONL 的完整事件结构"""
    event_id: str = Field(..., description="全局唯一事件 ID，格式 fb_YYYYMMDD_xxxx")
    type: FeedbackType
    session_id: str
    timestamp: float = Field(..., description="Unix 时间戳（秒）")
    payload: FeedbackPayload

    @staticmethod
    def from_request(req: FeedbackRequest, event_id: str) -> "FeedbackEvent":
        """从请求体构造已验证的 FeedbackEvent"""
        if req.type == "location_wrong":
            payload = LocationWrongPayload(**req.payload)
        elif req.type == "memory_wrong":
            payload = MemoryWrongPayload(**req.payload)
        elif req.type == "persona_wrong":
            payload = PersonaWrongPayload(**req.payload)
        else:
            raise ValueError(f"未知反馈类型: {req.type}")
        import time
        return FeedbackEvent(
            event_id=event_id,
            type=req.type,
            session_id=req.session_id,
            timestamp=time.time(),
            payload=payload,
        )
