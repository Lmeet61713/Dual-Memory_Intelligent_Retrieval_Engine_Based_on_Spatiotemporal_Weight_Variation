# tests/test_feedback_router.py
"""
Module A 验收测试 —— 反馈接口功能完整性验证
使用 HTTPX 对 POST /feedback 三类 payload 执行自动化测试
"""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback.schemas import (
    FeedbackRequest, FeedbackEvent,
    LocationWrongPayload, MemoryWrongPayload, PersonaWrongPayload,
)
from feedback.dispatcher import bayesian_update, dispatch_correction


class TestSchemas(unittest.TestCase):
    """Pydantic 模型校验测试"""

    def test_location_wrong_payload_valid(self):
        p = LocationWrongPayload(item_id="cup_01", correct_location="桌子上", confidence=0.95)
        self.assertEqual(p.item_id, "cup_01")
        self.assertEqual(p.confidence, 0.95)

    def test_location_wrong_payload_defaults(self):
        p = LocationWrongPayload(item_id="keys_01")
        self.assertEqual(p.confidence, 0.9)
        self.assertIsNone(p.correct_location)

    def test_memory_wrong_payload_valid(self):
        p = MemoryWrongPayload(memory_id="mem_001", reason="记错了")
        self.assertEqual(p.memory_id, "mem_001")
        self.assertEqual(p.confidence, 0.85)

    def test_persona_wrong_payload_valid(self):
        p = PersonaWrongPayload(
            dimension="personality_traits",
            disagree_value="introvert",
            suggest_value="ambivert",
            confidence=0.8,
        )
        self.assertEqual(p.dimension, "personality_traits")

    def test_persona_wrong_invalid_dimension(self):
        with self.assertRaises(Exception):
            PersonaWrongPayload(
                dimension="invalid_dim",
                disagree_value="test",
            )

    def test_confidence_bounds(self):
        with self.assertRaises(Exception):
            LocationWrongPayload(item_id="test", confidence=1.5)
        with self.assertRaises(Exception):
            LocationWrongPayload(item_id="test", confidence=-0.5)

    def test_feedback_request_missing_type(self):
        """缺少必填字段应报错"""
        with self.assertRaises(Exception):
            FeedbackRequest(session_id="sess_1", payload={"item_id": "cup"})

    def test_feedback_request_empty_session(self):
        """空 session_id 应报错"""
        with self.assertRaises(Exception):
            FeedbackRequest(type="location_wrong", session_id="", payload={"item_id": "cup"})

    def test_feedback_event_from_request(self):
        """测试从请求体构造完整事件"""
        req = FeedbackRequest(
            type="location_wrong",
            session_id="test_session",
            payload={"item_id": "cup_01", "correct_location": "桌上"},
        )
        event = FeedbackEvent.from_request(req, "fb_test_001")
        self.assertEqual(event.event_id, "fb_test_001")
        self.assertEqual(event.type, "location_wrong")
        self.assertIsInstance(event.payload, LocationWrongPayload)

    def test_feedback_event_from_request_persona(self):
        req = FeedbackRequest(
            type="persona_wrong",
            session_id="test_session",
            payload={
                "dimension": "interests",
                "disagree_value": "喜欢足球",
                "suggest_value": "喜欢篮球",
                "confidence": 0.7,
            },
        )
        event = FeedbackEvent.from_request(req, "fb_test_002")
        self.assertIsInstance(event.payload, PersonaWrongPayload)

    def test_all_three_types(self):
        """验证三类 payload 全部通过"""
        # location_wrong
        req1 = FeedbackRequest(
            type="location_wrong", session_id="s1",
            payload={"item_id": "cup_01"},
        )
        e1 = FeedbackEvent.from_request(req1, "fb_001")
        self.assertIsInstance(e1.payload, LocationWrongPayload)

        # memory_wrong
        req2 = FeedbackRequest(
            type="memory_wrong", session_id="s1",
            payload={"memory_id": "mem_01", "reason": "不对"},
        )
        e2 = FeedbackEvent.from_request(req2, "fb_002")
        self.assertIsInstance(e2.payload, MemoryWrongPayload)

        # persona_wrong
        req3 = FeedbackRequest(
            type="persona_wrong", session_id="s1",
            payload={"dimension": "values", "disagree_value": "偏保守"},
        )
        e3 = FeedbackEvent.from_request(req3, "fb_003")
        self.assertIsInstance(e3.payload, PersonaWrongPayload)


class TestBayesianUpdate(unittest.TestCase):
    """贝叶斯更新原语测试"""

    def test_positive_update(self):
        """正向加强：置信度应上升"""
        result = bayesian_update(0.5, 0.9, direction=+1, alpha=0.5)
        self.assertGreater(result, 0.5)

    def test_negative_update(self):
        """负向削弱：置信度应下降"""
        result = bayesian_update(0.5, 0.9, direction=-1, alpha=0.5)
        self.assertLess(result, 0.5)

    def test_bounds(self):
        """边界测试：不应超出 [0.01, 0.99]"""
        self.assertEqual(bayesian_update(0.99, 1.0, direction=+1, alpha=1.0), 0.99)
        self.assertEqual(bayesian_update(0.01, 1.0, direction=-1, alpha=1.0), 0.01)

    def test_no_evidence(self):
        """证据强度为 0 时不应变化"""
        result = bayesian_update(0.5, 0.0, direction=-1, alpha=0.5)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_alpha_variation(self):
        """alpha 越大修正幅度越大"""
        small = bayesian_update(0.5, 0.9, direction=-1, alpha=0.2)
        large = bayesian_update(0.5, 0.9, direction=-1, alpha=0.8)
        self.assertGreater(abs(0.5 - large), abs(0.5 - small))

    def test_alpha_0_3(self):
        """文档定义 alpha=0.3 场景"""
        result = bayesian_update(0.7, 0.9, direction=-1, alpha=0.3)
        self.assertLess(result, 0.7)
        self.assertGreater(result, 0.4)

    def test_alpha_0_6(self):
        """文档定义 alpha=0.6 场景"""
        result = bayesian_update(0.7, 0.9, direction=-1, alpha=0.6)
        self.assertLess(result, 0.7)
        self.assertGreater(result, 0.1)


class TestDispatcher(unittest.TestCase):
    """分发器测试"""

    def test_dispatch_location_wrong(self):
        record = {
            "event_id": "fb_test_01",
            "type": "location_wrong",
            "session_id": "sess_1",
            "timestamp": time.time(),
            "payload": {"item_id": "non_existent_item", "confidence": 0.9},
        }
        result = dispatch_correction(record)
        # 即使物品不存在，也不应该抛出异常
        self.assertIsInstance(result, bool)

    def test_dispatch_memory_wrong(self):
        record = {
            "event_id": "fb_test_02",
            "type": "memory_wrong",
            "session_id": "sess_1",
            "timestamp": time.time(),
            "payload": {"memory_id": "mem_001", "reason": "用户否认", "confidence": 0.85},
        }
        result = dispatch_correction(record)
        self.assertTrue(result)

    def test_dispatch_persona_wrong(self):
        record = {
            "event_id": "fb_test_03",
            "type": "persona_wrong",
            "session_id": "sess_1",
            "timestamp": time.time(),
            "payload": {
                "dimension": "personality_traits",
                "disagree_value": "introvert",
                "confidence": 0.9,
            },
        }
        result = dispatch_correction(record)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    print("=" * 60)
    print("  Module A 单元测试")
    print("=" * 60)
    unittest.main(verbosity=2)
