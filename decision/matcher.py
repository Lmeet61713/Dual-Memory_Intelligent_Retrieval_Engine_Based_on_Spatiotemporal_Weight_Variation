# decision/matcher.py
"""
双摄像头目标匹配器

- 画面流畅显示：每帧显示实时画面，叠加最近一次检测框
- 检测频率：每 30 帧跑一次 YOLO
- 匹配条件：两个摄像头都检测到同名物品且置信度 ≥ 0.6
- 匹配成功后传给 ConfidenceManager 进行权重/步长更新
- 消失检查由 _save_loop 调用，与 CHECK_INTERVAL 同步
"""
import threading
import time
import cv2
import numpy as np
from datetime import datetime
import config
from perception.camera_stream import GlobalCamera, MobileCamera
from perception.yolo_detector import YOLODetector
from decision.confidence_manager import ConfidenceManager

class Matcher:
    def __init__(self):
        self.global_cam = GlobalCamera()
        self.mobile_cam = MobileCamera()
        self.detector = YOLODetector()
        self.conf_manager = ConfidenceManager()

        self._running = False
        self._lock = threading.Lock()

        self._global_frame = 0
        self._mobile_frame = 0

        self._global_dets = []
        self._mobile_dets = []
        self._global_annotated = None
        self._mobile_annotated = None

        self._confirmed_items: dict = {}        # 已匹配物品

        self.scale_x = config.PHYSICAL_SPACE_WIDTH / config.CAM_FRAME_WIDTH
        self.scale_y = config.PHYSICAL_SPACE_HEIGHT / config.CAM_FRAME_HEIGHT

        self.MATCH_CONF = config.DETECTION_CONF_THRESHOLD          # 匹配条件：两个摄像头都检测到同名物品且置信度 ≥ 0.6
        self.DETECT_INTERVAL = 30

        # 提醒相关暂禁用
        self._pending_reminders: list = []
        self._reminder_lock = threading.Lock()

        self._stop_event = threading.Event()

    @property
    def is_running(self):
        return self._running

    # ==================== 绘制工具 ====================
    @staticmethod
    def _draw_boxes(frame, detections, scale_x=None, scale_y=None):
        img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class_name']
            conf = det['confidence']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            if scale_x is not None and scale_y is not None and det.get('center_physical'):
                px, py = det['center_physical']
                label += f" ({px:.1f},{py:.1f})"
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(img, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return img

    # ==================== 生命周期 ====================
    def start(self):
        if self._running:
            print("[Matcher] 已在运行中")
            return
        self._running = True
        self.global_cam.start()
        self.mobile_cam.start()
        threading.Thread(target=self._camera_loop, args=("global",), daemon=True).start()
        threading.Thread(target=self._camera_loop, args=("mobile",), daemon=True).start()
        threading.Thread(target=self._save_loop, daemon=True).start()
        print(f"[Matcher] 已启动 | 检测间隔:{self.DETECT_INTERVAL}帧 | 匹配置信度阈值:{self.MATCH_CONF}")

    def stop(self):
        if not self._running:
            return
        self._running = False
        time.sleep(0.5)
        self.conf_manager._sync_spatial_memory()
        self.global_cam.stop()
        self.mobile_cam.stop()
        self._stop_event.set()
        cv2.destroyAllWindows()
        print("[Matcher] 已停止")

    # ==================== 摄像头主循环 ====================
    def _camera_loop(self, cam_type: str):
        cam = self.global_cam if cam_type == "global" else self.mobile_cam
        window_name = "Global Camera (俯视)" if cam_type == "global" else "Mobile Camera (移动)"
        is_global = (cam_type == "global")
        local_count = 0
        cached_dets = []
        while self._running:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            local_count += 1
            do_detect = (local_count == 1 or local_count % self.DETECT_INTERVAL == 0)
            if do_detect:
                if is_global:
                    annotated, dets = self.detector.detect(frame, self.scale_x, self.scale_y)
                else:
                    if cam_type == "global":
                        annotated, dets = self.detector.detect(frame, scale_x=self.scale_x, scale_y=self.scale_y)
                    else:
                        annotated, dets = self.detector.detect(frame, use_mobile_model=True)
                cached_dets = dets
                display_frame = annotated
            else:
                if is_global:
                    display_frame = self._draw_boxes(frame, cached_dets, self.scale_x, self.scale_y)
                else:
                    display_frame = self._draw_boxes(frame, cached_dets)

            detect_label = "DETECT" if do_detect else " "       # 显示检测状态
            cv2.putText(display_frame,
                        f"{cam_type.upper()} | Frame:{local_count} | Objs:{len(cached_dets)} | {detect_label}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            with self._lock:
                if cam_type == "global":
                    self._global_dets = cached_dets
                    self._global_annotated = display_frame
                    self._global_frame = local_count
                else:
                    self._mobile_dets = cached_dets
                    self._mobile_annotated = display_frame
                    self._mobile_frame = local_count
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._running = False
                break
            time.sleep(0.03)

    # ==================== 匹配 + 消失检查 ====================
    def _save_loop(self):
        """每隔 CHECK_INTERVAL 秒执行一次匹配和消失检查"""
        while not self._stop_event.is_set():
            loop_start = time.time()            # 计时初始时间

            # 1. 双摄匹配（内部会更新物品的 last_seen）
            self._match_and_update()# 匹配并更新物品信息

            print("已检测到物品")

            # 2. 消失衰减检查
            self.conf_manager.check_missing(time.time())        # 消失衰减检查

            # 3. 等待到下一个间隔
            elapsed = time.time() - loop_start  # 已用时间
            sleep_time = config.CHECK_INTERVAL - elapsed    # 等待时间
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:           # 超时
                print("======================================================================")
                print(f"[Matcher] 警告: 匹配耗时 {elapsed:.1f}s 超过间隔 {config.CHECK_INTERVAL}s")

    def _match_and_update(self):
        """匹配双摄结果，并送给 confidence_manager 更新权重"""
        with self._lock:
            global_dets = list(self._global_dets)
            mobile_dets = list(self._mobile_dets)

        if not global_dets or not mobile_dets:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        matched_count = 0

        for g in global_dets:
            g_name = g['class_name']
            g_conf = g['confidence']
            if g_conf < self.MATCH_CONF:
                continue
            for m in mobile_dets:
                m_name = m['class_name']
                m_conf = m['confidence']
                if m_conf < self.MATCH_CONF:
                    continue
                if g_name == m_name:
                    print(f"找到同名物品:{g_name}")
                    coord = list(g.get('center_physical', g.get('center_pixel', [0, 0])))       # 坐标，默认为像素坐标
                    print(f"取得坐标:{coord}")
                    refs = self._find_nearby_refs(g, global_dets)
                    print(f"找到参考物:{refs}")
                    self.conf_manager.process_observation(
                        class_name=g_name,
                        located=coord,
                        yolo_confidence=round((g_conf + m_conf) / 2, 2),
                        features='',
                        space_id=0,
                        references=refs
                    )
                    print(f"成功写入将{g_name}写入数据")
                    matched_count += 1      # 匹配数量加一

                    # 高权重提醒（增加权重验证）
                    item = self.conf_manager.get_item(g_name)
                    if item and item['weight'] >= config.HIGH_WEIGHT_THRESHOLD:
                        reminder_text = self.conf_manager.reminder.try_trigger(g_name)
                        if reminder_text:
                            with self._reminder_lock:
                                self._pending_reminders.append(reminder_text)
                    else:
                        # 权重不达标，确保从提醒表中移除（修复重启后残留问题）
                        self.conf_manager.reminder.remove(g_name)
                    break

        if matched_count > 0:
            print(f"[Matcher] 在{now} 🎯 双摄匹配成功: {matched_count} 个物品，已推送到权重管理器")

    def _find_nearby_refs(self, target: dict, all_dets: list, max_refs: int = 2) -> list:
        tx, ty = target.get('center_physical', target['center_pixel'])
        candidates = []
        for det in all_dets:
            if det is target or det['class_name'] == target['class_name']:
                continue
            cx, cy = det.get('center_physical', det['center_pixel'])
            dist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            if dist <= config.REF_DISTANCE_THRESHOLD:
                candidates.append((dist, det['class_name']))
        candidates.sort(key=lambda x: x[0])
        return [name for _, name in candidates[:max_refs]]

    # ==================== 提醒队列接口（保留占位，暂时注释） ====================
    def drain_reminders(self) -> list:
        with self._reminder_lock:
            reminders = self._pending_reminders[:]
            self._pending_reminders.clear()
        return reminders