# perception/yolo_detector.py
"""
YOLOv8 目标检测器
- 加载模型 → 接收帧 → 返回检测列表
- 可选物理坐标映射（全局摄像头俯视用）
"""
from ultralytics import YOLO
import cv2
import config


class YOLODetector:
    """YOLOv8 检测器封装"""

    def __init__(self, model_path_global: str = None,model_path_mobile: str = None):
        if model_path_global is None:
            model_path_global = config.YOLO_MODEL_PATH_GLOBAL
        if model_path_mobile is None:
            model_path_mobile = config.YOLO_MODEL_PATH_MOBILE
        self.model_global = YOLO(model_path_global)
        self.model_mobile = YOLO(model_path_mobile)
        self._class_names = self.model_global.names  # {0: 'person', 1: 'cup', ...}

    def detect(self, frame, scale_x: float = None, scale_y: float = None,use_mobile_model: bool = False):
        """
        对单帧进行目标检测

        参数:
            frame:      np.ndarray (BGR格式)
            scale_x:    像素→物理 X 轴比例（厘米/像素），None 则不计算物理坐标
            scale_y:    像素→物理 Y 轴比例

        返回:
            (annotated_frame, detections)
            - annotated_frame: 已绘制边界框+标签的帧
            - detections: list[dict]
                {
                    'class_name': str,
                    'confidence': float,
                    'box': [x1, y1, x2, y2],
                    'center_pixel': (cx, cy),
                    'center_physical': (px, py) | None
                }
        """
        if frame is None:
            return frame, []

        model = self.model_mobile if use_mobile_model else self.model_global
        results = model(frame, verbose=False)
        annotated = results[0].plot() if results and len(results) > 0 else frame.copy()
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < config.DETECTION_CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    class_name = self._class_names.get(cls_id, f"unknown_{cls_id}")
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # 物理坐标映射（仅当 scale 参数传入时计算）
                    phys = None
                    if not use_mobile_model and scale_x is not None and scale_y is not None:
                        phys = (round(cx * scale_x, 2), round(cy * scale_y, 2))
                        label = f"{class_name} ({phys[0]},{phys[1]})"
                        cv2.putText(
                            annotated, label,
                            (int(x1), int(y1) - 10 if y1 > 20 else int(y1) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                        )

                    detections.append({
                        'class_name': class_name,
                        'confidence': round(conf, 3),
                        'box': [round(x1), round(y1), round(x2), round(y2)],
                        'center_pixel': (round(cx), round(cy)),
                        'center_physical': phys         # mobile = None
                    })

        return annotated, detections