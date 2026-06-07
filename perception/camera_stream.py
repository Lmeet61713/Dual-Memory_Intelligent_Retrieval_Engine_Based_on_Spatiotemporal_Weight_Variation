# perception/camera_stream.py
"""
摄像头驱动层
- GlobalCamera:  全局固定摄像头，俯瞰全空间
- MobileCamera:  移动摄像头，安装在可移动载体上
"""
import cv2
import threading
import time
import config


class Camera:
    """摄像头基类 —— 统一 read() 接口，内部维护采集线程"""

    def __init__(self, cam_id: int, name: str = "Camera"):
        self.cam_id = cam_id
        self.name = name
        self.cap = None
        self._running = False           # 运行状态
        self._frame = None
        self._lock = threading.Lock()   # 线程锁
        self._thread = None             # 后台线程

    # ---------- 生命周期 ----------
    def start(self):
        """打开摄像头并启动采集线程"""
        self.cap = cv2.VideoCapture(self.cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise IOError(f"[{self.name}] 无法打开摄像头 (id={self.cam_id})")

        #确定摄像头处于运行状态就分配线程
        self._running = True
        # target指定线程要执行的函数，daemon指定线程是否为守护线程，默认为False
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()        # 启动线程
        print(f"[{self.name}] 摄像头已启动")

    def stop(self):
        """停止采集并释放资源"""
        self._running = False
        if self._thread:
            # 等待线程结束，timeout指定等待时间，如果指定了timeout，则最多等待timeout秒
            self._thread.join(timeout=2)
        if self.cap:
            # cap是摄像头对象，cap.release()是释放摄像头资源
            self.cap.release()
        print(f"[{self.name}] 摄像头已关闭")

    def _update_loop(self):
        """后台线程：持续读取帧"""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        """
        获取最新一帧（非阻塞）
        防止多个线程同时访问共享数据导致冲突。尤其是主线程插入正在读取的副线程
        """
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    @property
    def is_running(self) -> bool:
        return self._running


class GlobalCamera(Camera):
    """全局摄像头 —— 固定安装，俯瞰全空间"""
    def __init__(self):
        super().__init__(cam_id=config.GLOBAL_CAM_ID, name="GlobalCamera")


class MobileCamera(Camera):
    """移动摄像头 —— 安装在可移动载体上"""
    def __init__(self):
        super().__init__(cam_id=config.MOBILE_CAM_ID, name="MobileCamera")


# ========================================
# 便捷工厂函数
# ========================================
def create_camera_pair() -> tuple[GlobalCamera, MobileCamera]:
    """创建并返回全局+移动摄像头实例（未启动）"""
    return GlobalCamera(), MobileCamera()

