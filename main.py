# main.py
"""
Homer 系统主入口
- 线程1: FastAPI 前端服务器
- 线程2: Matcher（按需启动，默认关闭）
"""

import sys
import os
import socket
from application.web_ui import start_server, set_matcher
from decision.matcher import Matcher
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件（main.py）所在的绝对路径。
os.chdir(BASE_DIR)                                      # 设置当前工作目录为 BASE_DIR
sys.path.insert(0, BASE_DIR)                      # 将当前目录添加到系统路径中
HOST_PREFIX = "http://localhost:"


def get_local_ip() -> str:
    """获取本机局域网 IP 地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    local_ip = get_local_ip()
    port = config.WEB_SERVER_PORT

    print()
    print("=" * 55)
    print("            🏠  Homer - 智能家庭助手")
    print("=" * 55)
    print()
    print("  🖥️  PC 端访问:")
    print(f"     http://localhost:{port}")
    print(f"     http://{local_ip}:{port}")
    print()
    print("  📱  移动端访问（同一局域网下）:")
    print(f"     http://{local_ip}:{port}/mobile")
    print()
    print("  📷  全局摄像头: id={}".format(config.GLOBAL_CAM_ID))
    print("  📷  移动摄像头: id={}".format(config.MOBILE_CAM_ID))
    print()
    print("=" * 55)
    print()

    # 1. 创建 Matcher 实例（不启动）
    matcher = Matcher()

    # 2. 注入到 web_ui 模块
    set_matcher(matcher)

    # 3. 启动 Web 服务器（主线程阻塞）
    start_server()

