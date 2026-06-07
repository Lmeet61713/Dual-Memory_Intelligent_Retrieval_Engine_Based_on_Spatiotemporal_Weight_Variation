# main.py
"""
Homer 系统主入口
- 线程1: FastAPI 前端服务器
- 线程2: Matcher（按需启动，默认关闭）
"""

import sys
import os
from application.web_ui import start_server, set_matcher
from decision.matcher import Matcher
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件（main.py）所在的绝对路径。
os.chdir(BASE_DIR)                                      # 设置当前工作目录为 BASE_DIR
sys.path.insert(0, BASE_DIR)                      # 将当前目录添加到系统路径中
HOST_PREFIX = "http://localhost:"


if __name__ == "__main__":
    print("=" * 50)
    print("  🏠 Homer - 智能家庭助手")
    print("=" * 50)
    print(f"  前端端口: {HOST_PREFIX}{config.WEB_SERVER_PORT}")
    print(f"  全局摄像头: id={config.GLOBAL_CAM_ID}")
    print(f"  移动摄像头: id={config.MOBILE_CAM_ID}")
    print("=" * 50)

    # 1. 创建 Matcher 实例（不启动）
    matcher = Matcher()

    # 2. 注入到 web_ui 模块
    set_matcher(matcher)

    # 3. 启动 Web 服务器（主线程阻塞）
    start_server()

## ======================================================================================================================

## 咕咕嘎嘎  ##

## ======================================================================================================================