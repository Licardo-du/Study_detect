"""cx_Freeze 备用打包脚本。

PyInstaller 是本项目推荐方案；如果 PyInstaller 在某台机器上无法正确收集
依赖，可以尝试使用本脚本生成 build/ 目录下的可执行程序。
"""

import sys
from pathlib import Path

from cx_Freeze import Executable, setup


# 项目根目录用于定位入口文件和需要复制的资源。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def include_file_if_exists(relative_path, target_path=None):
    """把存在的资源加入 include_files，缺失的可选资源直接跳过。"""
    source_path = PROJECT_ROOT / relative_path
    if not source_path.exists():
        return None
    return (str(source_path), target_path or str(relative_path))


# 这些资源会被复制到构建目录，确保 exe 启动后仍能找到模型和类别配置。
include_files = [
    item
    for item in (
        include_file_if_exists("data.yaml"),
        include_file_if_exists(Path("models") / "best.pt", str(Path("models") / "best.pt")),
        include_file_if_exists("README.md"),
        include_file_if_exists("LICENSE"),
    )
    if item is not None
]


# Windows GUI 程序使用 Win32GUI，避免运行时弹出额外控制台窗口。
base = "Win32GUI" if sys.platform == "win32" else None
target_name = "StudyBehaviorMonitor.exe" if sys.platform == "win32" else "StudyBehaviorMonitor"


build_options = {
    # packages 用于提示 cx_Freeze 显式收集动态导入的库。
    "packages": [
        "tkinter",
        "ultralytics",
        "cv2",
        "numpy",
        "PIL",
        "mss",
        "docx",
        "reportlab",
        "yaml",
        "torch",
    ],
    # include_files 复制模型、配置和说明文件。
    "include_files": include_files,
    # 排除测试相关模块，减少包体积。
    "excludes": ["pytest", "tests"],
}


setup(
    name="StudyBehaviorMonitor",
    version="1.0.0",
    description="Study behavior visual detection desktop application",
    options={"build_exe": build_options},
    executables=[
        Executable(
            str(PROJECT_ROOT / "app.py"),
            base=base,
            target_name=target_name,
        )
    ],
)
