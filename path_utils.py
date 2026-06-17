"""程序路径工具。

源码运行、PyInstaller 打包和 cx_Freeze 打包时，资源文件与运行数据所在位置不同。
本模块统一处理这些差异，避免模型、数据库、报告等路径在 exe 中失效。
"""

import sys
from pathlib import Path


def is_frozen_app():
    """判断当前程序是否运行在打包后的可执行文件中。"""
    return bool(getattr(sys, "frozen", False))


def resource_root():
    """返回只读资源根目录，用于读取随程序一起打包的模型和配置文件。"""
    if is_frozen_app():
        # PyInstaller 使用 sys._MEIPASS；cx_Freeze 通常使用 exe 所在目录。
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def runtime_root():
    """返回运行数据根目录，用于保存数据库、报告、导出文件和误报样本。"""
    if is_frozen_app():
        # 打包后写入 exe 同级目录，用户更容易找到，也避免写入 PyInstaller 内部目录。
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(relative_path, prefer_runtime=True):
    """定位资源文件；若 exe 同级目录存在同名资源，则优先使用外置资源。"""
    path = Path(relative_path)
    if path.is_absolute():
        return path

    if prefer_runtime:
        external_path = runtime_root() / path
        if external_path.exists():
            return external_path

    return resource_root() / path


def runtime_path(relative_path):
    """定位可写运行文件或目录。"""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return runtime_root() / path
