# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

这个 spec 文件把 app.py 作为桌面程序入口，并显式带上模型、配置和部分
第三方库的数据文件，避免打包后出现“源码能运行、exe 找不到资源”的问题。
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


# SPECPATH 是 PyInstaller 执行 spec 文件时提供的变量，指向 spec 所在目录。
PROJECT_ROOT = Path(SPECPATH).resolve().parent


def add_file_if_exists(relative_path, target_dir):
    """只有文件真实存在时才加入打包资源，避免缺少可选文件导致构建失败。"""
    source_path = PROJECT_ROOT / relative_path
    if source_path.exists():
        return [(str(source_path), target_dir)]
    return []


def safe_collect_data(package_name):
    """收集第三方库运行所需的数据文件；库未安装时不让 spec 直接崩溃。"""
    try:
        return collect_data_files(package_name)
    except Exception:
        return []


# 应用运行资源：best.pt 是默认模型，data.yaml 是数据集/类别配置。
datas = []
datas += add_file_if_exists("data.yaml", ".")
datas += add_file_if_exists("README.md", ".")
datas += add_file_if_exists("LICENSE", ".")
datas += add_file_if_exists(Path("models") / "best.pt", "models")

# 部分库包含字体、默认配置或模型辅助资源，需要一并复制到发布目录。
for package in ("ultralytics", "matplotlib", "reportlab"):
    datas += safe_collect_data(package)

# 只补充应用确实会动态用到的模块；大量收集 torch/numpy 子模块会显著放大包体积。
hiddenimports = [
    "PIL.Image",
    "PIL.ImageTk",
    "matplotlib.backends.backend_agg",
    "mss",
    "docx",
    "reportlab",
    "yaml",
    "lap",
]


block_cipher = None


# Analysis 负责扫描入口脚本、依赖、二进制文件和数据资源。
a = Analysis(
    [str(PROJECT_ROOT / "app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
        "tkinter.test",
        "torch.testing",
        "torch.utils.tensorboard",
        "tensorboard",
        "onnxruntime",
        "tensorflow",
        "torchaudio",
        "IPython",
        "jupyter",
        "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PYZ 会把纯 Python 模块压缩进运行包。
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# EXE 使用 windowed 模式，运行时不会额外弹出命令行窗口。
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StudyBehaviorMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# COLLECT 生成目录式发布包，适合 torch / ultralytics 这类大型依赖。
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StudyBehaviorMonitor",
)

# macOS 上额外生成 .app 包；Windows 会忽略这个变量。
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="StudyBehaviorMonitor.app",
        icon=None,
        bundle_identifier="com.local.study-behavior-monitor",
    )
