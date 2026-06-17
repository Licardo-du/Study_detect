#!/usr/bin/env bash
set -euo pipefail

# 定位项目根目录，保证脚本从任意目录执行都能找到 app.py。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"

# 构建机需要 Python 环境；发布后的 .app 不要求用户安装 Python。
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install -r requirements-build.txt

# 打包前做基础语法检查。
"${PYTHON_BIN}" -m py_compile app.py ui.py ai_core.py auth.py db.py network_utils.py visualization.py

# macOS 必须在 macOS 上打包，Windows 不能直接生成可用的 .app。
"${PYTHON_BIN}" -m PyInstaller --clean --noconfirm packaging/StudyBehaviorMonitor.spec

echo ""
echo "Build finished."
echo "Application: ${PROJECT_ROOT}/dist/StudyBehaviorMonitor/StudyBehaviorMonitor.app"
