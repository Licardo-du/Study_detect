"""程序启动入口。

这个文件保持极简，方便直接运行 `python app.py`，也方便 PyInstaller
把它作为桌面应用的打包入口。
"""

from ui import main


if __name__ == "__main__":
    # 只有直接运行 app.py 时才启动界面，作为模块导入时不会自动打开窗口。
    main()
