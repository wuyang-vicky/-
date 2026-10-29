@echo off
cd /d "%~dp0"
echo ========================================
echo   绣纹智绘 - 故宫纹样解析与创新生成演示
echo ========================================
echo 首次运行会自动安装依赖（需联网，约 1 分钟）...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败。请确认已安装 Python 3.9+ 并勾选“Add to PATH”。
    pause
    exit /b 1
)
echo 启动中，请稍候在浏览器打开 http://127.0.0.1:5000
python app.py
pause
