#!/bin/bash
cd "$(dirname "$0")"
echo "========================================"
echo "  绣纹智绘 - 故宫纹样解析与创新生成演示"
echo "========================================"
echo "首次运行会自动安装依赖（需联网，约 1 分钟）..."
python3 -m pip install -r requirements.txt || pip3 install -r requirements.txt
echo "启动中，请打开浏览器访问 http://127.0.0.1:5000"
python3 app.py
