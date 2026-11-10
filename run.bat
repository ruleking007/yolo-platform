@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 YOLO 训练平台...
python -m app.server %*
pause
