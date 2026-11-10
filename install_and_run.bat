@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/2] 安装依赖（首次运行需要几分钟）...
python -m pip install -r requirements.txt
echo.
echo [2/2] 启动平台...
python -m app.server
pause
