@echo off
chcp 65001 >nul
set OLLAMA_ORIGINS=*
echo ===================================================================
echo Запуск системы мониторинга воздушных объектов (3 метода + Ollama)
echo Открываем браузер: http://localhost:8000
echo ===================================================================
start http://localhost:8000
python server.py
pause
