#!/usr/bin/env bash
export OLLAMA_ORIGINS="*"
echo "Запуск системы мониторинга воздушных объектов (http://localhost:8000)..."
python3 server.py
