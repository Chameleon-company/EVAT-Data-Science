#!/bin/bash
# 啟動 Flask 伺服器腳本

cd "$(dirname "$0")"
echo "正在啟動 EV Traffic & Weather Model API..."
python3 api.py
