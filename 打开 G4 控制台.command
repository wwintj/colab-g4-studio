#!/bin/zsh
cd "$(dirname "$0")"
pkill -f "colab_panel.py" 2>/dev/null || true
pkill -f "deploy_colab_model.py --proxy-daemon" 2>/dev/null || true
nohup .venv/bin/python colab_panel.py --no-browser >/tmp/colab_panel.log 2>&1 &
sleep 1
open "http://127.0.0.1:18080"
echo "✅ G4 可视化网页控制台已在浏览器打开 (http://127.0.0.1:18080)"
