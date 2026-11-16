#!/bin/zsh
cd "$(dirname "$0")"
.venv/bin/python deploy_colab_model.py -y
echo ""
read -k 1 "?按任意键关闭此窗口..."
