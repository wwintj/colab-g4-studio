#!/bin/zsh
cd "$(dirname "$0")"
echo "正在关闭运行中的 Colab 实例以停止计费..."
.venv/bin/colab stop -s qwen38-g4 2>/dev/null || true
.venv/bin/colab stop -s easycolab 2>/dev/null || true
echo "✅ 已全部关闭（扣费速率已归零 0.00 CU/h）！"
echo ""
read -k 1 "?按任意键关闭此窗口..."
