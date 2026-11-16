#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${COLAB_G4_DIR:-$HOME/colab-g4-studio}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [[ -f "$SCRIPT_DIR/colab_panel.py" ]]; then
  INSTALL_DIR="$SCRIPT_DIR"
fi

echo "===================================================================="
echo " 🗑️  正在卸载 Colab G4 Pro 视频与大模型工作台..."
echo "===================================================================="

# 1. 自动关闭可能还在计费的云端 Colab 实例
if [[ -x "$INSTALL_DIR/.venv/bin/colab" ]]; then
  echo "🛑 正在检查并安全关闭云端 Colab 实例 (防止残留扣费)..."
  "$INSTALL_DIR/.venv/bin/colab" stop -s qwen38-g4 2>/dev/null || true
  "$INSTALL_DIR/.venv/bin/colab" stop -s easycolab 2>/dev/null || true
fi

# 2. 停止本机后台守护进程
echo "🛑 正在停止本机网页控制台与反代守护进程..."
pkill -f "colab_panel.py" 2>/dev/null || true
pkill -f "deploy_colab_model.py --proxy-daemon" 2>/dev/null || true

# 3. 清理全局命令与桌面快捷方式
rm -f "$HOME/.local/bin/g4-studio"
rm -f "$HOME/Desktop/打开 G4 控制台.command"

# 4. 清理程序目录（保留 ~/Movies/EasyColab 中已生成的视频文件）
if [[ -d "$INSTALL_DIR" ]]; then
  echo "🧹 正在移除安装目录: $INSTALL_DIR"
  rm -rf "$INSTALL_DIR"
fi

echo "===================================================================="
echo " ✅ 卸载完成！"
echo "    • 您之前生成的本地视频已安全保留在: ~/Movies/EasyColab/"
echo "    • 您 Google Drive 云盘中的模型文件 (MyDrive/models/) 均完好保留。"
echo "===================================================================="
