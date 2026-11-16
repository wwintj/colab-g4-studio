#!/bin/zsh
cd "$(dirname "$0")"
bash ./upgrade.sh
echo ""
read -k 1 "?按任意键关闭此窗口..."
