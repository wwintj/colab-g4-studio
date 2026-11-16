#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/wwintj/colab-g4-studio.git"
INSTALL_DIR="${COLAB_G4_DIR:-$HOME/colab-g4-studio}"
MODE="${1:-install}"

# If executed inside an existing checkout (e.g. ./install.sh), use that directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [[ -f "$SCRIPT_DIR/colab_panel.py" && -f "$SCRIPT_DIR/deploy_colab_model.py" ]]; then
  INSTALL_DIR="$SCRIPT_DIR"
fi

echo "===================================================================="
if [[ "$MODE" == "--upgrade" || "$MODE" == "upgrade" ]]; then
  echo " 🔄 正在一键升级 Colab G4 Pro 视频与大模型工作台..."
else
  echo " 🚀 正在一键安装 Colab G4 Pro 视频与大模型工作台..."
fi
echo "    • 安装目录: $INSTALL_DIR"
echo "===================================================================="

# 1. 检查并补齐基础系统工具 (git, python3, rclone, ffmpeg)
if [[ "$(uname -s)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    for pkg in git python3 rclone ffmpeg; do
      if ! command -v "$pkg" >/dev/null 2>&1; then
        echo "📦 正在通过 Homebrew 安装依赖: $pkg ..."
        brew install "$pkg"
      fi
    done
  fi
elif command -v apt-get >/dev/null 2>&1; then
  for pkg in git python3 python3-venv rclone ffmpeg; do
    if ! command -v "${pkg%%-*}" >/dev/null 2>&1; then
      sudo apt-get update -qq && sudo apt-get install -y -qq git python3 python3-venv rclone ffmpeg
      break
    fi
  done
fi

# 2. 拉取或更新 GitHub 仓库
if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "📥 正在从 GitHub 同步最新代码 ($REPO_URL)..."
  git -C "$INSTALL_DIR" fetch origin main
  git -C "$INSTALL_DIR" reset --hard origin/main
elif [[ ! -f "$INSTALL_DIR/colab_panel.py" ]]; then
  echo "📥 正在克隆仓库至 $INSTALL_DIR ..."
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
chmod +x ./*.sh ./*.command 2>/dev/null || true

# 3. 创建/更新 Python 虚拟环境 (.venv)
if [[ ! -x "$INSTALL_DIR/.venv/bin/python" ]]; then
  echo "🐍 正在创建独立 Python 虚拟环境 (.venv)..."
  python3 -m venv "$INSTALL_DIR/.venv"
fi

echo "📦 正在安装/更新 Python 核心依赖 (google-colab-cli 等)..."
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" --upgrade -q

# 4. 创建本机视频保存目录
mkdir -p "$HOME/Movies/EasyColab"

# 5. 创建全局快捷命令 g4-studio (写入 ~/.local/bin 并加入 PATH)
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/g4-studio" <<EOF
#!/usr/bin/env bash
cd "$INSTALL_DIR"
case "\${1:-panel}" in
  panel|start|"")
    pkill -f "colab_panel.py" 2>/dev/null || true
    nohup "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/colab_panel.py" --no-browser >/tmp/colab_panel.log 2>&1 &
    sleep 1
    if command -v open >/dev/null 2>&1; then open "http://127.0.0.1:18080"; fi
    echo "✅ 控制台已启动: http://127.0.0.1:18080"
    ;;
  stop)
    "$INSTALL_DIR/.venv/bin/colab" stop -s qwen38-g4 2>/dev/null || true
    "$INSTALL_DIR/.venv/bin/colab" stop -s easycolab 2>/dev/null || true
    echo "✅ 已关闭云端 Colab 实例（0.00 CU/h）"
    ;;
  upgrade|update)
    bash "$INSTALL_DIR/upgrade.sh"
    ;;
  uninstall|remove)
    bash "$INSTALL_DIR/uninstall.sh"
    ;;
  auth|login)
    echo "🔐 步骤 1/2: 登录 Google Colab CLI..."
    "$INSTALL_DIR/.venv/bin/colab" auth login || true
    echo "🔐 步骤 2/2: 绑定 Google Drive 免弹窗持久化授权..."
    rclone config create gdrive drive scope=drive --config "$INSTALL_DIR/.gdrive_rclone.conf"
    echo "✅ 账号与云盘永久绑定完成！"
    ;;
  *)
    echo "用法: g4-studio [start | stop | auth | upgrade | uninstall]"
    ;;
esac
EOF
chmod +x "$HOME/.local/bin/g4-studio"

for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
  if [[ -f "$rc" ]] && ! grep -q '.local/bin' "$rc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
  fi
done

# 6. 在 macOS 桌面生成「打开 G4 控制台.command」快捷方式（可选便捷入口）
if [[ "$(uname -s)" == "Darwin" && -d "$HOME/Desktop" ]]; then
  cat > "$HOME/Desktop/打开 G4 控制台.command" <<EOF
#!/bin/zsh
cd "$INSTALL_DIR"
./打开\ G4\ 控制台.command
EOF
  chmod +x "$HOME/Desktop/打开 G4 控制台.command"
fi

# 7. 重启本地控制台守护进程
echo "🌐 正在启动本地可视化网页控制台 (http://127.0.0.1:18080)..."
pkill -f "colab_panel.py" 2>/dev/null || true
nohup "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/colab_panel.py" --no-browser >/tmp/colab_panel.log 2>&1 &
sleep 1

if command -v open >/dev/null 2>&1 && [[ "$MODE" != "--no-browser" ]]; then
  open "http://127.0.0.1:18080" 2>/dev/null || true
fi

echo "===================================================================="
echo " ✅ 完成！本地控制台已运行在: http://127.0.0.1:18080"
echo "    • 桌面快捷方式: ~/Desktop/打开 G4 控制台.command"
echo "    • 终端快捷命令: g4-studio [start | stop | auth | upgrade | uninstall]"
if [[ ! -f "$INSTALL_DIR/.gdrive_rclone.conf" ]]; then
  echo "--------------------------------------------------------------------"
  echo " 💡 首次使用提示：请运行以下命令完成一次性的 Google Colab 与云盘登录："
  echo "    $HOME/.local/bin/g4-studio auth"
fi
echo "===================================================================="
