#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [[ -f "$SCRIPT_DIR/install.sh" ]]; then
  exec bash "$SCRIPT_DIR/install.sh" --upgrade
else
  curl -fsSL https://raw.githubusercontent.com/wwintj/colab-g4-studio/main/install.sh | bash -s -- --upgrade
fi
