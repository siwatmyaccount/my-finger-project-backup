#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "คำเตือน: ชุดนี้จัดเตรียมสำหรับ Raspberry Pi OS 64-bit (aarch64)"
fi

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip python3-tk \
  libgl1 libglib2.0-0 v4l-utils \
  mpg123 alsa-utils fontconfig

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

FONT_DIR="$HOME/.local/share/fonts"
mkdir -p "$FONT_DIR"
cp -f "$PROJECT_DIR/src_code/Sarabun-Regular.ttf" "$FONT_DIR/"
cp -f "$PROJECT_DIR/src_code/Sarabun-Bold.ttf" "$FONT_DIR/"
fc-cache -f >/dev/null

if ! id -nG "$USER" | tr ' ' '\n' | grep -qx video; then
  sudo usermod -aG video "$USER"
  echo "เพิ่ม $USER เข้ากลุ่ม video แล้ว กรุณาออกจากระบบแล้วเข้าใหม่หรือรีสตาร์ตหนึ่งครั้ง"
fi

echo "ติดตั้งเสร็จแล้ว รันตรวจสอบด้วย: ./verify_pi4.sh"
echo "จากนั้นเปิดโปรแกรมด้วย: ./run.sh"
