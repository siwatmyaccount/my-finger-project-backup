#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=0

echo "== ระบบ =="
uname -a
echo "architecture: $(uname -m)"

echo "== สิทธิ์ผู้ใช้ =="
id
if id -nG "$USER" | tr ' ' '\n' | grep -qx video; then
  echo "OK: ผู้ใช้อยู่ในกลุ่ม video"
else
  echo "ERROR: ผู้ใช้ยังไม่อยู่ในกลุ่ม video"
  echo "แก้ด้วย: sudo usermod -aG video \"$USER\" แล้วรีสตาร์ต"
  FAILED=1
fi

echo "== อุปกรณ์กล้อง =="
ls -l /dev/video* 2>/dev/null || true
if [[ -e /dev/video0 && -r /dev/video0 && -w /dev/video0 ]]; then
  echo "OK: อ่านและเขียน /dev/video0 ได้"
else
  echo "ERROR: /dev/video0 ไม่มีหรือไม่มีสิทธิ์ใช้งาน"
  FAILED=1
fi

if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --list-devices || true
  v4l2-ctl --device=/dev/video0 --all 2>/dev/null | sed -n '1,35p' || true
fi

echo "== ไฟล์และ Python =="
for path in \
  "$PROJECT_DIR/src_code/main.py" \
  "$PROJECT_DIR/src_code/hand_landmark.tflite" \
  "$PROJECT_DIR/.venv/bin/python"; do
  if [[ -e "$path" ]]; then
    echo "OK: $path"
  else
    echo "ERROR: ไม่พบ $path"
    FAILED=1
  fi
done

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "$PROJECT_DIR/.venv/bin/python" -m py_compile "$PROJECT_DIR/src_code/main.py" || FAILED=1
  "$PROJECT_DIR/.venv/bin/python" - <<'PY' || FAILED=1
import cv2
import customtkinter
import matplotlib
import numpy
from PIL import Image
try:
    from tflite_runtime.interpreter import Interpreter
    print("OK: ใช้ tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("OK: ใช้ tensorflow.lite")
print("OK: import ไลบรารีหลักสำเร็จ")
PY
fi

if [[ $FAILED -eq 0 ]]; then
  echo "พร้อมรัน: ./run.sh"
else
  echo "การตรวจสอบพบปัญหา โปรดแก้ข้อความ ERROR ด้านบน"
fi
exit "$FAILED"
