# my-finger-project-pi4 (backup)

**เวอร์ชันเสถียรบน Raspberry Pi 4**  
Build: `stable-last-crop-v5`  
สำรองจาก `my_finger_projectเวอชั่นเสถียร.zip` วันที่ 2026-09-12

โปรเจกต์ฝึกกายภาพมือด้วยกล้อง OpenCV และโมเดล TensorFlow Lite แสดงจุดโครงกระดูกมือแบบเรียลไทม์ พร้อมระบบจับเวลา รูปตัวอย่าง รายงาน และเสียงประกอบ

## ติดตั้งบน Raspberry Pi 4

ชุดติดตั้งเตรียมไว้สำหรับ Raspberry Pi OS แบบ Desktop 64-bit (`aarch64`) และกล้องที่ระบบมองเห็นเป็น `/dev/video0` ต้องมีอินเทอร์เน็ตตอนติดตั้ง เปิดโปรแกรมจาก Terminal บนหน้าจอ Pi เพื่อให้แสดง GUI ได้

Repository นี้เป็น Public ทุกคนเปิดดูและดาวน์โหลดได้โดยไม่ต้องเข้าสู่ระบบ

- ลิงก์แชร์โครงการ: https://github.com/siwatmyaccount/my-finger-project-backup
- ดาวน์โหลด ZIP: https://github.com/siwatmyaccount/my-finger-project-backup/archive/refs/heads/main.zip

คัดลอกลิงก์จากแถบที่อยู่ของเบราว์เซอร์เพื่อส่งต่อ หรือดาวน์โหลดผ่านปุ่ม Code → Download ZIP

```bash
git clone https://github.com/siwatmyaccount/my-finger-project-backup.git
cd my-finger-project-backup
chmod +x install_pi4.sh run.sh verify_pi4.sh
./install_pi4.sh
./verify_pi4.sh
./run.sh
```

หากสคริปต์เพิ่มผู้ใช้เข้ากลุ่ม `video` ให้รีสตาร์ตหนึ่งครั้งก่อนรัน:

```bash
sudo reboot
```

## รันพร้อมบันทึก log

```bash
cd ~/my-finger-project-backup
source .venv/bin/activate
cd src_code
python -u -X faulthandler main.py 2>&1 | tee tracking.log
```

เมื่อทำงานถูกต้องควรเห็นข้อความประมาณนี้:

```text
Build: stable-last-crop-v5
Camera /dev/video0: backend=V4L2, negotiated=320.0x240.0
Created TensorFlow Lite XNNPACK delegate for CPU
```

## ไฟล์สำคัญ

- `src_code/main.py` — โปรแกรมหลัก
- `src_code/hand_landmark.tflite` — โมเดลตรวจจุดมือที่โปรแกรมเรียกใช้
- `src_code/pictures/` — โลโก้ ภาพหน้าจอ และภาพตัวอย่างท่า 1–5
- `src_code/Voices/` — เสียง `001.mp3` ถึง `010.mp3`
- `requirements.txt` — รายการไลบรารี Python
- `install_pi4.sh` — ติดตั้งแพ็กเกจและสร้าง `.venv` ใหม่สำหรับ Pi
- `verify_pi4.sh` — ตรวจกล้อง สิทธิ์ ไฟล์ และไลบรารี
- `run.sh` — เปิดโปรแกรม

