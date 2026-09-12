# my-finger-project-backup

**เวอร์ชันเสถียรบน Raspberry Pi 4**  
Build: `stable-last-crop-v5`  
สำรองจาก `my_finger_projectเวอชั่นเสถียร.zip` วันที่ 2026-09-12

โปรเจกต์ฝึกกายภาพมือด้วยกล้อง OpenCV และโมเดล TensorFlow Lite แสดงจุดโครงกระดูกมือแบบเรียลไทม์ พร้อมระบบจับเวลา รูปตัวอย่าง รายงาน และเสียงประกอบ

## ติดตั้งบน Raspberry Pi 4

ชุดติดตั้งเตรียมไว้สำหรับ Raspberry Pi OS แบบ Desktop 64-bit (`aarch64`) และกล้องที่ระบบมองเห็นเป็น `/dev/video0` ต้องมีอินเทอร์เน็ตตอนติดตั้ง เปิดโปรแกรมจาก Terminal บนหน้าจอ Pi เพื่อให้แสดง GUI ได้

Repository นี้เป็น Private ผู้ดาวน์โหลดต้องมีสิทธิ์เข้าถึงและยืนยันตัวตน GitHub ก่อน clone หรือเลือก Code → Download ZIP หลังเข้าสู่ระบบ

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

## ไฟล์ที่ไม่นำขึ้น Git

ไม่รวม `venv/` จาก ZIP เพราะเป็น environment ของ Windows มีขนาดใหญ่มากและใช้บน Raspberry Pi ไม่ได้ สคริปต์ติดตั้งจะสร้าง environment ที่ถูกต้องให้ใหม่ ไม่รวม cache, log, build output และประวัติการฝึกหรือข้อมูลความปวดเดิม เพื่อให้เครื่องใหม่เริ่มด้วยข้อมูลว่าง

`src_code/main.py` ตรงกับไฟล์ใน ZIP ล่าสุดทุกไบต์ ไม่ได้เปลี่ยนระบบตรวจจับในขั้นตอนสำรองนี้ ส่วน `hand_landmark.tflite` ที่ ZIP ไม่ได้ใส่มา สกัดจาก `hand_landmarker.task` ใน ZIP เดียวกัน และเพิ่มชื่อฟอนต์ `Sarabun.ttf` ให้ตรงกับที่โค้ดเรียกใช้

ใช้คู่มือที่ root นี้และ `run.sh` เป็นหลัก ไฟล์ `src_code/README.md`, `Anti-Finger.sh` และ `Anti-Finger.service` เก็บจากต้นฉบับเพื่อสำรอง บางรายการอ้าง MediaPipe หรือพาธเครื่องเก่า ไม่ใช่คำสั่งติดตั้งชุดนี้

การตรวจชุดสำรองครอบคลุม syntax, ความครบถ้วนของไฟล์ และ checksum ยังไม่ได้ทดสอบติดตั้งชุดนี้บน Pi เครื่องใหม่ คำว่า “เวอร์ชันเสถียร” เป็นชื่อรุ่นสำรองตามที่เจ้าของโครงการระบุ

รุ่นนี้เป็นซอร์สสำรองที่ปรับให้ทำงานเสถียรบน Pi 4 จากผลทดสอบของเครื่องโครงการ การใช้กับผู้ป่วยควรอยู่ภายใต้คำแนะนำของบุคลากรทางการแพทย์และต้องทดสอบกับกล้อง แสง และตำแหน่งติดตั้งจริงก่อนใช้งาน
