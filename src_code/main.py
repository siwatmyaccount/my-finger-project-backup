try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from logging import root
from PIL import Image, ImageTk
import customtkinter as ctk
import cv2

import time, threading
import logging
from PIL import ImageOps
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from collections import defaultdict
import sys, os
import tkinter as tk
from tkinter import ttk
import random
import math
import numpy as np


HAND_CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                    (15,16),(13,17),(0,17),(17,18),(18,19),(19,20))


def overlay_tensor_indices(outputs, fallback_score):
    # Drawing requires image coordinates, never the world-landmark tensor.
    candidates = [d for d in outputs if np.prod(d['shape']) == 63]
    named = {d['name'].split(':')[0].split('/')[-1]: d for d in outputs}
    image = next((named[n] for n in ('Identity', 'landmarks', 'image_landmarks')
                  if n in named and np.prod(named[n]['shape']) == 63), None)
    if image is None and len(candidates) == 1:
        image = candidates[0]
    presence = named.get('Identity_1')
    score = presence['index'] if presence is not None and np.prod(presence['shape']) == 1 else fallback_score
    return (None if image is None else image['index']), score


def overlay_points(raw, box, input_size):
    points = np.asarray(raw, dtype=np.float64).reshape(21,3)
    if not np.isfinite(points).all():
        return None
    x1,y1,x2,y2 = box
    width,height = input_size
    xy = points[:,:2].copy()
    xy[:,0] = x1 + xy[:,0] / width * (x2-x1)
    xy[:,1] = y1 + xy[:,1] / height * (y2-y1)
    return xy


def draw_hand_overlay(frame, points):
    display = frame.copy()
    if points is None:
        return display
    points = np.asarray(points)
    if points.shape != (21,2) or not np.isfinite(points).all():
        return display
    h,w = frame.shape[:2]
    inside = (points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
    xy = np.rint(np.clip(points,-100000,100000)).astype(int)
    for first,last in HAND_CONNECTIONS:
        if inside[first] and inside[last]:
            cv2.line(display,tuple(xy[first]),tuple(xy[last]),(0,220,0),2,cv2.LINE_AA)
    for index in np.flatnonzero(inside):
        cv2.circle(display,tuple(xy[index]),3,(0,80,255),-1,cv2.LINE_AA)
    return display


def result_is_timely(now, captured, completed):
    # Completion freshness and camera age are distinct. Do not rejuvenate an
    # arbitrarily old frame just because inference finished a moment ago.
    return (completed > 0 and captured <= completed <= now
            and now-completed <= .7 and completed-captured <= 1.2
            and now-captured <= 1.5)


class PoseHoldGate:
    """Debounce observations, not GUI ticks. Durations are software defaults."""

    def __init__(self, acquire=0.10, grace=0.6, stale=0.8):
        self.acquire, self.grace, self.stale = acquire, grace, stale
        self.reset()

    def reset(self):
        self.context = None
        self.seq = None
        self.positive_since = None
        self.positive_samples = 0
        self.last_good = float("-inf")
        self.locked = False
        self.current_good = False

    def update(self, now, seq, captured, match, present, healthy, context):
        if context != self.context:
            self.reset()
            self.context = context
        if not healthy or captured > now or now - captured > self.stale:
            self.reset()
            self.context = context
            return False, "paused", False
        # Expire BEFORE ingesting a new observation, so a long gap cannot
        # continue an old lock when a fresh positive sample finally arrives.
        if self.locked and captured - self.last_good > self.grace:
            self.locked = False
            self.positive_since = None
            self.positive_samples = 0
        if seq != self.seq:
            self.seq = seq
            self.current_good = bool(match and present)
            if self.current_good:
                self.last_good = captured
                if self.positive_since is None:
                    self.positive_since = captured
                self.positive_samples += 1
                if self.positive_samples >= 2 and captured - self.positive_since >= self.acquire:
                    self.locked = True
            elif not self.locked:
                self.positive_since = None
                self.positive_samples = 0
        if self.locked and now - self.last_good <= self.grace:
            confirmed = self.current_good and now - captured <= 0.35
            return True, "stable" if confirmed else "grace", confirmed
        if self.locked:
            self.locked = False
            self.positive_since = None
            self.positive_samples = 0
        return False, "acquiring" if self.current_good else "paused", False

def select_hand_outputs(details):
    """Match semantic names or the official MediaPipe Identity export contract.

    Unknown/ambiguous exports fail visibly instead of treating handedness as
    hand presence or world coordinates as image pixel coordinates.
    """
    import re
    def size(d):
        return int(np.prod(d["shape"]))
    def name(d):
        return d["name"].lower().split(":")[0].split("/")[-1]
    named_landmarks = [d for d in details if size(d) == 63 and
                       ("ld_21_3d" in name(d) or
                        ("landmark" in name(d) and "world" not in name(d)))]
    named_presence = [d for d in details if size(d) == 1 and
                      any(word in name(d) for word in ("handflag", "presence", "hand_flag"))]
    named_handedness = [d for d in details if size(d) == 1 and "handedness" in name(d)]
    if len(named_landmarks) == len(named_presence) == 1:
        return (named_landmarks[0], named_presence[0],
                named_handedness[0] if len(named_handedness) == 1 else None)
    identities = {name(d): d for d in details if re.fullmatch(r"identity(?:_[123])?", name(d))}
    if (len(details) in (3, 4) and len(identities) == len(details)
            and all(key in identities for key in ("identity", "identity_1", "identity_2"))
            and size(identities["identity"]) == 63
            and size(identities["identity_1"]) == size(identities["identity_2"]) == 1
            and (len(details) == 3 or size(identities["identity_3"]) == 63)):
        return identities["identity"], identities["identity_1"], identities["identity_2"]
    # Older exports with exactly one landmark and one scalar are unambiguous.
    vectors = [d for d in details if size(d) == 63]
    scalars = [d for d in details if size(d) == 1]
    if len(details) == 2 and len(vectors) == len(scalars) == 1:
        return vectors[0], scalars[0], None
    summary = [(d["index"], d["name"], list(d["shape"])) for d in details]
    raise ValueError(f"Unknown hand model output mapping; inspect model: {summary}")

def comfortable_pose_match(points, pose, margin=6.0):
    """Small software tolerance around the original ranges, not clinical criteria."""
    low, high = {1:(120,200),2:(15,170),3:(0,110),4:(0,140),5:(30,180)}.get(pose,(120,200))
    count = 0
    for tip,base in ((8,5),(12,9),(16,13),(20,17)):
        first, second = points[tip]-points[base], points[0]-points[base]
        norm = float(np.linalg.norm(first)*np.linalg.norm(second))
        if norm <= 1e-6:
            continue
        angle = math.degrees(math.acos(float(np.clip(np.dot(first,second)/norm,-1,1))))
        count += max(0,low-margin) <= angle <= min(180,high+margin)
    return count >= 3


class HandPresenceLatch:
    """Brief display-only tolerance; never refresh from repeated GUI reads."""
    def __init__(self):
        self.last_seen = float('-inf')
        self.seq = None
    def update(self, now, seq, captured, present, healthy):
        if not healthy or not 0 <= now-captured < .8:
            self.last_seen = float('-inf')
            self.seq = seq
            return False
        if seq != self.seq:
            self.seq = seq
            if present:
                self.last_seen = captured
        return 0 <= now-self.last_seen <= 1.0


class StableLocalTracker:
    """Reuse the working model/crop path; bounded local search, no extra models."""
    def __init__(self, interpreter):
        self.model = interpreter
        inputs = interpreter.get_input_details()
        image, presence, _ = select_hand_outputs(interpreter.get_output_details())
        if len(inputs)!=1 or inputs[0]['dtype']!=np.float32:
            raise ValueError('Expected the original float32 hand model')
        self.input = inputs[0]['index']
        self.height,self.width = map(int,inputs[0]['shape'][1:3])
        self.image,self.presence = image['index'],presence['index']
        logging.info('Hand outputs: image=%s presence=%s',image['name'],presence['name'])
        self.box = None
        self.last_points = None
        self.last_seen = float('-inf')
        self.last_good = float('-inf')
        self.last_pose = None
        self.track_id = 0
        self.search_index = 0
        self.pending_box = None
        self.next_box_trial = 0.0
        self.retry_previous = False
        self.misses = 0
        self.last_score = float('nan')
        self.last_reason = "starting"

    @staticmethod
    def bounded_box(center, side, shape):
        h,w = shape[:2]
        cx,cy = center
        return (max(0,int(cx-side/2)),max(0,int(cy-side/2)),
                min(w,int(cx+side/2)),min(h,int(cy+side/2)))

    def search_box(self, frame):
        # Preserve the known working dark-contour proposal, but don't let a
        # bright background or a wrong large shadow veto all inference.
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),70,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[(0,0,w,h)]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:3]:
            if 1500 < cv2.contourArea(contour) < h*w*.95:
                x,y,cw,ch = cv2.boundingRect(contour)
                candidates.append(self.bounded_box((x+cw/2,y+ch/2),max(cw,ch)*1.6,frame.shape))
        # Overlapping local views help acquisition when the hand occupies only
        # part of the full image. Still only one search inference per sample.
        for fraction in (.25,.5,.75):
            candidates.append(self.bounded_box((w*fraction,h*.5),h*.95,frame.shape))
        result = candidates[self.search_index % len(candidates)]
        self.search_index += 1
        return result

    def infer(self, frame, box, threshold):
        x1,y1,x2,y2 = box
        crop = frame[y1:y2,x1:x2]
        if min(crop.shape[:2]) < 32:
            return None
        # Keep the hand proportions intact even for a tall/narrow crop.
        side=max(crop.shape[:2])
        left=(side-crop.shape[1])//2
        top=(side-crop.shape[0])//2
        padded=cv2.copyMakeBorder(crop,top,side-crop.shape[0]-top,
                                 left,side-crop.shape[1]-left,cv2.BORDER_CONSTANT,value=(0,0,0))
        data = cv2.resize(cv2.cvtColor(padded,cv2.COLOR_BGR2RGB),(self.width,self.height))
        self.model.set_tensor(self.input,data.astype(np.float32)[None]/255.0)
        self.model.invoke()
        score = float(self.model.get_tensor(self.presence).reshape(-1)[0])
        self.last_score = score
        if not np.isfinite(score) or not threshold <= score <= 1:
            self.last_reason = "low_presence"
            return None
        raw = np.asarray(self.model.get_tensor(self.image),dtype=np.float32).reshape(21,3)
        if not np.isfinite(raw).all():
            self.last_reason = "invalid_points"
            return None
        points = raw.copy()
        points[:,0] = x1-left + raw[:,0]*side/self.width
        points[:,1] = y1-top + raw[:,1]*side/self.height
        points[:,2] = raw[:,2]*side/self.width
        h,w=frame.shape[:2]
        visible=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
        palm=float(np.linalg.norm(points[9,:2]-points[0,:2]))
        # Seeing a hand does not require all fingertips to be usable for pose
        # assessment. Keep that stricter check below, exclusively for timing.
        if visible.sum()<8 or not 2<palm<max(w,h):
            self.last_reason = "point_geometry"
            return None
        self.last_reason = "found"
        return points

    def process(self, frame, captured, pose):
        h, w = frame.shape[:2]
        # Keep the last successful crop through transient low-presence results.
        # The v4 code cleared it after one miss, which made acquisition rotate
        # through unrelated full-frame tiles and caused long YES/NO oscillation.
        if self.box is not None and captured - self.last_seen <= 2.0:
            x1, y1, x2, y2 = self.box
            cx, cy = (x1+x2)/2, (y1+y2)/2
            base_side = max(x2-x1, y2-y1)
            recovery = self.misses % 6
            if recovery in (0, 1, 3):
                used_box = self.box
            elif recovery == 2:
                used_box = self.bounded_box((cx,cy), base_side*1.25, frame.shape)
            elif recovery == 4:
                used_box = self.bounded_box((cx,cy), base_side*1.55, frame.shape)
            else:
                used_box = self.search_box(frame)
            threshold = .40 if captured-self.last_seen <= 1.0 else .50
        else:
            used_box = self.search_box(frame)
            threshold = .50

        # One inference per new camera sample. Recovery choices are tried on
        # subsequent fresh frames, avoiding the old 800-890 ms double invokes.
        points = self.infer(frame, used_box, threshold)
        if points is None:
            self.misses += 1
            self.last_reason += f"_miss{self.misses}"
            if captured-self.last_seen > 2.0:
                self.box = None
            return None, False, self.track_id

        was_tracking = self.box is not None and captured-self.last_seen <= 2.0
        self.misses = 0
        self.box = used_box
        span=max(float(np.ptp(points[:,0])),float(np.ptp(points[:,1])),40.0)
        same=(self.last_points is not None and captured-self.last_seen<=2.0
              and np.linalg.norm(points[0,:2]-self.last_points[0,:2])<span*.75)
        if same:
            alpha=1-math.exp(-max(0,captured-self.last_seen)/.075)
            filtered=self.last_points+alpha*(points-self.last_points)
        else:
            self.track_id+=1
            self.last_good=float('-inf')
            filtered=points

        # Recenter gradually only when the hand approaches the crop border.
        center=(points[:,:2].min(axis=0)+points[:,:2].max(axis=0))/2
        proposed=self.bounded_box(center,span*1.85,frame.shape)
        x1,y1,x2,y2=self.box
        inset=min(x2-x1,y2-y1)*.10
        near_border=(points[:,0].min()<x1+inset or points[:,0].max()>x2-inset
                     or points[:,1].min()<y1+inset or points[:,1].max()>y2-inset)
        if near_border:
            old=np.asarray(self.box,dtype=float)
            new=np.asarray(proposed,dtype=float)
            self.box=tuple(np.rint(old*.65+new*.35).astype(int))

        margin=10.0 if self.last_pose==pose and captured-self.last_good<=.6 else 6.0
        visible=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
        pose_usable=bool(visible[[0,4,8,12,16,20]].all() and visible.sum()>=19)
        matched=pose_usable and comfortable_pose_match(filtered,pose,margin)
        if matched:
            self.last_good=captured
        self.last_points=filtered.copy()
        self.last_seen=captured
        self.last_pose=pose
        self.last_reason = "found_tracking" if was_tracking else "found_acquired"
        return filtered,matched,self.track_id


class AntiTriggerFingersApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        
        base_w = 1920
        base_h = 1080

        self.scale_w = self.screen_width / base_w
        self.scale_h = self.screen_height / base_h
        self.u_scale = min(self.scale_w, self.scale_h)

        self.title("AI-Powered Anti-trigger Fingers")
        self.bind("<Escape>", lambda e: self.on_close())
        self.configure(fg_color="#F4F7F6") 

        self.target_sets_var = tk.IntVar(value=3)
        self.target_hand_var = ctk.StringVar(value="ทั้งสอง")
        self.difficulty_var = ctk.StringVar(value="กลาง (5s)")

        self.key_held = False
        self.time_max = 5
        self.time_current = self.time_max
        self.hand_posit = 0
        self.still_hold = False
        self.current_pose = 1
        self.key = ""
        self.is_pass = False
        self.round = 0
        self.set = 0
        self.streak_days = 0
        self.chart_timeframe = 14
        
        self._cached_history_data = None
        self._last_history_read_time = 0

        self.pose_name = [
            "placeholder",
            "เหยียดมือตรง",
            "ทำมือคล้ายตะขอ",
            "กำมือ",
            "กำมือแบบเหยียดปลายนิ้ว",
            "งอโคนนิ้วแต่เหยียดปลายนิ้วมือ" 
        ]

        # Only the capture worker opens, reads and releases the device.
        self.camera_path = os.environ.get("CAMERA_DEVICE", "/dev/video0")
        self._stop_video = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_seq = 0
        self.frame_time = 0.0
        self.latest_match = False
        self.latest_overlay_points = None
        self.overlay_time = 0.0
        self.overlay_completed = 0.0
        self.result_completed = 0.0
        self.latest_present = False
        self.result_seq = -1
        self.result_track_id = 0
        self.ai_target_hand = "ทั้งสอง"
        self.result_target_hand = None
        self._match_gate = PoseHoldGate()
        self._presence_latch = HandPresenceLatch()
        self._match_confirmed = False
        self._match_state = "paused"
        self._match_updated = 0.0
        self._last_sensor_tick = time.monotonic()
        self.latest_env_text = "AI: กำลังเริ่มต้น..."
        self.latest_env_color = "#F39C12"
        self.result_time = 0.0
        self.result_pose = None
        self.ai_pose = self.current_pose
        self.camera_error = None
        self.ai_error = None
        self._closing = False
        self._video_job = None
        self._sensor_job = None
        self._displayed_seq = -1
        self._workers = []
        self._preview_only = os.environ.get("CAMERA_PREVIEW_ONLY") == "1"

        self.purple_bg = "#4A235A" 
        self.light_gray_bg = "#FFFFFF" 
        self.light_gray_bg_program = "transparent"
        self.red_btn = "#E74C3C"
        self.hover_red_bt = "#C0392B"
        self.yellow_btn = "#F39C12"
        self.hover_yellow_bt = "#D68910"
        self.green_btn = "#2ECC71"
        self.hover_green_bt = "#27AE60"
        self.blue_btn = "#3498DB"
        self.white_fg = "#ffffff"
        self.black_fg = "#2C3E50"

        self.font_large_title = ("Sarabun", int(32 * self.u_scale), "bold")
        self.font_medium_text = ("Sarabun", int(32 * self.u_scale), "bold") 
        self.font_timer = ("Sarabun", int(55 * self.u_scale), "bold")       
        self.font_pose_title = ("Sarabun", int(28 * self.u_scale), "bold")  
        self.font_pose_action = ("Sarabun", int(25 * self.u_scale), "bold") 
        self.font_btn_text = ("Sarabun", int(22 * self.u_scale), "bold")    
        self.font_small = ("Sarabun", int(18 * self.u_scale), "bold")       
        self.font_feedback = ("Sarabun", int(18 * self.u_scale), "bold")

        top_bar_h = int(105 * self.scale_h)
        logo_size = int(100 * self.u_scale)
        
        self.camera_width = int(640 * self.scale_w)
        self.camera_height = int(880 * self.scale_h)
        
        self.btn_h = int(65 * self.scale_h) 
        self.btn_w = int(160 * self.scale_w)
        
        self.timer_canvas_size = int(200 * self.u_scale)
        self.timer_pad = 8

        history_data = self.get_history_from_file()
        if history_data:
            dates = sorted([h['date'].date() for h in history_data])
            today = datetime.now().date()
            if dates[-1] == today or dates[-1] == today - timedelta(days=1):
                self.streak_days = 1
                for i in range(len(dates)-1, 0, -1):
                    if (dates[i] - dates[i-1]).days == 1:
                        self.streak_days += 1
                    elif (dates[i] - dates[i-1]).days > 1:
                        break

        self.top_bar_frame = ctk.CTkFrame(self, fg_color=self.purple_bg, height=top_bar_h, corner_radius=0)
        self.top_bar_frame.pack(side="top", fill="x")
        self.top_bar_frame.pack_propagate(False)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(BASE_DIR, "pictures", "logo.png")
        try:
            logo_image_pil = Image.open(logo_path)
            self.logo_photo = ctk.CTkImage(light_image=logo_image_pil, size=(logo_size, logo_size))
            self.logo_label = ctk.CTkLabel(self.top_bar_frame, image=self.logo_photo, text="")
            self.logo_label.pack(side="left", padx=int(20 * self.scale_w), pady=int(10 * self.scale_h))
        except FileNotFoundError:
            self.logo_label = ctk.CTkLabel(self.top_bar_frame, text="LOGO", font=("Sarabun", 18, "bold"), text_color=self.white_fg)
            self.logo_label.pack(side="left", padx=20, pady=15)

        self.app_title_label = ctk.CTkLabel(
            self.top_bar_frame, text="AI-Powered Anti-trigger Fingers", font=self.font_large_title, text_color=self.white_fg
        )
        self.app_title_label.pack(side="left", padx=12, pady=15)

        self.exit_btn = ctk.CTkButton(
            self.top_bar_frame, text="✕", font=("Arial", 24, "bold"), fg_color="#C0392B", text_color="#FFFFFF",
            hover_color="#992D22", width=50, height=50, corner_radius=25, command=self.on_close
        )
        self.exit_btn.pack(side="right", padx=20, pady=15)

        self.notif_frame = ctk.CTkFrame(self.top_bar_frame, fg_color=self.green_btn, corner_radius=10)
        self.notif_label = ctk.CTkLabel(
            self.notif_frame, text="", font=("Sarabun", int(14 * self.u_scale), "bold"), text_color="#FFFFFF"
        )
        self.notif_label.pack(side="left", padx=(12, 6), pady=6)
        self.notif_hide_job = None

        streak_color = "#F1C40F" if self.streak_days > 0 else "#95A5A6"
        self.streak_label = ctk.CTkLabel(
            self.top_bar_frame, text=f"🔥 ต่อเนื่อง: {self.streak_days} วัน", 
            font=("Sarabun", int(18 * self.u_scale), "bold"), text_color=streak_color
        )
        self.streak_label.pack(side="right", padx=20, pady=15)

        self.main_content_frame = ctk.CTkFrame(self, fg_color=self.light_gray_bg_program)
        self.main_content_frame.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        
        self.main_content_frame.grid_columnconfigure(0, weight=4) 
        self.main_content_frame.grid_columnconfigure(1, weight=3) 
        self.main_content_frame.grid_columnconfigure(2, weight=3) 
        self.main_content_frame.grid_rowconfigure(0, weight=1)

        self.camera_frame_bg = ctk.CTkFrame(self.main_content_frame, fg_color=self.light_gray_bg, corner_radius=12, border_width=1, border_color="#D5D8DC")
        self.camera_frame_bg.grid(row=0, column=0, padx=3, pady=1, sticky="nsew") 
        
        placeholder = Image.new("RGB", (self.camera_width, self.camera_height), (220, 220, 220))
        self.camera_photo = ctk.CTkImage(light_image=placeholder, size=(self.camera_width, self.camera_height))
        self.camera_label = ctk.CTkLabel(self.camera_frame_bg, image=self.camera_photo, text="", corner_radius=12)
        self.camera_label.pack(expand=True, padx=3, pady=3)

        self.match_badge = ctk.CTkLabel(
            self.camera_label, text="Match: NO", font=("Sarabun", int(16 * self.u_scale), "bold"), 
            fg_color="#FFFFFF", text_color="#E74C3C", corner_radius=6, padx=10, pady=2
        )
        self.match_badge.place(x=15, y=15)
        
        self.env_badge = ctk.CTkLabel(
            self.camera_label, text="ระยะกล้อง/แสง: ตรวจสอบ...", font=("Sarabun", int(12 * self.u_scale), "bold"), 
            fg_color="#2C3E50", text_color="#FFFFFF", corner_radius=6, padx=6, pady=2
        )
        self.env_badge.place(x=15, y=55)

        self.center_info_container = ctk.CTkFrame(self.main_content_frame, fg_color="transparent")
        self.center_info_container.grid(row=0, column=1, sticky="nsew", padx=3, pady=1)
        self.center_info_container.grid_rowconfigure(0, weight=1)
        self.center_info_container.grid_rowconfigure(1, weight=1)
        self.center_info_container.grid_rowconfigure(2, weight=1)
        self.center_info_container.grid_rowconfigure(3, weight=1)

        self.set_times_frame = ctk.CTkFrame(self.center_info_container, fg_color=self.light_gray_bg, corner_radius=12, border_width=1, border_color="#D5D8DC")
        self.set_times_frame.pack(side="top", fill="both", expand=True, pady=(0, 3))

        self.times_line_frame = ctk.CTkFrame(self.set_times_frame, fg_color="transparent")
        self.times_line_frame.pack(side="top", pady=(8, 1))
        
        self.Label_times_text = ctk.CTkLabel(self.times_line_frame, text="รอบที่ : ", font=self.font_medium_text, text_color=self.black_fg)
        self.Label_times_text.pack(side="left", padx=(14, 0))
        self.Label_set_times_number = ctk.CTkLabel(self.times_line_frame, text=f"{self.round}", font=self.font_medium_text, text_color=self.blue_btn)
        self.Label_set_times_number.pack(side="left", padx=(0, 14))

        self.sets_line_frame = ctk.CTkFrame(self.set_times_frame, fg_color="transparent")
        self.sets_line_frame.pack(side="top", pady=(1, 4))
        
        self.Label_set_text = ctk.CTkLabel(self.sets_line_frame, text="เซ็ตที่ : ", font=self.font_medium_text, text_color=self.black_fg)
        self.Label_set_text.pack(side="left", padx=(14, 0))
        self.Label_set_number = ctk.CTkLabel(self.sets_line_frame, text=f"{self.set}", font=self.font_medium_text, text_color=self.green_btn)
        self.Label_set_number.pack(side="left", padx=(0, 14))

        self.round_progress = ctk.CTkProgressBar(self.set_times_frame, width=int(280 * self.scale_w), height=14, fg_color="#EAECEE", progress_color=self.green_btn, corner_radius=8)
        self.round_progress.pack(pady=(2, 4))
        self.round_progress.set(0) 
        
        self.reward_label = ctk.CTkLabel(self.set_times_frame, text="🌟 พร้อมแล้ว เริ่มลุยเลย!", font=("Sarabun", int(16 * self.u_scale), "bold"), text_color="#F39C12")
        self.reward_label.pack(side="top", pady=(0, 4))

        self.pose_text_frame = ctk.CTkFrame(self.center_info_container, fg_color=self.light_gray_bg, corner_radius=12, border_width=1, border_color="#D5D8DC")
        self.pose_text_frame.pack(side="top", fill="both", expand=True, pady=(0, 3))
        
        self.Label_pose_thai_text = ctk.CTkLabel(self.pose_text_frame, text=f"ท่าที่ {self.current_pose}", font=self.font_pose_title, text_color=self.black_fg)
        self.Label_pose_thai_text.pack(side="top", pady=(8, 2))
        
        self.Label_pose_action_text = ctk.CTkLabel(
            self.pose_text_frame, text=f"{self.pose_name[self.current_pose]}", font=self.font_pose_action, 
            text_color=self.purple_bg, wraplength=int(300 * self.scale_w)
        )
        self.Label_pose_action_text.pack(side="top", pady=(0, 3), padx=8)

        self.pose_feedback_label = ctk.CTkLabel(self.pose_text_frame, text="💬 รอตรวจจับท่าทาง...", font=self.font_feedback, text_color="#7F8C8D")
        self.pose_feedback_label.pack(side="top", pady=(0, 3))

        self.popup_menu_frame = ctk.CTkFrame(self.center_info_container, fg_color=self.light_gray_bg, corner_radius=12, border_width=1, border_color="#D5D8DC")
        self.popup_menu_frame.pack(side="top", fill="both", expand=True, pady=(0, 3), padx=0)

        self.btn_select_hand = ctk.CTkButton(
            self.popup_menu_frame, text=f"🖐️ มือ: {self.target_hand_var.get()}  ▼ [แตะเพื่อเปลี่ยน]", font=self.font_small, 
            fg_color="#E5E7E9", text_color=self.black_fg, hover_color="#BDC3C7", height=45, corner_radius=10,
            command=self.open_hand_popup
        )
        self.btn_select_hand.pack(fill="x", padx=12, pady=4)

        self.btn_select_diff = ctk.CTkButton(
            self.popup_menu_frame, text=f"⏱️ ความยาก: {self.difficulty_var.get()}  ▼ [แตะเพื่อเปลี่ยน]", font=self.font_small, 
            fg_color="#E5E7E9", text_color=self.black_fg, hover_color="#BDC3C7", height=45, corner_radius=10,
            command=self.open_diff_popup
        )
        self.btn_select_diff.pack(fill="x", padx=12, pady=4)

        self.btn_select_target = ctk.CTkButton(
            self.popup_menu_frame, text=f"🎯 เป้าหมาย: {self.target_sets_var.get()} เซ็ต  ▼ [แตะเพื่อเปลี่ยน]", font=self.font_small, 
            fg_color="#E5E7E9", text_color=self.black_fg, hover_color="#BDC3C7", height=45, corner_radius=10,
            command=self.open_target_popup
        )
        self.btn_select_target.pack(fill="x", padx=12, pady=(4, 6))

        self.pain_frame = ctk.CTkFrame(self.center_info_container, fg_color="#FDF2E9", corner_radius=12, border_width=1, border_color="#F5B041")
        self.pain_frame.pack(side="top", fill="both", expand=True, pady=(0, 3))
        
        ctk.CTkLabel(self.pain_frame, text="📈 ประเมินความปวด (0-10)", font=("Sarabun", int(14 * self.u_scale), "bold"), text_color="#D35400").pack(pady=(4, 0))
        
        slider_inner = ctk.CTkFrame(self.pain_frame, fg_color="transparent")
        slider_inner.pack(fill="x", padx=10, pady=(0, 4))
        
        self.pain_var = ctk.IntVar(value=5)
        self.pain_val_lbl = ctk.CTkLabel(slider_inner, text="5", font=("Sarabun", 16, "bold"), text_color="#C0392B", width=25)
        self.pain_val_lbl.pack(side="left", padx=(2, 6))
        
        self.pain_slider = ctk.CTkSlider(slider_inner, from_=0, to=10, number_of_steps=10, variable=self.pain_var, command=lambda v: self.pain_val_lbl.configure(text=int(v)), progress_color="#E74C3C", button_color="#C0392B", height=22)
        self.pain_slider.pack(side="left", fill="x", expand=True, padx=4)
        
        self.pain_btn = ctk.CTkButton(slider_inner, text="บันทึก", font=self.font_small, fg_color="#E74C3C", hover_color="#C0392B", width=60, height=int(36 * self.scale_h), command=self.save_pain_log)
        self.pain_btn.pack(side="right", padx=(8, 2))

        self.controls_btn_frame = ctk.CTkFrame(self.center_info_container, fg_color="transparent")
        self.controls_btn_frame.pack(side="bottom", fill="x", pady=(4, 0))
        
        self.start_stop_button = ctk.CTkButton(
            self.controls_btn_frame, text="▶ เริ่มต้น", font=self.font_btn_text, fg_color=self.green_btn,
            text_color=self.white_fg, command=lambda: self.toggle_start_pause(), height=self.btn_h,
            hover_color=self.hover_green_bt, corner_radius=15
        )
        self.start_stop_button.pack(side="top", fill="x", pady=(0, 16))  
        
        self.reset_button = ctk.CTkButton(
            self.controls_btn_frame, text="🔄 รีเซ็ต", font=self.font_btn_text, fg_color=self.red_btn,
            text_color=self.white_fg, command=lambda: self.reset_action(), height=self.btn_h,
            hover_color=self.hover_red_bt, corner_radius=15
        )
        self.reset_button.pack(side="top", fill="x")

        self.right_info_container = ctk.CTkFrame(self.main_content_frame, fg_color="transparent")
        self.right_info_container.grid(row=0, column=2, sticky="nsew", padx=3, pady=1)

        self.timer_frame = ctk.CTkFrame(self.right_info_container, fg_color="transparent")
        self.timer_frame.pack(side="top", fill="x", pady=(0, 2), ipady=2)
        
        self.timer_canvas = ctk.CTkCanvas(self.timer_frame, width=self.timer_canvas_size, height=self.timer_canvas_size, bg="#F4F7F6", highlightthickness=0)
        self.timer_canvas.pack(pady=2)
        
        left_p = self.timer_pad
        right_p = self.timer_canvas_size - self.timer_pad
        self.timer_canvas.create_oval(left_p, left_p, right_p, right_p, outline=self.green_btn, width=12, tags="progress")
        center = self.timer_canvas_size // 2
        
        display_time = int(math.ceil(self.time_current))
        self.timer_text = self.timer_canvas.create_text(center, center, text=str(display_time), font=self.font_timer, fill=self.black_fg)

        self.breathing_label = ctk.CTkLabel(self.timer_frame, text="เตรียมพร้อมหายใจ...", font=("Sarabun", int(16 * self.u_scale), "bold"), text_color="#BDC3C7")
        self.breathing_label.pack(pady=(0, 2))

        img_size = int(220 * self.u_scale) 
        self.small_hand_bg = ctk.CTkFrame(self.right_info_container, fg_color=self.light_gray_bg, corner_radius=12, border_width=1, border_color="#D5D8DC")
        self.small_hand_bg.pack(side="top", fill="both", expand=True, pady=(0, 6)) 

        try:
            pose_path = os.path.join(BASE_DIR, "pictures", "EX_POSE", "pose1.png")
            small_hand_image_pil = Image.open(pose_path)
            self.small_hand_photo = ctk.CTkImage(light_image=small_hand_image_pil, size=(img_size, img_size))
            self.small_hand_label = ctk.CTkLabel(self.small_hand_bg, image=self.small_hand_photo, text="", corner_radius=8)
            self.small_hand_label.pack(expand=True, padx=4, pady=4)
        except FileNotFoundError:
            self.small_hand_label = ctk.CTkLabel(self.small_hand_bg, text="รูปตัวอย่าง", font=("Sarabun", 16), width=img_size, height=img_size)
            self.small_hand_label.pack(expand=True, padx=4, pady=4)

        self.log_button = ctk.CTkButton(
            self.right_info_container, text="📊 สถิติ & รายงานผล", font=self.font_btn_text, fg_color=self.purple_bg,
            text_color=self.white_fg, command=lambda: self.show_history_page(), height=self.btn_h,
            hover_color="#5B2C6F", corner_radius=15
        )
        self.log_button.pack(side="bottom", fill="x", pady=(2, 0)) 

        self.history_page = ctk.CTkFrame(self, fg_color="#F4F7F6")
        
        self.history_header_frame = ctk.CTkFrame(self.history_page, fg_color=self.purple_bg, height=top_bar_h, corner_radius=0)
        self.history_header_frame.pack(side="top", fill="x")
        self.history_header_frame.pack_propagate(False)

        self.history_title = ctk.CTkLabel(self.history_header_frame, text="📊 สรุปผลการฝึกซ้อมของคุณ", font=self.font_large_title, text_color=self.white_fg)
        self.history_title.pack(pady=int(15 * self.scale_h))
        
        self.history_body_frame = ctk.CTkFrame(self.history_page, fg_color="transparent")
        self.history_body_frame.pack(fill="both", expand=True, padx=int(15 * self.scale_w), pady=4)
        
        self.kpi_frame = ctk.CTkFrame(self.history_body_frame, fg_color="transparent")
        self.kpi_frame.pack(side="top", fill="x", pady=(0, 2))

        self.history_content_frame = ctk.CTkFrame(self.history_body_frame, fg_color="transparent")
        self.history_content_frame.pack(side="bottom", fill="x", pady=(2, 2))

        self.history_log_wrapper = ctk.CTkFrame(self.history_content_frame, fg_color="#FFFFFF", corner_radius=12, border_width=1, border_color="#D5D8DC")
        self.history_log_wrapper.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.log_title = ctk.CTkLabel(self.history_log_wrapper, text="📝 บันทึกประวัติย้อนหลัง", font=("Sarabun", int(18 * self.u_scale), "bold"), text_color=self.black_fg)
        self.log_title.pack(anchor="w", padx=12, pady=(4, 2))

        self.history_textbox = ctk.CTkTextbox(
            self.history_log_wrapper, height=int(140 * self.scale_h), font=("Sarabun", int(15 * self.u_scale)),
            text_color=self.black_fg, fg_color="#F8F9F9", corner_radius=8, border_width=1, border_color="#E5E7E9"
        )
        self.history_textbox.pack(fill="both", expand=True, padx=12, pady=(2, 6))

        self.back_button = ctk.CTkButton(
            self.history_content_frame, text="⬅️ กลับสู่หน้าหลัก", font=("Sarabun", int(24 * self.u_scale), "bold"), fg_color=self.yellow_btn,
            text_color=self.black_fg, hover_color=self.hover_yellow_bt, command=self.show_main_page,
            height=int(85 * self.scale_h), width=int(280 * self.scale_w), corner_radius=18
        )
        self.back_button.pack(side="right", padx=6, pady=4) 

        self.chart_container_wrapper = ctk.CTkFrame(self.history_body_frame, fg_color="#FFFFFF", corner_radius=15, border_width=1, border_color="#D5D8DC")
        self.chart_container_wrapper.pack(side="top", fill="both", expand=True, pady=(0, 2), ipady=2, ipadx=2)

        self.chart_container = tk.Frame(self.chart_container_wrapper, bg="#FFFFFF")
        self.chart_container.pack(fill="both", expand=True, padx=4, pady=4)
        
        self.pose_sounds = {1: ["001.mp3"], 2: ["002.mp3"], 3: ["003.mp3"], 4: ["004.mp3"], 5: ["005.mp3"]}
        self.current_chart = None

        self.running = False
        self.countdown_active = False
        self.countdown_job = None
        self.countdown_total = 0
        self.countdown_end_time = 0

        self.set_popup = ctk.CTkFrame(self, fg_color="#D5F5E3", corner_radius=15, border_width=2, border_color="#27AE60")
        self.set_popup_label = ctk.CTkLabel(
            self.set_popup, text="ขอแสดงความยินดีคุณทำครบ 1 เซตแล้ว\nต่อเซตที่ 2 เลย! 🎉", 
            font=("Sarabun", int(24 * self.u_scale), "bold"), text_color="#1E8449", justify="center"
        )
        self.set_popup_label.pack(padx=25, pady=20)

        self.goal_popup = ctk.CTkFrame(self, fg_color="#FCF3CF", corner_radius=15, border_width=2, border_color="#F1C40F")
        self.goal_popup_label = ctk.CTkLabel(
            self.goal_popup, text="สุดยอดมาก! 🎉\nคุณทำครบเป้าหมายที่ตั้งไว้แล้ว!", 
            font=("Sarabun", int(26 * self.u_scale), "bold"), text_color="#B7950B", justify="center"
        )
        self.goal_popup_label.pack(padx=25, pady=20)

        self.config_popup = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=15, border_width=2, border_color=self.purple_bg)
        
        self.config_popup_header = ctk.CTkFrame(self.config_popup, fg_color="transparent")
        self.config_popup_header.pack(fill="x", padx=15, pady=(12, 6))
        
        self.config_popup_title = ctk.CTkLabel(self.config_popup_header, text="เลือกการตั้งค่า", font=("Sarabun", 22, "bold"), text_color=self.purple_bg)
        self.config_popup_title.pack(side="left", padx=5)
        
        self.config_close_btn = ctk.CTkButton(
            self.config_popup_header, text="✕", font=("Arial", 20, "bold"), fg_color="#C0392B", text_color="#FFFFFF",
            hover_color="#992D22", width=40, height=40, corner_radius=20, command=self.close_config_popup
        )
        self.config_close_btn.pack(side="right", padx=5)

        self.config_options_container = ctk.CTkFrame(self.config_popup, fg_color="transparent")
        self.config_options_container.pack(padx=25, pady=(0, 20), fill="x", anchor="center")

        self.update_idletasks() 
        self.attributes("-fullscreen", True)

        self.check_sensor_loop()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._video_job = self.after(0, self._start_video)

    def show_config_popup(self):
        self.config_popup.place(relx=0.5, rely=0.5, anchor="center")
        self.config_popup.tkraise()

    def close_config_popup(self):
        self.config_popup.place_forget()

    def open_hand_popup(self):
        for widget in self.config_options_container.winfo_children():
            widget.destroy()
        self.config_popup_title.configure(text="🖐️ เลือกมือที่ต้องการฝึก")
        
        for h_val in ["ซ้าย", "ทั้งสอง", "ขวา"]:
            b = ctk.CTkButton(
                self.config_options_container, text=h_val, font=("Sarabun", 20, "bold"), height=60, width=280,
                fg_color=self.purple_bg if self.target_hand_var.get() == h_val else "#E5E7E9",
                text_color=self.white_fg if self.target_hand_var.get() == h_val else self.black_fg,
                hover_color="#5B2C6F", corner_radius=12, anchor="center",
                command=lambda val=h_val: self.select_hand(val)
            )
            b.pack(pady=10, anchor="center")
        self.show_config_popup()

    def open_diff_popup(self):
        for widget in self.config_options_container.winfo_children():
            widget.destroy()
        self.config_popup_title.configure(text="⏱️ เลือกระดับความยาก")
        
        for d_val in ["ง่าย (3s)", "กลาง (5s)", "ยาก (7s)"]:
            b = ctk.CTkButton(
                self.config_options_container, text=d_val, font=("Sarabun", 20, "bold"), height=60, width=280,
                fg_color=self.purple_bg if self.difficulty_var.get() == d_val else "#E5E7E9",
                text_color=self.white_fg if self.difficulty_var.get() == d_val else self.black_fg,
                hover_color="#5B2C6F", corner_radius=12, anchor="center",
                command=lambda val=d_val: self.select_difficulty(val)
            )
            b.pack(pady=10, anchor="center")
        self.show_config_popup()

    def open_target_popup(self):
        for widget in self.config_options_container.winfo_children():
            widget.destroy()
        self.config_popup_title.configure(text="🎯 เลือกเป้าหมายเซ็ตต่อวัน")
        
        for t_val in [3, 5]:
            b = ctk.CTkButton(
                self.config_options_container, text=f"{t_val} เซ็ต", font=("Sarabun", 20, "bold"), height=60, width=280,
                fg_color=self.purple_bg if self.target_sets_var.get() == t_val else "#E5E7E9",
                text_color=self.white_fg if self.target_sets_var.get() == t_val else self.black_fg,
                hover_color="#5B2C6F", corner_radius=12, anchor="center",
                command=lambda val=t_val: self.select_target(val)
            )
            b.pack(pady=10, anchor="center")
        self.show_config_popup()

    def select_hand(self, val):
        self.target_hand_var.set(val)
        self.btn_select_hand.configure(text=f"🖐️ มือ: {val}  ▼ [แตะเพื่อเปลี่ยน]")
        self.close_config_popup()

    def select_difficulty(self, val):
        self.difficulty_var.set(val)
        self.btn_select_diff.configure(text=f"⏱️ ความยาก: {val}  ▼ [แตะเพื่อเปลี่ยน]")
        self.change_difficulty(val)
        self.close_config_popup()

    def select_target(self, val):
        self.target_sets_var.set(val)
        self.btn_select_target.configure(text=f"🎯 เป้าหมาย: {val} เซ็ต  ▼ [แตะเพื่อเปลี่ยน]")
        self.close_config_popup()
        if self.current_chart is not None:
            self.draw_progress_chart()

    def change_chart_timeframe(self, val):
        self.chart_timeframe = val
        if hasattr(self, "ax_cached") and self.ax_cached:
            self.ax_cached.clear()
        self.draw_progress_chart()

    def trigger_set_popup(self, current_set):
        next_set = current_set + 1
        msg = f"ขอแสดงความยินดีคุณทำครบ {current_set} เซตแล้ว\nต่อเซตที่ {next_set} เลย! 🎉"
        self.set_popup_label.configure(text=msg)
        self.set_popup.place(relx=0.5, rely=0.5, anchor="center")
        self.set_popup.tkraise()
        self.after(3000, self.set_popup.place_forget)

    def trigger_goal_popup(self, target):
        msg = f"สุดยอดมาก! 🎉\nคุณทำครบเป้าหมาย {target} เซ็ตแล้ว!"
        self.goal_popup_label.configure(text=msg)
        self.goal_popup.place(relx=0.5, rely=0.5, anchor="center")
        self.goal_popup.tkraise()
        self.after(3500, self.goal_popup.place_forget)

    def change_difficulty(self, choice):
        if "3s" in choice: self.time_max = 3
        elif "7s" in choice: self.time_max = 7
        else: self.time_max = 5
        if not self.running:
            self.timer_reset()

    def save_pain_log(self):
        val = self.pain_var.get()
        self.write_log(f"PainLevel: {val}")
        self.pain_btn.configure(text="บันทึก! ✅", fg_color=self.green_btn)
        self.after(2000, lambda: self.pain_btn.configure(text="บันทึก", fg_color="#E74C3C"))

    def get_history_from_file(self):
        current_time = time.time()
        if self._cached_history_data is not None and (current_time - self._last_history_read_time) < 2.0:
            return self._cached_history_data

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FILE_PATH = os.path.join(BASE_DIR, "Anti-Finger.txt")
        DAILY_TARGET_REPS = self.target_sets_var.get() * 3
        
        poses_per_rep = 5
        reps_per_set = 3 
        daily_poses = defaultdict(int)
        daily_pain = defaultdict(list)

        if not os.path.exists(FILE_PATH):
            return []

        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("["):
                        continue
                    try:
                        date_str = line.split("]")[0][1:]
                        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        
                        if "PainLevel:" in line:
                            pain = int(line.split("PainLevel:")[1].strip())
                            daily_pain[date.date()].append(pain)
                        elif "ความปวดก่อน" in line:
                            try:
                                pain_str = line.split("ความปวดก่อน")[1].split("/")[0].strip()
                                if pain_str.isdigit():
                                    daily_pain[date.date()].append(int(pain_str))
                            except Exception:
                                pass

                        if "สำเร็จ" in line:
                            daily_poses[date.date()] += 1
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

        if not daily_poses and not daily_pain:
            return []

        history = []
        all_dates = set(daily_poses.keys()).union(set(daily_pain.keys()))
        if not all_dates:
            return []
            
        first_day = min(all_dates)
        last_day = max(all_dates)
        day = first_day

        while day <= last_day:
            poses = daily_poses.get(day, 0)
            reps = poses / float(poses_per_rep) if poses_per_rep > 0 else 0
            sets_done = int(reps // reps_per_set)
            progress = min((reps / float(DAILY_TARGET_REPS)) * 100.0, 100.0) if DAILY_TARGET_REPS > 0 else 0.0
            
            avg_pain = None
            if daily_pain.get(day):
                avg_pain = sum(daily_pain[day]) / len(daily_pain[day])

            history.append({
                'date': datetime.combine(day, datetime.min.time()),
                'poses': poses,
                'reps': int(reps),
                'sets_done': sets_done,
                'progress': progress,
                'pain': avg_pain
            })
            day += timedelta(days=1)

        history.sort(key=lambda x: x['date'])
        self._cached_history_data = history
        self._last_history_read_time = current_time
        return history
        
    def update_kpi_dashboard(self, history):
        for widget in self.kpi_frame.winfo_children():
            widget.destroy()

        total_days = len(history)
        goal_met_days = sum(1 for h in history if h['progress'] >= 100)
        goal_met_pct = (goal_met_days / total_days * 100) if total_days > 0 else 0
        total_sets = sum(h['sets_done'] for h in history)
        pains = [h['pain'] for h in history if h['pain'] is not None]
        avg_pain = sum(pains) / len(pains) if pains else 0

        title_font = ("Sarabun", int(14 * self.u_scale), "bold")
        val_font = ("Sarabun", int(20 * self.u_scale), "bold")

        kpis = [
            ("🔥 ต่อเนื่อง (Streak)", f"{self.streak_days} วัน", "#E67E22"),
            ("🎯 วันที่ทำครบเป้า", f"{goal_met_pct:.0f}%", "#27AE60"),
            ("📚 เซ็ตสะสมรวม", f"{total_sets} เซ็ต", "#2980B9"),
            ("📉 ความปวดเฉลี่ย", f"{avg_pain:.1f}/10", "#8E44AD")
        ]

        for title, val, color in kpis:
            card = ctk.CTkFrame(self.kpi_frame, fg_color="#FFFFFF", corner_radius=10, border_width=1, border_color="#D5D8DC")
            card.pack(side="left", fill="both", expand=True, padx=3)
            ctk.CTkLabel(card, text=title, font=title_font, text_color="#7F8C8D").pack(pady=(4, 0))
            ctk.CTkLabel(card, text=val, font=val_font, text_color=color).pack(pady=(0, 4))

    def draw_progress_chart(self):
        for widget in self.chart_container.winfo_children():
            widget.destroy()
            
        history = self.get_history_from_file()
        if not history:
            label = tk.Label(self.chart_container, text="ไม่มีข้อมูลการฝึกซ้อม ลองไปฝึกสักเซ็ตสิครับ! 💪", bg="#FFFFFF", font=("Sarabun", 14, "bold"), fg="#7F8C8D")
            label.pack(fill="both", expand=True)
            return

        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            font_path = os.path.join(BASE_DIR, "Sarabun.ttf")
            if os.path.exists(font_path):
                thai_font_prop = fm.FontProperties(fname=font_path, size=11)
            else:
                thai_font_prop = fm.FontProperties(family='Tahoma', size=11)

            main_frame = ctk.CTkFrame(self.chart_container, fg_color="transparent")
            main_frame.pack(fill="both", expand=True)

            control_frame = ctk.CTkFrame(main_frame, fg_color="#F8F9F9", width=int(250 * self.scale_w), corner_radius=12, border_width=1, border_color="#D5D8DC")
            control_frame.pack(side="right", fill="y", padx=(8, 0))
            control_frame.pack_propagate(False) 

            chart_frame = tk.Frame(main_frame, bg="white")
            chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

            tf_frame = ctk.CTkFrame(chart_frame, fg_color="#F8F9F9", corner_radius=8, height=32)
            tf_frame.pack(side="top", fill="x", pady=(0, 4))

            ctk.CTkLabel(tf_frame, text="⏳ ระยะเวลากราฟ:", font=("Sarabun", 13, "bold"), text_color=self.black_fg).pack(side="left", padx=8)

            for tf_val, tf_text in [(7, "7"), (14, "14"), (30, "30"), (0, "All")]:
                btn_fg = self.purple_bg if self.chart_timeframe == tf_val else "#BDC3C7"
                btn = ctk.CTkButton(
                    tf_frame, text=tf_text, font=("Sarabun", 12, "bold"), height=24, width=42,
                    fg_color=btn_fg, text_color="white", corner_radius=5,
                    command=lambda v=tf_val: self.change_chart_timeframe(v)
                )
                btn.pack(side="left", padx=3, pady=3)

            ctk.CTkLabel(control_frame, text="📅 เลือกดูวันที่", font=("Sarabun", 15, "bold"), text_color=self.purple_bg).pack(anchor='w', pady=(6, 4), padx=12)
            
            date_var = tk.StringVar()
            date_combo = ttk.Combobox(control_frame, textvariable=date_var, width=14, state='readonly', font=("Sarabun", 13))
            date_combo.pack(anchor='w', padx=12, pady=(0, 8))

            ctk.CTkLabel(control_frame, text="📌 สัญลักษณ์", font=("Sarabun", 14, "bold"), text_color=self.black_fg).pack(anchor='w', pady=(4, 2), padx=12)
            ctk.CTkLabel(control_frame, text="🔴 แดง: < 50%", font=("Sarabun", 12, "bold"), text_color="#E74C3C").pack(anchor='w', padx=12)
            ctk.CTkLabel(control_frame, text="🟢 เขียว: >= 50%", font=("Sarabun", 12, "bold"), text_color="#2ECC71").pack(anchor='w', padx=12)
            ctk.CTkLabel(control_frame, text="🟣 ม่วง: ระดับความปวด", font=("Sarabun", 12, "bold"), text_color="#9B59B6").pack(anchor='w', padx=12)

            fb_container = ctk.CTkFrame(control_frame, fg_color="#FEF9E7", corner_radius=10, border_width=1, border_color="#F1C40F")
            fb_container.pack(fill="x", padx=8, pady=6, ipady=6)
            
            feedback_label = ctk.CTkLabel(fb_container, text="คลิกที่จุดบนกราฟ\nเพื่อดูรายละเอียด", font=("Sarabun", 13, "bold"), text_color="#5D6D7E", justify="left")
            feedback_label.pack(padx=6, pady=4)

            history_dict = {h['date'].date(): h for h in history}
            today = datetime.now().date()
            
            display_history = []
            if self.chart_timeframe > 0:
                start_date = today - timedelta(days=self.chart_timeframe - 1)
                for i in range(self.chart_timeframe):
                    d = start_date + timedelta(days=i)
                    if d in history_dict:
                        display_history.append(history_dict[d])
                    else:
                        display_history.append({
                            'date': datetime.combine(d, datetime.min.time()),
                            'poses': 0, 'reps': 0, 'sets_done': 0, 'progress': 0.0, 'pain': None
                        })
            else:
                start_date = min(history[0]['date'].date(), today) if history else today
                days_diff = (today - start_date).days + 1
                for i in range(max(1, days_diff)):
                    d = start_date + timedelta(days=i)
                    if d in history_dict:
                        display_history.append(history_dict[d])
                    else:
                        display_history.append({
                            'date': datetime.combine(d, datetime.min.time()),
                            'poses': 0, 'reps': 0, 'sets_done': 0, 'progress': 0.0, 'pain': None
                        })

            fig, ax = plt.subplots(figsize=(5, 1.4), dpi=90) 
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            self.ax_cached = ax 

            dates = [h['date'] for h in display_history]
            progresses = [h['progress'] for h in display_history]
            pains = [h['pain'] for h in display_history]

            colors = ['#2ECC71' if p >= 50 else '#E74C3C' for p in progresses]
            ax.bar(dates, progresses, width=0.6, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

            ax.set_xticks(dates)
            ax.set_xticklabels([d.strftime('%d %b') for d in dates], rotation=15)
            
            if len(dates) > 10:
                step = math.ceil(len(dates) / 10)
                for i, label in enumerate(ax.xaxis.get_ticklabels()):
                    if i % step != 0 and i != len(dates) - 1:
                        label.set_visible(False)

            if any(p is not None for p in pains):
                ax2 = ax.twinx()
                p_dates = [d for d, p in zip(dates, pains) if p is not None]
                p_vals = [p for p in pains if p is not None]
                ax2.plot(p_dates, p_vals, color='#9B59B6', marker='s', linestyle='-.', linewidth=1.5, markersize=4)
                ax2.set_ylim(-1, 12) 
                ax2.spines['right'].set_color('#9B59B6')
                ax2.tick_params(axis='y', colors='#9B59B6')

            ax.set_ylabel("ความสำเร็จ (%)", fontproperties=thai_font_prop, color='#7F8C8D', labelpad=4)
            ax.set_ylim(0, 120)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            date_list = [h['date'].strftime('%d-%b-%Y') for h in history]
            date_combo['values'] = date_list
            if date_list:
                date_combo.set(date_list[-1])

            points = []
            for i, p in enumerate(progresses):
                points.append((dates[i], p, display_history[i]['sets_done'], pains[i], i))

            def update_feedback(event=None):
                selected_date_str = date_var.get()
                if not selected_date_str: return
                try:
                    selected_date = datetime.strptime(selected_date_str, '%d-%b-%Y').date()
                    idx = next((i for i, h in enumerate(display_history) if h['date'].date() == selected_date), None)
                    if idx is not None:
                        prog = display_history[idx]['progress']
                        sets = display_history[idx]['sets_done']
                        pain = display_history[idx]['pain']
                        pain_txt = f"{pain:.1f}/10" if pain is not None else "-"
                        feedback_label.configure(text=f"📌 {selected_date_str}\nสำเร็จ: {prog:.0f}% | เซ็ต: {sets}\nความปวด: {pain_txt}")
                except Exception: pass

            def on_click(event):
                if event.inaxes is None: return
                for date, prog, sets, pain, idx in points:
                    xdata = mdates.date2num(date)
                    if abs(event.xdata - xdata) < 0.5:
                        pain_txt = f"{pain:.1f}/10" if pain is not None else "-"
                        feedback_label.configure(text=f"📌 {date.strftime('%d-%b-%Y')}\nสำเร็จ: {prog:.0f}% | เซ็ต: {sets}\nความปวด: {pain_txt}")
                        date_var.set(date.strftime('%d-%b-%Y'))
                        break

            date_combo.bind("<<ComboboxSelected>>", update_feedback)
            canvas.mpl_connect("button_press_event", on_click)
            update_feedback()
            self.current_chart = (fig, canvas)
        except Exception as e:
            print(f"Error drawing chart: {e}")

    def play_sounds_sequential(self, filename):
        def _play(f=filename):
            try:
                import os
                if not f.endswith(".mp3"):
                    f += ".mp3"
                
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                sound_path = os.path.join(BASE_DIR, "Voices", f)
                
                if os.path.exists(sound_path):
                    os.system(f"mpg123 -q '{sound_path}' &> /dev/null || aplay -q '{sound_path}' &> /dev/null")
                else:
                    print(f"[Sound Error] ไม่พบไฟล์เสียงที่: {sound_path}")
            except Exception as e:
                print(f"[Sound Error] {e}")
        threading.Thread(target=_play, daemon=True).start()

    def load_history(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FILE_PATH = os.path.join(BASE_DIR, "Anti-Finger.txt")
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = ["ยังไม่มีประวัติการฝึกซ้อม\n"]
        
        if len(lines) > 2000:
            lines = lines[-2000:]

        self.history_textbox.configure(state="normal")
        self.history_textbox.delete("1.0", "end")
        self.history_textbox.insert("end", "".join(lines))
        self.history_textbox.see("end")
        self.history_textbox.configure(state="disabled")

    def show_main_page(self):
        self.history_page.pack_forget()
        self.main_content_frame.pack(side="top", fill="both", expand=True, padx=6, pady=2)
        self.play_sounds_sequential("010.mp3")

    def show_history_page(self):
        self.main_content_frame.pack_forget()
        self.play_sounds_sequential("009.mp3")
        self.history_page.pack(side="top", fill="both", expand=True)
        self.update_kpi_dashboard(self.get_history_from_file())
        self.draw_progress_chart()
        self.load_history()

    def write_log(self, message):
        now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_message = f"{now} เซ็ตที่ {self.set} รอบที่ {self.round} : {message}"
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            FILE_PATH = os.path.join(BASE_DIR, "Anti-Finger.txt")
            with open(FILE_PATH, "a", encoding="utf-8") as f:
                f.write(log_message + "\n")
        except Exception:
            pass

    def _start_video(self):
        if self._closing:
            return
        cv2.setNumThreads(1)
        self.cam_thread = threading.Thread(
            target=self._camera_feed_loop, name="camera", daemon=True)
        self._workers.append(self.cam_thread)
        self.cam_thread.start()
        if not self._preview_only:
            self.ai_thread = threading.Thread(
                target=self._ai_processing_loop, name="inference", daemon=True)
            self._workers.append(self.ai_thread)
            self.ai_thread.start()
        self._poll_camera()

    def _update_camera_label(self, pil_image, match, env_txt, env_col):
        # All calls to Tk/CustomTkinter stay on the main thread.
        if pil_image is not None:
            self.camera_photo.configure(light_image=pil_image)
        self.match_badge.configure(
            text="มือ: YES" if match else "มือ: NO",
            text_color="#2ECC71" if match else "#E74C3C")
        self.env_badge.configure(text=env_txt, text_color=env_col)

    def _camera_feed_loop(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_path, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {self.camera_path}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this.
            logging.info("Camera %s: backend=%s, negotiated=%sx%s",
                         self.camera_path, cap.getBackendName(),
                         cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                         cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            failures = 0
            while not self._stop_video.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    with self.frame_lock:
                        self.latest_frame = None
                        self.camera_error = f"อ่านกล้องไม่ได้ ({failures}/10)"
                    if failures >= 10:
                        raise RuntimeError("Camera read failed 10 times; check device/USB")
                    self._stop_video.wait(0.1)
                    continue
                failures = 0
                frame = cv2.flip(frame, 1)
                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_seq += 1
                    self.frame_time = time.monotonic()
                    self.camera_error = None
        except Exception as exc:
            logging.exception("Camera worker failed")
            with self.frame_lock:
                self.latest_frame = None
                self.camera_error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            logging.info("Camera worker stopped; capture released")

    def _poll_camera(self):
        if self._closing:
            return
        try:
            now = time.monotonic()
            with self.frame_lock:
                self.ai_pose = self.current_pose
                self.ai_target_hand = self.target_hand_var.get()
                frame = self.latest_frame
                seq, captured = self.frame_seq, self.frame_time
                camera_error, ai_error = self.camera_error, self.ai_error
                timely = result_is_timely(now,self.result_time,self.result_completed)
                fresh_result = self.result_pose == self.current_pose and timely
                healthy = (fresh_result and not ai_error and not camera_error and not self._preview_only
                           and frame is not None and 0 <= now-captured < .8
                           and self.result_target_hand == self.ai_target_hand)
                context = (self.current_pose,self.ai_target_hand,self.running,self.countdown_active,self.result_track_id)
                match,self._match_state,self._match_confirmed = self._match_gate.update(
                    now,self.result_seq,self.result_completed,self.latest_match,self.latest_present,healthy,context)
                self._match_updated = now
                overlay_fresh = (not ai_error and not camera_error and not self._preview_only
                                 and result_is_timely(now,self.overlay_time,self.overlay_completed)
                                 and now-self.overlay_completed <= 1.0 and now-captured < .8)
                points = self.latest_overlay_points if overlay_fresh else None
                presence_healthy = (not ai_error and not camera_error and not self._preview_only
                                    and frame is not None and 0 <= now-captured < .8 and timely)
                hand_detected = self._presence_latch.update(
                    now,self.result_seq,self.result_completed,self.latest_present,presence_healthy)
                currently_seen = self.latest_present and timely and 0 <= now-self.result_completed <= 1.0
                overlay_key = (seq, self.overlay_time if points is not None else None)
                env_txt, env_col = self.latest_env_text, self.latest_env_color
            pil_image = None
            if frame is None or now - captured > 1.0 or camera_error:
                match = False
                if frame is not None and self._displayed_seq != (seq, None):
                    pil_image = ImageOps.fit(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                        (self.camera_width, self.camera_height),
                        method=Image.Resampling.BILINEAR)
                    self._displayed_seq = (seq, None)
                elif frame is None and self._displayed_seq is not None:
                    pil_image = Image.new("RGB", (self.camera_width, self.camera_height), (220,220,220))
                    self._displayed_seq = None
                env_txt = "กล้อง: " + (camera_error or "ยังไม่มีเฟรมใหม่")
                env_col = "#E74C3C"
            else:
                if overlay_key != self._displayed_seq:
                    pil_image = ImageOps.fit(
                        Image.fromarray(cv2.cvtColor(draw_hand_overlay(frame, points), cv2.COLOR_BGR2RGB)),
                        (self.camera_width, self.camera_height),
                        method=Image.Resampling.BILINEAR)
                    self._displayed_seq = overlay_key
                if self._preview_only:
                    match, env_txt, env_col = False, "ทดสอบกล้อง: ปิด AI", "#F39C12"
                elif ai_error:
                    match, env_txt, env_col = False, "AI: " + ai_error, "#E74C3C"
                elif not fresh_result:
                    match, env_txt, env_col = False, "AI: รอผลใหม่", "#F39C12"
            if match and self._match_state == "grace":
                env_txt,env_col = "กำลังติดตามต่อ...", "#F39C12"
            self.set_hand_posit_state(match)
            if hand_detected and not currently_seen:
                env_txt,env_col = "ติดตามมือขาดช่วง · กำลังตรวจซ้ำ", "#F39C12"
            elif hand_detected:
                env_txt = "พบมือ · " + ("ยืนยันท่าแล้ว" if self._match_confirmed else
                                        "กำลังติดตามต่อ" if match else "ยังยืนยันท่าไม่ได้")
            self._update_camera_label(pil_image, hand_detected, env_txt, env_col)
        except Exception:
            self.set_hand_posit_state(False)
            logging.exception("Camera GUI update failed")
        finally:
            if not self._closing:
                self._video_job = self.after(50, self._poll_camera)

    def _ai_processing_loop(self):
        try:
            self._run_ai_inference()
        except Exception as exc:
            logging.exception("AI worker failed; camera preview remains independent")
            with self.frame_lock:
                self.latest_match = False
                self.result_time = 0.0
                self.ai_error = str(exc)

    def _run_ai_inference(self):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmark.tflite")
        ai_threads=max(1,min(2,int(os.environ.get("AI_THREADS","2"))))
        logging.info("TFLite CPU threads=%s",ai_threads)
        interpreter = Interpreter(model_path=model_path,num_threads=ai_threads)
        interpreter.allocate_tensors()
        tracker = StableLocalTracker(interpreter)
        last_seq = -1
        log_started = time.monotonic()
        inference_times = []
        found_samples = 0
        # One model call per sample; deferred retries read a new frame.
        while not self._stop_video.wait(.06):
            with self.frame_lock:
                if self.latest_frame is None or self.frame_seq == last_seq:
                    continue
                frame = self.latest_frame.copy()
                last_seq,captured = self.frame_seq,self.frame_time
                pose,target = self.ai_pose,self.ai_target_hand
            if time.monotonic()-captured > .8:
                continue
            started = time.monotonic()
            points,matched,track_id = tracker.process(frame,captured,pose)
            completed = time.monotonic()
            inference_times.append((completed-started)*1000)
            if completed-captured > 1.2:
                # A newly completed, excessively old result is not fresh evidence.
                points,matched = None,False
                tracker.last_reason = "result_too_old"
            found_samples += points is not None
            if time.monotonic()-log_started >= 3:
                logging.info("Tracking found=%s/%s mean=%.0fms max=%.0fms age=%.0fms score=%.3f reason=%s",
                             found_samples,len(inference_times),sum(inference_times)/len(inference_times),
                             max(inference_times),(time.monotonic()-captured)*1000,
                             tracker.last_score,tracker.last_reason)
                inference_times.clear()
                found_samples=0
                log_started=time.monotonic()
            with self.frame_lock:
                self.latest_match = matched
                self.latest_present = points is not None
                if points is not None:
                    self.latest_overlay_points = points[:,:2].copy()
                    self.overlay_time = captured
                    self.overlay_completed = completed
                self.latest_env_text = "สภาพแวดล้อม: ปกติ" if points is not None else "กำลังค้นหามือ..."
                self.latest_env_color = "#2ECC71" if points is not None else "#F39C12"
                self.result_time,self.result_pose = captured,pose
                self.result_completed = completed
                self.result_seq,self.result_track_id = last_seq,track_id
                self.result_target_hand = target

    def set_hand_posit_state(self, match):
        self.is_pose_currently_matched = match

    def timer_reset(self):
        self.time_current = self.time_max
        self.update_timer_display()
        self.reset_pic()
        self.breathing_label.configure(text="เตรียมพร้อม...", text_color="#BDC3C7")
        try:
            self.update_EX_pose()
        except Exception:
            pass

    def reset_pic(self):
        self.timer_canvas.delete("progress")
        l = self.timer_pad
        r = self.timer_canvas_size - self.timer_pad
        self.timer_canvas.create_oval(l, l, r, r, outline=self.green_btn, width=10, tags="progress")

    def update_timer_display(self):
        try:
            progress = max(0.0, min(1.0, (self.time_max - self.time_current) / float(self.time_max)))
            extent = 360 * progress
            self.timer_canvas.delete("progress")
            l = self.timer_pad
            r = self.timer_canvas_size - self.timer_pad
            self.timer_canvas.create_arc(l, l, r, r, start=-90, extent=-extent, style="arc", width=10, outline=self.green_btn, tags="progress")
            
            display_time = int(math.ceil(self.time_current))
            self.timer_canvas.itemconfig(self.timer_text, text=str(display_time))
        except Exception:
            pass

    def reset_feedback(self):
        self.pose_feedback_label.configure(text="💬 ลุยท่าต่อไปได้เลย!", text_color="#F39C12")

    def update_text(self):
        self.Label_pose_action_text.configure(text=f"{self.pose_name[self.current_pose]}")
        self.Label_pose_thai_text.configure(text=f"ท่าที่ {self.current_pose}")

    def update_round(self):
        self.Label_set_times_number.configure(text=f"{self.round}")
        self.Label_set_number.configure(text=f"{self.set}")
        self.round_progress.set(self.round / 3.0)

        target = self.target_sets_var.get()
        if self.set == 0 and self.round == 0:
            reward_msg = "🌟 พร้อมแล้ว เริ่มลุยเลย!"
            color = "#F39C12"
        elif self.round == 0 and self.set > 0:
            reward_msg = f"🏆 สุดยอด! จบเซ็ตที่ {self.set} แล้ว!"
            color = "#27AE60"
        else:
            reward_msg = f"🔥 ลุยต่อ! (เซ็ตที่ {self.set + 1}/{target} | รอบที่ {self.round}/3)"
            color = "#D35400"
        self.reward_label.configure(text=reward_msg, text_color=color)

    def update_EX_pose(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        try:
            img_size = int(220 * self.u_scale)
            pose_path = os.path.join(BASE_DIR, "pictures", "EX_POSE", f"pose{self.current_pose}.png")
            small_hand_image_pil = Image.open(pose_path)
            self.small_hand_photo = ctk.CTkImage(light_image=small_hand_image_pil, size=(img_size, img_size))
            self.small_hand_label.configure(image=self.small_hand_photo)
        except:
            pass

    def toggle_start_pause(self):
        if self.start_stop_button.cget("text") == "▶ เริ่มต้น":
            self.start_stop_button.configure(text="⏸ หยุด", fg_color=self.yellow_btn, hover_color=self.hover_yellow_bt)
            self.start_pose_countdown(2)
            self.play_sounds_sequential("006.mp3")
        else:
            self.start_stop_button.configure(text="▶ เริ่มต้น", fg_color=self.green_btn, hover_color=self.hover_green_bt)
            self.running = False
            self._cancel_countdown()
            self.play_sounds_sequential("007.mp3")

    def start_pose_countdown(self, seconds: int):
        self._cancel_countdown()
        self.countdown_active = True
        self.countdown_total = max(1, seconds)
        self.countdown_end_time = time.time() + self.countdown_total
        self.countdown_job = self.after(0, self._animate_countdown)

    def _animate_countdown(self):
        if not self.countdown_active:
            return
        now = time.time()
        remaining = self.countdown_end_time - now
        if remaining <= 0:
            self.countdown_active = False
            self.countdown_job = None
            self.running = True
            self.is_pose_currently_matched = False
            try:
                display_time = int(math.ceil(self.time_current))
                self.timer_canvas.itemconfig(self.timer_text, text=str(display_time))
                self.timer_canvas.delete("progress")
                l = self.timer_pad
                r = self.timer_canvas_size - self.timer_pad
                self.timer_canvas.create_oval(l, l, r, r, outline=self.green_btn, width=10, tags="progress")
            except Exception:
                pass
            try:
                self.play_sounds_sequential(self.pose_sounds[self.current_pose][0])
            except Exception:
                pass
            return

        frac = max(0.0, min(1.0, remaining / float(self.countdown_total)))
        extent = 360 * frac
        try:
            self.timer_canvas.delete("progress")
            l = self.timer_pad
            r = self.timer_canvas_size - self.timer_pad
            self.timer_canvas.create_arc(l, l, r, r, start=-90, extent=-extent, style="arc", width=10, outline="#FFA500", tags="progress")
            secs = int(math.ceil(remaining))
            self.timer_canvas.itemconfig(self.timer_text, text=str(secs))
        except Exception:
            pass

        try:
            self.countdown_job = self.after(50, self._animate_countdown)
        except Exception:
            self.countdown_active = False
            self.countdown_job = None

    def _cancel_countdown(self):
        if self.countdown_active:
            self.countdown_active = False
            if self.countdown_job:
                try:
                    self.after_cancel(self.countdown_job)
                except Exception:
                    pass
                self.countdown_job = None
            try:
                display_time = int(math.ceil(self.time_current))
                self.timer_canvas.itemconfig(self.timer_text, text=str(display_time))
                self.timer_canvas.delete("progress")
                l = self.timer_pad
                r = self.timer_canvas_size - self.timer_pad
                self.timer_canvas.create_oval(l, l, r, r, outline=self.green_btn, width=10, tags="progress")
            except Exception:
                pass

    def reset_action(self):
        self.running = False
        try:
            self._cancel_countdown()
        except Exception:
            pass
        self.start_stop_button.configure(text="▶ เริ่มต้น", fg_color=self.green_btn, hover_color=self.hover_green_bt)
        self.round = 0
        self.set = 0
        self.current_pose = 1
        self.timer_reset()
        self.update_text()
        self.update_round()
        self.reset_feedback()
        try:
            self.play_sounds_sequential("008.mp3")
        except Exception:
            pass

    def on_close(self):
        if self._closing:
            return
        self._closing = True
        self.running = False
        self.set_hand_posit_state(False)
        self._stop_video.set()
        for attr in ("_video_job", "_sensor_job", "countdown_job", "notif_hide_job"):
            job = getattr(self, attr, None)
            if job is not None:
                self.after_cancel(job)
                setattr(self, attr, None)
        self._close_started = time.monotonic()
        self._close_warned = False
        self._wait_for_video_stop()

    def _wait_for_video_stop(self):
        # Keep Tk alive while workers finish. Never release during cap.read().
        if any(worker.is_alive() for worker in self._workers):
            if not self._close_warned and time.monotonic() - self._close_started > 3:
                logging.error("Waiting for camera/AI native call to finish; "
                              "if stuck, identify and terminate this application's PID")
                self.env_badge.configure(text="กำลังรอปิดกล้อง/AI...", text_color="#E74C3C")
                self._close_warned = True
            self.after(50, self._wait_for_video_stop)
            return
        for worker in self._workers:
            worker.join(timeout=0)
        self.destroy()

    def check_sensor_loop(self):
        if self._closing:
            return
        now = time.monotonic()
        dt = max(0.0,now-self._last_sensor_tick)
        self._last_sensor_tick = now
        if dt > .5:
            self._match_gate.reset()
            self.set_hand_posit_state(False)
            self._match_confirmed = False
        if self.running:
            with self.frame_lock:
                matched = (getattr(self, "is_pose_currently_matched", False)
                           and self.latest_frame is not None
                           and self.camera_error is None
                           and self.ai_error is None
                           and self.result_pose == self.current_pose
                           and self.result_target_hand == self.target_hand_var.get()
                           and result_is_timely(now,self.result_time,self.result_completed)
                           and now-self.frame_time < .8
                           and now-self._match_updated < .25
                           and dt <= .5)
            if matched:
                if self.time_current > 0:
                    self.time_current = max(0.0,self.time_current-min(dt,.25)) 
                    self.update_timer_display()
                    elapsed = self.time_max - self.time_current
                    if self.time_max == 3:
                        msg, color = ("สูดหายใจเข้า...", self.blue_btn) if elapsed <= 1 else ("ผ่อนลมหายใจ...", self.green_btn)
                    else: 
                        msg, color = ("สูดหายใจเข้า...", self.blue_btn) if elapsed <= 2 else ("ผ่อนลมหายใจ...", self.green_btn)
                    self.breathing_label.configure(text=msg, text_color=color)

                if self.time_current <= 0 and self._match_confirmed and now-self.result_completed <= .35:
                    self._on_pose_success()
            else:
                self.breathing_label.configure(text="⏸ พักเวลาไว้ · จัดมือใหม่เมื่อพร้อม", text_color="#E74C3C")

        try:
            self._sensor_job = self.after(100, self.check_sensor_loop)
        except Exception:
            pass

    def _on_pose_success(self):
        try:
            completed_pose_name = self.pose_name[self.current_pose]
            self.write_log(f"ท่า {completed_pose_name} สำเร็จ! 🌟")
            self.show_success_notification(completed_pose_name)
            
            encouragements = ["เก่งมาก! 👏", "ยอดเยี่ยม! 🌟", "เป๊ะมาก! ✨", "สุดยอด! 🔥"]
            self.pose_feedback_label.configure(text=f"🎉 สำเร็จ! {random.choice(encouragements)}", text_color="#27AE60")
            self.breathing_label.configure(text="ผ่อนคลาย...", text_color="#BDC3C7")
        except Exception:
            pass

        self.current_pose += 1
        if self.current_pose > 5:
            self.current_pose = 1
            self.round += 1
            
            if self.round >= 3:
                self.round = 0
                self.set += 1
                target = self.target_sets_var.get()
                if self.set >= target:
                    self.trigger_goal_popup(target)
                    self.after(3500, self.reset_action)
                else:
                    self.trigger_set_popup(self.set)

        try:
            self.update_round()
            self.timer_reset()
            self.update_EX_pose()
            self.update_text()
            self.after(2500, self.reset_feedback)
            
            sound_file = self.pose_sounds.get(self.current_pose, [None])[0]
            if sound_file:
                self.play_sounds_sequential(sound_file)
        except Exception:
            pass

    def show_success_notification(self, pose_name):
        message = f"คุณทำท่า{pose_name}สำเร็จแล้ว!"
        self.notif_label.configure(text=message)
        self.notif_frame.pack(side="right", padx=15, pady=12)
        if self.notif_hide_job:
            self.after_cancel(self.notif_hide_job)
        self.notif_hide_job = self.after(2500, self._hide_notification)

    def _hide_notification(self):
        self.notif_frame.pack_forget()
        self.notif_hide_job = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    logging.info("Build: stable-last-crop-v5")
    app = AntiTriggerFingersApp()
    app.mainloop()