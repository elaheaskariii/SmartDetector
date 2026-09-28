"""
Smart Detector - تشخیص چرخش سر با احراز هویت چهره
---------------------------------------------------
نظارت بر چرخش سر در بازه‌های ۲۰ ثانیه‌ای:
  • ۳ حرکت → چراغ زرد
  • ۵ حرکت → چراغ قرمز + بوق + قفل تا ریست
"""

from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
import winsound
from insightface.app import FaceAnalysis


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class Config:
    # --- فایل مرجع ---
    reference_image: str = r"C:\Users\pars\Downloads\Eitaa Desktop\me.jpg"

    # --- پارامترهای منطق ---
    time_window: float = 20.0 # طول پنجره (ثانیه)
    max_turns: int = 5 # آستانه‌ی قرمز + بوق
    yellow_turns: int = 3 # آستانه‌ی زرد
    similarity_threshold: float = 0.35
    other_person_min_sim: float = 0.15
    turn_ratio_threshold: float = 0.05 # آستانه‌ی چرخش سر

    # --- کارایی ---
    process_width: int = 480
    face_check_every: int = 8
    mp_check_every: int = 2

    # --- دوربین ---
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    # --- رنگ‌ها (BGR) ---
    color_ok: Tuple[int, int, int] = (0, 255, 0)
    color_warn: Tuple[int, int, int] = (0, 215, 255)
    color_alert: Tuple[int, int, int] = (0, 0, 255)
    color_text: Tuple[int, int, int] = (0, 255, 255)
    color_off: Tuple[int, int, int] = (80, 80, 80)
    color_white: Tuple[int, int, int] = (255, 255, 255)

    # --- فرکانس‌های بوق ---
    alarm_pattern: Tuple[Tuple[int, int], ...] = (
        (2000, 80), (2500, 80), (2000, 80), (2800, 120), (2200, 100),
    )


CFG = Config()

# ایندکس‌های MediaPipe (ثابت و خوانا)
NOSE_IDX = 1
LEFT_FACE_IDX = 127
RIGHT_FACE_IDX = 356


# ============================================================
# Alarm (Thread)
# ============================================================
class Alarm:
    """پخش بوق در نخ جداگانه با قابلیت توقف تمیز."""

    def __init__(self, pattern: Tuple[Tuple[int, int], ...]) -> None:
        self._pattern = pattern
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for freq, dur in self._pattern:
                if self._stop_event.is_set():
                    return
                try:
                    winsound.Beep(freq, dur)
                except Exception as exc: # noqa: BLE001
                    print(f"[Alarm] sound error: {exc}")
                    return


# ============================================================
# تشخیص جهت سر
# ============================================================
class HeadDirectionDetector:
    """محاسبه‌ی جهت سر (RIGHT / LEFT / CENTER) از لندمارک‌های صورت."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def detect(
        self, landmarks, frame_width: int
    ) -> Tuple[str, float]:
        nose_x = landmarks[NOSE_IDX].x * frame_width
        left_x = landmarks[LEFT_FACE_IDX].x * frame_width
        right_x = landmarks[RIGHT_FACE_IDX].x * frame_width

        face_center = (left_x + right_x) / 2.0
        face_width = abs(right_x - left_x)

        if face_width < 1e-6:
            return "CENTER", 0.0

        ratio = (nose_x - face_center) / face_width

        if ratio > self._threshold:
            return "RIGHT", ratio
        if ratio < -self._threshold:
            return "LEFT", ratio
        return "CENTER", ratio


# ============================================================
# کلاس اصلی دتکتور
# ============================================================
class SmartDetector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.head_dir = HeadDirectionDetector(cfg.turn_ratio_threshold)
        self.alarm = Alarm(cfg.alarm_pattern)

        # مدل‌ها
        print("Loading InsightFace...")
        self._face_app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self._face_app.prepare(ctx_id=0, det_size=(320, 320))
        print("InsightFace ready.")

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        # امبدینگ مرجع
        self._reference_embedding = self._load_reference_embedding()

        # دوربین
        self._cap = self._open_camera()

        # وضعیت
        self._reset_state()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    def _load_reference_embedding(self) -> np.ndarray:
        path = self.cfg.reference_image
        if not os.path.exists(path):
            print(f"Reference image not found: {path}")
            sys.exit(1)

        img = cv2.imread(path)
        faces = self._face_app.get(img)
        if not faces:
            print("No face in reference image!")
            sys.exit(1)

        print("Reference loaded.")
        return faces[0].normed_embedding

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not opened!")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reset_state(self) -> None:
        self.turn_count = 0
        self.last_turn_direction: Optional[str] = None
        self.window_start: Optional[float] = None

        self.frame_counter = 0
        self.identity_verified = False
        self.other_person_present = False
        self.similarity_val = 0.0
        self.current_direction = "NO FACE"
        self.ratio_val = 0.0
        self.alert_locked = False

        self._last_faces = []

    # --------------------------------------------------------
    # حلقه‌ی اصلی
    # --------------------------------------------------------
    def run(self) -> None:
        print("Camera started. Q=quit S=pause R=reset")
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    print("Frame read error")
                    break

                frame = self._preprocess(frame)
                self._analyze(frame)
                self._update_logic()
                self._render(frame)

                if not self._handle_keys():
                    break
        finally:
            self._cleanup()

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # آینه‌ای کردن (حس دوربین سلفی)
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        if w > self.cfg.process_width:
            scale = self.cfg.process_width / w
            frame = cv2.resize(frame, (self.cfg.process_width, int(h * scale)))
        return frame

    # --------------------------------------------------------
    # تحلیل فریم
    # --------------------------------------------------------
    def _analyze(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.frame_counter += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face mesh (با نمونه‌گیری)
        mp_result = None
        if self.frame_counter % self.cfg.mp_check_every == 0:
            mp_result = self._face_mesh.process(rgb)

        # InsightFace (با نمونه‌گیری)
        if self.frame_counter % self.cfg.face_check_every == 0:
            self._last_faces = self._face_app.get(frame)

        # احراز هویت
        my_face_bbox = self._verify_identity()

        # جهت سر
        self._detect_direction(mp_result, my_face_bbox, w, h)

    def _verify_identity(self):
        """چهره‌ی خودی رو پیدا می‌کنه. bbox برمی‌گردونه یا None."""
        self.identity_verified = False
        self.other_person_present = False
        self.similarity_val = 0.0
        my_face_bbox = None

        for face in self._last_faces:
            sim = float(np.dot(self._reference_embedding, face.normed_embedding))
            if sim >= self.cfg.similarity_threshold:
                self.identity_verified = True
                self.similarity_val = sim
                my_face_bbox = face.bbox
                break
            if sim > self.cfg.other_person_min_sim:
                self.other_person_present = True

        if not self.identity_verified and self._last_faces:
            self.other_person_present = True
            self.similarity_val = max(
                float(np.dot(self._reference_embedding, f.normed_embedding))
                for f in self._last_faces
            )
        return my_face_bbox

    def _detect_direction(self, mp_result, my_face_bbox, w: int, h: int) -> None:
        self.current_direction = "NO FACE"
        self.ratio_val = 0.0

        if not (mp_result and mp_result.multi_face_landmarks and self.identity_verified):
            return

        landmarks_list = mp_result.multi_face_landmarks
        my_landmarks = self._pick_my_landmarks(landmarks_list, my_face_bbox, w, h)

        if my_landmarks is not None:
            self.current_direction, self.ratio_val = self.head_dir.detect(
                my_landmarks, w
            )

    @staticmethod
    def _pick_my_landmarks(landmarks_list, my_bbox, w: int, h: int):
        """انتخاب لندمارک مربوط به چهره‌ی خودی از بین چند چهره."""
        if len(landmarks_list) == 1:
            return landmarks_list[0].landmark

        # مرکز تقریبی چهره‌ی خودی
        if my_bbox is not None:
            x1, y1, x2, y2 = my_bbox
            target_cx, target_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            target_cx, target_cy = w / 2.0, h / 2.0

        best_lms, best_dist = None, float("inf")
        for face_lms in landmarks_list:
            lm = face_lms.landmark
            cx = sum(p.x for p in lm) / len(lm) * w
            cy = sum(p.y for p in lm) / len(lm) * h
            dist = (cx - target_cx) ** 2 + (cy - target_cy) ** 2
            if dist < best_dist:
                best_dist, best_lms = dist, lm
        return best_lms

    # --------------------------------------------------------
    # منطق اصلی
    # --------------------------------------------------------
    def _update_logic(self) -> None:
        now = time.perf_counter()

        # شروع پنجره
        if (
            self.identity_verified
            and self.window_start is None
            and not self.alert_locked
        ):
            self._start_window(now)

        # شمارش چرخش‌ها
        if (
            self.window_start is not None
            and self.identity_verified
            and not self.alert_locked
        ):
            self._count_turn()

        # تایمر پنجره
        self._tick_window(now)

    def _start_window(self, now: float) -> None:
        self.window_start = now
        self.turn_count = 0
        self.last_turn_direction = None
        print(">>> Window started (20s).")

    def _count_turn(self) -> None:
        if self.current_direction in ("LEFT", "RIGHT"):
            # فقط وقتی جهت با آخرین جهت شمرده‌شده فرق داره، حرکت جدید شمرده می‌شه
            if self.current_direction != self.last_turn_direction:
                self.last_turn_direction = self.current_direction
                self.turn_count += 1
                print(
                    f"Turn: {self.current_direction} | "
                    f"count: {self.turn_count} | ratio={self.ratio_val:+.3f}"
                )

                if self.turn_count >= self.cfg.max_turns:
                    self.alert_locked = True
                    print(">>> ALERT! Reached 5 turns. Press R to reset. <<<")
                    self.alarm.start()

        elif self.current_direction == "CENTER":
            # برگشت به وسط → اجازه‌ی شمارش حرکت بعدی (حتی هم‌جهت)
            self.last_turn_direction = None

    def _tick_window(self, now: float) -> None:
        if self.window_start is None or self.alert_locked:
            return

        elapsed = now - self.window_start
        if elapsed >= self.cfg.time_window:
            print("--- Window finished ---")
            print("Turns in this window:", self.turn_count)
            print("Normal. Starting new 20s window.")
            self.turn_count = 0
            self.last_turn_direction = None
            self.window_start = now

    def _remaining_time(self, now: float) -> float:
        if self.window_start is None:
            return self.cfg.time_window
        if self.alert_locked:
            return 0.0
        return max(0.0, self.cfg.time_window - (now - self.window_start))

    # --------------------------------------------------------
    # رندر
    # --------------------------------------------------------
    def _render(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        now = time.perf_counter()

        self._draw_info_panel(frame, now)
        self._draw_alarm_overlay(frame, w)
        self._draw_status_light(frame, w, now)

        cv2.imshow("Smart Detector", frame)

    def _draw_info_panel(self, frame: np.ndarray, now: float) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (320, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        if self.identity_verified:
            id_text, id_color = "YOU", self.cfg.color_ok
        elif self.other_person_present:
            id_text, id_color = "OTHER", self.cfg.color_alert
        else:
            id_text, id_color = "NOBODY", self.cfg.color_alert

        remaining = self._remaining_time(now)
        rows = (
            (f"Identity: {id_text}", id_color, 0.65),
            (f"Similarity: {self.similarity_val:.3f}", self.cfg.color_text, 0.55),
            (f"Turns: {self.turn_count}/{self.cfg.max_turns}", self.cfg.color_text, 0.55),
            (f"Time: {remaining:.1f}s", self.cfg.color_text, 0.55),
        )
        for i, (text, color, scale) in enumerate(rows):
            cv2.putText(
                frame, text, (15, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2,
            )

        dir_color = (
            self.cfg.color_ok
            if self.current_direction == "CENTER"
            else (0, 165, 255)
        )
        cv2.putText(
            frame,
            f"Head: {self.current_direction} ({self.ratio_val:+.2f})",
            (15, 165),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 2,
        )

    def _draw_alarm_overlay(self, frame: np.ndarray, w: int) -> None:
        if not self.alert_locked:
            return

        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 255
        cv2.addWeighted(red_overlay, 0.35, frame, 0.65, 0, frame)

        cv2.putText(
            frame, "WARNING!", (w - 240, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.cfg.color_alert, 3,
        )
        cv2.putText(
            frame, "STOP MOVING", (w - 240, 145),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cfg.color_alert, 2,
        )

    def _draw_status_light(self, frame: np.ndarray, w: int, now: float) -> None:
        cx, cy, r = w - 45, 45, 25

        if self.alert_locked:
            # قرمز چشمک‌زن
            blink = int(now * 5) % 2
            color = self.cfg.color_alert if blink == 0 else (0, 0, 150)
            cv2.circle(frame, (cx, cy), r, color, -1)
            cv2.circle(frame, (cx, cy), r, self.cfg.color_white, 2)

        elif self.turn_count >= self.cfg.yellow_turns:
            # زرد
            cv2.circle(frame, (cx, cy), r, self.cfg.color_warn, -1)
            cv2.circle(frame, (cx, cy), r, self.cfg.color_white, 2)
            cv2.putText(
                frame, "CAUTION", (w - 200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cfg.color_warn, 2,
            )
        else:
            # خاموش
            cv2.circle(frame, (cx, cy), r, self.cfg.color_off, -1)

    # --------------------------------------------------------
    # کیبورد
    # --------------------------------------------------------
    def _handle_keys(self) -> bool:
        """True = ادامه بده، False = خروج."""
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("Quit.")
            return False

        if key in (ord("s"), ord("S")):
            return self._handle_pause()

        if key in (ord("r"), ord("R")):
            self._handle_reset()

        return True

    def _handle_pause(self) -> bool:
        print("PAUSED. Press S to resume.")
        while True:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 in (ord("s"), ord("S")):
                print("RESUMED.")
                if self.identity_verified and not self.alert_locked:
                    self._start_window(time.perf_counter())
                return True
            if k2 in (ord("q"), ord("Q"), 27):
                print("Stopped.")
                return False

    def _handle_reset(self) -> None:
        self.alert_locked = False
        self.turn_count = 0
        self.last_turn_direction = None
        self.window_start = None
        self.alarm.stop()
        print("Reset. Waiting for your face to start a new 20s window.")

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    def _cleanup(self) -> None:
        self.alarm.stop()
        self._cap.release()
        cv2.destroyAllWindows()
        self._face_mesh.close()
        print("Stopped.")


# ============================================================
# Entry point
# ============================================================
def main() -> None:
    SmartDetector(CFG).run()


if __name__ == "__main__":
    main()
