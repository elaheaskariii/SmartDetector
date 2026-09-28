import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import winsound
from insightface.app import FaceAnalysis

# ============================================================
REFERENCE_IMAGE = r"C:\Users\pars\Downloads\Eitaa Desktop\me.jpg"
# ============================================================

TIME_WINDOW = 20
MAX_TURNS = 5 # آستانه‌ی هشدار قرمز + بوق
YELLOW_TURNS = 3 # آستانه‌ی چراغ زرد
SIMILARITY_THRESHOLD = 0.35

TURN_RATIO_THRESHOLD = 0.05

PROCESS_WIDTH = 480
FACE_CHECK_EVERY = 8
MP_CHECK_EVERY = 2

# ============================================================
print("Loading InsightFace...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))
print("InsightFace ready.")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    refine_landmarks=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# ============================================================
if not os.path.exists(REFERENCE_IMAGE):
    print("Reference image not found!")
    exit()

ref_img = cv2.imread(REFERENCE_IMAGE)
ref_faces = app.get(ref_img)

if len(ref_faces) == 0:
    print("No face in reference image!")
    exit()

reference_embedding = ref_faces[0].normed_embedding
print("Reference loaded.")

# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not opened!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera started. Q=quit S=pause R=reset")

# ============================================================
# متغیرها
# ============================================================
turn_count = 0
last_turn_direction = None # ⭐️ آخرین جهتی که شمرده شده (RIGHT/LEFT/None)
window_start = None

frame_counter = 0
identity_verified = False
other_person_present = False
similarity_val = 0.0
current_direction = "NO FACE"

alert_locked = False
last_faces = []

# برای صدا
alarm_stop_event = threading.Event()
alarm_thread = None


# ============================================================
# توابع
# ============================================================
def get_head_direction(landmarks, width):
    nose = landmarks[1]
    left_face = landmarks[127]
    right_face = landmarks[356]

    nose_x = nose.x * width
    left_x = left_face.x * width
    right_x = right_face.x * width

    face_center = (left_x + right_x) / 2
    face_width = abs(right_x - left_x)

    if face_width < 1e-6:
        return "CENTER", 0.0

    ratio = (nose_x - face_center) / face_width

    if ratio > TURN_RATIO_THRESHOLD:
        return "RIGHT", ratio
    elif ratio < -TURN_RATIO_THRESHOLD:
        return "LEFT", ratio
    return "CENTER", ratio


def get_face_center_from_landmarks(landmarks, w, h):
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def play_alarm_repeating(stop_event):
    """بوق تیز و سریع تا وقتی stop_event ست بشه"""
    while not stop_event.is_set():
        try:
            winsound.Beep(2000, 80)
            if stop_event.is_set():
                break
            winsound.Beep(2500, 80)
            if stop_event.is_set():
                break
            winsound.Beep(2000, 80)
            if stop_event.is_set():
                break
            winsound.Beep(2800, 120)
            if stop_event.is_set():
                break
            winsound.Beep(2200, 100)
        except Exception as e:
            print("Sound error:", e)
            break


def start_alarm():
    global alarm_thread
    alarm_stop_event.clear()
    alarm_thread = threading.Thread(
        target=play_alarm_repeating,
        args=(alarm_stop_event,),
        daemon=True
    )
    alarm_thread.start()


# ============================================================
# حلقه اصلی
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read error")
        break

    # آینه‌ای کردن تصویر (حس دوربین سلفی)
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    if w > PROCESS_WIDTH:
        scale = PROCESS_WIDTH / w
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))
    h, w = frame.shape[:2]

    frame_counter += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_counter % MP_CHECK_EVERY == 0:
        mp_result = face_mesh.process(rgb)
    else:
        mp_result = None

    if frame_counter % FACE_CHECK_EVERY == 0:
        last_faces = app.get(frame)

    # ---- پیدا کردن چهره‌ی خودت ----
    identity_verified = False
    other_person_present = False
    similarity_val = 0.0
    my_face_bbox = None

    for f in last_faces:
        emb = f.normed_embedding
        sim = float(np.dot(reference_embedding, emb))
        if sim >= SIMILARITY_THRESHOLD:
            identity_verified = True
            similarity_val = sim
            my_face_bbox = f.bbox
            break
        else:
            if sim > 0.15:
                other_person_present = True

    if not identity_verified and len(last_faces) > 0:
        other_person_present = True
        similarity_val = max(
            float(np.dot(reference_embedding, f.normed_embedding))
            for f in last_faces
        )

    # ---- تشخیص جهت سر ----
    current_direction = "NO FACE"
    ratio_val = 0.0

    if mp_result and mp_result.multi_face_landmarks and identity_verified:
        my_landmarks = None
        n_faces_mp = len(mp_result.multi_face_landmarks)

        if n_faces_mp == 1:
            my_landmarks = mp_result.multi_face_landmarks[0].landmark
        else:
            if my_face_bbox is not None:
                bx1, by1, bx2, by2 = my_face_bbox
                my_cx = (bx1 + bx2) / 2
                my_cy = (by1 + by2) / 2
            else:
                my_cx = w / 2
                my_cy = h / 2

            best_dist = 1e18
            for face_lms in mp_result.multi_face_landmarks:
                cx, cy = get_face_center_from_landmarks(face_lms.landmark, w, h)
                d = (cx - my_cx) ** 2 + (cy - my_cy) ** 2
                if d < best_dist:
                    best_dist = d
                    my_landmarks = face_lms.landmark

        if my_landmarks is not None:
            current_direction, ratio_val = get_head_direction(my_landmarks, w)

    # ---- شروع پنجره ----
    if identity_verified and window_start is None and not alert_locked:
        window_start = time.perf_counter()
        turn_count = 0
        last_turn_direction = None
        print(">>> Window started (20s).")

    # ---- شمارش چرخش ----
    if (window_start is not None
            and identity_verified
            and not alert_locked):

        if current_direction in ("LEFT", "RIGHT"):
            # ⭐️⭐️⭐️ منطق اصلاح‌شده ⭐️⭐️⭐️
            # اگه جهت با آخرین جهت شمرده‌شده فرق داشته باشه، حرکت جدید شمرده می‌شه.
            # این باعث می‌شه «راست-چپ-راست» یا «راست-راست» (اگه وسط CENTER نری)
            # هم درست شمرده بشن — به شرط برگشتن به CENTER بین دو حرکت یکسان.
            if current_direction != last_turn_direction:
                last_turn_direction = current_direction
                turn_count += 1
                print(f"Turn: {current_direction} | count: {turn_count} | ratio={ratio_val:+.3f}")

                # ⭐️ هشدار قرمز + بوق در لحظه‌ی رسیدن به ۵
                if turn_count >= MAX_TURNS:
                    alert_locked = True
                    print(">>> ALERT! Reached 5 turns. Press R to reset. <<<")
                    start_alarm()

        elif current_direction == "CENTER":
            # وقتی سرت برمی‌گرده وسط، اجازه بده حرکت بعدی (حتی هم‌جهت) شمرده بشه
            last_turn_direction = None

    # ---- تایمر پنجره ----
    now = time.perf_counter()

    if window_start is None:
        remaining = TIME_WINDOW
    elif alert_locked:
        remaining = 0.0
    else:
        elapsed = now - window_start
        remaining = max(0.0, TIME_WINDOW - elapsed)

        if elapsed >= TIME_WINDOW:
            print("--- Window finished ---")
            print("Turns in this window:", turn_count)
            print("Normal. Starting new 20s window.")
            turn_count = 0
            last_turn_direction = None
            window_start = now

    # ============================================================
    # رسم روی تصویر
    # ============================================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (320, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    if identity_verified:
        id_text, id_color = "YOU", (0, 255, 0)
    elif other_person_present:
        id_text, id_color = "OTHER", (0, 0, 255)
    else:
        id_text, id_color = "NOBODY", (0, 0, 255)

    cv2.putText(frame, f"Identity: {id_text}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, id_color, 2)
    cv2.putText(frame, f"Similarity: {similarity_val:.3f}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Turns: {turn_count}/{MAX_TURNS}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Time: {remaining:.1f}s", (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    dir_color = (0, 255, 0) if current_direction == "CENTER" else (0, 165, 255)
    cv2.putText(frame, f"Head: {current_direction} ({ratio_val:+.2f})",
                (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 2)

    # ⭐️⭐️⭐️ چراغ‌های وضعیت گوشه‌ی بالا-راست ⭐️⭐️⭐️
    # چراغ سبز = عادی، زرد = به ۳ رسید، قرمز = به ۵ رسید
    light_x, light_y = w - 45, 45
    light_r = 25

    if alert_locked:
        # قرمز چشمک‌زن
        blink = int(time.perf_counter() * 5) % 2
        circle_color = (0, 0, 255) if blink == 0 else (0, 0, 150)
        cv2.circle(frame, (light_x, light_y), light_r, circle_color, -1)
        cv2.circle(frame, (light_x, light_y), light_r, (255, 255, 255), 2)

        # لایه‌ی قرمز روی کل تصویر
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 255
        frame = cv2.addWeighted(red_overlay, 0.35, frame, 0.65, 0)

        cv2.putText(frame, "WARNING!", (w - 240, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.putText(frame, "STOP MOVING", (w - 240, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif turn_count >= YELLOW_TURNS:
        # ⭐️ چراغ زرد (بدون صدا)
        cv2.circle(frame, (light_x, light_y), light_r, (0, 215, 255), -1)
        cv2.circle(frame, (light_x, light_y), light_r, (255, 255, 255), 2)
        cv2.putText(frame, "CAUTION", (w - 200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    else:
        # چراغ خاکستری (خاموش)
        cv2.circle(frame, (light_x, light_y), light_r, (80, 80, 80), -1)

    cv2.imshow("Smart Detector", frame)

    # ---- کلیدها ----
    key = cv2.waitKey(30) & 0xFF

    if key in (ord('q'), ord('Q'), 27):
        alarm_stop_event.set()
        print("Quit.")
        break

    elif key in (ord('s'), ord('S')):
        print("PAUSED. Press S to resume.")
        while True:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 in (ord('s'), ord('S')):
                print("RESUMED.")
                if identity_verified and not alert_locked:
                    window_start = time.perf_counter()
                    turn_count = 0
                    last_turn_direction = None
                break
            elif k2 in (ord('q'), ord('Q'), 27):
                alarm_stop_event.set()
                cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                print("Stopped.")
                exit()

    elif key in (ord('r'), ord('R')):
        alert_locked = False
        turn_count = 0
        last_turn_direction = None
        window_start = None
        alarm_stop_event.set()
        print("Reset. Waiting for your face to start a new 20s window.")

# ============================================================
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
alarm_stop_event.set()
print("Stopped.")
