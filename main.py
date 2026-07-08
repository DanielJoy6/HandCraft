import threading
import os
from collections import deque, Counter
import cv2
from pynput.keyboard import Controller as KeywordController
from pynput.mouse import Button, Controller as MouseController

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp

# ============================================================
# CONFIG
# ============================================================

keyboard = KeywordController()
mouse = MouseController()

# Bounding-box padding (px) around detected hand before cropping for classification
PADDING = 20

# LEFT HAND "movement zones" -- 4 adjustable rectangles.
# Coordinates are (x1, y1, x2, y2) in FULL-FRAME pixel space.
MOVEMENT_ZONES = {
    "up": (130, 80, 210, 180),
    "down": (130, 290, 210, 390),
    "left": (20, 190, 120, 290),
    "right": (210, 190, 310, 290),
}

# RIGHT HAND -> mouse look (camera rotation)
# Every frame the right hand is detected, we compute how far its center
# moved since the previous frame and translate that into a mouse delta.
# Tune MOUSE_SENSITIVITY to taste -- higher = faster camera turn per px
# of physical hand movement.
MOUSE_SENSITIVITY = 3.5
# Ignore tiny jitter below this many pixels of hand movement per frame
# so an essentially-still hand doesn't cause camera drift.
MOUSE_DEADZONE_PX = 3

# Number of recent raw predictions to majority-vote over before a gesture
# label is considered "confirmed" and allowed to trigger an action. This
# smooths out single-frame misclassifications at the cost of a small delay
# (VOTE_WINDOW frames' worth of prediction latency) before a new gesture
# takes effect.
VOTE_WINDOW = 3

# ============================================================
# MODEL LOADING
# ============================================================
# model1 -> LEFT hand gesture classifier (fist / open hand -> inventory toggle)
# model2 -> RIGHT hand gesture classifier (fist / index / pinky -> clicks)
LEFT_HAND_MODEL_PATH = "model1.keras"
RIGHT_HAND_MODEL_PATH = "model2.keras"

LEFT_HAND_LABELS = {
    0: "Open (Idle)",
    1: "Fist (Inventory)",
}
RIGHT_HAND_LABELS = {
    0: "Idle (Fist)",
    1: "Left Click",
    2: "Right Click",
}


def load_or_warn(path, hand_name):
    if os.path.exists(path):
        return load_model(path)
    print(f"[WARNING] '{path}' not found. {hand_name}-hand classification will be disabled.")
    return None


left_hand_model = load_or_warn(LEFT_HAND_MODEL_PATH, "Left")
right_hand_model = load_or_warn(RIGHT_HAND_MODEL_PATH, "Right")

# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    model_complexity=0,
)

# ============================================================
# ASYNC PREDICTION THREADS (one per hand, same pattern as before)
# ============================================================
left_lock = threading.Lock()
latest_left_crop = None
left_label = 0  # majority-voted, confirmed label
left_vote_history = deque(maxlen=VOTE_WINDOW)
new_left_crop_event = threading.Event()

right_lock = threading.Lock()
latest_right_crop = None
right_label = 0  # majority-voted, confirmed label
right_vote_history = deque(maxlen=VOTE_WINDOW)
new_right_crop_event = threading.Event()


def majority_vote(history):
    """Return the most common label in the recent-prediction history.
    Ties fall back to the most recent raw prediction so the system doesn't
    freeze on a class boundary."""
    if not history:
        return 0
    counts = Counter(history)
    top_label, top_count = counts.most_common(1)[0]
    tied = [label for label, count in counts.items() if count == top_count]
    if len(tied) > 1:
        return history[-1]
    return top_label

def letterbox_resize(img, target_size=224):
    """Resize keeping aspect ratio, padding with black to reach a square."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def make_predict_loop(model, lock, get_latest_crop, history, set_label, event):
    def predict_loop():
        while True:
            event.wait()
            event.clear()
            with lock:
                crop = get_latest_crop()
            if crop is not None and model is not None and crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                square = letterbox_resize(crop_rgb, 224)
                img_array = preprocess_input(square.astype("float32"))
                img_array = np.expand_dims(img_array, axis=0)
                preds = model(img_array, training=False).numpy()
                raw_label = int(np.argmax(preds))
                history.append(raw_label)
                set_label(majority_vote(history))

    return predict_loop


def get_left_crop():
    return latest_left_crop


def set_left_label(v):
    global left_label
    left_label = v


def get_right_crop():
    return latest_right_crop


def set_right_label(v):
    global right_label
    right_label = v


left_thread = threading.Thread(
    target=make_predict_loop(
        left_hand_model, left_lock, get_left_crop, left_vote_history, set_left_label, new_left_crop_event
    ),
    daemon=True,
)
right_thread = threading.Thread(
    target=make_predict_loop(
        right_hand_model, right_lock, get_right_crop, right_vote_history, set_right_label, new_right_crop_event
    ),
    daemon=True,
)
left_thread.start()
right_thread.start()


# ============================================================
# HELPERS
# ============================================================
def get_hand_box(landmarks, frame_w, frame_h, padding=PADDING):
    """Given mediapipe landmarks (already normalized 0-1) return a clamped
    pixel bounding box (x1, y1, x2, y2) with padding."""
    x_coords = [lm.x * frame_w for lm in landmarks.landmark]
    y_coords = [lm.y * frame_h for lm in landmarks.landmark]
    x1, y1 = int(min(x_coords)), int(min(y_coords))
    x2, y2 = int(max(x_coords)), int(max(y_coords))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(frame_w, x2 + padding), min(frame_h, y2 + padding)
    return x1, y1, x2, y2


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def check_movement_zone(center, zones):
    cx, cy = center
    for name, (zx1, zy1, zx2, zy2) in zones.items():
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return name
    return None


# ------------------------------------------------------------
# LEFT HAND: movement (position, edge-triggered hold) + inventory (gesture)
# ------------------------------------------------------------
current_direction = None


def update_movement(direction):
    global current_direction
    key_map = {"up": "w", "down": "s", "left": "a", "right": "d"}
    if direction == current_direction:
        return
    if current_direction is not None:
        keyboard.release(key_map[current_direction])
    if direction is not None:
        keyboard.press(key_map[direction])
    current_direction = direction


def send_inventory_command(label, prev_label):
    """label 1 = open inventory, label 0 = idle/reset (close inventory).
    Minecraft's inventory key ('e') toggles open/closed, so we press it
    once on each transition into a new state rather than holding it."""
    if label == prev_label:
        return
    if label == 1:
        print("Open inventory")
        keyboard.press("e")
        keyboard.release("e")
    elif label == 0:
        print("Reset / close inventory")
        keyboard.press("e")
        keyboard.release("e")


# ------------------------------------------------------------
# RIGHT HAND: mouse look (position delta) + clicks (gesture)
# ------------------------------------------------------------
prev_right_center = None


def update_mouse_look(center):
    """Move the mouse by the delta in the right hand's position since last
    frame, scaled by MOUSE_SENSITIVITY, to rotate the in-game camera."""
    global prev_right_center
    if center is None:
        prev_right_center = None
        return
    if prev_right_center is not None:
        dx = center[0] - prev_right_center[0]
        dy = center[1] - prev_right_center[1]
        if abs(dx) >= MOUSE_DEADZONE_PX or abs(dy) >= MOUSE_DEADZONE_PX:
            mouse.move(int(dx * MOUSE_SENSITIVITY), int(dy * MOUSE_SENSITIVITY))
    prev_right_center = center


def send_click_command(label, prev_label):
    """label 0 = idle/fist -> nothing (release left if held)
    label 1 = index finger up -> left click
    label 2 = pinky up -> right click"""
    if label == prev_label:
        return
    if label == 1:
        print("Left click")
        mouse.press(Button.left)
        mouse.release(Button.left)
    elif label == 2:
        print("Right click")
        mouse.press(Button.right)
        mouse.release(Button.right)
    elif label == 0:
        mouse.release(Button.left)


# ============================================================
# CAMERA / VIDEO OUTPUT SETUP
# ============================================================
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_w = frame_w // 2

# ============================================================
# MAIN LOOP STATE
# ============================================================
active_zone = None
last_left_box = None
last_right_box = None
prev_left_label = 0
prev_right_label = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Detection runs on a clean frame -- overlays are drawn afterward
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    active_zone = None
    last_left_box = None
    left_center = None
    left_hand_crop = None

    last_right_box = None
    right_center = None
    right_hand_crop = None

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            box = get_hand_box(landmarks, frame_w, frame_h)
            x1, y1, x2, y2 = box
            center = box_center(box)

            if center[0] < half_w:
                # LEFT hand: position -> movement, crop -> inventory gesture
                last_left_box = box
                left_center = center
                active_zone = check_movement_zone(left_center, MOVEMENT_ZONES)
                left_hand_crop = frame[y1:y2, x1:x2]
            else:
                # RIGHT hand: position -> mouse look, crop -> click gesture
                last_right_box = box
                right_center = center
                right_hand_crop = frame[y1:y2, x1:x2]

    # --- LEFT hand actions ---
    update_movement(active_zone)

    if left_hand_crop is not None and left_hand_crop.size > 0:
        with left_lock:
            latest_left_crop = left_hand_crop.copy()
        new_left_crop_event.set()

    last_left_label = left_label
    send_inventory_command(last_left_label, prev_left_label)
    prev_left_label = last_left_label

    # --- RIGHT hand actions ---
    update_mouse_look(right_center)

    if right_hand_crop is not None and right_hand_crop.size > 0:
        with right_lock:
            latest_right_crop = right_hand_crop.copy()
        new_right_crop_event.set()

    last_right_label = right_label
    send_click_command(last_right_label, prev_right_label)
    prev_right_label = last_right_label

    # ------------------------------------------------------------
    # Overlays / HUD (drawn AFTER detection, never before)
    # ------------------------------------------------------------
    for name, (zx1, zy1, zx2, zy2) in MOVEMENT_ZONES.items():
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)

    if last_left_box:
        x1, y1, x2, y2 = last_left_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, left_center, 4, (255, 0, 0), -1)

    if last_right_box:
        x1, y1, x2, y2 = last_right_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, right_center, 4, (255, 0, 0), -1)

    cv2.putText(frame, f"Zone: {active_zone or '-'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Left hand: {LEFT_HAND_LABELS.get(last_left_label, '?')}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Right hand: {RIGHT_HAND_LABELS.get(last_right_label, '?')}",
                (half_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release any held keys/buttons on exit
if current_direction is not None:
    keyboard.release({"up": "w", "down": "s", "left": "a", "right": "d"}[current_direction])
mouse.release(Button.left)

camera.release()
cv2.destroyAllWindows()
