import os
import cv2
import mediapipe as mp
import numpy as np

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

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

SAVE_DIR = 'TrainingImages'
CLASS_NAME = 'two'
PADDING = 20
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

camera = cv2.VideoCapture(0)
frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
existing = [f for f in os.listdir(SAVE_DIR) if CLASS_NAME in f]
img_count = len(existing)

print(f"Collecting images for class: '{CLASS_NAME}'")
print(f"Already have {img_count} images for this class")
print("Press L to save image, Q to quit")

while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_crop = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * frame_w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * frame_h for lm in hand_landmarks.landmark]
            x1, y1 = max(0, int(min(x_coords)) - PADDING), max(0, int(min(y_coords)) - PADDING)
            x2, y2 = min(frame_w, int(max(x_coords)) + PADDING), min(frame_h, int(max(y_coords)) + PADDING)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hand_crop = frame[y1:y2, x1:x2]

    status = f"Class: {CLASS_NAME} | Saved: {img_count} | L=save Q=quit"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if hand_crop is not None:
        cv2.putText(frame, "Hand detected!", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        if hand_crop is not None and hand_crop.size > 0:
            filename = f"{CLASS_NAME}_{img_count:04d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            square_crop = letterbox_resize(hand_crop, 224)
            cv2.imwrite(filepath, square_crop)
            img_count += 1
            print(f"Saved {filepath}")
        else:
            print("No hand detected")

camera.release()
cv2.destroyAllWindows()
print(f"Done! Collected {img_count} total images for class '{CLASS_NAME}'")