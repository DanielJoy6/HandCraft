import os
import cv2
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

y = []
x = []

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

TARGET_SIZE = (224, 224)
mappings = {
    "zero": 0,
    "one": 1,
    "two": 2,
}

for filename in os.listdir("TrainingImages"):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Check filename matches a known class
    label = None
    for key, val in mappings.items():
        if key in filename:
            label = val
            break
    if label is None:
        #print(f"Skipping unrecognized file: {filename}")
        continue

    y.append(label)

    image_path = os.path.join("TrainingImages", filename)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_np = np.array(img)
        square = letterbox_resize(img_np, 224)  # no-op if already square, safe otherwise
        img_array = preprocess_input(square.astype("float32"))
        x.append(img_array)
x = np.array(x)
y = np.array(y, dtype=np.int32)
print(f"Loaded {len(x)} training images, shape: {x.shape}")
y = to_categorical(y, num_classes=len(list(mappings.keys())))

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

model2 = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ]
)
print("MODEL 2")
model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model2.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val),
    verbose=1,
)

model2.save("model2.keras")
