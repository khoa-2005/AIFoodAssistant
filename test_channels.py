from ultralytics import YOLO
from PIL import Image
import numpy as np

image_path = r"c:\Trituenhantao\bai2\food_dataset\demo\Banh beo\23.jpg"
image = Image.open(image_path).convert("RGB")

model = YOLO("best.pt")

print("--- 1. PIL Image ---")
results = model.predict(image, conf=0.1, verbose=False)
for box in results[0].boxes:
    print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.4f}")

print("--- 2. numpy array (RGB) ---")
results = model.predict(np.array(image), conf=0.1, verbose=False)
for box in results[0].boxes:
    print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.4f}")

print("--- 3. numpy array (BGR) ---")
results = model.predict(np.array(image)[:, :, ::-1], conf=0.1, verbose=False)
for box in results[0].boxes:
    print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.4f}")
