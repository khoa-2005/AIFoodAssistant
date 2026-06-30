from ultralytics import YOLO
import os

image_path = r"c:\Trituenhantao\bai2\food_dataset\demo\Banh beo\23.jpg"

models = ["best.pt", "bestv8n.pt", "bestv8s.pt"]

for m in models:
    if os.path.exists(m):
        print(f"--- Testing {m} ---")
        try:
            model = YOLO(m)
            results = model.predict(image_path, conf=0.1, verbose=False)
            boxes = results[0].boxes
            print(f"Found {len(boxes)} boxes")
            for box in boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"  - {name}: {conf:.4f}")
        except Exception as e:
            print(f"Error loading/predicting with {m}: {e}")
    else:
        print(f"{m} does not exist")
