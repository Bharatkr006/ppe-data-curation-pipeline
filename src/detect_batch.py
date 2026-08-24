import os
from ultralytics import YOLO
import pandas as pd
model = YOLO('yolov8n.pt')
frame_files = sorted(os.listdir("data/frames"))

all_detections = []

for frame_file in frame_files:
    frame_path = os.path.join("data/frames", frame_file)
    results = model(frame_path)
    
    for box in results[0].boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]
        confidence = float(box.conf)
        bbox = box.xyxy[0].tolist()
        all_detections.append({
            "frame_id": frame_file,
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": bbox
        })

df = pd.DataFrame(all_detections)
df.to_csv("outputs/detections.csv", index=False)
print(f"Total detections across {len(frame_files)} frames: {len(df)}")