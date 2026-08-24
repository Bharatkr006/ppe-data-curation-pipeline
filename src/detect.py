from ultralytics import YOLO
import pandas as pd

model = YOLO('yolov8n.pt')

results = model('data/test.webp')

detections = []
for box in results[0].boxes:
    cls_id = int(box.cls)
    class_name = model.names[cls_id]
    confidence = float(box.conf)
    bbox = box.xyxy[0].tolist()
    detections.append({
        "frame_id": "test.webp",
        "class": class_name,
        "confidence": round(confidence, 3),
        "bbox": bbox
    })

df = pd.DataFrame(detections)
df.to_csv("outputs/detections.csv", index=False)
print(df)