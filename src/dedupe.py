import cv2
import numpy as np
import os
import pandas as pd

frame_dir = "data/frames"
frame_files = sorted(os.listdir(frame_dir))

results = []
prev_frame = None

for frame_file in frame_files:
    frame_path = os.path.join(frame_dir, frame_file)
    frame = cv2.imread(frame_path)
    
    if prev_frame is None:
        diff_score = None
    else:
        diff = cv2.absdiff(frame, prev_frame)
        diff_score = diff.mean()
    
    results.append({
        "frame_id": frame_file,
        "diff_score": diff_score
    })
    
    prev_frame = frame

df = pd.DataFrame(results)
threshold = df["diff_score"].quantile(0.2)
df["is_duplicate"] = df["diff_score"] < threshold
df.to_csv("outputs/dedupe_scores.csv", index=False)
print(f"Threshold: {threshold:.2f}")
print(df["is_duplicate"].value_counts())