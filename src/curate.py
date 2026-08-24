import pandas as pd

df = pd.read_csv("outputs/detections.csv")

def triage(confidence):
    if confidence >= 0.7:
        return "auto_accept"
    elif confidence >= 0.4:
        return "needs_review"
    else:
        return "reject"

df["status"] = df["confidence"].apply(triage)
frame_summary = df.groupby("frame_id").agg(
    total_detections=("class", "count"),
    needs_review_count=("status", lambda x: (x == "needs_review").sum()),
    reject_count=("status", lambda x: (x == "reject").sum()),
    min_confidence=("confidence", "min")
).reset_index()

frame_summary["review_ratio"] = frame_summary["needs_review_count"] / frame_summary["total_detections"]
frame_summary["flag_for_review"] = (frame_summary["review_ratio"] > 0.3) | (frame_summary["min_confidence"] < 0.3)

frame_summary.to_csv("outputs/frame_review_queue.csv", index=False)
print(frame_summary["flag_for_review"].value_counts())