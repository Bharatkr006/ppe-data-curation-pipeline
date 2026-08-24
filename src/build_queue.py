import pandas as pd

review_df = pd.read_csv("outputs/frame_review_queue.csv")
dedupe_df = pd.read_csv("outputs/dedupe_scores.csv")

merged = review_df.merge(dedupe_df, on="frame_id")

merged["final_review_needed"] = merged["flag_for_review"] & (~merged["is_duplicate"])

merged.to_csv("outputs/final_review_queue.csv", index=False)

total_frames = len(merged)
needs_review = merged["final_review_needed"].sum()
print(f"Total frames: {total_frames}")
print(f"Needs human review: {needs_review} ({needs_review/total_frames*100:.1f}%)")
print(f"Auto-resolved/skipped: {total_frames - needs_review} ({(total_frames-needs_review)/total_frames*100:.1f}%)")