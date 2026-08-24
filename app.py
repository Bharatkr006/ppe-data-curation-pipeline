import streamlit as st
import pandas as pd
from PIL import Image
import os

st.title("PPE Data Curation Pipeline")

df = pd.read_csv("outputs/final_review_queue.csv")

total = len(df)
needs_review = df["final_review_needed"].sum()
st.metric("Total Frames", total)
st.metric("Needs Human Review", f"{needs_review} ({needs_review/total*100:.1f}%)")
st.metric("Auto-Resolved", f"{total - needs_review} ({(total-needs_review)/total*100:.1f}%)")

st.subheader("Review Queue")
queue = df[df["final_review_needed"] == True].sort_values("needs_review_count", ascending=False)
st.dataframe(queue[["frame_id", "total_detections", "needs_review_count", "min_confidence"]])

st.subheader("Preview a Flagged Frame")
selected_frame = st.selectbox("Choose a frame", queue["frame_id"].tolist())
if selected_frame:
    img_path = os.path.join("data/frames", selected_frame)
    st.image(Image.open(img_path), caption=selected_frame)