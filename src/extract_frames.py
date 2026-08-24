import cv2

cap = cv2.VideoCapture("data/test.mp4")

frame_count = 0
saved_count = 0
save_every = 1  # save 1 out of every 5 frames

while True:
    success, frame = cap.read()
    
    if not success:
        break  # video ended
    
    if frame_count % save_every == 0:
        filename = f"data/frames/frame_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1
    
    frame_count += 1

cap.release()
print(f"Total frames read: {frame_count}, saved: {saved_count}")