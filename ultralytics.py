from ultralytics import YOLO
import cv2
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------------------- Setup ----------------------
video_path = 'factory_2.mp4'
model = YOLO('yolov8n.pt')  # Faster but less accurate
log_file = "presence_log.csv"
json_file = "tracking_output.json"
heatmap_file = "heatmap_output.png"

FPS = 30  # Approximate FPS of the input video
video_width = 1280
video_height = 720

# ---------------------- Tracking and CSV Logging ----------------------
cap = cv2.VideoCapture(video_path)
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Track_ID", "X", "Y", "W", "H"])  # header

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, persist=True, classes=[0], verbose=False)
        frame_num += 1

        boxes = results[0].boxes
        if boxes.id is not None:
            for i, box in enumerate(boxes.xywh):
                track_id = int(boxes.id[i])
                x, y, w, h = map(int, box)
                writer.writerow([frame_num, track_id, x, y, w, h])

        print(f"Processed frame: {frame_num}")

cap.release()
print(f"\n Tracking log saved to: {log_file}")

# ---------------------- JSON Generation ----------------------
df = pd.read_csv(log_file)
json_output = {}

for frame, group in df.groupby("Frame"):
    frame_key = f"frame_{int(frame)}"
    entries = []
    for _, row in group.iterrows():
        trackid = int(row["Track_ID"])
        bbox = [int(row["X"]), int(row["Y"]), int(row["W"]), int(row["H"])]
        timestamp = round(row["Frame"] / FPS, 2)
        entries.append([trackid, bbox, timestamp])
    json_output[frame_key] = entries

with open(json_file, "w") as f:
    json.dump(json_output, f, indent=4)

print(f" JSON file saved as '{json_file}'")

# ---------------------- Heatmap Generation ----------------------
heat_data = np.zeros((video_height, video_width), dtype=np.float32)

for _, row in df.iterrows():
    cx = int(row["X"] + row["W"] // 2)
    cy = int(row["Y"] + row["H"] // 2)
    if 0 <= cx < video_width and 0 <= cy < video_height:
        heat_data[cy, cx] += 1

heat_data_blurred = cv2.GaussianBlur(heat_data, (0, 0), sigmaX=15, sigmaY=15)
norm_heat = heat_data_blurred / np.max(heat_data_blurred)

plt.figure(figsize=(14, 7))
sns.heatmap(norm_heat, cmap="hot", cbar=True)
plt.title("Human Movement Heatmap")
plt.xlabel("X-axis (pixels)")
plt.ylabel("Y-axis (pixels)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(heatmap_file, dpi=300)
plt.show()

print(f" Heatmap saved as '{heatmap_file}'")
