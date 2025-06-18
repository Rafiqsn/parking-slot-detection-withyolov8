import os

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

model = YOLO("./models/car_detection.pt")
video = cv2.VideoCapture("./video/input9.mp4")

save_dir = "cropped_dataset"
os.makedirs(save_dir, exist_ok=True)

fps = video.get(cv2.CAP_PROP_FPS)
max_seconds = 10
max_frames = int(fps * max_seconds)

tracker = DeepSort(max_age=30)  # Inisialisasi DeepSORT

frame_idx = 0
while frame_idx < max_frames:
    ret, frame = video.read()
    if not ret:
        print(f"Frame {frame_idx} gagal dibaca.")
        break

    results = model(frame)[0]
    dets = []
    for det in results.boxes.data:
        cls_id = int(det[5])
        # Ganti 0 jika class mobil berbeda
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = map(float, det[:4])
        conf = float(det[4])
        w = x2 - x1
        h = y2 - y1
        dets.append(([x1, y1, w, h], conf, "car"))

    tracks = tracker.update_tracks(dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cropped = frame[y1:y2, x1:x2]
        car_dir = f"{save_dir}/car_{track_id}"
        os.makedirs(car_dir, exist_ok=True)
        if cropped.size > 0:
            cv2.imwrite(f"{car_dir}/frame_{frame_idx}.jpg", cropped)

    frame_idx += 1
