import json

import cv2
from ultralytics import YOLO

# Load model deteksi
model = YOLO("models/car_detection.pt")

# Buka video
cap = cv2.VideoCapture("video/input9.mp4")

frame_idx = 0
all_bboxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek
    results = model(frame)[0]

    # Ambil bbox mobil (asumsi class mobil = 0)
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            bbox = [x1, y1, x2, y2]

            # Simpan ke list
            all_bboxes.append({"frame": frame_idx, "bbox": bbox})

    frame_idx += 1

cap.release()
