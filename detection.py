from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Load model YOLOv8 kamu
yolo_model = YOLO("models/car_detection.pt")  # pastikan path ini sesuai

def run_detection(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Jalankan prediksi YOLO
        results = yolo_model.predict(source=frame, save=False, conf=0.4, iou=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Gambar bounding box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

        out.write(frame)

    cap.release()
    if out:
        out.release()
