import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

from cnn_resnet18 import extract_feature_from_bbox, load_embeder_from_pt
from tracking import Tracker


# Load model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO("models/car_detection.pt")
    detector.to(device)
    embedder = load_embeder_from_pt("models/embedder_triplet.pt").to(device)
    return detector, Tracker(embedder)


detector, tracker = load_model()

# Static config
VIDEO_LIST = [f"video/input{i}.mp4" for i in range(1, 11)]
SLOT_POLYGON_LIST = [f"anotasi/slot_polygons{i}.json" for i in range(1, 11)]

# UI
st.title("Deteksi Slot Parkir Kosong")
video_idx = st.selectbox("Pilih Video Parkiran", list(range(1, 11)))
video_path = VIDEO_LIST[video_idx - 1]
slot_path = SLOT_POLYGON_LIST[video_idx - 1]

# Tampilkan video player
st.video(video_path)

# --- Session State untuk menyimpan hasil deteksi ---
if "slots" not in st.session_state:
    st.session_state.slots = None

# Tombol untuk mendeteksi kondisi parkir saat ini
if st.button("Cari Parkir Sekarang"):
    # Ambil frame terakhir dari video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 5)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Gagal mengambil frame dari video.")
    else:
        # Load slot
        with open(slot_path) as f:
            slots = json.load(f)
        st.session_state.slots = slots  # Simpan ke session state

        # Deteksi mobil
        results = detector(frame)[0]
        detections = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2])

        # Tracking
        tracks = tracker.update(frame, detections)

        # Buat mapping bbox->track_id
        bbox_id_map = {}
        for track_id, bbox in tracks:
            # Cari deteksi dengan IOU tertinggi
            best_iou = 0
            best_det = None
            for det in detections:
                # Hitung IOU
                xx1 = max(bbox[0], det[0])
                yy1 = max(bbox[1], det[1])
                xx2 = min(bbox[2], det[2])
                yy2 = min(bbox[3], det[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area2 = (det[2] - det[0]) * (det[3] - det[1])
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_det = tuple(det)
            if best_det is not None and best_iou > 0.3:
                bbox_id_map[best_det] = track_id

        # Gambar bounding box dan ID (jika ada)
        for det in detections:
            x1, y1, x2, y2 = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if tuple(det) in bbox_id_map:
                track_id = bbox_id_map[tuple(det)]
                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Cek slot kosong
        slot_status = {}
        for slot in slots:
            slot_id = slot["id"]
            pts = np.array(slot["points"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            occupied = False

            # Cek setiap mobil (track) apakah ada di slot ini
            for det in detections:
                cx = int((det[0] + det[2]) / 2)
                cy = int((det[1] + det[3]) / 2)
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    occupied = True
                    break

            color = (0, 0, 255) if occupied else (0, 255, 0)
            slot_status[slot_id] = not occupied  # True jika kosong, False jika terisi
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            # Outline hitam + teks putih
            text_pos = tuple(pts[0][0])
            cv2.putText(
                frame,
                f"ID:{slot_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ID:{slot_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        kosong = [str(slot_id) for slot_id, kosong in slot_status.items() if kosong]
        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption="Hasil Deteksi",
        )

        if kosong:
            st.success(f"Slot kosong ditemukan: {', '.join(kosong)}")
        else:
            st.warning("Tidak ada slot kosong yang terdeteksi.")

# Tombol proses video, hanya aktif jika sudah ada hasil deteksi (slots)
if st.session_state.slots is not None:
    if st.button("Proses Video Hasil Deteksi"):
        with st.spinner("Mempersiapkan video hasil deteksi..."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # File sementara untuk hasil video
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(
                temp_video.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            progress_bar = st.progress(0)
            preview_placeholder = st.empty()

            frame_idx = 0
            slots = st.session_state.slots  # Ambil dari session state
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Deteksi mobil
                results = detector(frame)[0]
                detections = []
                for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                    if int(cls) == 0:
                        x1, y1, x2, y2 = map(int, box)
                        detections.append([x1, y1, x2, y2])

                # Tracking
                tracks = tracker.update(frame, detections)

                # Buat mapping bbox->track_id
                bbox_id_map = {}
                for track_id, bbox in tracks:
                    best_iou = 0
                    best_det = None
                    for det in detections:
                        xx1 = max(bbox[0], det[0])
                        yy1 = max(bbox[1], det[1])
                        xx2 = min(bbox[2], det[2])
                        yy2 = min(bbox[3], det[3])
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        inter = w * h
                        area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        area2 = (det[2] - det[0]) * (det[3] - det[1])
                        iou = inter / (area1 + area2 - inter + 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best_det = tuple(det)
                    if best_det is not None and best_iou > 0.3:
                        bbox_id_map[best_det] = track_id

                # Gambar bounding box dan ID (jika ada)
                for det in detections:
                    x1, y1, x2, y2 = det
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    if tuple(det) in bbox_id_map:
                        track_id = bbox_id_map[tuple(det)]
                        cv2.putText(
                            frame,
                            f"ID:{track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                # Cek slot kosong untuk setiap slot pada setiap frame
                slot_status = {}
                for slot in slots:
                    slot_id = slot["id"]
                    pts = np.array(slot["points"], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    occupied = False

                    for det in detections:
                        cx = int((det[0] + det[2]) / 2)
                        cy = int((det[1] + det[3]) / 2)
                        if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                            occupied = True
                            break

                    color = (0, 0, 255) if occupied else (0, 255, 0)
                    slot_status[slot_id] = (
                        not occupied
                    )  # True jika kosong, False jika terisi
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                    text_pos = tuple(pts[0][0])
                    cv2.putText(
                        frame,
                        f"ID:{slot_id}",
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Outline hitam
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"ID:{slot_id}",
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # Teks putih
                        1,
                        cv2.LINE_AA,
                    )

                out.write(frame)

                # Update progress bar
                frame_idx += 1
                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                # Show preview every 50 frames
                if frame_idx % 50 == 0:
                    preview_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        caption=f"Preview Frame {frame_idx}",
                    )

            cap.release()
            out.release()
            progress_bar.empty()
            preview_placeholder.empty()

            # Baca file hasil video untuk diunduh
            with open(temp_video.name, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="Unduh Video Hasil Deteksi",
                data=video_bytes,
                file_name="hasil_deteksi.mp4",
                mime="video/mp4",
            )

            # Hapus file sementara setelah diunduh
            Path(temp_video.name).unlink(missing_ok=True)
