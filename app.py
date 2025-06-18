import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
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

# --- Konfigurasi video dari Google Drive ---
# Mapping: video_id -> (drive_file_id, slot_polygon_path, local_filename)
VIDEO_CONFIG = [
    {
        "id": 1,
        "drive_id": "1g452_pqKTBNNoIGE6SnwOThapTiDuvpj",  # Ganti dengan ID Google Drive asli
        "slot_path": "anotasi/slot_polygons1.json",
        "filename": "video/input1.mp4",
    },
    {
        "id": 2,
        "drive_id": "1jSudC2pEu4n8b0CSP5RwI7td26Wl5JhM",
        "slot_path": "anotasi/slot_polygons2.json",
        "filename": "video/input2.mp4",
    },
    {
        "id": 3,
        "drive_id": "1w93ospu_e89NADDfO5kk5PzhasxhePcU",
        "slot_path": "anotasi/slot_polygons3.json",
        "filename": "video/input3.mp4",
    },
    {
        "id": 4,
        "drive_id": "1-CiiBqf37OPX_L0EVYP57WTeyw3Wdroa",
        "slot_path": "anotasi/slot_polygons4.json",
        "filename": "video/input4.mp4",
    },
    {
        "id": 5,
        "drive_id": "1k1Es8k8ESV1_w3kA4L3dBYB9NBVq0BPP",
        "slot_path": "anotasi/slot_polygons5.json",
        "filename": "video/input5.mp4",
    },
    {
        "id": 6,
        "drive_id": "14djCdke8349btuPnUDig_Nm3uvsFjyHO",
        "slot_path": "anotasi/slot_polygons6.json",
        "filename": "video/input6.mp4",
    },
    {
        "id": 7,
        "drive_id": "1gb0IoEuLt7MjTWfABOp6bLL206oNrHBP",
        "slot_path": "anotasi/slot_polygons7.json",
        "filename": "video/input7.mp4",
    },
    {
        "id": 8,
        "drive_id": "1HIz11DiNss44M-KigpLsf6JgbvrqIoPO",
        "slot_path": "anotasi/slot_polygons8.json",
        "filename": "video/input8.mp4",
    },
    {
        "id": 9,
        "drive_id": "1PUTm9vx0LgVFrpmgSJL85W2AVJXEdm5_",
        "slot_path": "anotasi/slot_polygons9.json",
        "filename": "video/input9.mp4",
    },
    {
        "id": 10,
        "drive_id": "1B62L3YQQXTWGfBpYp03YykUb_HM4kPM_",
        "slot_path": "anotasi/slot_polygons10.json",
        "filename": "video/input10.mp4",
    },
]


def download_from_gdrive(drive_id, dest_path):
    # Download file dari Google Drive (public) menggunakan requests
    # Untuk file besar, gunakan confirm token
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": drive_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        params = {"id": drive_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


# Static config
VIDEO_LIST = [f"video/input{i}.mp4" for i in range(1, 11)]
SLOT_POLYGON_LIST = [f"anotasi/slot_polygons{i}.json" for i in range(1, 11)]

# UI
st.title("Deteksi Slot Parkir Kosong")
video_options = [cfg["id"] for cfg in VIDEO_CONFIG]
video_idx = st.selectbox("Pilih Video Parkiran", video_options)
video_cfg = next(cfg for cfg in VIDEO_CONFIG if cfg["id"] == video_idx)
video_path = video_cfg["filename"]
slot_path = video_cfg["slot_path"]
drive_id = video_cfg["drive_id"]

# Cek apakah video sudah ada, jika belum download dulu
video_file = Path(video_path)
if not video_file.exists():
    # Pastikan folder tujuan ada
    video_file.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Mengunduh video ID {drive_id} dari Google Drive..."):
        try:
            download_from_gdrive(drive_id, video_path)
            st.success("Video berhasil diunduh.")
        except Exception as e:
            st.error(f"Gagal mengunduh video: {e}")
            st.stop()

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
