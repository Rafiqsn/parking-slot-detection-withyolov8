import streamlit as st
import tempfile
import os
import uuid
from detection import run_detection

st.set_page_config(page_title="Deteksi Parkir", layout="centered")

st.title("🚗 Deteksi Mobil & Tempat Parkir Kosong")
st.markdown("Unggah video lalu klik tombol untuk mendeteksi tempat parkir kosong.")

uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "mov", "avi", "mpeg"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    input_video_path = tfile.name

    st.video(input_video_path)

    if st.button("🔍 Cari Tempat Parkir"):
        st.write("⏳ Sedang memproses video...")

        unique_id = str(uuid.uuid4())[:8]
        os.makedirs("output", exist_ok=True)
        output_video_path = os.path.join("output", f"result_{unique_id}.mp4")

        run_detection(input_video_path, output_video_path)

        st.success("✅ Deteksi selesai! Lihat hasil di bawah ini:")

        # Pastikan video bisa dibaca kembali
        with open(output_video_path, "rb") as file:
            video_bytes = file.read()
            st.video(video_bytes)
