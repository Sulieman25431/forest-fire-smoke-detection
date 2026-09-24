import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Forest Fire & Smoke Detection", layout="wide")

st.title("🔥 Real-Time Forest Fire & Smoke Detection System")
st.markdown("Upload a video or use a webcam stream to run real-time YOLOv8 detection.")

# Load Trained YOLO Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar Setup
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
input_source = st.sidebar.radio("Select Input Source", ("Upload Video File", "Live Webcam"))

# ----------------- OPTION 1: VIDEO FILE UPLOAD -----------------
if input_source == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        alert_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference
            results = model.predict(source=frame, conf=confidence_threshold)
            
            # Check for detected classes
            fire_detected = False
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    if class_name.lower() in ['fire', 'smoke']:
                        fire_detected = True

            # Display real-time alert banner
            if fire_detected:
                alert_box.error("⚠️ ALERT: Fire or Smoke Hazard Detected!")
            else:
                alert_box.success("✅ Status: Clear (No Hazards Detected)")

            # Render detection bounding boxes on frame
            annotated_frame = results[0].plot()

            # Display updated frame in Streamlit dashboard
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()

# ----------------- OPTION 2: LIVE WEBCAM STREAM -----------------
elif input_source == "Live Webcam":
    st.info("Click Start below to enable webcam feed.")
    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        alert_box = st.empty()

        while cap.isOpened() and run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam feed.")
                break

            # Run YOLO inference
            results = model.predict(source=frame, conf=confidence_threshold)

            fire_detected = False
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    if class_name.lower() in ['fire', 'smoke']:
                        fire_detected = True

            if fire_detected:
                alert_box.error("⚠️ ALERT: Fire or Smoke Hazard Detected!")
            else:
                alert_box.success("✅ Status: Clear (No Hazards Detected)")

            annotated_frame = results[0].plot()
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()