import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="PPE Compliance Detector", layout="wide")
st.title("PPE Compliance Detector")
st.write("Upload an image to detect PPE items and PPE violations. Faces can be blurred for privacy.")

DEFAULT_WEIGHTS = Path("best.pt")
if not DEFAULT_WEIGHTS.exists():
    st.warning("best.pt was not found in the app folder. Upload or copy the trained YOLO weights before deployment.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@st.cache_resource
def load_detector(weights_path):
    return YOLO(str(weights_path))

def blur_faces_in_bgr_image(image_bgr, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), blur_kernel=(51, 51)):
    output = image_bgr.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    for (x, y, w, h) in faces:
        roi = output[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        kx = blur_kernel[0] if blur_kernel[0] % 2 == 1 else blur_kernel[0] + 1
        ky = blur_kernel[1] if blur_kernel[1] % 2 == 1 else blur_kernel[1] + 1
        output[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (kx, ky), 0)
    return output, len(faces)

def draw_predictions(result, image_bgr):
    output = image_bgr.copy()
    parsed = []
    if result.boxes is None or len(result.boxes) == 0:
        return output, parsed
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = result.names.get(cls_id, str(cls_id)) if isinstance(result.names, dict) else result.names[cls_id]
        color = (0, 200, 0)
        if "no_" in label or label == "none":
            color = (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        parsed.append({"class_name": label, "confidence": conf, "box": [x1, y1, x2, y2]})
    return output, parsed

weights_path = st.text_input("YOLO weights path", value=str(DEFAULT_WEIGHTS))
conf_threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
iou_threshold = st.slider("IoU threshold", min_value=0.05, max_value=0.95, value=0.45, step=0.05)
blur_faces = st.checkbox("Blur faces for privacy", value=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    detector = load_detector(weights_path)
    result = detector.predict(source=image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
    annotated_bgr, predictions = draw_predictions(result, image_bgr)

    faces_blurred = 0
    if blur_faces:
        annotated_bgr, faces_blurred = blur_faces_in_bgr_image(annotated_bgr)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(annotated_rgb, caption=f"Detections complete | Faces blurred: {faces_blurred}", use_container_width=True)
    with col2:
        st.subheader("Predictions")
        if predictions:
            st.dataframe(pd.DataFrame(predictions), use_container_width=True)
        else:
            st.info("No detections above the selected threshold.")

        unsafe_hits = [p for p in predictions if "no_" in p["class_name"] or p["class_name"] == "none"]
        if unsafe_hits:
            st.error("Violation classes detected.")
        else:
            st.success("No violation classes detected above threshold.")\n
