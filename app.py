from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="PPE Compliance Detector", layout="wide")
st.title("PPE Compliance Detector")
st.write("YOLO detects PPE classes and displays compliance status.")

DEFAULT_WEIGHTS = Path("best.pt")
REQUIRED_DEFAULT = ["Helmet", "Vest", "Boots"]

POSITIVE_PPE = {"Helmet", "Gloves", "Vest", "Boots", "Goggles"}
NEGATIVE_TO_POSITIVE = {
    "no_helmet": "Helmet",
    "no_goggle": "Goggles",
    "no_gloves": "Gloves",
    "no_boots": "Boots",
}
ALL_REQUIRED_OPTIONS = ["Helmet", "Vest", "Boots", "Gloves", "Goggles"]


@st.cache_resource
def load_detector(weights_path):
    return YOLO(str(weights_path))


def normalize_label(label):
    text = str(label).strip()
    lower = text.lower()
    if lower in {"helmet", "gloves", "vest", "boots", "goggles", "person", "none"}:
        return lower.title()
    if lower in NEGATIVE_TO_POSITIVE:
        return lower
    return text


def parse_detections(result, conf_threshold):
    rows = []
    if result.boxes is None or len(result.boxes) == 0:
        return rows

    for box in result.boxes:
        conf = float(box.conf[0].item())
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0].item())
        raw = result.names.get(cls_id, str(cls_id)) if isinstance(result.names, dict) else result.names[cls_id]
        label = normalize_label(raw)

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        rows.append({
            "class": label,
            "confidence": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    return rows


def check_compliance(detections, required_ppe, use_negative_classes=True):
    detected = {d["class"] for d in detections}

    present_ppe = sorted([name for name in detected if name in POSITIVE_PPE])
    missing_required = [ppe for ppe in required_ppe if ppe not in detected]

    violations = []
    if use_negative_classes:
        for neg, pos in NEGATIVE_TO_POSITIVE.items():
            if neg in detected and pos not in detected:
                violations.append(neg)

    status = "COMPLIANT" if not missing_required and not violations else "NON-COMPLIANT"

    return status, present_ppe, missing_required, sorted(violations)


# Sidebar controls
weights_path = st.sidebar.text_input("YOLO weights path", value=str(DEFAULT_WEIGHTS))
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold", 0.05, 0.95, 0.45, 0.05)
required_ppe = st.sidebar.multiselect("Required PPE", ALL_REQUIRED_OPTIONS, default=REQUIRED_DEFAULT)
use_negative_classes = st.sidebar.checkbox("Use no_* violation classes", value=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    detector = load_detector(weights_path)

    result = detector.predict(
        source=image_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )[0]

    detections = parse_detections(result, conf_threshold)

    status, present_ppe, missing_required, violations = check_compliance(
        detections, required_ppe, use_negative_classes
    )

    # ✅ FIXED IMAGE (no blue tint, no stretch, smaller labels)
    annotated_bgr = result.plot(line_width=1, font_size=0.5)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(annotated_rgb, width=700)

    with col2:
        if status == "COMPLIANT":
            st.success("Status: COMPLIANT")
        else:
            st.error("Status: NON-COMPLIANT")

        st.write("Present PPE:", ", ".join(present_ppe) if present_ppe else "None")
        st.write("Missing required PPE:", ", ".join(missing_required) if missing_required else "None")
        st.write("Detected violations:", ", ".join(violations) if violations else "None")

        st.subheader("YOLO detections")
        if detections:
            st.dataframe(pd.DataFrame(detections), use_container_width=True)
        else:
            st.info("No detections above threshold.")

else:
    st.info("Upload an image to run PPE compliance detection.")
