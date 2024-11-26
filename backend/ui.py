import cv2
import numpy as np
import streamlit as st
from classify import Classifier
from detect import Detector
from PIL import Image

st.set_page_config(page_title="Military Aircraft Detector", layout="wide")


@st.cache_resource
def load_models():
    detector = Detector("models/yolo-detect-identify.pt")
    classifier = Classifier("models/custom-classification.keras")
    return detector, classifier


detector, classifier = load_models()


st.title("🛩️ Military Aircraft Detection & Classification")
st.markdown(
    """
This application detects and identifies military aircraft in images. 
1. First, it locates aircraft using YOLO object detection
2. Then, it classifies each detected aircraft using a CNN model
"""
)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    img_array = np.array(image)

    detections = detector.detect(img_array)

    if not detections:
        st.warning("No aircraft detected in the image!")
    else:
        with col2:
            st.subheader("Detection Results")

            img_with_boxes = img_array.copy()

            for i, detection in enumerate(detections, 1):

                point1 = (detection.x1, detection.y1)
                point2 = (detection.x2, detection.y2)

                cv2.rectangle(
                    img_with_boxes,
                    point1,
                    point2,
                    (0, 255, 0),
                    2,
                )

                cropped = img_array[
                    detection.y1 : detection.y2, detection.x1 : detection.x2
                ]

                resized = cv2.resize(cropped, (256, 256))
                normalized = resized / 255.0
                expanded = np.expand_dims(normalized, axis=0)

                class_id, confidence = classifier.classify(expanded)

                label = f"Aircraft {i}: Class {class_id} ({confidence:.2f})"
                cv2.putText(
                    img_with_boxes,
                    label,
                    (detection.x1, detection.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            st.image(img_with_boxes, use_column_width=True)

            st.subheader("Detection Summary")
            st.info(f"Found {len(detections)} aircraft in the image")


st.markdown(
    """
---
Created with ❤️ using YOLO and CNN models
"""
)
