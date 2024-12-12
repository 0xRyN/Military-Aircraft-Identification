from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from classify import Classifier
from consts import CLASSES
from detect import Detection, Detector
from PIL import Image

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
PAGE_TITLE = "Military Aircraft Detector"
APP_TITLE = "🛩️ Military Aircraft Detection & Classification"
APP_DESCRIPTION = """
This application detects and identifies military aircraft in images. 
1. First, it locates aircraft using YOLO object detection
2. Then, it classifies each detected aircraft using a CNN model
"""


@dataclass
class ModelPaths:
    detector: str = "models/yolo-detect-identify.pt"
    classifier: str = "models/custom-classification.keras"


class AircraftDetectorUI:
    """Handles the UI logic for aircraft detection application."""

    @staticmethod
    @st.cache_resource
    def _load_models(
        detector_path: str, classifier_path: str
    ) -> Tuple[Detector, Classifier]:
        """Load detection and classification models."""
        try:
            detector = Detector(detector_path)
            classifier = Classifier(classifier_path)
            return detector, classifier
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()

    def __init__(self, model_paths: ModelPaths):
        """Initialize the UI with model paths."""
        self.model_paths = model_paths
        self.setup_page()
        self.load_models()

    def setup_page(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(page_title=PAGE_TITLE, layout="wide")
        st.title(APP_TITLE)
        st.markdown(APP_DESCRIPTION)

    def load_models(self) -> None:
        """Initialize model instances."""
        self.detector, self.classifier = self._load_models(
            self.model_paths.detector, self.model_paths.classifier
        )

    def show_upload_section(self) -> Optional[Image.Image]:
        """Display and handle the image upload section."""
        return st.file_uploader("Choose an image...", type=SUPPORTED_FORMATS)

    def process_image(self, uploaded_file: Image.Image) -> None:
        """Process the uploaded image and display results."""
        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            if "detections" not in st.session_state:
                st.session_state.detections = []
            if "classifications" not in st.session_state:
                st.session_state.classifications = {}

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            detections = self.detector.detect(img_array)
            st.session_state.detections = detections

            img_with_boxes = img_array.copy()
            for idx, box in enumerate(detections):
                cv2.rectangle(
                    img_with_boxes,
                    (int(box.x1), int(box.y1)),
                    (int(box.x2), int(box.y2)),
                    (0, 255, 0),
                    2,
                )

                label = f"ID {idx+1}"
                cv2.putText(
                    img_with_boxes,
                    label,
                    (int(box.x1), int(box.y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            with col2:
                st.subheader("Detected Aircraft")
                st.image(img_with_boxes, use_column_width=True)

            st.subheader("Detection Summary")
            st.info(f"Found {len(detections)} aircraft in the image")

            if len(detections) > 0:
                classify_button = st.button("Classify Aircraft")
                if classify_button:
                    self.classify_detections(img_array, detections)

    def classify_detections(
        self, img_array: np.ndarray, detections: List[Detection]
    ) -> None:
        """Classify each detected aircraft and display results."""
        img_with_classes = img_array.copy()
        classifications = {}

        for idx, box in enumerate(detections):

            cropped_img = img_array[
                int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)
            ]
            if cropped_img.size == 0:
                classifications[idx] = "Unknown"
                continue

            resized_img = cv2.resize(cropped_img, (256, 256))
            expanded = np.expand_dims(resized_img, axis=0).astype(np.float32)
            class_id, confidence = self.classifier.classify(expanded)
            st.text(f"Class ID: {class_id}, Confidence: {confidence}")
            aircraft_class = CLASSES[class_id]

            cv2.putText(
                img_with_classes,
                str(aircraft_class + "[{:.2f}]".format(confidence)),
                (int(box.x1), int(box.y2) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        st.session_state.classifications = classifications

        st.subheader("Classified Aircraft")
        st.image(img_with_classes, use_column_width=True)

        st.success("Classification completed!")
        for idx, cls in classifications.items():
            st.write(f"**Aircraft {idx+1}:** {cls}")


def main():
    """Main application entry point."""
    model_paths = ModelPaths()
    app = AircraftDetectorUI(model_paths)

    uploaded_file = app.show_upload_section()
    if uploaded_file:
        app.process_image(uploaded_file)

    st.markdown(
        """
        ---
        Created with ❤️ using YOLO and CNN models
        """
    )


if __name__ == "__main__":
    main()
