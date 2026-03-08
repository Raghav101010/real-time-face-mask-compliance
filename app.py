import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import gdown
import os

st.title("Real-Time Face Mask Detection with Live Analytics")

MODEL_PATH = "face_mask_detector1.h5"
MODEL_URL = "https://drive.google.com/uc?id=1LmR-juV4KXUmJvuDu5hxB8K0oDdMJukF"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_models():
    detector = MTCNN()
    mask_model = load_model(MODEL_PATH, compile=False)
    return detector, mask_model

detector, mask_model = load_models()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Live analytics counters
        self.mask_count = 0
        self.no_mask_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        faces = detector.detect_faces(img)

        # Reset counters for current frame
        self.mask_count = 0
        self.no_mask_count = 0

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = img[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            face_resized = cv2.resize(face_img, (150,150))
            face_resized = face_resized / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            pred = mask_model.predict(face_resized, verbose=0)[0][0]

            if pred > 0.5:
                label = "Mask"
                color = (0,255,0)
                self.mask_count += 1
            else:
                label = "No Mask"
                color = (0,0,255)
                self.no_mask_count += 1

            # Draw bounding box & label
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        # Overlay live analytics
        analytics_text = f"Mask: {self.mask_count} | No Mask: {self.no_mask_count}"
        cv2.putText(img, analytics_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="mask-detection",
    video_processor_factory=VideoProcessor
)