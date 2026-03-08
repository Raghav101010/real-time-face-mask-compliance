import streamlit as st
import cv2
import numpy as np
import av

from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("Real-Time Face Mask Detection")

@st.cache_resource
def load_models():
    face_detector = MTCNN()
    mask_model = load_model("face_mask_detector1.h5", compile=False)
    return face_detector, mask_model

face_detector, mask_model = load_models()


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB for MTCNN
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_detector.detect_faces(rgb)

        for face in faces:

            x, y, w, h = face['box']

            x = max(0, x)
            y = max(0, y)

            face_img = img[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (150,150))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            pred = mask_model.predict(face_img, verbose=0)[0][0]

            if pred > 0.5:
                label = "Mask"
                color = (0,255,0)
            else:
                label = "No Mask"
                color = (0,0,255)

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

            cv2.putText(
                img,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="mask-detection",
    video_processor_factory=VideoProcessor
)