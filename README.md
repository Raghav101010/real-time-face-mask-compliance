# Real-Time Face Mask Detector

This is a **real-time face mask detection app** built using:

- **MTCNN** for face detection
- **TensorFlow/Keras** for mask/no mask classification
- **Streamlit + Streamlit-webrtc** for live video processing

The model is downloaded dynamically from Google Drive using `gdown`, so the repository stays lightweight.

## Features

- Real-time face detection from webcam
- Mask/No Mask classification with bounding boxes
- Live analytics sidebar showing detection statistics
- Lightweight, production-ready, and deployable on Hugging Face Spaces

## How to Run Locally

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd face-mask-detector-app