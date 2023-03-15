import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

# Set up OpenCV video capture object
cap = cv2.VideoCapture(0)

# Set up YOLO model for object detection
model = YOLO("yolov8s.pt")

# Set up Streamlit app
st.title("Webcam Live Stream")

video_buffer = st.empty()

while True:
    ret, frame = cap.read()

    if ret:
        # Perform object detection on the captured frame using YOLO model
        result = model(frame)

        # Display the annotated frame in Streamlit
        video_buffer.image(result.show(), channels="BGR")

    else:
        break

cap.release()
st.balloons()
