import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

# Set up OpenCV video capture object
cap = cv2.VideoCapture(0)

st.title("YOLOv8 Webcam Live Stream")
st.sidebar.title("Select Webcam")

# Add webcam selection option in the sidebar
cam_option = st.sidebar.selectbox("Select Webcam", ("Webcam 0", "Webcam 1", "Webcam 2"))

# Set the video capture index based on the selected webcam option
if cam_option == "Webcam 0":
    cap = cv2.VideoCapture(0)
elif cam_option == "Webcam 1":
    cap = cv2.VideoCapture(1)
elif cam_option == "Webcam 2":
    cap = cv2.VideoCapture(2)

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Create a video buffer to display the live stream
video_buffer = st.empty()

while True:
    # Read frame from video capture object
    ret, frame = cap.read()

    if ret:
        # Convert color space from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection using YOLOv8
        results = model(frame)

        # Draw bounding boxes on the detected objects
        for box in results.xyxy[0]:
            x1, y1, x2, y2, _ = box.tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Display the frame with bounding boxes
        video_buffer.image(frame)

    else:
        break

# Release the video capture object
cap.release()

st.balloons()
