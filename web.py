import cv2
import streamlit as st
import numpy as np



# Set up OpenCV video capture object
cap = cv2.VideoCapture(0)
st.title("Webcam Live Stream")

video_buffer = st.empty()

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_buffer.image(frame)


    else:
        break

cap.release()
st.balloons()
