import cv2
import streamlit as st
import time
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a function to detect and save a person's video
def detect_person(cam_index):
    cap = cv2.VideoCapture(cam_index)
    # Define the codec and create a VideoWriter object to write the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    timestamp = int(time.time())
    video_name = f'person_detected_{timestamp}.mp4'
    out = cv2.VideoWriter(f'pic/{video_name}', fourcc, 20.0, (640, 480))
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Write the frame to the video file
        out.write(frame)
        cv2.imshow('frame', frame)
        # Wait for the 'q' key to be pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam, release the video writer, and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Create a Streamlit app
st.title("Person Detection with Webcam")
st.write("Press the button below to start detecting people with your webcam.")

# Add a selectbox to choose one of three different cameras
cam_index = st.selectbox("Select a camera:", [0, 1, 2])

if st.button("Detect Person"):
    detect_person(cam_index)
    # Generate a timestamp name for the video
    timestamp = int(time.time())
    video_name = f'person_detected_{timestamp}.mp4'
    # Display a success message and the video with the timestamp in the filename
    st.success(f"Person detected and video saved as {video_name} in 'pic' folder!")
    st.video(f"pic/{video_name}")

