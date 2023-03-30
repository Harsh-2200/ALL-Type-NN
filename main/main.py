import cv2
import streamlit as st
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a function to detect and save a person's image
def detect_person():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Generate a timestamp name for the image
            timestamp = int(time.time())
            image_name = f'person_detected_{timestamp}.jpg'
            # Save the frame as an image with the timestamp name
            cv2.imwrite(image_name, frame)
            # Display a success message
            st.success(f"Person detected and image saved as {image_name}!")
        cv2.imshow('frame', frame)
        # Wait for the 'q' key to be pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Create a Streamlit app
st.title("Person Detection with Webcam")
st.write("Press the button below to start detecting people with your webcam.")

if st.button("Detect Person"):
    detect_person()