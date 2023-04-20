import cv2
import streamlit as st
import time
import os



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
plate_cascade = cv2.CascadeClassifier('haarcascade_license_plate_rus_16stages.xml')
full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')





def detect_object(cascade_file, camera_source):
    cap = cv2.VideoCapture(camera_source)
    
    
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    timestamp = int(time.time())
    video_name = f'{object_to_detect.lower().replace(" ", "_")}_detected_{timestamp}.mp4'
    out = cv2.VideoWriter(f'pic/{video_name}', fourcc, 20.0, (640, 480))
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = cascade_file.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            
            
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()





st.title("Smart Survillance")
st.write("Select the object and camera source you want to detect with your webcam.")



object_to_detect = st.selectbox("Object to detect", ("Person (full body)", "Person (lower body)", "Face", "Car license plate"))



camera_source = st.selectbox("Select camera", (0, 1, 2))

if object_to_detect == "Person (full body)":
    cascade_file = full_body_cascade
elif object_to_detect == "Person (Upper body)":
    cascade_file = upper_body_cascade
elif object_to_detect == "Face":
    cascade_file = face_cascade
else:
    cascade_file = plate_cascade

if st.button("Detect Object"):
    detect_object(cascade_file, camera_source)
    timestamp = int(time.time())
    video_name = f'{object_to_detect.lower().replace(" ", "_")}_detected_{timestamp}.mp4'
    st.success(f"{object_to_detect} detected and video saved as {video_name} in 'pic' folder!")
    st.video(f"pic/{video_name}")

