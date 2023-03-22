import streamlit as st
import cv2 
import numpy as np
import tempfile
from pathlib import Path

st.title('Video Enhancement')

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:
    # Create a temporary file to save the video
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Read the video
    cap = cv2.VideoCapture(temp_file.name)
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer to save the enhanced video
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = Path('enhanced_video.mp4')
    out = cv2.VideoWriter(str(out_file), codec, fps, (width, height))

    # Apply image enhancement to each frame and write it to the output video
    super_res = cv2.dnn_superres.DnnSuperResImpl_create()
    super_res.readModel('LapSRN_x4.pb')
    super_res.setModel('lapsrn', 4)

    st.text("Processing video...")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (height // 2, width // 2))

        # Apply image enhancement
        enhanced_frame = super_res.upsample(frame)

        # Write the enhanced frame to the output video
        out.write(enhanced_frame)

    st.text("Video processing complete!")

    # Download the enhanced video
    st.download_button(
        label="Download Enhanced Video",
        data=open(str(out_file), 'rb').read(),
        file_name="enhanced_video.mp4"
    )

    # Release the video resources
    cap.release()
    out.release()

else:
    st.write("Please upload a video to enhance.")
