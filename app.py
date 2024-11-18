import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# App Title
st.title("Jump Analysis App")
st.write("Upload a video to analyze jumps dynamically.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

if uploaded_file:
    # Save uploaded video to a temporary file
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Video uploaded successfully!")

    # Load the video
    cap = cv2.VideoCapture("uploaded_video.mp4")

    # Extract Y-coordinates of the right hip
    hip_y_positions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            hip_y_positions.append(right_hip.y)
        frame_count += 1
    cap.release()

    # Plot the Y-coordinates
    st.write("### Hip Y-Position Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hip_y_positions, label="Right Hip Y-Position")
    ax.set_title("Right Hip Y-Position Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y-Coordinate")
    ax.legend()
    st.pyplot(fig)

    st.success(f"Video processed! Total frames analyzed: {frame_count}")
else:
    st.warning("Please upload a video.")

