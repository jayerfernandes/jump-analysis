import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import subprocess
from scipy.signal import find_peaks

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# App Title
st.title("Jump Analysis with Visual Overlays and Counting")
st.write("Upload a video to see pose keypoints, connections visualized, and jump classification.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

# Parameters for classification
big_jump_threshold = 0.07  # Adjust this based on the video
small_jump_threshold = 0.03

if uploaded_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.success("Video uploaded successfully!")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Prepare output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_path = "output_with_overlays_and_counts.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Y-coordinate data
    hip_y_positions = []
    frame_count = 0

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Track Y-coordinate of right hip
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            hip_y_positions.append(right_hip.y)

        # Write the frame to the output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Classify and count jumps
    hip_y_positions = np.array(hip_y_positions)
    peaks, properties = find_peaks(-hip_y_positions, prominence=small_jump_threshold)

    # Calculate displacements
    displacements = properties['prominences']
    big_jumps = [i for i, d in enumerate(displacements) if d >= big_jump_threshold]
    small_jumps = [i for i, d in enumerate(displacements) if d < big_jump_threshold]

    big_jump_count = len(big_jumps)
    small_jump_count = len(small_jumps)

    # Display jump counts
    st.write(f"### Jump Counts")
    st.write(f"Big Jumps: {big_jump_count}")
    st.write(f"Pogos (Small Jumps): {small_jump_count}")

    # Display the processed video
    st.write("### Processed Video with Visual Overlays and Jump Classification")
    st.video(out_path)

    # Clean up the uploaded temporary file
    os.remove(video_path)
else:
    st.warning("Please upload a video.")
