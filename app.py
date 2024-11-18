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

    # Create a temporary directory for frames
    frames_dir = tempfile.mkdtemp()
    frame_count = 0

    hip_y_positions = []

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

        # Save processed frame to the temporary directory
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

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

    # Reopen the video to add jump classification overlays
    cap = cv2.VideoCapture(video_path)
    out_path = "output_with_overlays_and_counts.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Process the video with jump classification and overlays
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for overlay processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Display jump classifications on the video
        for i, peak in enumerate(peaks):
            y_position = hip_y_positions[peak]
            if i in big_jumps:
                cv2.putText(frame, "Big Jump", (50, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Pogo", (50, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video with overlays
        out.write(frame)

    cap.release()
    out.release()

    # Display the processed video with overlays and jump classification
    st.write("### Processed Video with Visual Overlays and Jump Classification")
    st.video(out_path)

    # Clean up the uploaded temporary file
    os.remove(video_path)
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
