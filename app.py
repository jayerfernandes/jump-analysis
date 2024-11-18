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

    frames_dir = tempfile.mkdtemp()
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

        # Save processed frame to a temporary directory
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Use FFmpeg to compile frames into a video
    output_video_path = "output_with_overlays_and_counts.mp4"
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-framerate", "30",  # Set the frame rate
        "-i", os.path.join(frames_dir, "frame_%04d.png"),  # Input frame pattern
        "-c:v", "libx264",  # Use H.264 codec for compatibility
        "-pix_fmt", "yuv420p",  # Ensure compatibility with most players
        output_video_path,
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Display the output video
    st.write("### Processed Video with Visual Overlays and Jump Classification")
    st.video(output_video_path)

    # Clean up temporary files
    os.remove(video_path)
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
