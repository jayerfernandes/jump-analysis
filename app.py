import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
from scipy.signal import find_peaks
import subprocess

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# App Title
st.title("Jump Analysis with Visual Overlays and Jump Classification")
st.write("Upload a video to see pose keypoints, connections, and jump analysis.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

if uploaded_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.success("Video uploaded successfully!")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Prepare to save frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary directory for frames
    frames_dir = tempfile.mkdtemp()

    # Variables for jump classification and counting
    hip_y_positions = []
    frame_count = 0
    big_jumps = []
    small_jumps = []

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
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Track Y-coordinate of the left hip (or right hip)
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            hip_y_positions.append(left_hip.y)  # Y-coordinate of the left hip

        # Save the frame to the temporary directory
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Perform peak detection on hip Y-positions to detect jumps
    hip_y_positions = np.array(hip_y_positions)
    peaks, properties = find_peaks(-hip_y_positions, prominence=0.01)  # Find peaks (representing jumps)
    
    # Classify jumps based on prominence
    displacements = properties['prominences']
    big_jump_threshold = 0.07  # Threshold for big jumps
    big_jumps = [i for i, d in enumerate(displacements) if d >= big_jump_threshold]
    small_jumps = [i for i, d in enumerate(displacements) if d < big_jump_threshold]

    # Display the results
    total_jumps = len(peaks)
    big_jump_count = len(big_jumps)
    small_jump_count = len(small_jumps)

    # Show jump counts on the Streamlit UI
    st.write(f"Total Jumps: {total_jumps}")
    st.write(f"Big Jumps: {big_jump_count}")
    st.write(f"Small Jumps: {small_jump_count}")

    # Use FFmpeg to compile frames into a video with annotations
    output_video_path = "output_with_overlays.mp4"  # Using .mp4 format
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Process the video frames again to add jump counter text to each frame
    frame_count = 0
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)

        # Create a black bar at the top of the frame
        bar_height = 100  # Height of the black bar (you can adjust this)
        frame[:bar_height, :] = (0, 0, 0)  # Set the top part of the frame to black

        # Adding text inside the black bar (larger font, visible)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2  # Larger font size
        font_thickness = 4  # Thicker font
        color = (255, 255, 255)  # White text on black background
        line_type = cv2.LINE_AA

        # Add text for total jumps, big jumps, small jumps
        cv2.putText(frame, f'Total Jumps: {total_jumps}', (50, 50), font, font_scale, color, font_thickness, lineType=line_type)
        cv2.putText(frame, f'Big Jumps: {big_jump_count}', (50, 100), font, font_scale, color, font_thickness, lineType=line_type)
        cv2.putText(frame, f'Pogos: {small_jump_count}', (50, 150), font, font_scale, color, font_thickness, lineType=line_type)

        # Write the processed frame with overlays to the output video
        out.write(frame)

    out.release()

    # Run FFmpeg to create the video
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

    # Display the processed video with overlays
    st.write("### Processed Video with Visual Overlays and Jump Classification")
    st.video(output_video_path)

    # Clean up the uploaded temporary file and frames
    os.remove(video_path)
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
