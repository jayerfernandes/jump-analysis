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

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Track Y-coordinate of the left hip
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            hip_y_positions.append(left_hip.y)

        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Perform jump detection
    hip_y_positions = np.array(hip_y_positions)
    peaks, properties = find_peaks(-hip_y_positions, prominence=0.01)
    
    # Classify jumps
    displacements = properties['prominences']
    big_jump_threshold = 0.07
    big_jumps = [i for i, d in enumerate(displacements) if d >= big_jump_threshold]
    small_jumps = [i for i, d in enumerate(displacements) if d < big_jump_threshold]

    total_jumps = len(peaks)
    big_jump_count = len(big_jumps)
    small_jump_count = len(small_jumps)

    # Display stats in Streamlit
    st.write(f"Total Jumps: {total_jumps}")
    st.write(f"Big Jumps: {big_jump_count}")
    st.write(f"Small Jumps: {small_jump_count}")

    # Create output video with text overlays using FFmpeg
    output_video_path = "output_with_overlays.mp4"

    # Process saved frames again to add text overlays
    for i in range(frame_count):  # Use the actual frame_count from earlier
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)
        
        if frame is not None:  # Check if frame was read successfully
            # Add text overlays with adjusted positioning and style
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            color = (0, 255, 0)  # Green
            
            # Add black background for better text visibility
            def put_text_with_background(img, text, position):
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                # Draw black background rectangle
                cv2.rectangle(img, 
                            (position[0] - 10, position[1] - text_height - 10),
                            (position[0] + text_width + 10, position[1] + 10),
                            (0, 0, 0),
                            -1)
                # Draw text
                cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Add text with backgrounds
            put_text_with_background(frame, f'Big Jumps: {big_jump_count}', (50, 50))
            put_text_with_background(frame, f'Pogos: {small_jump_count}', (50, 100))
            put_text_with_background(frame, f'Total Jumps: {total_jumps}', (50, 150))

            # Save the frame with overlays
            cv2.imwrite(frame_path, frame)

    # Use FFmpeg to compile frames into video
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",  # Added for faster encoding
        output_video_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Display the processed video
    st.write("### Processed Video with Visual Overlays and Jump Classification")
    st.video(output_video_path)

    # Cleanup
    os.remove(video_path)
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
