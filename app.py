
import streamlit as st

# App Title
st.title("Jump Analysis App")
st.write("Upload a video to analyze jumps dynamically.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

if uploaded_file:
    st.write("Video uploaded successfully!")
    st.write(f"Filename: {uploaded_file.name}")
else:
    st.write("Please upload a video.")
