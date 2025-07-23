import os
import streamlit as st
import cv2
import numpy as np
import joblib
import sys
import random
import time

# Disable oneDNN custom operations to avoid floating-point round-off logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Test OpenCV
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ImportError:
    print("OpenCV is not installed.")

# Check Python environment
print("Python executable:", sys.executable)

# Define a dictionary mapping class numbers to class names
class_mapping = {
    0: "Abuse",
    1: "Arrest",
    2: "Arson",
    3: "Assault",
    4: "Burglary",
    5: "Fighting",
    6: "Shooting",
    7: "Shoplifting",
    8: "Stealing",
    9: "Vandalism",
   10: "Robbery",
   11: "Roadaccident",
   12: "Normal",
   13: "Explosion"
}

# Function to extract color histograms from a video
def extract_color_histograms(video_path):
    cap = cv2.VideoCapture(video_path)
    histograms = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
        progress_bar.progress((i + 1) / frame_count)
    cap.release()
    return np.array(histograms)

# Function to preprocess video data and extract features
def preprocess_video(video_path):
    histograms = extract_color_histograms(video_path)
    # Aggregate histograms by taking the mean
    aggregated_histogram = np.mean(histograms, axis=0)
    return aggregated_histogram

# Function to make predictions
def make_predictions(model, feature_vector):
    # Make prediction using the selected model
    prediction = model.predict([feature_vector])
    
    # Convert class number to class name
    class_name = class_mapping[prediction[0]]
    return class_name

# Function to get a random frame from the video
def get_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_index = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame

# Function to get video duration
def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration_seconds = frame_count / fps
    duration_minutes = int(duration_seconds // 60)
    duration_seconds = int(duration_seconds % 60)
    cap.release()
    return duration_minutes, duration_seconds

# Set page configuration
st.set_page_config(page_title="Surveillance Detection")

# Center heading
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Surveillance Detection</h1>", unsafe_allow_html=True)

# Add description
st.markdown("**Upload a video file and choose a model to detect activities in the video.**")

# Create file upload widget
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Create dropdown widget to choose model
model_paths = {
    'Random forest': "D:/saved_ml_model/model2.joblib",
    'XGBoost': "D:/saved_ml_model/model7.joblib",
    'MLP Multilayer Perceptron': "D:/saved_ml_model/clf9.joblib",
    'Extra tree classifier': "D:/saved_ml_model/clf10.joblib",
    'LightGBM': "D:/saved_ml_model/model8.joblib",
    'Convolution Neural Network': "D:/updated mods/cnn.joblib",
    'Recurrent Neural Network': "D:/sic/saved_dl_model/rnn_model.joblib",
    'Feedforward Neural Network': "D:/sic/saved_dl_model/fnn_model.joblib"
}
selected_model = st.selectbox("Choose Model:", list(model_paths.keys()))

# Flag to track if predictions have been made
predictions_made = False

if uploaded_file is not None and not predictions_made:
    predictions_made = True  # Set the flag to True after making predictions

    # Save the uploaded video file to a temporary location
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Get video duration
    duration_minutes, duration_seconds = get_video_duration("temp_video.mp4")

    # Display video duration
    st.write(f"Video Duration: {duration_minutes} minutes {duration_seconds} seconds")

    # Get a random frame from the video
    random_frame = get_random_frame("temp_video.mp4")

    # Display the random frame as a thumbnail
    st.image(random_frame, caption='Thumbnail from the uploaded video', use_column_width=True)

    # Preprocess video and extract features (using color histograms)
    feature_vector = preprocess_video("temp_video.mp4")

    # Load the selected model based on the user's choice
    model_path = model_paths[selected_model]
    try:
        if 'Neural Network' in selected_model:
            import dill
            import keras
        model = joblib.load(model_path)
    except ImportError as e:
        missing_module = str(e).split(' ')[-1]
        st.error(f"The '{missing_module}' module is not installed. Please install it using 'pip install {missing_module}'.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
    else:
        # Make predictions using the selected model and feature vector
        try:
            prediction = make_predictions(model, feature_vector)
            # Display prediction
            st.success(f'Prediction: **{prediction}**')
        except Exception as e:
            st.error(f"An error occurred while making predictions: {e}")
