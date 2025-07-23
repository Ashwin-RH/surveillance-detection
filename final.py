import os
import streamlit as st
import cv2
import numpy as np
import joblib
import sys
import random


st.write("Python version:", sys.version)


from huggingface_hub import hf_hub_download

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Test OpenCV
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ImportError:
    print("OpenCV is not installed.")

# Set page configuration
st.set_page_config(page_title="Surveillance Detection")

# 💅 Custom styling and header



st.markdown("<h1 style='text-align: center;'>Surveillance Detection</h1>", unsafe_allow_html=True)
st.markdown("**Upload a video file and choose a model to detect activities in the video.**")


# ✅ Session state initialization
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

# 🧹 Reset logic
if st.session_state.reset_trigger:
    st.session_state.reset_trigger = False
    st.stop()

# 📤 File upload
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="uploaded_video")
if uploaded_file is None:
    st.info("Please upload a video to begin analysis.")




# 👇 Hugging Face model paths (cached after 1st download)
model_paths = {
    'Random forest': hf_hub_download(repo_id="Ashwinharagi/surveillance-randomforest", filename="model2.joblib"),
    'XGBoost': hf_hub_download(repo_id="Ashwinharagi/surveillance-xgboost", filename="model7.joblib"),
    'MLP Multilayer Perceptron': hf_hub_download(repo_id="Ashwinharagi/surveillance-Multilayer-Perceptron-mlp", filename="clf9.joblib"),
    'LightGBM': hf_hub_download(repo_id="Ashwinharagi/surveillance-lightgbm", filename="model8.joblib"),
}
 
    # 'Extra tree classifier': "D:/saved_ml_model/clf10.joblib",
    # 'Convolution Neural Network': "D:/sic/saved_dl_model/cnn_model.joblib",
    # 'Recurrent Neural Network': "D:/sic/saved_dl_model/rnn_model.joblib",
    # 'Feedforward Neural Network': "D:/sic/saved_dl_model/fnn_model.joblib"

selected_model = st.selectbox("Choose Model:", list(model_paths.keys()))

# 🧾 Class mapping
class_mapping = {
    0: "Abuse", 1: "Arrest", 2: "Arson", 3: "Assault", 4: "Burglary",
    5: "Fighting", 6: "Shooting", 7: "Shoplifting", 8: "Stealing",
    9: "Vandalism", 10: "Robbery", 11: "Roadaccident", 12: "Normal", 13: "Explosion"
}

# 📹 Video processing functions
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

def preprocess_video(video_path):
    histograms = extract_color_histograms(video_path)
    return np.mean(histograms, axis=0)

def make_predictions(model, feature_vector):
    prediction = model.predict([feature_vector])
    return class_mapping[prediction[0]]

def get_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None
    random_frame_index = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps == 0:
        return 0, 0
    total_sec = frame_count / fps
    return int(total_sec // 60), int(total_sec % 60)

# ✅ Main logic
if uploaded_file is not None:
    #st.video(uploaded_file)

    # Save video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Show video duration
    minutes, seconds = get_video_duration("temp_video.mp4")
    st.write(f"Video Duration: {minutes} minutes {seconds} seconds")

    # Show random frame
    frame = get_random_frame("temp_video.mp4")
    if frame is not None:
        st.image(frame, caption="Thumbnail from the uploaded video", use_container_width=True)
    else:
        st.warning("Could not extract a frame from the video.")

    # 🟢 Start Analysis Button
    if st.button("Start Analysis"):
        with st.spinner('Analyzing video and extracting features...'):
            feature_vector = preprocess_video("temp_video.mp4")

        # Load model
        model_path = model_paths[selected_model]
        try:
            if 'Neural Network' in selected_model:
                import dill
                import keras
            model = joblib.load(model_path)
        except ImportError as e:
            missing = str(e).split(' ')[-1]
            st.error(f"The '{missing}' module is not installed. Run: `pip install {missing}`")
        except Exception as e:
            st.error(f"Model loading failed: {e}")
        else:
            try:
                prediction = make_predictions(model, feature_vector)
                st.success(f'Prediction: **{prediction}**')
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# 🔁 Reset Button
if st.button("Reset"):
    # Clear session state keys manually (uploaded video + any temp file)
    if "uploaded_video" in st.session_state:
        del st.session_state["uploaded_video"]
    st.session_state.reset_trigger = False
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")
    st.rerun()
    
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: #EEAECA;
    background: radial-gradient(circle, rgba(238, 174, 202, 1) 0%, rgba(148, 187, 233, 1) 100%);
    color: black;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    border-top: 1px solid #e0e0e0;
}
.footer a {
    color: #1f77b4;
    text-decoration: none;
    font-weight: 500;
}
</style>

<div class="footer">
    Made with ❤️ by Ashwin Haragi, Likith M, Pavan Kumar V Kulkarni</a>
</div>
""", unsafe_allow_html=True)


