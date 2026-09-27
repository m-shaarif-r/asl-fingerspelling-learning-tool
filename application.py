import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# -------------------- LAZY LOAD MODEL + DETECTOR --------------------
@st.cache_resource
def load_resources():
    from inference_post_training import detector, device, labels, model, offset
    return model, detector, device, labels, offset


# -------------------- UI SETUP --------------------
st.set_page_config(page_title="ASL (Sign Language) Learning Tool")
st.title("ASL (Sign Language) Learning Tool")
st.write("Obtain real-time feedback by practicing your ASL skills, right in your browser!")

with st.expander("How this works", expanded=False):
    st.write(
        "Your webcam video is processed to run the model but is not stored. "
        "Each frame is passed through a MediaPipe hand landmarker to find your "
        "hand, then a small PyTorch CNN classifies the letter you're signing. "
        "Simple geometric checks on the hand landmarks add specific coaching "
        "tips (e.g. 'tuck your thumb in')."
    )

# -------------------- TARGET SIGN SELECTION --------------------
if "target_label" not in st.session_state:
    st.session_state.target_label = "A"

st.markdown("### Choose a sign to practice:")

cols = st.columns(5)
for col, letter in zip(cols, ["A", "B", "C", "L", "W"]):
    with col:
        if st.button(letter, use_container_width=True):
            st.session_state.target_label = letter

target_label = st.session_state.target_label
st.markdown(f"### Show this sign: '{target_label}'")


# -------------------- SHARED STATE BETWEEN WEBRTC THREAD AND UI THREAD --------------------
class SharedState:
    """The webrtc video callback runs on its own thread, so results are handed
    back to the main Streamlit thread through this small thread-safe box."""

    def __init__(self):
        self.lock = threading.Lock()
        self.label = ""
        self.feedback = ""
        self.target_label = "A"


if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()

shared_state = st.session_state.shared_state
with shared_state.lock:
    shared_state.target_label = target_label

# Load the model in the main thread BEFORE starting the stream, so the first
# video frame isn't blocked by a slow model load (and no Streamlit calls are
# needed from the WebRTC worker thread).
with st.spinner("Loading model..."):
    model, detector, device, labels, offset = load_resources()


def run_geometric_checks(target_label, label, landmarks, feedback):
    """Adds sign-specific coaching tips on top of the base model prediction.
    These are simple heuristics on MediaPipe's 21 hand landmarks, not part of
    the trained model itself."""

    if target_label == "A":
        try:
            finger_tips, finger_bases = [8, 12, 16, 20], [5, 9, 13, 17]
            folded_fingers = sum(
                1 for tip, base in zip(finger_tips, finger_bases) if landmarks[tip].y > landmarks[base].y
            )
            thumb_tucked = landmarks[4].x < landmarks[3].x
            if label == target_label:
                if folded_fingers < 4:
                    feedback += " | Fold your fingers more."
                if not thumb_tucked:
                    feedback += " | Tuck your thumb in."
        except Exception:
            pass

    elif target_label == "B":
        try:
            finger_tips, finger_bases = [8, 12, 16, 20], [5, 9, 13, 17]
            extended_fingers = sum(
                1 for tip, base in zip(finger_tips, finger_bases) if landmarks[tip].y < landmarks[base].y
            )
            thumb_across = abs(landmarks[4].x - landmarks[0].x) < 0.1
            if label == target_label:
                if extended_fingers < 4:
                    feedback += " | Extend all fingers fully."
                if not thumb_across:
                    feedback += " | Place your thumb across your palm."
        except Exception:
            pass

    elif target_label == "C":
        try:
            finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
            curved_fingers = sum(1 for tip, base in finger_pairs if 0.05 < abs(landmarks[tip].y - landmarks[base].y) < 0.25)
            if label == target_label:
                if curved_fingers < 3:
                    feedback += " | Curve your fingers to form a 'C' shape."
                else:
                    feedback += " | Good curvature."
        except Exception:
            pass

    elif target_label == "L":
        try:
            index_extended = landmarks[8].y < landmarks[5].y
            folded_count = sum(1 for tip, base in zip([12, 16, 20], [9, 13, 17]) if landmarks[tip].y > landmarks[base].y)
            thumb_extended = abs(landmarks[4].x - landmarks[2].x) > 0.1
            if label == target_label:
                if not index_extended:
                    feedback += " | Raise your index finger."
                if folded_count < 3:
                    feedback += " | Fold the other fingers."
                if not thumb_extended:
                    feedback += " | Extend your thumb outward."
        except Exception:
            pass

    elif target_label == "W":
        try:
            finger_tips, finger_bases = [8, 12, 16, 20], [5, 9, 13, 17]
            extended_count = sum(1 for tip, base in zip(finger_tips, finger_bases) if landmarks[tip].y < landmarks[base].y)
            if label == target_label:
                if extended_count != 3:
                    feedback += " | Show exactly three fingers."
                else:
                    feedback += " | Good finger count."
        except Exception:
            pass

    return feedback


def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 480))
    img_output = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    label = ""
    feedback = ""

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        h_img, w_img, _ = img.shape

        for lm in landmarks:
            cv2.circle(img_output, (int(lm.x * w_img), int(lm.y * h_img)), 4, (0, 255, 0), cv2.FILLED)

        x_list = [lm.x * w_img for lm in landmarks]
        y_list = [lm.y * h_img for lm in landmarks]
        x_min, x_max = int(min(x_list)), int(max(x_list))
        y_min, y_max = int(min(y_list)), int(max(y_list))

        x1, y1 = max(x_min - offset, 0), max(y_min - offset, 0)
        x2, y2 = min(x_max + offset, w_img), min(y_max + offset, h_img)

        img_crop = img[y1:y2, x1:x2]

        if img_crop.size != 0:
            img_input = cv2.resize(img_crop, (224, 224))
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_input = (img_input - mean[None, None, :]) / std[None, None, :]

            img_input = np.transpose(img_input, (2, 0, 1))
            img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_input)
                probs = torch.softmax(output, dim=1)
                index = torch.argmax(output, dim=1).item()

            label = labels[index]
            confidence = probs[0][index].item()

            with shared_state.lock:
                current_target = shared_state.target_label

            if label == current_target:
                feedback = (
                    f"Correct! Good '{current_target}' sign."
                    if confidence > 0.8
                    else f"Looks like '{current_target}', but refine your hand shape."
                )
            else:
                feedback = (
                    f"That looks like '{label}', not '{current_target}'."
                    if confidence > 0.8
                    else "Unclear sign — try again."
                )

            feedback = run_geometric_checks(current_target, label, landmarks, feedback)

            cv2.rectangle(img_output, (x1, y1 - 50), (x1 + 150, y1), (255, 0, 255), cv2.FILLED)
            cv2.putText(img_output, label, (x1 + 10, y1 - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(img_output, (x1, y1), (x2, y2), (255, 0, 255), 4)

    with shared_state.lock:
        shared_state.label = label
        shared_state.feedback = feedback

    return av.VideoFrame.from_ndarray(img_output, format="bgr24")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    try:
        return process_frame(frame)
    except Exception:
        # Never let an error kill the stream; just show the raw frame.
        return frame


# -------------------- WEBCAM STREAM (RUNS IN THE VISITOR'S BROWSER) --------------------
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

ctx = webrtc_streamer(
    key="asl-learning-tool",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

label_placeholder = st.empty()
feedback_placeholder = st.empty()

if ctx.state.playing:
    while ctx.state.playing:
        with shared_state.lock:
            current_label = shared_state.label
            current_feedback = shared_state.feedback
        label_placeholder.markdown(f"## Detected Sign: '{current_label}'" if current_label else "")
        feedback_placeholder.markdown(f"### Feedback: {current_feedback}" if current_feedback else "")
        time.sleep(0.2)
else:
    st.info("Click **Start** above and allow camera access to begin practicing.")
