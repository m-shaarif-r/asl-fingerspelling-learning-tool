import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
import time

# -------------------- LAZY LOAD (NEW) --------------------
@st.cache_resource
def load_resources():
    from inference_post_training import model, detector, device, labels, offset
    return model, detector, device, labels, offset

model = detector = device = labels = offset = None

# --------------------UI Setup--------------------
st.set_page_config(page_title = "ASL (Sign language) Learning Tool")
st.title("ASL (Sign Language) Learning Tool")
st.write("Obtain real-time feedback by practicing your ASL skills!")

# -------------------- TARGET SIGN SELECTION --------------------
if "target_label" not in st.session_state:
    st.session_state.target_label = "A"

st.markdown("### Choose a sign to practice:")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("A"):
        st.session_state.target_label = "A"
with col2:
    if st.button("B"):
        st.session_state.target_label = "B"
with col3:
    if st.button("C"):
        st.session_state.target_label = "C"
with col4:
    if st.button("L"):
        st.session_state.target_label = "L"
with col5:
    if st.button("W"):
        st.session_state.target_label = "W"

target_label = st.session_state.target_label
st.markdown(f"### Show this sign: '{target_label}'")

frame_placeholder = st.empty()
label_placeholder = st.empty()
feedback_placeholder = st.empty()
label = ""

# -------------------- CAMERA STATE (NEW) --------------------
if "capture" not in st.session_state:
    st.session_state.capture = None

# -------------------- SESSION STATE --------------------
if 'on' not in st.session_state:
    st.session_state.on = False
    button = st.button('Begin Session')
else:
    if st.session_state.on:
        button = st.button('End Session')
    else:
        button = st.button('Begin Session')

if button:
    st.session_state.on = not st.session_state.on

    # -------------------- START SESSION --------------------
    if st.session_state.on:
        if st.session_state.capture is None:
            st.session_state.capture = cv2.VideoCapture(0)

    # -------------------- END SESSION --------------------
    else:
        if st.session_state.capture is not None:
            st.session_state.capture.release()
            st.session_state.capture = None
        warning = st.warning("Session ended.")
        time.sleep(1.5)
        warning.empty()

    st.rerun()

# -------------------- MAIN LOOP --------------------
while st.session_state.on:

    # -------------------- LOAD MODEL (LAZY) --------------------
    if model is None:
        model, detector, device, labels, offset = load_resources()

    capture = st.session_state.capture

    success, img = capture.read()
    if not success:
        break

    # -------------------- RESIZE EARLY --------------------
    img = cv2.resize(img, (640, 480))

    imgOutput = img.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        h_img, w_img, _ = img.shape

        for lm in landmarks:
            x_lm = int(lm.x * w_img)
            y_lm = int(lm.y * h_img)
            cv2.circle(imgOutput, (x_lm, y_lm), 4, (0, 255, 0), cv2.FILLED)

        x_list = [lm.x * w_img for lm in landmarks]
        y_list = [lm.y * h_img for lm in landmarks]

        x_min, x_max = int(min(x_list)), int(max(x_list))
        y_min, y_max = int(min(y_list)), int(max(y_list))

        x1 = max(x_min - offset, 0)
        y1 = max(y_min - offset, 0)
        x2 = min(x_max + offset, w_img)
        y2 = min(y_max + offset, h_img)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            imgInput = cv2.resize(imgCrop, (224, 224))
            imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
            imgInput = imgInput.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            imgInput = (imgInput - mean[None, None, :]) / std[None, None, :]

            imgInput = np.transpose(imgInput, (2, 0, 1))
            imgInput = torch.from_numpy(imgInput).unsqueeze(0).to(device)

            # ------------------ PREDICTION ------------------
            with torch.no_grad():
                output = model(imgInput)
                probs = torch.softmax(output, dim=1)
                index = torch.argmax(output, dim=1).item()

            label = labels[index]
            confidence = probs[0][index].item()

            # ------------------ FEEDBACK LOGIC ------------------
            feedback = ""

            # Basic correctness + confidence
            if label == target_label:
                if confidence > 0.8:
                    feedback = f"Correct! Good '{target_label}' sign."
                else:
                    feedback = f"Looks like '{target_label}', but refine your hand shape."
            else:
                if confidence > 0.8:
                    feedback = f"That looks like '{label}', not '{target_label}'."
                else:
                    feedback = "Unclear sign — try again."

            # ------------------ GEOMETRIC CHECK (A-specific) ------------------
            if target_label == "A":
                # A = fingers folded, thumb tucked
                try:
                    # Finger tips vs bases (y-axis check)
                    folded_fingers = 0
                    finger_tips = [8, 12, 16, 20]
                    finger_bases = [5, 9, 13, 17]

                    for tip, base in zip(finger_tips, finger_bases):
                        if landmarks[tip].y > landmarks[base].y:
                            folded_fingers += 1

                    thumb_tip = landmarks[4]
                    thumb_ip = landmarks[3]

                    thumb_tucked = thumb_tip.x < thumb_ip.x  # simple heuristic

                    if label == target_label:
                        if folded_fingers < 4:
                            feedback += " | Fold your fingers more."
                        if not thumb_tucked:
                            feedback += " | Tuck your thumb in."

                except:
                    pass
            # ------------------ GEOMETRIC CHECK (B-specific) ------------------
            if target_label == "B":
                try:
                    finger_tips = [8, 12, 16, 20]
                    finger_bases = [5, 9, 13, 17]

                    extended_fingers = 0
                    for tip, base in zip(finger_tips, finger_bases):
                        if landmarks[tip].y < landmarks[base].y:
                            extended_fingers += 1

                    thumb_tip = landmarks[4]
                    palm_center = landmarks[0]

                    thumb_across = abs(thumb_tip.x - palm_center.x) < 0.1

                    if label == target_label:
                        if extended_fingers < 4:
                            feedback += " | Extend all fingers fully."
                        if not thumb_across:
                            feedback += " | Place your thumb across your palm."

                except:
                    pass
            # ------------------ GEOMETRIC CHECK (C-specific) ------------------
            if target_label == "C":
                try:
                    # Check curvature via distance between tip and base
                    curved_fingers = 0
                    finger_pairs = [(8,5), (12,9), (16,13), (20,17)]

                    for tip, base in finger_pairs:
                        dist = abs(landmarks[tip].y - landmarks[base].y)
                        if 0.05 < dist < 0.25:
                            curved_fingers += 1

                    if label == target_label:
                        if curved_fingers < 3:
                            feedback += " | Curve your fingers to form a 'C' shape."
                        else:
                            feedback += " | Good curvature."

                except:
                    pass
            # ------------------ GEOMETRIC CHECK (L-specific) ------------------
            if target_label == "L":
                try:
                    # Index extended
                    index_extended = landmarks[8].y < landmarks[5].y

                    # Other fingers folded
                    folded_count = 0
                    for tip, base in zip([12,16,20], [9,13,17]):
                        if landmarks[tip].y > landmarks[base].y:
                            folded_count += 1

                    # Thumb extended sideways
                    thumb_extended = abs(landmarks[4].x - landmarks[2].x) > 0.1

                    if label == target_label:
                        if not index_extended:
                            feedback += " | Raise your index finger."
                        if folded_count < 3:
                            feedback += " | Fold the other fingers."
                        if not thumb_extended:
                            feedback += " | Extend your thumb outward."

                except:
                    pass
            # ------------------ GEOMETRIC CHECK (W-specific) ------------------
            if target_label == "W":
                try:
                    # Count extended fingers
                    finger_tips = [8, 12, 16, 20]
                    finger_bases = [5, 9, 13, 17]

                    extended = []
                    for tip, base in zip(finger_tips, finger_bases):
                        extended.append(landmarks[tip].y < landmarks[base].y)

                    extended_count = sum(extended)

                    if label == target_label:
                        if extended_count != 3:
                            feedback += " | Show exactly three fingers."
                        else:
                            feedback += " | Good finger count."

                except:
                    pass

            cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 150, y1), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x1 + 10, y1 - 15),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

            feedback_placeholder.markdown(f"### Feedback: {feedback}")

    frame_placeholder.image(imgOutput, channels="BGR", width=640)

    if label:
        label_placeholder.markdown(f"## Detected Sign: '{label}'")

    # Loop 'throttle'
    time.sleep(0.03)

# -------------------- CLEANUP --------------------
if st.session_state.capture is not None:
    st.session_state.capture.release()