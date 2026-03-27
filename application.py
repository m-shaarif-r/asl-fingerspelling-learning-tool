import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from inference_post_training import model, detector, device, labels, offset

# --------------------UI Setup--------------------
st.set_page_config(page_title = "ASL Learning Tool")
st.title("ASL Learning Tool")
st.write("Obtain real-time feedback by practicing your ASL skills!")

# -------------------- NEW: TARGET SIGN --------------------
target_label = "A"
st.markdown(f"### Show this sign: '{target_label}'")

frame_placeholder = st.empty()
label_placeholder = st.empty()
feedback_placeholder = st.empty()  # NEW
label = ""
capture = cv2.VideoCapture(0)

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
    if not st.session_state.on:
        capture.release()
        warning = st.warning("Session ended.")
        time.sleep(2.5)
        warning.empty()
    st.rerun()

while st.session_state.on:
    success, img = capture.read()
    if not success:
        break

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

        w = x_max - x_min
        h = y_max - y_min

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
                probs = torch.softmax(output, dim=1)  # NEW
                index = torch.argmax(output, dim=1).item()

            label = labels[index]
            confidence = probs[0][index].item()  # NEW

            # ------------------ FEEDBACK LOGIC ------------------
            feedback = ""

            # Basic correctness + confidence
            if label == target_label:
                if confidence > 0.8:
                    feedback = "Correct! Good 'A' sign."
                else:
                    feedback = "Looks like 'A', but refine your hand shape."
            else:
                if confidence > 0.8:
                    feedback = f"That looks like '{label}', not 'A'."
                else:
                    feedback = "Unclear sign — try again."

            # ------------------ GEOMETRIC CHECK (A-specific) ------------------
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

            # ------------------ DISPLAY ------------------
            cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 150, y1), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x1 + 10, y1 - 15),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

            feedback_placeholder.markdown(f"### Feedback: {feedback}")  # NEW

    frame_placeholder.image(imgOutput, channels="BGR", width=640)
    if label:
        label_placeholder.markdown(f"## Detected Sign: '{label}'")

capture.release()