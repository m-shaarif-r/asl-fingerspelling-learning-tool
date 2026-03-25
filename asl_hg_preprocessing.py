import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ------------------ CONFIG ------------------
model_path = "hand_landmarker.task"
input_root = "ASL-HG"              # your dataset folder
output_root = "Processed_ASL"      # output folder
imgSize = 300
offset = 20

# ------------------ MEDIAPIPE SETUP ------------------
base_options = BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ------------------ PROCESS DATASET ------------------

for label in os.listdir(input_root):
    label_path = os.path.join(input_root, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing: {label}")

    output_label_path = os.path.join(output_root, label)
    os.makedirs(output_label_path, exist_ok=True)

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # ------------------ MEDIAPIPE INPUT ------------------
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        timestamp = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            h_img, w_img, _ = img.shape

            # ------------------ COMPUTE BOUNDING BOX ------------------
            x_list = [lm.x * w_img for lm in landmarks]
            y_list = [lm.y * h_img for lm in landmarks]

            x_min, x_max = int(min(x_list)), int(max(x_list))
            y_min, y_max = int(min(y_list)), int(max(y_list))

            w = x_max - x_min
            h = y_max - y_min

            x, y = x_min, y_min

            # ------------------ CROP WITH OFFSET ------------------
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, w_img)
            y2 = min(y + h + offset, h_img)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                crop_h, crop_w, _ = imgCrop.shape
                aspectRatio = crop_h / crop_w

                # ------------------ RESIZE + CENTER ------------------
                if aspectRatio > 1:
                    k = imgSize / crop_h
                    wCal = math.ceil(k * crop_w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / crop_w
                    hCal = math.ceil(k * crop_h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # ------------------ SAVE ------------------
                save_path = os.path.join(output_label_path, img_name)
                cv2.imwrite(save_path, imgWhite)

        # Optional debug (comment out for speed)
        # cv2.imshow("Processed", imgWhite)
        # cv2.waitKey(1)

print("Processing complete.")