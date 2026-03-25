import cv2
import mediapipe as mp
import numpy as np
import time
import torch

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ------------------ CONFIG ------------------
model_path = "hand_landmarker.task"
pytorch_model_path = "Model/pytorch_model.pth"

offset = 20

labels = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
]

# ------------------ LOAD MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(pytorch_model_path, map_location=device)
model.to(device)
model.eval()

# ------------------ MEDIAPIPE ------------------
base_options = BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ------------------ OPENCV ------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
# ------------------ MAIN LOOP ------------------
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()

        # Convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            h_img, w_img, _ = img.shape

            # Draw a small green dot for each landmark on the output image
            for lm in landmarks:
                x_lm = int(lm.x * w_img)
                y_lm = int(lm.y * h_img)
                cv2.circle(imgOutput, (x_lm, y_lm), 4, (0, 255, 0), cv2.FILLED)

            # ------------------ BOUNDING BOX ------------------
            x_list = [lm.x * w_img for lm in landmarks]
            y_list = [lm.y * h_img for lm in landmarks]

            x_min, x_max = int(min(x_list)), int(max(x_list))
            y_min, y_max = int(min(y_list)), int(max(y_list))

            w = x_max - x_min
            h = y_max - y_min

            # ------------------ CROP ------------------
            x1 = max(x_min - offset, 0)
            y1 = max(y_min - offset, 0)
            x2 = min(x_max + offset, w_img)
            y2 = min(y_max + offset, h_img)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:

                # ------------------ PREPROCESS (match training transforms) ------------------
                # Resize to match training input
                imgInput = cv2.resize(imgCrop, (224, 224))

                # Convert BGR (OpenCV) -> RGB
                imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)

                # Scale to [0,1]
                imgInput = imgInput.astype(np.float32) / 255.0

                # Normalize using training mean/std
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                imgInput = (imgInput - mean[None, None, :]) / std[None, None, :]

                # HWC -> CHW
                imgInput = np.transpose(imgInput, (2, 0, 1))

                # Add batch dimension and move to device
                imgInput = torch.from_numpy(imgInput).unsqueeze(0).to(device)

                # ------------------ PREDICTION ------------------
                with torch.no_grad():
                    output = model(imgInput)
                    index = torch.argmax(output, dim=1).item()

                label = labels[index]

                # ------------------ DISPLAY ------------------
                cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 150, y1), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, label, (x1 + 10, y1 - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)

        cv2.imshow("Image", imgOutput)

        if cv2.waitKey(1) & 0xFF == 27:
            break