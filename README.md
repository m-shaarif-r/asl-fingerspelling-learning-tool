# ASL Fingerspelling Learning Tool

A real-time American Sign Language (ASL) fingerspelling coach that runs in your browser. Show a letter to your webcam and get instant feedback on whether your hand shape is correct — and specific tips on how to fix it if it isn't.

**[Try the live demo →](#)** *(update this link once deployed — see [Deployment](#deployment) below)*

## Why this project

Learning ASL fingerspelling on your own is hard: static images and diagrams can't tell you whether *your* hand shape is actually right. This tool gives live, corrective feedback the way a human tutor would — closing a real accessibility gap for people trying to learn ASL without in-person instruction.

## How it works

1. **Hand detection** — [MediaPipe's Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) locates 21 landmark points on the hand in each webcam frame.
2. **Cropping & preprocessing** — the hand region is cropped, resized to 224×224, and normalized to match the model's training distribution.
3. **Classification** — a ResNet-18 (transfer-learned in PyTorch) classifies the cropped image as one of the 26 ASL alphabet letters.
4. **Rule-based coaching** — for a subset of letters (A, B, C, L, W), geometric heuristics on the MediaPipe landmarks (finger extension, thumb position, curvature) generate specific corrective feedback, e.g. *"Tuck your thumb in"* or *"Extend all fingers fully."*

```
Webcam frame → MediaPipe hand landmarks → crop + normalize → ResNet-18 → predicted letter
                                                                      ↓
                                                    landmark geometry → coaching tip
```

## Tech stack

| Layer | Tool |
|---|---|
| UI / app framework | [Streamlit](https://streamlit.io/) |
| In-browser webcam capture | [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) (WebRTC, so video is captured client-side, not on the server) |
| Hand landmark detection | [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| Sign classification | PyTorch, ResNet-18 (transfer learning) |
| Image processing | OpenCV, NumPy |

## Project structure

```
.
├── application.py                  # Streamlit app (live demo entry point)
├── inference_post_training.py      # Model + MediaPipe loading, and a standalone desktop OpenCV demo
├── model_training.py               # Trains the ResNet-18 classifier on preprocessed hand crops
├── asl_hg_preprocessing.py         # One-off script: crops raw dataset images to hand-only, centered squares
├── Model/pytorch_model.pth         # Trained model checkpoint
├── hand_landmarker.task            # MediaPipe hand landmark model
├── requirements.txt                # Python dependencies
└── packages.txt                    # System (apt) dependencies for deployment
```

## Currently supported letters

The live corrective-feedback logic currently covers **A, B, C, L, and W**. The underlying classifier recognizes all 26 letters, but hand-crafted coaching tips only exist for these five so far — extending coverage to the rest of the alphabet is the most natural next step (see [Roadmap](#roadmap)).

## Run it locally

If you'd rather run this on your own machine (e.g. for GPU inference, or to try the standalone OpenCV version):

1. **Clone the repo and create a virtual environment** (Python 3.12 recommended):
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app** (browser UI, recommended):
   ```bash
   streamlit run application.py
   ```
   Press `Ctrl+C` in the terminal to stop it.

   **Or**, run the standalone desktop version (opens a native OpenCV window, no browser):
   ```bash
   python inference_post_training.py
   ```
   Press `Esc` in the video window to quit.

## Deployment

This app is designed to deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub (make sure `Model/pytorch_model.pth` and `hand_landmarker.task` are committed — they're the app's only large binary assets).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click **New app**.
3. Point it at this repo, branch, and set the main file path to `application.py`.
4. **Before clicking Deploy**, open **Advanced settings** and explicitly select **Python 3.11** (or 3.10) from the Python version dropdown. This matters: `mediapipe` does not publish wheels for Python 3.13/3.14, and Community Cloud has been defaulting new apps to newer Python versions that break this dependency. Community Cloud only lets you set this at initial deploy time — changing it later means deleting and redeploying the app.
5. Deploy. Streamlit Cloud will pick up `requirements.txt` and `packages.txt` automatically.
6. Once live, visitors just click the link and allow camera access — no install required. Update the demo link at the top of this README once you have it.

> Because the app uses `streamlit-webrtc`, the webcam feed is captured in the *visitor's* browser and streamed to the app for inference — it does not try to open a camera device on the server (which wouldn't exist on a hosted platform).

## Training your own model

`model_training.py` trains the ResNet-18 classifier from a zipped, preprocessed dataset (folders of cropped hand images per letter, in `train`/`test` splits). `asl_hg_preprocessing.py` produces that preprocessed dataset from raw images by detecting the hand with MediaPipe and cropping/centering it. Update the paths at the top of each script to point at your own dataset if you want to retrain or fine-tune.

## Roadmap

- Extend rule-based coaching feedback to the remaining letters of the alphabet
- Track practice streaks / progress over a session
- Support two-handed signs and letters that require motion (e.g. J, Z)

## License

*(Add a license if you'd like this to be reusable — MIT is a common choice for portfolio projects.)*
