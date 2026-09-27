# ASL Fingerspelling Learning Tool

A real-time American Sign Language (ASL) fingerspelling coach that runs in your browser. Show a letter to your webcam and get instant feedback on whether your hand shape is correct, and tips on how to fix it if it isn't.

**[Try the live demo →](#)** *(update this link once deployed — see [Deployment](#deployment) below)*

## Why this project

Learning ASL fingerspelling on your own is hard: static images and diagrams can't tell you whether *your* hand shape is actually right. This tool gives live, corrective feedback the way a human tutor would, attempting to close a real accessibility gap for people trying to learn ASL without in-person instruction.

## How it works

1. **Hand detection**: [MediaPipe's Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) locates 21 landmark points on the hand in each webcam frame.
2. **Cropping & preprocessing**: the hand region is cropped, resized to 224×224, and normalized to match the model's training distribution.
3. **Classification**: a ResNet-18 (transfer-learned in PyTorch) classifies the cropped image as one of the 26 ASL alphabet letters.
4. **Rule-based coaching**: for a subset of letters (A, B, C, L, W), geometric heuristics on the MediaPipe landmarks (finger extension, thumb position, curvature) generate specific corrective feedback, e.g. *"Tuck your thumb in"* or *"Extend all fingers fully."*

## Currently supported letters

The live corrective-feedback logic currently covers **A, B, C, L, and W**. The underlying classifier recognizes all 26 letters, but hand-crafted coaching tips only exist for these five so far.
