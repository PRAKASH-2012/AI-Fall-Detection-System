---
title: AI Fall Detection System
emoji: 🛟
colorFrom: blue
colorTo: green
sdk: gradio
app_file: web_app.py
---

# AI Fall Detection System

A Python computer-vision prototype that uses webcam video, MediaPipe pose landmarks, OpenCV and temporal rules to estimate body states and flag possible falls.

## Features

- Live webcam-based pose estimation
- Skeleton landmark visualization and person bounding box
- Heuristic posture labels: standing, walking, sitting, lying, falling
- Temporal fall confirmation and audio alert on Windows
- FPS and status overlay
- Left/right wrist movement detection, with direction arrows and dashboard status
- Full visible-pose bounding box with preview pixel coordinates for the body center and key joints
- Command-line options for camera selection, resolution, feature listing, and audio disable
- Browser demo: analyze a short video recorded with your webcam or uploaded as a file

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam
- Windows, macOS or Linux (audio alert uses Windows `winsound`; other platforms print a fallback alert)

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, use Command Prompt:

```bat
venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

Press **Q** in the video window to exit. Change `CAMERA_INDEX` in `config.py` if your webcam is not camera 0.

Useful runtime options:

```powershell
python main.py --list-features
python main.py --no-camera
python main.py --no-gui
python main.py --camera-index 1 --frame-width 640 --frame-height 480
python main.py --disable-audio
```

The app includes a Tkinter dashboard with Start, Stop and Quit controls, a live status log, posture telemetry, left/right hand-motion status, body bounding-box coordinates, and visible key-joint positions. The video feed annotates the body center and visible joint coordinates. Coordinates are pixels in the mirrored preview. Run `python main.py` normally to open the dashboard and webcam detection window.

## Browser demo deployment

The Hugging Face Space runs `web_app.py` and accepts a short webcam recording or uploaded video (up to 1,800 frames). It returns an annotated video and posture/body-position summary; it does not continuously monitor the webcam. To publish it, create a **public Gradio Space** on Hugging Face, then push this project to that Space's Git repository. The Space provides a persistent `https://<username>-<space-name>.hf.space` URL. The desktop dashboard remains available separately with `python main.py`.

## Project structure

```text
AI_Fall_Detection_System/
├── .gitignore
├── LICENSE
├── main.py
├── web_app.py
├── config.py
├── pose_estimator.py
├── fall_detector.py
├── hand_motion.py
├── requirements.txt
├── packages.txt
├── README.md
└── utils/
    ├── __init__.py
    └── alerts.py
```

## Important limitations

This is an educational prototype using hand-designed thresholds, not a trained or clinically validated fall classifier. The displayed fall-confirmation percentage is progress through a frame threshold, not a calibrated probability. Results depend on camera angle, lighting, visibility and body orientation. A person lying down may be mistaken for a fall, and some falls may be missed. Test cautiously with safe, controlled movements; do not deliberately fall. Do not use this prototype as the sole safety, medical or emergency-response system.

Hand motion is estimated from wrist landmark movement relative to the detected body size. It identifies left/right wrist movement, not individual finger positions or specific hand gestures.

Body localization uses MediaPipe's visible pose landmarks and requires at least one visible shoulder and hip. If the person is clipped by the camera frame or landmarks are obscured, the box and coordinates represent only visible joints; full-body position cannot be inferred reliably from unseen body parts.

Pose tracking uses a higher-detail model and relaxed detection thresholds to help with weaker lighting and partial visibility. Keep the torso (at least one shoulder and one hip) in view; the preview displays guidance when no usable pose is found.
