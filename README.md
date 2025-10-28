**# NicheApple

Face driven reactions with OpenCV and MediaPipe.  
Shows an Apple face in a second window that reacts to your expressions.

## Setup
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe numpy

## Run
python apple_mouth_react.py

Press c to recalibrate  
Press q to quit
**# NicheApple — Face-Driven Apple Reactions (OpenCV + MediaPipe)

[▶️ Demo Video](https://youtu.be/56XS_4H8a_w)

NicheApple is a tiny real-time face-reaction toy. It uses a webcam feed and MediaPipe Face Mesh to classify a few simple expressions and swaps an on-screen apple sprite accordingly.

## Features

- **Neutral** – baseline calibrated from your face.
- **Tongue-Out (closed lips ok)** – detects a protruding tongue without requiring a wide open mouth.
- **Shock** – big mouth + brows/eyes up.
- **Cry** – mouth corners pulled down with a neutral mouth opening.
- **One-key Recalibration** – press `c` for a quick neutral recalibration (~2s).
- **Camera Switching** – press `[` and `]` to cycle cameras at runtime.
- **Lightweight Logging** – periodic metrics print to the terminal for quick tuning.

> ⚠️ I may add a **“mad/angry”** expression in a future update. For now, please enjoy the current set—and feel free to contact me if you hit any issues.

---

## Quick Start (macOS / Apple Silicon)

### 1) Create and activate a Python 3.11 venv
```bash
cd /path/to/NicheApple
/opt/homebrew/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python -V   # should show 3.11.x
