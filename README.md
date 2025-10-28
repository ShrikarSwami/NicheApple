# Apple Face States

Apple Face States is a tiny OpenCV app that watches your face and flips a separate Apple window between five images:

- neutral
- tongue
- shock
- angry
- cry

The camera window stays clean with no drawings. The Apple window swaps PNGs based on simple mouth and eyebrow signals from MediaPipe Face Mesh.

## Quick start
pip install opencv-python mediapipe numpy

assets/
  apple_neutral.png
  apple_tongue.png
  apple_shock.png
  apple_angry.png
  apple_cry.png
# each 1024x1024 PNG, framed the same

python apple_face_states_clean.py

Press q or Esc to quit.

## Tuning
Open mouth cutoff: 0.28
Big open for shock: 0.40
Tongue red ratio: 0.18
Brow anger gap: 0.09
Corner drop for cry: 0.22

Raise or lower these in the script if your lighting or camera needs it.
