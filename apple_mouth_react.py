import cv2
import numpy as np
import mediapipe as mp

SHOW_CAMERA_FEED = True         # camera shows with no overlays
APPLE_WINDOW_SIZE = 620         # pixels for the Apple window

ASSET_FILES = {
    "neutral": "assets/apple_neutral.png",
    "tongue":  "assets/apple_tongue.png",
    "shock":   "assets/apple_shock.png",
    "angry":   "assets/apple_angry.png",
    "cry":     "assets/apple_cry.png",
}

def load_png(path, label):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        canvas = np.full((600, 600, 3), 255, np.uint8)
        cv2.putText(canvas, f"{label.upper()}", (80, 315),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
        return canvas
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def fit_square(img, size=600):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    out = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    top = (size - out.shape[0]) // 2
    bottom = size - out.shape[0] - top
    left = (size - out.shape[1]) // 2
    right = size - out.shape[1] - left
    return cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))

APPLE = {k: fit_square(load_png(v, k), 600) for k, v in ASSET_FILES.items()}

mpfm = mp.solutions.face_mesh
FACE_MESH = mpfm.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def idx_set(conns):
    s = set()
    for a, b in conns: s.add(a); s.add(b)
    return sorted(s)

LIPS = idx_set(mpfm.FACEMESH_LIPS)
L_EYE = idx_set(mpfm.FACEMESH_LEFT_EYE)
R_EYE = idx_set(mpfm.FACEMESH_RIGHT_EYE)
L_BROW = idx_set(mpfm.FACEMESH_LEFT_EYEBROW)
R_BROW = idx_set(mpfm.FACEMESH_RIGHT_EYEBROW)

LEFT_CORNER, RIGHT_CORNER, UPPER_CENTER, LOWER_CENTER = 61, 291, 13, 14

def pts_xy(lm, idxs, w, h):
    return np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs])

def mouth_metrics(frame, face):
    h, w = frame.shape[:2]
    lm = face.landmark
    lips = pts_xy(lm, LIPS, w, h)
    x1, y1 = lips[:,0].min(), lips[:,1].min()
    x2, y2 = lips[:,0].max(), lips[:,1].max()
    mw = max(1, x2-x1); mh = max(1, y2-y1)
    mar = mh / float(mw)

    L = (int(lm[LEFT_CORNER].x*w),  int(lm[LEFT_CORNER].y*h))
    R = (int(lm[RIGHT_CORNER].x*w), int(lm[RIGHT_CORNER].y*h))
    U = (int(lm[UPPER_CENTER].x*w), int(lm[UPPER_CENTER].y*h))
    D = (int(lm[LOWER_CENTER].x*w), int(lm[LOWER_CENTER].y*h))
    center_y = (U[1] + D[1]) // 2
    corners_y = (L[1] + R[1]) // 2
    corner_drop = (corners_y - center_y) / float(mh)

    sx = int(x1 + 0.12*mw); ex = int(x2 - 0.12*mw)
    sy = int(y1 + 0.28*mh); ey = int(y2 - 0.18*mh)
    sx, sy = max(0, sx), max(0, sy)
    ex, ey = min(w-1, ex), min(h-1, ey)
    inner = frame[sy:ey, sx:ex].copy() if ex>sx and ey>sy else None

    return mar, inner, corner_drop

def red_tongue(inner_bgr):
    if inner_bgr is None or inner_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, 70, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return (float(cv2.countNonZero(mask)) / mask.size) > 0.18

def eye_center(lm, idxs, w, h):
    p = pts_xy(lm, idxs, w, h)
    return p[:,0].mean(), p[:,1].mean()

def brow_eye_gap_norm(frame, face):
    h, w = frame.shape[:2]
    lm = face.landmark
    Leye = eye_center(lm, L_EYE, w, h)
    Reye = eye_center(lm, R_EYE, w, h)
    Lbrow_y = pts_xy(lm, L_BROW, w, h)[:,1].mean()
    Rbrow_y = pts_xy(lm, R_BROW, w, h)[:,1].mean()
    eye_y = (Leye[1] + Reye[1]) / 2.0
    brow_y = (Lbrow_y + Rbrow_y) / 2.0
    ipd = max(np.hypot(Leye[0]-Reye[0], Leye[1]-Reye[1]), 1.0)
    return (brow_y - eye_y) / ipd

def decide_state(frame, face):
    mar, inner, corner_drop = mouth_metrics(frame, face)
    gap = brow_eye_gap_norm(frame, face)
    tongue = red_tongue(inner)

    if mar > 0.28 and tongue:
        return "tongue"
    if mar > 0.40 and not tongue:
        return "shock"
    if corner_drop > 0.22 and mar <= 0.28:
        return "cry"
    if gap < 0.09 and mar <= 0.28:
        return "angry"
    return "neutral"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found"); return

    cv2.namedWindow("Apple", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Apple", APPLE_WINDOW_SIZE, APPLE_WINDOW_SIZE)

    state, stable = "neutral", 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = FACE_MESH.process(rgb)

        next_state = "neutral"
        if out.multi_face_landmarks:
            next_state = decide_state(frame, out.multi_face_landmarks[0])

        if next_state == state:
            stable = min(stable + 1, 12)
        else:
            stable -= 1
            if stable <= -2:
                state = next_state
                stable = 0

        cv2.imshow("Apple", fit_square(APPLE[state], 600))
        if SHOW_CAMERA_FEED:
            cv2.imshow("Camera", frame)   # no drawings added

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
