# apple_mouth_react.py
import cv2, numpy as np, mediapipe as mp, time, os
from collections import deque

# ---- config ----
SHOW_CAMERA_FEED = True
APPLE_WINDOW_SIZE = 620
LOG_EVERY = 15  # frames

ASSET_FILES = {
    "neutral": "assets/apple_neutral.png",
    "tongue":  "assets/apple_tongue.png",
    "shock":   "assets/apple_shock.png",
    "angry":   "assets/apple_angry.png",   # not prioritized now
    "cry":     "assets/apple_cry.png",
}
MUSIC_WAV = "assets/music_wink_start.wav"
MUSIC_MP3 = "assets/music_wink_start.mp3"

# ---------- camera helpers ----------
PREF_BACKEND = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0

def open_cam(index: int):
    cap = cv2.VideoCapture(index, PREF_BACKEND)
    if not cap.isOpened():
        cap.release()
        return None
    return cap

def cycle_camera(curr_index: int, step: int = 1, max_try: int = 6):
    tried = set()
    idx = curr_index
    for _ in range(max_try):
        idx = (idx + step) % max_try
        if idx in tried:
            continue
        tried.add(idx)
        cap = open_cam(idx)
        if cap is not None:
            return idx, cap
    return None, None

# ---------------- assets ----------------
def load_png(path, label):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        canvas = np.full((600, 600, 3), 255, np.uint8)
        cv2.putText(canvas, label.upper(), (80, 315),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
        return canvas
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def fit_square(img, size=600):
    h, w = img.shape[:2]
    s = size / max(h, w)
    out = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    top = (size - out.shape[0]) // 2
    bottom = size - out.shape[0] - top
    left = (size - out.shape[1]) // 2
    right = size - out.shape[1] - left
    return cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))

APPLE = {k: fit_square(load_png(v, k), 600) for k, v in ASSET_FILES.items()}

# ---------------- mediapipe ----------------
mpfm = mp.solutions.face_mesh
FACE_MESH = mpfm.FaceMesh(static_image_mode=False, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.6,
                          min_tracking_confidence=0.6)

# Eyes (robust refs)
L_OUT, L_IN, L_UP, L_DN = 33, 133, 159, 145
R_OUT, R_IN, R_UP, R_DN = 263, 362, 386, 374

# Mouth keypoints
LEFT_CORNER, RIGHT_CORNER, UPPER_CENTER, LOWER_CENTER = 61, 291, 13, 14

def xy(lm, i, w, h):
    p = lm[i]; return int(p.x*w), int(p.y*h)

def eye_metrics(lm, w, h):
    # per-eye normalized openings + averaged EAR
    lw = np.hypot(*(np.array(xy(lm, L_OUT, w, h)) - np.array(xy(lm, L_IN, w, h))))
    rw = np.hypot(*(np.array(xy(lm, R_OUT, w, h)) - np.array(xy(lm, R_IN, w, h))))
    l_open = abs(xy(lm, L_UP, w, h)[1]-xy(lm, L_DN, w, h)[1]) / max(lw, 1.0)
    r_open = abs(xy(lm, R_UP, w, h)[1]-xy(lm, R_DN, w, h)[1]) / max(rw, 1.0)
    ear = 0.5*(l_open + r_open)
    ipd = np.hypot(*(np.array(xy(lm, L_IN, w, h)) - np.array(xy(lm, R_IN, w, h))))
    return l_open, r_open, ear, max(ipd, 1.0)

# ---------- mouth boxes (returns 7 values) ----------
def mouth_boxes(frame, lm, w, h):
    L = xy(lm, LEFT_CORNER, w, h); R = xy(lm, RIGHT_CORNER, w, h)
    U = xy(lm, UPPER_CENTER, w, h); D = xy(lm, LOWER_CENTER, w, h)

    mw = max(1, abs(R[0]-L[0])); mh = max(1, abs(D[1]-U[1]))
    mar = mh / float(mw)
    center_y = (U[1]+D[1])//2; corners_y = (L[1]+R[1])//2
    corner_drop = (corners_y - center_y) / float(mh)

    # inner mouth (tight, tries to avoid lips)
    x1, x2 = min(L[0], R[0]), max(L[0], R[0])
    y1, y2 = min(U[1], D[1]), max(U[1], D[1])
    sx = int(x1 + 0.20*mw); ex = int(x2 - 0.20*mw)
    sy = int(y1 + 0.22*mh); ey = int(y2 - 0.05*mh)
    H, W = frame.shape[:2]
    sx, sy = max(0, sx), max(0, sy); ex, ey = min(W-1, ex), min(H-1, ey)
    inner = frame[sy:ey, sx:ex].copy() if ex>sx and ey>sy else None

    # expanded tongue search zone: just above to well below lower lip
    bx1 = int(x1 - 0.12*mw); bx2 = int(x2 + 0.12*mw)
    by1 = int(D[1] - 0.10*mh)
    by2 = int(D[1] + 0.90*mh)
    bx1, by1 = max(0, bx1), max(0, by1); bx2, by2 = min(W-1, bx2), min(H-1, by2)
    below = frame[by1:by2, bx1:bx2].copy() if bx2>bx1 and by2>by1 else None

    # return: mar, corner_drop, inner, below-lip ROI, lower lip y, mouth w/h
    return mar, corner_drop, inner, below, D[1], mw, mh

# ---------- tongue detector (focus on protrusion below lower lip) ----------
def has_tongue(inner_bgr, belowlip_bgr, lower_lip_y, mw, mh):
    def red_mask(bgr):
        if bgr is None or bgr.size == 0: return None, None
        bgr = cv2.GaussianBlur(bgr, (3,3), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[...,2] = cv2.equalizeHist(hsv[...,2])
        m1 = cv2.inRange(hsv, (0, 70, 60),   (12, 255, 255))
        m2 = cv2.inRange(hsv, (165, 70, 60), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
        k = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
        return mask, hsv

    for roi in (belowlip_bgr, inner_bgr):
        mask, hsv = red_mask(roi)
        if mask is None: continue
        nz = cv2.countNonZero(mask)
        if nz == 0: continue

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        H,W = mask.shape[:2]

        frac_overall = nz / float(H*W)
        frac_blob    = (w*h) / float(H*W)
        height_ratio = h / float(H)
        cy = y + 0.5*h

        below_mid    = cy > 0.52*H
        tall_enough  = height_ratio > 0.30
        sat_mean     = cv2.mean(hsv[...,1], mask=mask)[0]

        # score for logs
        score = (
            0.35*min(frac_blob/0.18, 1.0) +
            0.25*min(height_ratio/0.38, 1.0) +
            0.20*(1.0 if below_mid else 0.0) +
            0.20*min(max((sat_mean-70)/80.0, 0.0), 1.0)
        )

        ok = (frac_overall > 0.10 and frac_blob > 0.10 and tall_enough and below_mid and sat_mean > 70)
        if ok:
            return True, float(score)

    return False, 0.0

# ----- eyebrows / gap / together -----
def idx_set(conns):
    s = set()
    for a, b in conns: s.add(a); s.add(b)
    return sorted(s)

def eyebrow_groups(lm, w, h):
    LB = idx_set(mpfm.FACEMESH_LEFT_EYEBROW)
    RB = idx_set(mpfm.FACEMESH_RIGHT_EYEBROW)
    nose_x = xy(lm, 1, w, h)[0]

    def inner_outer(idxs):
        pts = [(i, xy(lm, i, w, h)) for i in idxs]
        pts.sort(key=lambda t: abs(t[1][0]-nose_x))
        n = max(1, len(pts)//3)
        inner = np.array([p[1] for p in pts[:n]])
        outer = np.array([p[1] for p in pts[-n:]])
        return inner, outer

    Li, Lo = inner_outer(LB); Ri, Ro = inner_outer(RB)
    return Li, Lo, Ri, Ro

def brow_signals(lm, w, h, ipd):
    Li, Lo, Ri, Ro = eyebrow_groups(lm, w, h)
    Li_y, Lo_y, Ri_y, Ro_y = Li[:,1].mean(), Lo[:,1].mean(), Ri[:,1].mean(), Ro[:,1].mean()
    inner_raise = 0.5*((Lo_y - Li_y)/ipd + (Ro_y - Ri_y)/ipd)  # + = inner higher

    Leye_y = 0.5*(xy(lm, L_UP, w, h)[1] + xy(lm, L_DN, w, h)[1])
    Reye_y = 0.5*(xy(lm, R_UP, w, h)[1] + xy(lm, R_DN, w, h)[1])
    brow_y = 0.25*(Li_y+Lo_y+Ri_y+Ro_y)
    gap = (brow_y - 0.5*(Leye_y+Reye_y)) / ipd

    Lc, Rc = Li.mean(axis=0), Ri.mean(axis=0)
    inter_inner = np.hypot(*(Lc - Rc)) / ipd
    return inner_raise, gap, inter_inner

# ---------------- calibration ----------------
class Calib:
    def __init__(self):
        self.ready = False
        self.mars = deque(maxlen=90)
        self.ears = deque(maxlen=90)
        self.gaps = deque(maxlen=90)
        self.inner_raise = deque(maxlen=90)
        self.inter_inner = deque(maxlen=90)
        self.phils = deque(maxlen=90)
        self.noses = deque(maxlen=90)
        self.lopens = deque(maxlen=90)
        self.ropens = deque(maxlen=90)

    def push(self, mar, l_open, r_open, ear, gap, ir, ii, phil=None, nosew=None):
        self.mars.append(mar)
        self.ears.append(ear)
        self.gaps.append(gap)
        self.inner_raise.append(ir)
        self.inter_inner.append(ii)
        self.lopens.append(l_open)
        self.ropens.append(r_open)
        if phil is not None: self.phils.append(phil)
        if nosew is not None: self.noses.append(nosew)

    def finalize(self):
        if len(self.mars) < 30: return False
        self.MAR0 = float(np.median(self.mars))
        self.EAR0 = float(np.median(self.ears))
        self.GAP0 = float(np.median(self.gaps))
        self.IR0  = float(np.median(self.inner_raise))
        self.II0  = float(np.median(self.inter_inner))
        self.L0   = float(np.median(self.lopens))
        self.R0   = float(np.median(self.ropens))
        self.PHIL0 = float(np.median(self.phils)) if self.phils else 0.10
        self.NOSE0 = float(np.median(self.noses)) if self.noses else 0.35
        self.ready = True
        return True

# ---------------- knobs (relative to neutral) ----------------
MOUTH_OPEN_DELTA   = 0.08
MOUTH_BIG_DELTA    = 0.20
EYES_WIDE_DELTA    = 0.05
BROWS_UP_DELTA     = 0.06
CORNERS_DOWN_ABS   = 0.40

# Wink heuristics: closed eye < 55% of baseline, other > 85% baseline
WINK_CLOSE_RATIO   = 0.55
WINK_OPEN_RATIO    = 0.85
WINK_COOLDOWN_FR   = 20  # frames between triggers

# ---------------- audio (pygame.mixer) ----------------
_mixer_ok = False
def audio_init():
    global _mixer_ok
    try:
        import pygame
        from pygame import mixer
        pygame.mixer.init()  # just mixer, no window
        _mixer_ok = True
        # try WAV, then MP3 fallback
        path = MUSIC_WAV if os.path.exists(MUSIC_WAV) else (MUSIC_MP3 if os.path.exists(MUSIC_MP3) else None)
        if path:
            pygame.mixer.music.load(path)
        else:
            print("Audio: no music file found. Add assets/music_wink_start.wav (preferred) or .mp3")
    except Exception as e:
        print(f"Audio init failed: {e}")
        _mixer_ok = False

def audio_play_once():
    if not _mixer_ok:
        return
    try:
        import pygame
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()  # play once
    except Exception as e:
        print(f"Audio play failed: {e}")

# ---------------- decision (tongue first) ----------------
def decide_state(frame, face, calib):
    h, w = frame.shape[:2]
    lm = face.landmark

    l_open, r_open, ear, ipd = eye_metrics(lm, w, h)
    mar, corner_drop, inner, below, lower_y, mw, mh = mouth_boxes(frame, lm, w, h)
    ir, gap, ii = brow_signals(lm, w, h, ipd)
    # nose metrics not needed for tongue/wink but kept for logs
    Uphil, Unosew = 0.0, 0.0

    tongue_ok, tongue_score = has_tongue(inner, below, lower_y, mw, mh)

    if not calib.ready:
        return "neutral", (mar, ear, gap, ir, ii, Uphil, Unosew, tongue_score, l_open, r_open)

    # relative gates
    mouth_open = mar > calib.MAR0 + MOUTH_OPEN_DELTA
    mouth_big  = mar > calib.MAR0 + MOUTH_BIG_DELTA
    eyes_wide  = ear > calib.EAR0 + EYES_WIDE_DELTA
    brows_up   = gap > calib.GAP0 + BROWS_UP_DELTA

    # state priority: shock > tongue > cry > neutral
    if mouth_big and (brows_up or eyes_wide):
        return "shock", (mar, ear, gap, ir, ii, Uphil, Unosew, tongue_score, l_open, r_open)

    if tongue_ok and not (brows_up or eyes_wide):
        return "tongue", (mar, ear, gap, ir, ii, Uphil, Unosew, tongue_score, l_open, r_open)

    if (corner_drop > CORNERS_DOWN_ABS) and (ir > calib.IR0 + 0.05) and not mouth_open:
        return "cry", (mar, ear, gap, ir, ii, Uphil, Unosew, tongue_score, l_open, r_open)

    return "neutral", (mar, ear, gap, ir, ii, Uphil, Unosew, tongue_score, l_open, r_open)

# ---------------- main loop ----------------
def main():
    audio_init()

    cam_idx = 0
    cap = open_cam(cam_idx)
    if cap is None:
        print("No camera found"); return

    cv2.namedWindow("Apple", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Apple", APPLE_WINDOW_SIZE, APPLE_WINDOW_SIZE)

    state, stable = "neutral", 0
    calib, t0 = Calib(), time.time()
    print("Calibrating (~2s). Keep a relaxed neutral face...")

    log_ctr = 0
    music_started = False
    wink_cooldown = 0  # frames left

    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = FACE_MESH.process(rgb)

        next_state = "neutral"
        l_open = r_open = 0.0

        if out.multi_face_landmarks:
            f = out.multi_face_landmarks[0]
            lm = f.landmark
            l_open, r_open, ear, ipd = eye_metrics(lm, *frame.shape[1::-1])
            mar, _, inner, below, lower_y, mw, mh = mouth_boxes(frame, lm, *frame.shape[1::-1])
            ir, gap, ii = brow_signals(lm, *frame.shape[1::-1], ipd)

            if not calib.ready:
                # push for calibration
                calib.push(mar, l_open, r_open, ear, gap, ir, ii)
                if time.time() - t0 > 2.0 and calib.finalize():
                    print("Calibration done.")
                next_state = "neutral"
            else:
                next_state, metrics = decide_state(frame, f, calib)
                # logging
                log_ctr = (log_ctr + 1) % LOG_EVERY
                if log_ctr == 0:
                    mar_m, ear_m, gap_m, ir_m, ii_m, ph_m, nw_m, tscore, lo_m, ro_m = metrics
                    print(f"STATE={next_state:7s} MAR={mar_m:.2f} EAR={ear_m:.2f} GAP={gap_m:.2f} "
                          f"IR={ir_m:.2f} II={ii_m:.2f} LOPEN={lo_m:.2f} ROPEN={ro_m:.2f} TONGUE={tscore:.2f}")

                # ---- wink detection -> start music (once) ----
                if wink_cooldown > 0:
                    wink_cooldown -= 1
                else:
                    left_closed  = l_open < calib.L0 * WINK_CLOSE_RATIO
                    right_open   = r_open > calib.R0 * WINK_OPEN_RATIO
                    right_closed = r_open < calib.R0 * WINK_CLOSE_RATIO
                    left_open    = l_open > calib.L0 * WINK_OPEN_RATIO

                    wink_now = (left_closed and right_open) or (right_closed and left_open)
                    if wink_now:
                        if not music_started:
                            audio_play_once()
                            music_started = True
                            print("♫ Wink detected — music started.")
                        wink_cooldown = WINK_COOLDOWN_FR

        else:
            next_state = "neutral"

        # hysteresis for apple sprite
        if next_state == state:
            stable = min(stable + 1, 12)
        else:
            stable -= 1
            if stable <= -2:
                state = next_state
                stable = 0

        cv2.imshow("Apple", fit_square(APPLE[state], 600))
        if SHOW_CAMERA_FEED:
            cv2.imshow("Camera", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if k == ord('c'):
            calib, t0 = Calib(), time.time()
            print("Recalibrating: neutral face please...")
        elif k == ord(']'):  # next camera
            new_idx, new_cap = cycle_camera(cam_idx, +1)
            if new_cap is not None:
                cap.release()
                cap = new_cap
                cam_idx = new_idx
                print(f"Switched camera to index {cam_idx}")
            else:
                print("No other camera found.")
        elif k == ord('['):  # previous camera
            new_idx, new_cap = cycle_camera(cam_idx, -1)
            if new_cap is not None:
                cap.release()
                cap = new_cap
                cam_idx = new_idx
                print(f"Switched camera to index {cam_idx}")
            else:
                print("No other camera found.")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# ---------------- end ----------------
