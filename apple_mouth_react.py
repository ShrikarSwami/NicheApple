import cv2, numpy as np, mediapipe as mp, time
from collections import deque

# ===================== config =====================
SHOW_CAMERA_FEED = True         # raw camera (no overlays)
APPLE_WINDOW_SIZE = 620         # size of apple window
LOG_EVERY = 15                  # print metrics every N frames

ASSET_FILES = {
    "neutral": "assets/apple_neutral.png",
    "tongue":  "assets/apple_tongue.png",
    "shock":   "assets/apple_shock.png",
    "angry":   "assets/apple_angry.png",
    "cry":     "assets/apple_cry.png",
}

# ======= knobs (relative to calibrated neutral) =======
MOUTH_OPEN_DELTA      = 0.08
MOUTH_BIG_DELTA       = 0.20
EYES_WIDE_DELTA       = 0.05

BROWS_UP_DELTA        = 0.06
BROWS_LOW_DELTA       = 0.05
BROWS_TOGETHER_DEL    = 0.05
BROWS_INNER_DOWN      = 0.05

CORNERS_DOWN_ABS      = 0.40  # absolute, not relative

NOSE_SCRUNCH_PHIL_DEL = 0.03  # philtrum shorter than neutral
NOSE_SCRUNCH_NOST_DEL = 0.03  # nose bbox wider than neutral

# stricter angry extras so it feels intentional
ANGRY_IR_EXTRA        = 0.03
ANGRY_II_EXTRA        = 0.06

# ===================== assets =====================
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

# ===================== mediapipe =====================
mpfm = mp.solutions.face_mesh
FACE_MESH = mpfm.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Eyes (robust refs)
L_OUT, L_IN, L_UP, L_DN = 33, 133, 159, 145
R_OUT, R_IN, R_UP, R_DN = 263, 362, 386, 374

# Mouth keypoints
LEFT_CORNER, RIGHT_CORNER, UPPER_CENTER, LOWER_CENTER = 61, 291, 13, 14

def xy(lm, i, w, h):
    p = lm[i]; return int(p.x*w), int(p.y*h)

def eye_metrics(lm, w, h):
    lw = np.hypot(*(np.array(xy(lm, L_OUT, w, h)) - np.array(xy(lm, L_IN, w, h))))
    rw = np.hypot(*(np.array(xy(lm, R_OUT, w, h)) - np.array(xy(lm, R_IN, w, h))))
    l_open = abs(xy(lm, L_UP, w, h)[1]-xy(lm, L_DN, w, h)[1]) / max(lw, 1.0)
    r_open = abs(xy(lm, R_UP, w, h)[1]-xy(lm, R_DN, w, h)[1]) / max(rw, 1.0)
    ear = 0.5*(l_open + r_open)
    ipd = np.hypot(*(np.array(xy(lm, L_IN, w, h)) - np.array(xy(lm, R_IN, w, h))))
    return ear, max(ipd, 1.0)

# ========== mouth boxes (tight inner + expanded big ROI) ==========
def mouth_boxes(frame, lm, w, h):
    L = xy(lm, LEFT_CORNER, w, h); R = xy(lm, RIGHT_CORNER, w, h)
    U = xy(lm, UPPER_CENTER, w, h); D = xy(lm, LOWER_CENTER, w, h)

    mw = max(1, abs(R[0]-L[0])); mh = max(1, abs(D[1]-U[1]))
    mar = mh / float(mw)
    center_y = (U[1]+D[1])//2; corners_y = (L[1]+R[1])//2
    corner_drop = (corners_y - center_y) / float(mh)

    # inner mouth (tight) — reduces lip contamination
    x1, x2 = min(L[0], R[0]), max(L[0], R[0])
    y1, y2 = min(U[1], D[1]), max(U[1], D[1])
    sx = int(x1 + 0.20*mw); ex = int(x2 - 0.20*mw)
    sy = int(y1 + 0.22*mh); ey = int(y2 - 0.06*mh)
    H, W = frame.shape[:2]
    sx, sy = max(0, sx), max(0, sy)
    ex, ey = min(W-1, ex), min(H-1, ey)
    inner = frame[sy:ey, sx:ex].copy() if ex>sx and ey>sy else None

    # expanded mouth (captures long/protruding tongue)
    bx1 = int(x1 - 0.15*mw); bx2 = int(x2 + 0.15*mw)
    by1 = y1;                 by2 = int(y2 + 0.70*mh)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(W-1, bx2), min(H-1, by2)
    big = frame[by1:by2, bx1:bx2].copy() if bx2>bx1 and by2>by1 else None

    return mar, corner_drop, inner, big

# ========== tongue detection ==========
def _red_mask_hsv(bgr):
    if bgr is None or bgr.size == 0: return None, None
    bgr = cv2.GaussianBlur(bgr, (3,3), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[...,2] = cv2.equalizeHist(hsv[...,2])
    m1 = cv2.inRange(hsv, (0,   70, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 70, 60), (180,255, 255))
    m = cv2.bitwise_or(m1, m2)
    k = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    return m, hsv

def has_tongue(inner_bgr, big_bgr):
    """
    returns True if we believe tongue is visible.
    strategy:
      1) try inner ROI (cleanest)
      2) fallback to big ROI (catches long protrudes)
      - require blob to be low, tall-ish, sufficiently red/saturated
    """
    for roi in (inner_bgr, big_bgr):
        mask, hsv = _red_mask_hsv(roi)
        if mask is None: 
            continue
        nz = cv2.countNonZero(mask)
        if nz == 0:
            continue

        H, W = mask.shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        frac_overall = nz / float(H*W)
        frac_blob    = (w*h) / float(H*W)
        height_ratio = h / float(H)
        cy = y + h/2.0
        below_mid = cy > (0.52*H)
        sat_mean = cv2.mean(hsv[...,1], mask=mask)[0]

        # LAB a* ensures "reddish/pink" not just bright
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        a_mean = cv2.mean(lab[...,1], mask=mask)[0]

        ok = (frac_overall > 0.12 and frac_blob > 0.10 and height_ratio > 0.28
              and below_mid and sat_mean > 70 and a_mean > 145)
        if ok:
            return True
    return False

# ========== brows / gap / together / slant ==========
def idx_set(conns):
    s = set()
    for a,b in conns: s.add(a); s.add(b)
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

# ========== nose scrunch ==========
NOSE = idx_set(mpfm.FACEMESH_NOSE)

def nose_metrics(lm, w, h, ipd):
    pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in NOSE])
    x1, y1 = pts[:,0].min(), pts[:,1].min()
    x2, y2 = pts[:,0].max(), pts[:,1].max()
    nose_w = (x2 - x1) / max(ipd, 1.0)    # nostril flare proxy
    U = int(lm[UPPER_CENTER].y * h)
    philtrum = (y2 - U) / max(ipd, 1.0)   # shorter when scrunched
    return philtrum, nose_w

# ===================== calibration =====================
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

    def push(self, mar, ear, gap, ir, ii, phil=None, nosew=None):
        self.mars.append(mar); self.ears.append(ear); self.gaps.append(gap)
        self.inner_raise.append(ir); self.inter_inner.append(ii)
        if phil is not None: self.phils.append(phil)
        if nosew is not None: self.noses.append(nosew)

    def finalize(self):
        if len(self.mars) < 30: return False
        self.MAR0  = float(np.median(self.mars))
        self.EAR0  = float(np.median(self.ears))
        self.GAP0  = float(np.median(self.gaps))
        self.IR0   = float(np.median(self.inner_raise))
        self.II0   = float(np.median(self.inter_inner))
        self.PHIL0 = float(np.median(self.phils)) if self.phils else 0.10
        self.NOSE0 = float(np.median(self.noses)) if self.noses else 0.35
        self.ready = True
        return True

# ===================== state decision =====================
def decide_state(frame, lm, calib):
    h, w = frame.shape[:2]

    ear, ipd = eye_metrics(lm, w, h)
    mar, corner_drop, inner, big = mouth_boxes(frame, lm, w, h)
    ir, gap, ii = brow_signals(lm, w, h, ipd)
    phil, nosew = nose_metrics(lm, w, h, ipd)
    tongue = has_tongue(inner, big)

    if not calib.ready:
        return "neutral", (mar, ear, gap, ir, ii, phil, nosew)

    # relative gates
    mouth_open = mar > calib.MAR0 + MOUTH_OPEN_DELTA
    mouth_big  = mar > calib.MAR0 + MOUTH_BIG_DELTA
    eyes_wide  = ear > calib.EAR0 + EYES_WIDE_DELTA
    brows_up   = gap > calib.GAP0 + BROWS_UP_DELTA
    brows_low  = gap < calib.GAP0 - BROWS_LOW_DELTA
    brows_together = ii < calib.II0 - BROWS_TOGETHER_DEL
    inner_down = ir < calib.IR0 - BROWS_INNER_DOWN
    inner_up   = ir > calib.IR0 + BROWS_INNER_DOWN
    scrunch = (phil < calib.PHIL0 - NOSE_SCRUNCH_PHIL_DEL) and \
              (nosew > calib.NOSE0 + NOSE_SCRUNCH_NOST_DEL)

    # extra strictness for angry "intent"
    ir_low_enough   = ir < (calib.IR0 - (BROWS_INNER_DOWN + ANGRY_IR_EXTRA))
    ii_small_enough = ii < (calib.II0 - (BROWS_TOGETHER_DEL + ANGRY_II_EXTRA))

    # priority:
    # 1) shock
    if mouth_big and (brows_up or eyes_wide):
        return "shock", (mar, ear, gap, ir, ii, phil, nosew)
    # 2) tongue (no need to require mouth_open; large protrudes still win)
    if tongue and not (brows_up or eyes_wide):
        return "tongue", (mar, ear, gap, ir, ii, phil, nosew)
    # 3) cry
    if (corner_drop > CORNERS_DOWN_ABS) and inner_up and not mouth_open:
        return "cry", (mar, ear, gap, ir, ii, phil, nosew)
    # 4) angry (stacked cues so it doesn’t false positive)
    if (brows_low and not mouth_open and not eyes_wide):
        if (brows_together or inner_down) and (scrunch or ir_low_enough or ii_small_enough):
            return "angry", (mar, ear, gap, ir, ii, phil, nosew)

    return "neutral", (mar, ear, gap, ir, ii, phil, nosew)

# ===================== main =====================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found"); return

    cv2.namedWindow("Apple", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Apple", APPLE_WINDOW_SIZE, APPLE_WINDOW_SIZE)

    state, stable = "neutral", 0
    calib, t0 = Calib(), time.time()
    print("Calibrating (~2s). Keep a relaxed neutral face...")

    log_ctr = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = FACE_MESH.process(rgb)

        next_state = "neutral"
        if out.multi_face_landmarks:
            lm = out.multi_face_landmarks[0].landmark

            # one pass of metrics, use for both calibration and decision
            h, w = frame.shape[:2]
            ear, ipd = eye_metrics(lm, w, h)
            mar, corner_drop, inner, big = mouth_boxes(frame, lm, w, h)
            ir, gap, ii = brow_signals(lm, w, h, ipd)
            phil, nosew = nose_metrics(lm, w, h, ipd)

            if not calib.ready:
                calib.push(mar, ear, gap, ir, ii, phil, nosew)
                if time.time() - t0 > 2.0 and calib.finalize():
                    print("Calibration done.")
                # while calibrating: hold neutral apple
                next_state = "neutral"
            else:
                # already calibrated: decide
                # reuse metrics instead of recomputing
                next_state, _ = decide_state(frame, lm, calib)

                # logging for tuning
                log_ctr = (log_ctr + 1) % LOG_EVERY
                if log_ctr == 0:
                    print(f"STATE={next_state:7s} MAR={mar:.2f} EAR={ear:.2f} "
                          f"GAP={gap:.2f} IR={ir:.2f} II={ii:.2f} PH={phil:.2f} NW={nosew:.2f}")
        else:
            next_state = "neutral"

        # hysteresis to avoid flicker
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
        if k in (27, ord('q')): break
        if k == ord('c'):
            calib, t0 = Calib(), time.time()
            print("Recalibrating: neutral face please...")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
