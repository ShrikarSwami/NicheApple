import cv2, numpy as np, mediapipe as mp, time
from collections import deque

SHOW_CAMERA_FEED = True
APPLE_WINDOW_SIZE = 620

ASSET_FILES = {
    "neutral": "assets/apple_neutral.png",
    "tongue":  "assets/apple_tongue.png",
    "shock":   "assets/apple_shock.png",
    "angry":   "assets/apple_angry.png",
    "cry":     "assets/apple_cry.png",
}

# ---------- helpers: assets ----------
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
    scale = size / max(h, w)
    out = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    top = (size - out.shape[0]) // 2
    bottom = size - out.shape[0] - top
    left = (size - out.shape[1]) // 2
    right = size - out.shape[1] - left
    return cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))

APPLE = {k: fit_square(load_png(v, k), 600) for k, v in ASSET_FILES.items()}

# ---------- mediapipe ----------
mpfm = mp.solutions.face_mesh
FACE_MESH = mpfm.FaceMesh(static_image_mode=False, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.6,
                          min_tracking_confidence=0.6)

# Eye landmarks (stable)
L_OUT, L_IN, L_UP, L_DN = 33, 133, 159, 145
R_OUT, R_IN, R_UP, R_DN = 263, 362, 386, 374

# Mouth landmarks
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

def mouth_metrics(frame, lm, w, h):
    L = xy(lm, LEFT_CORNER, w, h); R = xy(lm, RIGHT_CORNER, w, h)
    U = xy(lm, UPPER_CENTER, w, h); D = xy(lm, LOWER_CENTER, w, h)
    mw = max(1, abs(R[0]-L[0])); mh = max(1, abs(D[1]-U[1]))
    mar = mh / float(mw)
    center_y = (U[1]+D[1])//2; corners_y = (L[1]+R[1])//2
    corner_drop = (corners_y - center_y) / float(mh)

    # inner mouth crop (tighter sides to exclude lips)
    # inner mouth crop (wider & deeper so big tongues fit)
    x1, x2 = min(L[0], R[0]), max(L[0], R[0])
    y1, y2 = min(U[1], D[1]), max(U[1], D[1])
    sx = int(x1 + 0.18*mw)       # was 0.22
    ex = int(x2 - 0.18*mw)       # was 0.22
    sy = int(y1 + 0.22*mh)       # was 0.27
    ey = int(y2 - 0.02*mh)       # was 0.12  (captures lower tongue)
    sx, sy = max(0, sx), max(0, sy)
    ex, ey = min(w-1, ex), min(h-1, ey)
    inner = frame[sy:ey, sx:ex].copy() if ex>sx and ey>sy else None

    return mar, corner_drop, inner

def red_tongue(inner_bgr):
    if inner_bgr is None or inner_bgr.size == 0:
        return False, 0.0
    # denoise a hair
    inner_bgr = cv2.GaussianBlur(inner_bgr, (3,3), 0)
    hsv = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
    hsv[...,2] = cv2.equalizeHist(hsv[...,2])

    # red mask
    m1 = cv2.inRange(hsv, (0, 90, 60),  (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 90, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return False, 0.0

    # largest red blob stats (size + where in mouth)
    H, W = mask.shape[:2]
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    area_largest = w*h
    frac_largest = area_largest / float(H*W)
    frac_overall = float(cv2.countNonZero(mask)) / (H*W)
    height_ratio = h / float(H)             # tall tongue = big number
    cy = y + h/2.0                          # vertical center of blob
    below_mid = cy > (0.55*H)               # tongue sits in lower half
    sat_mean = cv2.mean(hsv[...,1], mask=mask)[0]

    ok = (frac_overall > 0.16 and frac_largest > 0.14 and
          height_ratio > 0.30 and below_mid and sat_mean > 90)
    return ok, frac_overall

# eyebrow groups via mesh constants
def eyebrow_groups(lm, w, h):
    LB = set(i for a,b in mpfm.FACEMESH_LEFT_EYEBROW for i in (a,b))
    RB = set(i for a,b in mpfm.FACEMESH_RIGHT_EYEBROW for i in (a,b))
    nose_x = xy(lm, 1, w, h)[0]

    def inner_outer(index_set):
        pts = [(i, xy(lm, i, w, h)) for i in index_set]
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
    # positive = inner higher than outer; negative = inner lower (angry slant)
    inner_raise = 0.5*((Lo_y - Li_y)/ipd + (Ro_y - Ri_y)/ipd)

    Leye_y = 0.5*(xy(lm, L_UP, w, h)[1] + xy(lm, L_DN, w, h)[1])
    Reye_y = 0.5*(xy(lm, R_UP, w, h)[1] + xy(lm, R_DN, w, h)[1])
    brow_y = 0.25*(Li_y+Lo_y+Ri_y+Ro_y)
    gap = (brow_y - 0.5*(Leye_y+Reye_y)) / ipd

    # distance between inner brow centroids (together if smaller)
    Lc, Rc = Li.mean(axis=0), Ri.mean(axis=0)
    inter_inner = np.hypot(*(Lc - Rc)) / ipd
    return inner_raise, gap, inter_inner

# Nose set for bbox/metrics
def idx_set(conns):
    s = set()
    for a, b in conns: s.add(a); s.add(b)
    return sorted(s)

NOSE = idx_set(mpfm.FACEMESH_NOSE)  # new

def nose_metrics(lm, w, h, ipd):
    # bbox of nose region
    pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in NOSE])
    x1, y1 = pts[:,0].min(), pts[:,1].min()
    x2, y2 = pts[:,0].max(), pts[:,1].max()
    nose_w = (x2 - x1) / max(ipd, 1.0)    # nostril flare proxy

    # philtrum = distance from nose base to upper lip center
    U = int(lm[UPPER_CENTER].y * h)        # inner upper lip landmark (13)
    nose_base_y = y2                       # bottom of nose bbox
    philtrum = (nose_base_y - U) / max(ipd, 1.0)

    return philtrum, nose_w


# ---------- calibration ----------
class Calib:
    def __init__(self):
        self.ready = False
        self.mars = deque(maxlen=90)
        self.ears = deque(maxlen=90)
        self.gaps = deque(maxlen=90)
        self.inner_raise = deque(maxlen=90)
        self.inter_inner = deque(maxlen=90)
        self.phils = deque(maxlen=90)      # NEW
        self.noses = deque(maxlen=90)      # NEW

    def push(self, mar, ear, gap, ir, ii, phil=None, nosew=None):
        self.mars.append(mar)
        self.ears.append(ear)
        self.gaps.append(gap)
        self.inner_raise.append(ir)
        self.inter_inner.append(ii)
        if phil is not None: self.phils.append(phil)
        if nosew is not None: self.noses.append(nosew)

    def finalize(self):
        if len(self.mars) < 30: return False
        self.MAR0 = float(np.median(self.mars))
        self.EAR0 = float(np.median(self.ears))
        self.GAP0 = float(np.median(self.gaps))
        self.IR0  = float(np.median(self.inner_raise))
        self.II0  = float(np.median(self.inter_inner))
        # NEW baselines
        self.PHIL0 = float(np.median(self.phils)) if self.phils else 0.10
        self.NOSE0 = float(np.median(self.noses)) if self.noses else 0.35
        self.ready = True
        return True

# Tuning knobs (relative to your neutral)
MOUTH_OPEN_DELTA   = 0.10
MOUTH_BIG_DELTA    = 0.22
EYES_WIDE_DELTA    = 0.05
BROWS_UP_DELTA     = 0.06
BROWS_LOW_DELTA    = 0.05
BROWS_TOGETHER_DEL = 0.04
BROWS_INNER_DOWN   = 0.04
CORNERS_DOWN_ABS   = 0.40
NOSE_SCRUNCH_PHIL_DEL = 0.03
NOSE_SCRUNCH_NOST_DEL = 0.03
ANGRY_IR_EXTRA = 0.03   
ANGRY_II_EXTRA = 0.06   




def decide_state(frame, face, calib):
    h, w = frame.shape[:2]
    lm = face.landmark

    ear, ipd = eye_metrics(lm, w, h)
    mar, corner_drop, inner = mouth_metrics(frame, lm, w, h)  # capture corner_drop & inner
    ir, gap, ii = brow_signals(lm, w, h, ipd)
    phil, nosew = nose_metrics(lm, w, h, ipd)
    tongue, _ = red_tongue(inner)  # detect tongue here

    if not calib.ready:
        # while calibrating we just pass metrics back for logging
        return "neutral", (mar, ear, gap, ir, ii, phil, nosew)

    # gates (relative to neutral baselines)
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

    # priority: SHOCK before TONGUE
    if mouth_big and (brows_up or eyes_wide):
        return "shock", (mar, ear, gap, ir, ii, phil, nosew)
    if mouth_open and tongue and not (brows_up or eyes_wide):
        return "tongue", (mar, ear, gap, ir, ii, phil, nosew)
    if (corner_drop > CORNERS_DOWN_ABS) and inner_up and not mouth_open:
        return "cry", (mar, ear, gap, ir, ii, phil, nosew)
    if (brows_low and not mouth_open and not eyes_wide):
        if ( (brows_together or inner_down) and (scrunch or ir_low_enough or ii_small_enough) ):
            return "angry", (mar, ear, gap, ir, ii, phil, nosew)
    return "neutral", (mar, ear, gap, ir, ii, phil, nosew)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found"); return

    cv2.namedWindow("Apple", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Apple", APPLE_WINDOW_SIZE, APPLE_WINDOW_SIZE)

    state, stable = "neutral", 0
    calib, t0 = Calib(), time.time()
    print("Calibrating (~2s). Keep a relaxed neutral face...")

    log_every = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = FACE_MESH.process(rgb)

        next_state = "neutral"
        if out.multi_face_landmarks:
            f = out.multi_face_landmarks[0]
            if not calib.ready:
                h, w = frame.shape[:2]; lm = f.landmark
                ear, ipd = eye_metrics(lm, w, h)
                mar, _, _ = mouth_metrics(frame, lm, w, h)
                ir, gap, ii = brow_signals(lm, w, h, ipd)
                phil, nosew = nose_metrics(lm, w, h, ipd)  # NEW
                calib.push(mar, ear, gap, ir, ii, phil, nosew)
                if time.time() - t0 > 2.0 and calib.finalize():
                    print("Calibration done.")
            next_state, metrics = decide_state(frame, f, calib)
            # print metrics ~2x/sec to help tune (no overlay)
            log_every = (log_every + 1) % 15
            if log_every == 0 and calib.ready:
                mar, ear, gap, ir, ii, phil, nosew = metrics
                print(f"STATE={next_state:7s} MAR={mar:.2f} EAR={ear:.2f} GAP={gap:.2f} IR={ir:.2f} II={ii:.2f} PH={phil:.2f} NW={nosew:.2f}")
        else:
            next_state = "neutral"

        if next_state == state:
            stable = min(stable + 1, 12)
        else:
            stable -= 1
            if stable <= -2:
                state = next_state
                stable = 0

        cv2.imshow("Apple", fit_square(APPLE[state], 600))
        if SHOW_CAMERA_FEED: cv2.imshow("Camera", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('c'):
            calib, t0 = Calib(), time.time()
            print("Recalibrating: neutral face please...")
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
