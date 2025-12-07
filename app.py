# app.py (Render-optimized)
import io
import os
import time
import gc
from typing import Tuple

from PIL import Image
import numpy as np
import cv2
from flask import Flask, jsonify, make_response, request, url_for, send_from_directory

# Reduce thread/multi-threaded BLAS usage (lowers memory)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Minimum visibility to draw a landmark / connection
MIN_VISIBILITY = 0.40


# ---------------------
# Utilities
# ---------------------
def save_jpg_from_bgr(bgr_img: np.ndarray, path: str, quality: int = 85) -> None:
    """Save numpy BGR image as JPEG (cv2 uses BGR)."""
    # Use cv2.imencode to avoid PIL conversions
    ok, enc = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        with open(path, "wb") as f:
            f.write(enc.tobytes())


def pil_resize_and_to_bgr(file_storage, max_size: int = 640) -> np.ndarray:
    """
    Read file (werkzeug.FileStorage), convert to RGB, resize keeping aspect ratio,
    and return an OpenCV-compatible BGR numpy array.
    """
    img = Image.open(file_storage.stream).convert("RGB")
    # Resize preserving aspect ratio, max dimension = max_size
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img)
    # RGB -> BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # reset stream pointer for potential reuse (defensive)
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass
    return bgr


def to_px(landmark, image_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    h, w = image_shape[0], image_shape[1]
    return (int(landmark.x * w), int(landmark.y * h))


def euclid(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def calculate_angle_pts(a_pt, b_pt, c_pt):
    a = np.array(a_pt).astype(float)
    b = np.array(b_pt).astype(float)
    c = np.array(c_pt).astype(float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return angle


def annotate_landmarks(image: np.ndarray, landmarks):
    """
    Draw landmarks & connections on BGR image using MediaPipe PoseLandmark indices.
    `landmarks` is a sequence of normalized landmarks.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # create pixel list with visibility
    pts = []
    for i, lm in enumerate(landmarks):
        vis = getattr(lm, "visibility", 1.0)
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y, vis))

    # Use mp.solutions.pose.POSE_CONNECTIONS dynamically to avoid heavy globals
    import mediapipe as mp_local
    connections = mp_local.solutions.pose.POSE_CONNECTIONS
    # Draw connections
    for connection in connections:
        s = connection[0].value if hasattr(connection[0], "value") else int(connection[0])
        e = connection[1].value if hasattr(connection[1], "value") else int(connection[1])
        if s < len(pts) and e < len(pts):
            xs, ys, vs = pts[s]
            xe, ye, ve = pts[e]
            if vs >= MIN_VISIBILITY and ve >= MIN_VISIBILITY:
                cv2.line(img, (xs, ys), (xe, ye), (20, 200, 20), 2, cv2.LINE_AA)

    # Draw landmarks + indices
    for i, (x, y, v) in enumerate(pts):
        if v >= MIN_VISIBILITY:
            cv2.circle(img, (x, y), 4, (0, 180, 255), -1)
            cv2.putText(img, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 2, (150, 150, 150), 1)
    return img


# ---------------------
# Scoring logic (left as-is but minor safety guards)
# ---------------------
def score_from_views(front_lm, rear_lm, side_lm, front_img, rear_img, side_img):
    breakdown = {}
    total = 0.0

    # (Same checks as your original, with try/except guards)
    try:
        import mediapipe as mp_local
        PoseLandmark = mp_local.solutions.pose.PoseLandmark

        # FRONT
        if front_lm:
            lw = front_lm[PoseLandmark.LEFT_WRIST.value]
            rw = front_lm[PoseLandmark.RIGHT_WRIST.value]
            la = front_lm[PoseLandmark.LEFT_ANKLE.value]
            ra = front_lm[PoseLandmark.RIGHT_ANKLE.value]
            ls = front_lm[PoseLandmark.LEFT_SHOULDER.value]
            rs = front_lm[PoseLandmark.RIGHT_SHOULDER.value]

            def near_bottom(pt, tol=0.15):
                return (1.0 - pt.y) < tol

            hands_feet_ground = all([near_bottom(x) for x in (lw, rw, la, ra)])
            wr_dist = abs(lw.x - rw.x)
            sh_dist = abs(ls.x - rs.x)
            ratio = wr_dist / (sh_dist + 1e-8)
            s1 = 1.0 if (hands_feet_ground and 0.75 <= ratio <= 1.25) else (0.5 if hands_feet_ground else 0.0)
            breakdown["front_hands_feet"] = s1
            total += s1

            # foot straight
            try:
                lk_angle = calculate_angle_pts(
                    to_px(front_lm[PoseLandmark.LEFT_KNEE.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.LEFT_ANKLE.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.LEFT_FOOT_INDEX.value], front_img.shape),
                )
                rk_angle = calculate_angle_pts(
                    to_px(front_lm[PoseLandmark.RIGHT_KNEE.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.RIGHT_ANKLE.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.RIGHT_FOOT_INDEX.value], front_img.shape),
                )
                s2 = 0.5 if (lk_angle > 160 and rk_angle > 160) else 0.0
            except Exception:
                s2 = 0.0
            breakdown["front_foot_straight"] = s2
            total += s2

            # hand straight
            try:
                left_hand_ang = calculate_angle_pts(
                    to_px(front_lm[PoseLandmark.LEFT_SHOULDER.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.LEFT_ELBOW.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.LEFT_WRIST.value], front_img.shape),
                )
                right_hand_ang = calculate_angle_pts(
                    to_px(front_lm[PoseLandmark.RIGHT_SHOULDER.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.RIGHT_ELBOW.value], front_img.shape),
                    to_px(front_lm[PoseLandmark.RIGHT_WRIST.value], front_img.shape),
                )
                s3 = 0.5 if (left_hand_ang > 160 and right_hand_ang > 160) else 0.0
            except Exception:
                s3 = 0.0
            breakdown["front_hand_straight"] = s3
            total += s3

        # REAR
        if rear_lm:
            def near_bottom_rel(pt, tol=0.15):
                return (1.0 - pt.y) < tol
            lw = rear_lm[PoseLandmark.LEFT_WRIST.value]
            rw = rear_lm[PoseLandmark.RIGHT_WRIST.value]
            la = rear_lm[PoseLandmark.LEFT_ANKLE.value]
            ra = rear_lm[PoseLandmark.RIGHT_ANKLE.value]
            ls = rear_lm[PoseLandmark.LEFT_SHOULDER.value]
            rs = rear_lm[PoseLandmark.RIGHT_SHOULDER.value]

            hands_feet_ground_rear = all([near_bottom_rel(x) for x in (lw, rw, la, ra)])
            wr_dist = abs(lw.x - rw.x)
            sh_dist = abs(ls.x - rs.x)
            ratio = wr_dist / (sh_dist + 1e-8)
            r1 = 1.0 if (hands_feet_ground_rear and 0.75 <= ratio <= 1.25) else (0.5 if hands_feet_ground_rear else 0.0)
            breakdown["rear_hands_feet"] = r1
            total += r1

            # foot straight
            try:
                lk_angle = calculate_angle_pts(
                    to_px(rear_lm[PoseLandmark.LEFT_KNEE.value], rear_img.shape),
                    to_px(rear_lm[PoseLandmark.LEFT_ANKLE.value], rear_img.shape),
                    to_px(rear_lm[PoseLandmark.LEFT_FOOT_INDEX.value], rear_img.shape),
                )
                rk_angle = calculate_angle_pts(
                    to_px(rear_lm[PoseLandmark.RIGHT_KNEE.value], rear_img.shape),
                    to_px(rear_lm[PoseLandmark.RIGHT_ANKLE.value], rear_img.shape),
                    to_px(rear_lm[PoseLandmark.RIGHT_FOOT_INDEX.value], rear_img.shape),
                )
                rear_foot_straight = 0.5 if (lk_angle > 160 and rk_angle > 160) else 0.0
            except Exception:
                rear_foot_straight = 0.0
            breakdown["rear_foot_straight"] = rear_foot_straight
            total += rear_foot_straight

            # head between arms
            try:
                left_ear = (rear_lm[PoseLandmark.LEFT_EAR.value].x, rear_lm[PoseLandmark.LEFT_EAR.value].y)
                right_ear = (rear_lm[PoseLandmark.RIGHT_EAR.value].x, rear_lm[PoseLandmark.RIGHT_EAR.value].y)
                left_biceps = ((rear_lm[PoseLandmark.LEFT_SHOULDER.value].x + rear_lm[PoseLandmark.LEFT_ELBOW.value].x) / 2.0,
                               (rear_lm[PoseLandmark.LEFT_SHOULDER.value].y + rear_lm[PoseLandmark.LEFT_ELBOW.value].y) / 2.0)
                right_biceps = ((rear_lm[PoseLandmark.RIGHT_SHOULDER.value].x + rear_lm[PoseLandmark.RIGHT_ELBOW.value].x) / 2.0,
                                (rear_lm[PoseLandmark.RIGHT_SHOULDER.value].y + rear_lm[PoseLandmark.RIGHT_ELBOW.value].y) / 2.0)
                dleft = euclid(left_ear, left_biceps)
                dright = euclid(right_ear, right_biceps)
                if dleft < 0.06 and dright < 0.06:
                    headscore = 1.0
                elif dleft < 0.12 and dright < 0.12:
                    headscore = 0.5
                else:
                    headscore = 0.0
            except Exception:
                headscore = 0.0
            breakdown["rear_head_between_arms"] = headscore
            total += headscore

        # SIDE
        if side_lm:
            try:
                l_kang = calculate_angle_pts(
                    to_px(side_lm[PoseLandmark.LEFT_HIP.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.LEFT_KNEE.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.LEFT_ANKLE.value], side_img.shape),
                )
                r_kang = calculate_angle_pts(
                    to_px(side_lm[PoseLandmark.RIGHT_HIP.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.RIGHT_KNEE.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.RIGHT_ANKLE.value], side_img.shape),
                )
                avg_knee = (l_kang + r_kang) / 2.0
                sk = 2.0 if 170 <= avg_knee <= 180 else (1.5 if avg_knee > 120 else 0.5)
            except Exception:
                sk = 0.5
            breakdown["side_knee"] = sk
            total += sk

            try:
                l_eang = calculate_angle_pts(
                    to_px(side_lm[PoseLandmark.LEFT_SHOULDER.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.LEFT_ELBOW.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.LEFT_WRIST.value], side_img.shape),
                )
                r_eang = calculate_angle_pts(
                    to_px(side_lm[PoseLandmark.RIGHT_SHOULDER.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.RIGHT_ELBOW.value], side_img.shape),
                    to_px(side_lm[PoseLandmark.RIGHT_WRIST.value], side_img.shape),
                )
                avg_elbow = (l_eang + r_eang) / 2.0
                se = 2.0 if 170 <= avg_elbow <= 180 else (1.5 if avg_elbow > 120 else 0.5)
            except Exception:
                se = 0.5
            breakdown["side_elbow"] = se
            total += se

            # head neutral - side
            try:
                mid_sh = ((side_lm[PoseLandmark.LEFT_SHOULDER.value].x + side_lm[PoseLandmark.RIGHT_SHOULDER.value].x) / 2.0,
                          (side_lm[PoseLandmark.LEFT_SHOULDER.value].y + side_lm[PoseLandmark.RIGHT_SHOULDER.value].y) / 2.0)
                nose = (side_lm[PoseLandmark.NOSE.value].x, side_lm[PoseLandmark.NOSE.value].y)
                v = np.array(nose) - np.array(mid_sh)
                vert = np.array([0.0, -1.0])
                cosang = np.dot(v, vert) / ((np.linalg.norm(v) * np.linalg.norm(vert)) + 1e-8)
                cosang = np.clip(cosang, -1.0, 1.0)
                head_tilt_deg = np.degrees(np.arccos(cosang))
                side_head = 1.0 if head_tilt_deg < 15 else 0.5
            except Exception:
                side_head = 0.5
            breakdown["side_head"] = side_head
            total += side_head

            # fingertip to heel distance (side)
            try:
                lw_px = to_px(side_lm[PoseLandmark.LEFT_WRIST.value], side_img.shape)
                la_px = to_px(side_lm[PoseLandmark.LEFT_ANKLE.value], side_img.shape)
                rw_px = to_px(side_lm[PoseLandmark.RIGHT_WRIST.value], side_img.shape)
                ra_px = to_px(side_lm[PoseLandmark.RIGHT_ANKLE.value], side_img.shape)
                d1 = euclid(lw_px, la_px)
                d2 = euclid(rw_px, ra_px)
                dmin = min(d1, d2)
                small_px, medium_px = side_img.shape[0] / 500.0 * 5, side_img.shape[0] / 500.0 * 20
                if dmin < small_px:
                    sf = 2.0
                elif dmin < medium_px:
                    sf = 1.5
                else:
                    sf = 1.0
            except Exception:
                sf = 1.0
            breakdown["side_fingertips_heels"] = sf
            total += sf

            side_proj = 1.0 if (("side_elbow" in breakdown and breakdown["side_elbow"] >= 2.0) and ("side_knee" in breakdown and breakdown["side_knee"] >= 2.0)) else 0.5
            breakdown["side_projection"] = side_proj
            total += side_proj

    except Exception as e:
        # If scoring raises, log and continue with whatever breakdown we have
        print("Scoring error:", e)

    total = round(min(total, 10.0), 2)

    annotated_front = annotate_landmarks(front_img, front_lm) if front_lm else front_img
    annotated_rear = annotate_landmarks(rear_img, rear_lm) if rear_lm else rear_img
    annotated_side = annotate_landmarks(side_img, side_lm) if side_lm else side_img

    return {"total": total, "breakdown": breakdown, "annotated": {"front": annotated_front, "rear": annotated_rear, "side": annotated_side}}


# ---------------------
# Routes
# ---------------------
@app.route("/", methods=["GET"])
def home():
    return "YogaAlign API — POST to /api/score with form-data (front, side, rear)."


@app.route('/api/score', methods=['POST'])
def api_score():
    # Basic file checks
    if not (request.files.get("front") and request.files.get("side") and request.files.get("rear")):
        return make_response(jsonify({"error": "Please upload rear, front and side images."}), 400)

    try:
        # Compress & load images (small memory footprint)
        front_img = pil_resize_and_to_bgr(request.files["front"], max_size=640)
        side_img = pil_resize_and_to_bgr(request.files["side"], max_size=640)
        rear_img = pil_resize_and_to_bgr(request.files["rear"], max_size=640)

        # Run mediapipe pose within a short-lived context (lazy load)
        import mediapipe as mp_local
        with mp_local.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            front_res = pose.process(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
            rear_res = pose.process(cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB))
            side_res = pose.process(cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB))

        if not (front_res.pose_landmarks and rear_res.pose_landmarks and side_res.pose_landmarks):
            return make_response(jsonify({"error": "Pose not detected in one or more images."}), 400)

        front_lm = front_res.pose_landmarks.landmark
        rear_lm = rear_res.pose_landmarks.landmark
        side_lm = side_res.pose_landmarks.landmark

        result = score_from_views(front_lm, rear_lm, side_lm, front_img, rear_img, side_img)

        # Save annotated images and return external URLs
        t = int(time.time())
        annotated_paths = {}
        for k, img in result["annotated"].items():
            fname = f"{k}_annotated_{t}.jpg"
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            save_jpg_from_bgr(img, fpath, quality=80)
            annotated_paths[k] = url_for('static', filename=f"{fname}", _external=True)

        payload = {"total": result["total"], "breakdown": result["breakdown"], "images": annotated_paths}
        response = make_response(jsonify(payload), 200)
        response.headers["Access-Control-Allow-Origin"] = "*"

        # explicit cleanup
        del front_res, rear_res, side_res
        gc.collect()

        return response

    except Exception as e:
        print("API error:", e)
        # don't leak internal stack traces to client
        return make_response(jsonify({"error": "Internal server error"}), 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
