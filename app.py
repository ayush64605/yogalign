# app.py
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import (Flask, jsonify, make_response, render_template, request,
                   url_for)

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# We'll NOT use mp_drawing.draw_landmarks here for annotation; we'll draw manually.
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Minimum visibility to draw a landmark / connection
MIN_VISIBILITY = 0.40


# ---------------------
# Helpers
# ---------------------
def to_px(landmark, image_shape):
    """Convert normalized landmark to pixel (x,y) tuple"""
    h, w = image_shape[0], image_shape[1]
    return (int(landmark.x * w), int(landmark.y * h))


def euclid(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def calculate_angle_pts(a_pt, b_pt, c_pt):
    """Angle in degrees given pixel points a,b,c (angle at b)."""
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


def annotate_landmarks(image, landmarks):
    """
    Draw only high-confidence landmarks and connections.
    Also draw small numeric labels and some key angles (elbow/knee).
    landmarks: list of normalized landmarks (MediaPipe)
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Build pixel list with visibility
    pts = []
    for i, lm in enumerate(landmarks):
        vis = getattr(lm, "visibility", 1.0)
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y, vis))

    # Draw connections first (so lines under the dots)
    for connection in POSE_CONNECTIONS:
        # connection elements may be enums; ensure we use their int index
        s = connection[0].value if hasattr(connection[0], "value") else int(connection[0])
        e = connection[1].value if hasattr(connection[1], "value") else int(connection[1])
        if s < len(pts) and e < len(pts):
            xs, ys, vs = pts[s]
            xe, ye, ve = pts[e]
            if vs >= MIN_VISIBILITY and ve >= MIN_VISIBILITY:
                cv2.line(img, (xs, ys), (xe, ye), (20, 200, 20), 2, cv2.LINE_AA)

    # Draw landmarks (dots + index)
    for i, (x, y, v) in enumerate(pts):
        if v >= MIN_VISIBILITY:
            cv2.circle(img, (x, y), 5, (0, 180, 255), -1)  # visible landmark: orange dot
            cv2.putText(img, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # optionally show faint dot for low confidence (comment out if undesired)
            cv2.circle(img, (x, y), 3, (150, 150, 150), 1)

    # Compute and draw angles for knees and elbows (if visible)
    # LEFT/RIGHT indices from MediaPipe PoseLandmark
    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LEL = mp_pose.PoseLandmark.LEFT_ELBOW.value
    REL = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LWR = mp_pose.PoseLandmark.LEFT_WRIST.value
    RWR = mp_pose.PoseLandmark.RIGHT_WRIST.value
    LHIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RHIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    LKNE = mp_pose.PoseLandmark.LEFT_KNEE.value
    RKNE = mp_pose.PoseLandmark.RIGHT_KNEE.value
    LANK = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RANK = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    def safe_angle(i_a, i_b, i_c):
        if i_a < len(pts) and i_b < len(pts) and i_c < len(pts):
            xa, ya, va = pts[i_a]
            xb, yb, vb = pts[i_b]
            xc, yc, vc = pts[i_c]
            if va >= MIN_VISIBILITY and vb >= MIN_VISIBILITY and vc >= MIN_VISIBILITY:
                return calculate_angle_pts((xa, ya), (xb, yb), (xc, yc))
        return None

    # Elbows
    left_elbow_ang = safe_angle(LSH, LEL, LWR)
    right_elbow_ang = safe_angle(RSH, REL, RWR)
    if left_elbow_ang is not None:
        x, y, _ = pts[LEL]
        cv2.putText(img, f"{int(left_elbow_ang)}\u00B0", (x+6, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    if right_elbow_ang is not None:
        x, y, _ = pts[REL]
        cv2.putText(img, f"{int(right_elbow_ang)}\u00B0", (x+6, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Knees
    left_knee_ang = safe_angle(LHIP, LKNE, LANK)
    right_knee_ang = safe_angle(RHIP, RKNE, RANK)
    if left_knee_ang is not None:
        x, y, _ = pts[LKNE]
        cv2.putText(img, f"{int(left_knee_ang)}\u00B0", (x+6, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    if right_knee_ang is not None:
        x, y, _ = pts[RKNE]
        cv2.putText(img, f"{int(right_knee_ang)}\u00B0", (x+6, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    return img


# ---------------------
# Main scoring function (keeps the same checks you defined)
# ---------------------
def score_from_views(front_lm, rear_lm, side_lm, front_img, rear_img, side_img):
    """
    Input: landmarks lists (normalized), images (BGR)
    Returns: dict with total score, breakdown, and annotated images
    """

    breakdown = {}
    total = 0.0

    # ---------- FRONT VIEW ----------
    if front_lm:
        # Example checks: hands/feet on ground and shoulder width, foot & hand straight
        h, w = front_img.shape[:2]
        # landmarks
        lw = front_lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = front_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        la = front_lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = front_lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ls = front_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = front_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # near-bottom test (normalized coordinates)
        def near_bottom(pt, tol=0.15):
            return (1.0 - pt.y) < tol

        hands_feet_ground = all([near_bottom(x) for x in (lw, rw, la, ra)])
        wr_dist = abs(lw.x - rw.x)
        sh_dist = abs(ls.x - rs.x)
        ratio = wr_dist / (sh_dist + 1e-8)
        if hands_feet_ground and 0.75 <= ratio <= 1.25:
            s1 = 1.0
        elif hands_feet_ground:
            s1 = 0.5
        else:
            s1 = 0.0
        breakdown['front_hands_feet'] = s1
        total += s1

        # foot straight (proxy): use knee-ankle-foot angle > 160
        try:
            lk_angle = calculate_angle_pts(to_px(front_lm[mp_pose.PoseLandmark.LEFT_KNEE.value], front_img.shape),
                                           to_px(front_lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], front_img.shape),
                                           to_px(front_lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value], front_img.shape))
            rk_angle = calculate_angle_pts(to_px(front_lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], front_img.shape),
                                           to_px(front_lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], front_img.shape),
                                           to_px(front_lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value], front_img.shape))
            if lk_angle > 160 and rk_angle > 160:
                s2 = 0.5
            else:
                s2 = 0.0
        except Exception:
            s2 = 0.0
        breakdown['front_foot_straight'] = s2
        total += s2

        # hand straight (proxy): shoulder-elbow-wrist angles close to 180
        try:
            left_hand_ang = calculate_angle_pts(to_px(front_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], front_img.shape),
                                                to_px(front_lm[mp_pose.PoseLandmark.LEFT_ELBOW.value], front_img.shape),
                                                to_px(front_lm[mp_pose.PoseLandmark.LEFT_WRIST.value], front_img.shape))
            right_hand_ang = calculate_angle_pts(to_px(front_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], front_img.shape),
                                                 to_px(front_lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value], front_img.shape),
                                                 to_px(front_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value], front_img.shape))
            if left_hand_ang > 160 and right_hand_ang > 160:
                s3 = 0.5
            else:
                s3 = 0.0
        except Exception:
            s3 = 0.0
        breakdown['front_hand_straight'] = s3
        total += s3

    # ---------- REAR VIEW ----------
    if rear_lm:
        # hands/feet ground and foot straight and head-between-arms (partial)
        def near_bottom_rel(pt, tol=0.15):
            return (1.0 - pt.y) < tol
        lw = rear_lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = rear_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        la = rear_lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = rear_lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ls = rear_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = rear_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        hands_feet_ground_rear = all([near_bottom_rel(x) for x in (lw, rw, la, ra)])
        wr_dist = abs(lw.x - rw.x)
        sh_dist = abs(ls.x - rs.x)
        ratio = wr_dist / (sh_dist + 1e-8)
        if hands_feet_ground_rear and 0.75 <= ratio <= 1.25:
            r1 = 1.0
        elif hands_feet_ground_rear:
            r1 = 0.5
        else:
            r1 = 0.0
        breakdown['rear_hands_feet'] = r1
        total += r1

        # foot straight
        try:
            lk_angle = calculate_angle_pts(to_px(rear_lm[mp_pose.PoseLandmark.LEFT_KNEE.value], rear_img.shape),
                                           to_px(rear_lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], rear_img.shape),
                                           to_px(rear_lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value], rear_img.shape))
            rk_angle = calculate_angle_pts(to_px(rear_lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], rear_img.shape),
                                           to_px(rear_lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], rear_img.shape),
                                           to_px(rear_lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value], rear_img.shape))
            rear_foot_straight = 0.5 if (lk_angle > 160 and rk_angle > 160) else 0.0
        except Exception:
            rear_foot_straight = 0.0
        breakdown['rear_foot_straight'] = rear_foot_straight
        total += rear_foot_straight

        # head between arms: ear-to-biceps distance proxy
        try:
            left_ear = (rear_lm[mp_pose.PoseLandmark.LEFT_EAR.value].x, rear_lm[mp_pose.PoseLandmark.LEFT_EAR.value].y)
            right_ear = (rear_lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x, rear_lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y)
            left_biceps = ((rear_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + rear_lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x)/2.0,
                           (rear_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + rear_lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)/2.0)
            right_biceps = ((rear_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + rear_lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x)/2.0,
                            (rear_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y + rear_lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)/2.0)
            # normalized distance
            dleft = euclid(left_ear, left_biceps)
            dright = euclid(right_ear, right_biceps)
            # empirical thresholds
            if dleft < 0.06 and dright < 0.06:
                headscore = 1.0
            elif dleft < 0.12 and dright < 0.12:
                headscore = 0.5
            else:
                headscore = 0.0
        except Exception:
            headscore = 0.0
        breakdown['rear_head_between_arms'] = headscore
        total += headscore

    # ---------- SIDE VIEW ----------
    if side_lm:
        # knee and elbow angles (use average)
        try:
            l_kang = calculate_angle_pts(to_px(side_lm[mp_pose.PoseLandmark.LEFT_HIP.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.LEFT_KNEE.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], side_img.shape))
            r_kang = calculate_angle_pts(to_px(side_lm[mp_pose.PoseLandmark.RIGHT_HIP.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], side_img.shape))
            avg_knee = (l_kang + r_kang) / 2.0
            if 170 <= avg_knee <= 180:
                sk = 2.0
            elif avg_knee > 120:
                sk = 1.5
            else:
                sk = 0.5
        except Exception:
            sk = 0.5
        breakdown['side_knee'] = sk
        total += sk

        try:
            l_eang = calculate_angle_pts(to_px(side_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.LEFT_ELBOW.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.LEFT_WRIST.value], side_img.shape))
            r_eang = calculate_angle_pts(to_px(side_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value], side_img.shape),
                                         to_px(side_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value], side_img.shape))
            avg_elbow = (l_eang + r_eang) / 2.0
            if 170 <= avg_elbow <= 180:
                se = 2.0
            elif avg_elbow > 120:
                se = 1.5
            else:
                se = 0.5
        except Exception:
            se = 0.5
        breakdown['side_elbow'] = se
        total += se

        # head neutral (side): nose vs shoulder vertical alignment proxy
        try:
            mid_sh = ((side_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + side_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)/2.0,
                      (side_lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + side_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/2.0)
            nose = (side_lm[mp_pose.PoseLandmark.NOSE.value].x, side_lm[mp_pose.PoseLandmark.NOSE.value].y)
            v = np.array(nose) - np.array(mid_sh)
            vert = np.array([0.0, -1.0])
            cosang = np.dot(v, vert) / ((np.linalg.norm(v) * np.linalg.norm(vert)) + 1e-8)
            cosang = np.clip(cosang, -1.0, 1.0)
            head_tilt_deg = np.degrees(np.arccos(cosang))
            side_head = 1.0 if head_tilt_deg < 15 else 0.5
        except Exception:
            side_head = 0.5
        breakdown['side_head'] = side_head
        total += side_head

        # fingertip to heel distance (side)
        try:
            lw_px = to_px(side_lm[mp_pose.PoseLandmark.LEFT_WRIST.value], side_img.shape)
            la_px = to_px(side_lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], side_img.shape)
            rw_px = to_px(side_lm[mp_pose.PoseLandmark.RIGHT_WRIST.value], side_img.shape)
            ra_px = to_px(side_lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], side_img.shape)
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
        breakdown['side_fingertips_heels'] = sf
        total += sf

        # body projection
        side_proj = 1.0 if ( (('side_elbow' in breakdown and breakdown['side_elbow']>=2.0) and ('side_knee' in breakdown and breakdown['side_knee']>=2.0)) ) else 0.5
        breakdown['side_projection'] = side_proj
        total += side_proj

    total = round(min(total, 10.0), 2)

    # Annotate images (use improved annotation)
    annotated_front = annotate_landmarks(front_img, front_lm) if front_lm else front_img
    annotated_rear = annotate_landmarks(rear_img, rear_lm) if rear_lm else rear_img
    annotated_side = annotate_landmarks(side_img, side_lm) if side_lm else side_img

    return {"total": total, "breakdown": breakdown, "annotated": {"front": annotated_front, "rear": annotated_rear, "side": annotated_side}}


# ---------------------
# Flask routes
# ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        rear_file = request.files.get("rear")
        front_file = request.files.get("front")
        side_file = request.files.get("side")
        if not (rear_file and front_file and side_file):
            return render_template("index.html", error="Please upload rear, front and side images.")

        t = int(time.time())
        rear_path = os.path.join(UPLOAD_FOLDER, f"rear_{t}.jpg")
        front_path = os.path.join(UPLOAD_FOLDER, f"front_{t}.jpg")
        side_path = os.path.join(UPLOAD_FOLDER, f"side_{t}.jpg")
        rear_file.save(rear_path); front_file.save(front_path); side_file.save(side_path)

        # run pose
        front_img = cv2.imread(front_path)
        rear_img = cv2.imread(rear_path)
        side_img = cv2.imread(side_path)

        front_res = POSE.process(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
        rear_res = POSE.process(cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB))
        side_res = POSE.process(cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB))

        if not (front_res.pose_landmarks and rear_res.pose_landmarks and side_res.pose_landmarks):
            return render_template("index.html", error="Pose not detected in one or more images. Make sure whole body is visible and images are well-lit.")

        front_lm = front_res.pose_landmarks.landmark
        rear_lm = rear_res.pose_landmarks.landmark
        side_lm = side_res.pose_landmarks.landmark

        result = score_from_views(front_lm, rear_lm, side_lm, front_img, rear_img, side_img)

        # Save annotated images
        annotated_paths = {}
        for k, img in result["annotated"].items():
            fname = f"{k}_annotated_{t}.jpg"
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            cv2.imwrite(fpath, img)
            # Build a URL for the template using Flask's static route
            annotated_paths[k] = url_for('static', filename=f'uploads/{fname}')

        return render_template("result.html", total=result["total"], breakdown=result["breakdown"], images=annotated_paths)

    return render_template("index.html", error=None)


@app.route('/api/score', methods=['POST'])
def api_score():
    """API endpoint to accept three images and return JSON with scores and annotated image URLs.
    This can be called from the mobile app (expo) to receive a JSON response instead of an HTML page.
    """
    rear_file = request.files.get("rear")
    front_file = request.files.get("front")
    side_file = request.files.get("side")
    if not (rear_file and front_file and side_file):
        resp = make_response(jsonify({"error": "Please upload rear, front and side images."}), 400)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    t = int(time.time())
    rear_path = os.path.join(UPLOAD_FOLDER, f"rear_{t}.jpg")
    front_path = os.path.join(UPLOAD_FOLDER, f"front_{t}.jpg")
    side_path = os.path.join(UPLOAD_FOLDER, f"side_{t}.jpg")
    rear_file.save(rear_path); front_file.save(front_path); side_file.save(side_path)

    front_img = cv2.imread(front_path)
    rear_img = cv2.imread(rear_path)
    side_img = cv2.imread(side_path)

    front_res = POSE.process(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
    rear_res = POSE.process(cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB))
    side_res = POSE.process(cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB))

    if not (front_res.pose_landmarks and rear_res.pose_landmarks and side_res.pose_landmarks):
        resp = make_response(jsonify({"error": "Pose not detected in one or more images. Make sure whole body is visible and images are well-lit."}), 400)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    front_lm = front_res.pose_landmarks.landmark
    rear_lm = rear_res.pose_landmarks.landmark
    side_lm = side_res.pose_landmarks.landmark

    result = score_from_views(front_lm, rear_lm, side_lm, front_img, rear_img, side_img)

    annotated_paths = {}
    for k, img in result["annotated"].items():
        fname = f"{k}_annotated_{t}.jpg"
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        cv2.imwrite(fpath, img)
        annotated_paths[k] = url_for('static', filename=f'uploads/{fname}', _external=True)

    payload = {
        "total": result["total"],
        "breakdown": result["breakdown"],
        "images": annotated_paths,
    }
    resp = make_response(jsonify(payload), 200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    # Bind to 0.0.0.0 so devices on the same LAN (e.g. Expo on phone)
    # can reach the Flask server. When testing from an emulator, use
    # the emulator-specific loopback addresses instead (e.g. Android: 10.0.2.2).
    app.run(host="0.0.0.0", debug=True)
