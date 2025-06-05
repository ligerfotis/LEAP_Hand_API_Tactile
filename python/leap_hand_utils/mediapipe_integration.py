# mediapipe_integration.py (updated)
import time
import queue

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.framework.formats import landmark_pb2

# ------------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------------

# MediaPipe confidence thresholds (set to 0.8 as requested)
MIN_DET_CONF = 0.8           # min_detection_confidence
MIN_TRACK_CONF = 0.8         # min_tracking_confidence

# Threshold‐logic parameters for on‐screen joint angles:
HISTORY_LENGTH = 3           # number of frames over which we compare oldest vs. newest
ANGLE_THRESHOLD_DEG = 1.0    # if |newest – oldest| ≤ 1°, we do NOT update

# ------------------------------------------------------------------------------
# Helper: calculate the 3D angle at landmark_B formed by landmark_A–landmark_B–landmark_C
# ------------------------------------------------------------------------------
def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle ABC in degrees, given three 3D points a, b, c (as [x, y, z]).
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ------------------------------------------------------------------------------
# Smoothing for palm‐centric angles (Thumb abduction, twist, etc.)
# ------------------------------------------------------------------------------
_smoothed = {}
_alpha = 0.5


def get_smoothed_angle(label: str, new_angle: float) -> int:
    """
    Exponential smoothing for labelled angle streams.
    """
    if label in _smoothed:
        val = int(_alpha * new_angle + (1 - _alpha) * _smoothed[label])
    else:
        val = int(new_angle)
    _smoothed[label] = val
    return val


# ------------------------------------------------------------------------------
# Overlay a list of text lines onto an image (used for the palm‐centric angles dict)
# ------------------------------------------------------------------------------
def overlay_text(
    image: np.ndarray,
    lines: list[str],
    start_x: int = 10,
    start_y: int = 30,
    font_scale: float = 0.8,
    thickness: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    for i, line in enumerate(lines):
        y = start_y + i * 30
        cv2.putText(
            image,
            line,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return image


# ------------------------------------------------------------------------------
# Which triplets of MediaPipe landmark indices to measure (A, B, C) → angle at B
# ------------------------------------------------------------------------------
JOINT_TRIPLETS = [
    # Thumb: CMC (0–1–2), MCP (1–2–3), IP (2–3–4)
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    # Index finger: MCP (0–5–6), PIP (5–6–7), DIP (6–7–8)
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    # Middle finger: MCP (0–9–10), PIP (9–10–11), DIP (10–11–12)
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    # Ring finger: MCP (0–13–14), PIP (13–14–15), DIP (14–15–16)
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    # Pinky finger: MCP (0–17–18), PIP (17–18–19), DIP (18–19–20)
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
]

# ------------------------------------------------------------------------------
# For each (hand_id, joint_idx), we’ll keep:
#   1) A deque of the last HISTORY_LENGTH raw angles (float),
#   2) The last “displayed” angle (float) that we drew on screen.
# ------------------------------------------------------------------------------
angle_history: dict[tuple[int, int], deque] = {}
last_displayed_angle: dict[tuple[int, int], float] = {}

# ------------------------------------------------------------------------------
# MediapipeStream: replace HandLandmarker with mp.solutions.hands.Hands
# ------------------------------------------------------------------------------

class MediapipeStream:
    def __init__(self, camera_source: int = 0):
        self.camera_source = camera_source
        self.running = False
        self.capture = None

        # Initialize mp.solutions.hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DET_CONF,
            min_tracking_confidence=MIN_TRACK_CONF,
        )

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.camera_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.camera_source}")

    def stop_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def draw_landmarks(self, img: np.ndarray, hand_landmarks_list) -> np.ndarray:
        out = img.copy()
        for hand_landmarks in hand_landmarks_list:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks.landmark
                ]
            )
            self.mp_drawing.draw_landmarks(
                out,
                proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        return out

    def stream_loop(self, update_callback, width: int, height: int):
        """
        Capture → process via mp.solutions.hands → annotate → callback(img, angles)

        Each frame: runs Hands.process(), draws skeleton + joint angles with threshold‐smoothing,
        then computes palm‐centric angles dict for “Thumb”, “Index”, “Middle”, “Ring” exactly as before.
        """
        self.running = True
        self.start_capture()

        frame_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # The “fingers” mapping for palm‐centric (same as before)
        fingers = {
            "Thumb": None,
            "Index": [5, 6, 7, 8],
            "Middle": [9, 10, 11, 12],
            "Ring": [13, 14, 15, 16],
        }

        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                continue

            # Mirror + BGR→RGB
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            annotated = frame.copy()
            angles: dict[str, dict[str, int]] = {}

            if results.multi_hand_landmarks and results.multi_handedness:
                # Just handle the first detected hand for the palm‐centric angles dict:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                hand_id = int(handedness.classification[0].index)

                # Build a (21,3) array of normalized (x,y,z)
                lm_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                # Draw all landmarks + connections on the annotated image
                annotated = self.draw_landmarks(annotated, results.multi_hand_landmarks)

                # Pixel coords for each landmark (used to place “triplet” angle text)
                lm_pixels = [
                    (int(lm.x * frame_w), int(lm.y * frame_h))
                    for lm in hand_landmarks.landmark
                ]

                # ──────────────────────────────────────────────────────────
                # 1) For each JOINT_TRIPLETS, compute raw_angle; threshold‐smooth;
                #    draw the integer angle next to landmark B.
                # ──────────────────────────────────────────────────────────
                for joint_idx, (i_a, i_b, i_c) in enumerate(JOINT_TRIPLETS):
                    A = lm_array[i_a]
                    B = lm_array[i_b]
                    C = lm_array[i_c]
                    raw_angle = calculate_angle(A, B, C)

                    key = (hand_id, joint_idx)
                    if key not in angle_history:
                        angle_history[key] = deque(maxlen=HISTORY_LENGTH)
                        last_displayed_angle[key] = raw_angle

                    angle_history[key].append(raw_angle)
                    displayed_angle = last_displayed_angle[key]

                    if len(angle_history[key]) == HISTORY_LENGTH:
                        oldest = angle_history[key][0]
                        newest = angle_history[key][-1]
                        if abs(newest - oldest) > ANGLE_THRESHOLD_DEG:
                            displayed_angle = newest

                    last_displayed_angle[key] = displayed_angle

                    x_text, y_text = lm_pixels[i_b]
                    y_text = max(y_text - 10, 10)
                    cv2.putText(
                        annotated,
                        f"{int(round(displayed_angle))}°",
                        (x_text, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # ──────────────────────────────────────────────────────────
                # 2) Build palm‐centric frame (wrist=0, index-MCP=5, pinky-MCP=17),
                #    rotate landmarks into that frame, compute per‐finger angles.
                # ──────────────────────────────────────────────────────────
                lm_world = [(p.x, p.y, p.z) for p in hand_landmarks.landmark]
                w = np.array(lm_world[0])
                i_mcp = np.array(lm_world[5])
                p_mcp = np.array(lm_world[17])

                xv = i_mcp - w
                yv = p_mcp - w
                if np.linalg.norm(np.cross(xv, yv)) < 1e-4:
                    xv = np.array([1.0, 0.0, 0.0])
                    yv = np.array([0.0, 1.0, 0.0])

                zv = np.cross(xv, yv)
                zv /= np.linalg.norm(zv)
                xv /= np.linalg.norm(xv)
                yv = np.cross(zv, xv)
                R = np.vstack([xv, yv, zv]).T  # world→palm

                P = [(R.T @ (np.array(p) - w)) for p in lm_world]

                # Helper to compute angle in palm frame
                def ang(a_idx: int, b_idx: int, c_idx: int) -> float:
                    return calculate_angle(P[a_idx], P[b_idx], P[c_idx])

                # 2a) Thumb: abduction, MCP, IP, twist
                abd = get_smoothed_angle("Thumb_abd", ang(4, 1, 17))
                mcp_t = get_smoothed_angle("Thumb_MCP", ang(1, 2, 3))
                pip_t = get_smoothed_angle("Thumb_IP", ang(2, 3, 4))

                td = P[4] - P[2]
                idv = P[8] - P[5]
                td /= np.linalg.norm(td)
                idv /= np.linalg.norm(idv)
                tw_raw = int(np.degrees(np.arccos(np.clip(np.dot(td, idv), -1.0, 1.0))))
                tw = get_smoothed_angle("Thumb_twist", tw_raw)

                angles["Thumb"] = dict(
                    abduction=abd,
                    mcp=mcp_t,
                    pip=pip_t,
                    twist=tw,
                )

                # 2b) Index / Middle / Ring: MCP, PIP, DIP + abduction
                for name in ["Index", "Middle", "Ring"]:
                    idxs = fingers[name]
                    mcp_angle = get_smoothed_angle(
                        f"{name}_MCP",
                        ang(idxs[0] - 1, idxs[0], idxs[1]),
                    )
                    pip_angle = get_smoothed_angle(
                        f"{name}_PIP",
                        ang(idxs[0], idxs[1], idxs[2]),
                    )
                    dip_angle = get_smoothed_angle(
                        f"{name}_DIP",
                        ang(idxs[1], idxs[2], idxs[3]),
                    )
                    angles[name] = dict(mcp=mcp_angle, pip=pip_angle, dip=dip_angle)

                    # abduction (Index vs. Middle, Middle vs. Ring, etc.)
                    if name == "Index":
                        abd_ix = get_smoothed_angle("Index_abd", ang(6, 5, 9))
                        angles[name]["abduction"] = abd_ix
                    elif name == "Middle":
                        abd_md = get_smoothed_angle("Middle_abd", ang(10, 9, 13))
                        angles[name]["abduction"] = abd_md
                    elif name == "Ring":
                        abd_rg = get_smoothed_angle("Ring_abd", ang(14, 13, 17))
                        angles[name]["abduction"] = abd_rg

                # 2c) Pinky is not used for teleop, so we skip it from structured dict.
                #     (We already drew its per‐joint angles above.)

                # 2d) Build the on‐screen text lines for the palm‐centric angles dict
                lines = []
                for f, v in angles.items():
                    if f == "Thumb":
                        lines.append(
                            f"{f:6s} Abd {v['abduction']:3d}°  MCP {v['mcp']:3d}°  "
                            f"PIP {v['pip']:3d}°  Twt {v['twist']:3d}°"
                        )
                    else:
                        lines.append(
                            f"{f:6s} Abd {v['abduction']:3d}°  MCP {v['mcp']:3d}°  "
                            f"PIP {v['pip']:3d}°  DIP {v['dip']:3d}°"
                        )

                annotated = overlay_text(
                    annotated,
                    lines,
                    start_x=10,
                    start_y=20,
                    font_scale=0.6,
                    thickness=1,
                    color=(0, 255, 0),
                )
            # end if any hand detected

            # 3) Resize to match requested (width, height) before calling back
            annotated = cv2.resize(
                annotated,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            update_callback(annotated, angles)

        # Cleanup
        self.stop_capture()

    def stop(self):
        self.running = False
        if hasattr(self, "hands"):
            self.hands.close()
