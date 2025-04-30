# mediapipe_integration.py
import queue
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


# ─── Helper Functions ───────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)
    return int(np.degrees(angle))


_smoothed = {}
_alpha = 0.5


def get_smoothed_angle(label, new_angle):
    if label in _smoothed:
        val = int(_alpha * new_angle + (1 - _alpha) * _smoothed[label])
    else:
        val = new_angle
    _smoothed[label] = val
    return val


def overlay_text(image, lines, start_x=10, start_y=30, font_scale=0.8, thickness=2, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        y = start_y + i * 30
        cv2.putText(image, line, (start_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return image


# ─── Main Stream Class ──────────────────────────────────────────────────────

class MediapipeStream:
    def __init__(self, model_path='models/hand_landmarker.task', camera_source='/dev/video0'):
        self.model_path = model_path
        self.camera_source = camera_source
        self.running = False
        self.capture = None

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._results = queue.Queue()

        def _on_result(result, output_image, timestamp_ms):
            self._results.put((result, output_image, timestamp_ms))

        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=_on_result
        )
        self.landmarker = HandLandmarker.create_from_options(opts)

        self.drawing_utils = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self.landmark_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        self.conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.camera_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.camera_source}")

    def stop_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def draw_landmarks(self, img, result):
        out = img.copy()
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand
                ])
                self.drawing_utils.draw_landmarks(
                    out, proto,
                    self.hand_connections,
                    self.landmark_style,
                    self.conn_style
                )
        return out

    # in mediapipe_integration.py

    def stream_loop(self, update_callback, width, height):
        """
        Capture → detect_async → wait for callback result → annotate → callback(img, angles)

        Uses a palm-centric frame (wrist + two MCPs) to remove global roll, so
        rotating the whole forearm does NOT change finger angles.
        """
        import numpy as np
        self.running = True
        self.start_capture()

        scale_factor = 3
        target_w = width * scale_factor
        target_h = height * scale_factor

        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_i = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            self.landmarker.detect_async(mp_i, ts)

            try:
                result, _, _ = self._results.get(timeout=0.05)
            except queue.Empty:
                continue

            annotated = self.draw_landmarks(rgb, result)
            angles = {}

            if result.hand_landmarks:
                # 1) fetch raw image-space landmarks
                lm = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]

                # 2) palm frame: wrist (0), index-MCP (5), pinky-MCP (17)
                w = np.array(lm[0])
                i_mcp = np.array(lm[5])
                p_mcp = np.array(lm[17])

                xv = i_mcp - w
                yv = p_mcp - w
                if np.linalg.norm(np.cross(xv, yv)) < 1e-4:
                    xv = np.array([1.0, 0.0, 0.0])
                    yv = np.array([0.0, 1.0, 0.0])

                zv = np.cross(xv, yv);
                zv /= np.linalg.norm(zv)
                xv /= np.linalg.norm(xv)
                yv = np.cross(zv, xv)
                R = np.vstack([xv, yv, zv]).T  # world→palm

                # 3) landmarks in palm frame
                P = [(R.T @ (np.array(p) - w)) for p in lm]

                # 4) angle helper
                def ang(a, b, c):
                    return calculate_angle(a, b, c)

                fingers = {
                    'Thumb': None,
                    'Index': [5, 6, 7, 8],
                    'Middle': [9, 10, 11, 12],
                    'Ring': [13, 14, 15, 16],
                }

                # 5) primary flexions & thumb-twist
                for name in fingers:
                    if name == 'Thumb':
                        abd = get_smoothed_angle("Thumb_abd",
                                                 ang(P[4], P[1], P[17]))
                        mcp = get_smoothed_angle("Thumb_MCP",
                                                 ang(P[1], P[2], P[3]))
                        pip = get_smoothed_angle("Thumb_IP",
                                                 ang(P[2], P[3], P[4]))

                        # twist = angle between thumb (4-2) and index (8-5)
                        td = P[4] - P[2]
                        idv = P[8] - P[5]
                        td /= np.linalg.norm(td)
                        idv /= np.linalg.norm(idv)
                        tw = int(np.degrees(np.arccos(
                            np.clip(np.dot(td, idv), -1.0, 1.0))))
                        tw = get_smoothed_angle("Thumb_twist", tw)

                        angles['Thumb'] = dict(
                            abduction=abd,
                            mcp=mcp,
                            pip=pip,
                            twist=tw
                        )
                    else:
                        idx = fingers[name]
                        mcp = get_smoothed_angle(
                            f"{name}_MCP",
                            ang(P[idx[0] - 1], P[idx[0]], P[idx[1]])
                        )
                        pip = get_smoothed_angle(
                            f"{name}_PIP",
                            ang(P[idx[0]], P[idx[1]], P[idx[2]])
                        )
                        dip = get_smoothed_angle(
                            f"{name}_DIP",
                            ang(P[idx[1]], P[idx[2]], P[idx[3]])
                        )
                        angles[name] = dict(mcp=mcp, pip=pip, dip=dip)

                # 6) abductions for I/M/R
                angles['Index']['abduction'] = get_smoothed_angle(
                    "Index_abd",
                    ang(P[6], P[5], P[9]))
                angles['Middle']['abduction'] = get_smoothed_angle(
                    "Middle_abd",
                    ang(P[10], P[9], P[13]))
                angles['Ring']['abduction'] = get_smoothed_angle(
                    "Ring_abd",
                    ang(P[18], P[17], P[13]))

            # 7) overlay text
            lines = []
            for f, v in angles.items():
                if f == 'Thumb':
                    lines.append(f"{f:6s} Abd {v['abduction']:3d}°  MCP {v['mcp']:3d}°  "
                                 f"PIP {v['pip']:3d}°  Twt {v['twist']:3d}°")
                else:
                    lines.append(f"{f:6s} Abd {v['abduction']:3d}°  MCP {v['mcp']:3d}°  "
                                 f"PIP {v['pip']:3d}°  DIP {v['dip']:3d}°")

            annotated = overlay_text(annotated, lines,
                                     start_x=10, start_y=20,
                                     font_scale=0.6, thickness=1,
                                     color=(0, 255, 0))
            annotated = cv2.resize(annotated,
                                   (target_w, target_h),
                                   interpolation=cv2.INTER_LINEAR)

            update_callback(annotated, angles)

        self.stop_capture()

    def stop(self):
        self.running = False


def proj_plane(v, n):
    """project vector v on plane with normal n"""
    n = n / (np.linalg.norm(n) + 1e-9)
    return v - np.dot(v, n) * n


def angle_plane(a, b, c, plane_n):
    """
    angle a-b-c measured **inside** the plane whose normal is plane_n
    (used for abduction)
    """
    ba = proj_plane(a - b, plane_n)
    bc = proj_plane(c - b, plane_n)
    return calculate_angle(ba, np.zeros(3), bc)
