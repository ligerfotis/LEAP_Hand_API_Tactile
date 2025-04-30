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
        Capture → detect_async → wait for callback result → annotate → callback(annotated_image, angles)
        Direct per-finger flexions/spans for Index/Middle/Ring, and for Thumb:
          • CMC abduction (brings thumb toward/away from palm plane)
          • MCP flexion
          • IP flexion
          • Twist (opposition)
        Overlay drawn in smaller font.
        """
        self.running = True
        self.start_capture()

        scale_factor = 3
        target_w = width * scale_factor
        target_h = height * scale_factor

        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int(time.time() * 1000)
            self.landmarker.detect_async(mp_img, ts)

            try:
                result, _, _ = self._results.get(timeout=0.05)
            except queue.Empty:
                continue

            annotated = self.draw_landmarks(frame_rgb, result)
            angles = {}

            if result.hand_landmarks:
                lm = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]

                fingers = {
                    'Thumb': None,
                    'Index': [5, 6, 7, 8],
                    'Middle': [9, 10, 11, 12],
                    'Ring': [13, 14, 15, 16],
                }

                for name in fingers:
                    if name == 'Thumb':
                        # CMC abduction
                        abd = get_smoothed_angle("Thumb_abd", calculate_angle(lm[4], lm[1], lm[17]))                        # MCP flexion
                        mcp = get_smoothed_angle("Thumb_MCP", calculate_angle(lm[1], lm[2], lm[3]))
                        # IP flexion
                        pip = get_smoothed_angle("Thumb_IP", calculate_angle(lm[2], lm[3], lm[4]))
                        # Twist opposition
                        thumb_dir = np.array(lm[4]) - np.array(lm[2])
                        index_dir = np.array(lm[8]) - np.array(lm[5])
                        td = thumb_dir / (np.linalg.norm(thumb_dir) + 1e-6)
                        idv = index_dir / (np.linalg.norm(index_dir) + 1e-6)
                        tw_angle = int(np.degrees(np.arccos(np.clip(np.dot(td, idv), -1.0, 1.0))))
                        twt = get_smoothed_angle("Thumb_twist", tw_angle)

                        angles['Thumb'] = {
                            'abduction': abd,
                            'mcp': mcp,
                            'pip': pip,
                            'twist': twt
                        }
                    else:
                        idx = fingers[name]
                        mcp = get_smoothed_angle(f"{name}_MCP",
                                                 calculate_angle(lm[idx[0] - 1], lm[idx[0]], lm[idx[1]]))
                        pip = get_smoothed_angle(f"{name}_PIP",
                                                 calculate_angle(lm[idx[0]], lm[idx[1]], lm[idx[2]]))
                        dip = get_smoothed_angle(f"{name}_DIP",
                                                 calculate_angle(lm[idx[1]], lm[idx[2]], lm[idx[3]]))
                        angles[name] = {'abduction': 0, 'mcp': mcp, 'pip': pip, 'dip': dip}

                # spans for Index/Middle/Ring
                angles['Index']['abduction'] = get_smoothed_angle("Index_abd",
                                                                  calculate_angle(lm[6], lm[5], lm[9]))
                angles['Middle']['abduction'] = get_smoothed_angle("Middle_abd",
                                                                   calculate_angle(lm[10], lm[9], lm[13]))
                angles['Ring']['abduction'] = get_smoothed_angle("Ring_abd",
                                                                 calculate_angle(lm[18], lm[17], lm[13]))

            # build overlay
            lines = []
            for f, v in angles.items():
                if f == 'Thumb':
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
                annotated, lines,
                start_x=10, start_y=20,
                font_scale=0.6, thickness=1, color=(0, 255, 0)
            )

            annotated = cv2.resize(
                annotated, (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )
            update_callback(annotated, angles)

        self.stop_capture()

    def stop(self):
        self.running = False
