# leap_hand_gui_new.py
import glob
import threading
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import time
import cv2  # for image conversion in replay
from sensor_manager import SensorManager
from trajectory_manager import TrajectoryManager
from configuration_manager import ConfigurationManager
from finger_teach_manager import FingerTeachManager
from leap_node import LeapNode
import leap_hand_utils as lhu
from PIL import Image, ImageTk
# Import the MediaPipe integration module.
from mediapipe_integration import MediapipeStream

# Global smoothing parameters.
GLOBAL_SMOOTHING_STEPS = 30
GLOBAL_SMOOTHING_DELAY = 20
import pybullet as p
import pybullet_data
import os


def find_available_cameras():
    """
    Scan only the /dev/video* entries that actually exist.
    For each, read its name from sysfs, skip Intel/RealSense/Digit devices,
    then attempt to open it to verify it returns a frame. Return a list of
    (idx, label) where label = "idx: <device_name> (WxH)".
    """
    cams = []
    blacklist = ("intel", "realsense", "digit")

    # 1) List all /dev/video* nodes
    video_nodes = glob.glob("/dev/video*")
    for node in video_nodes:
        # Extract the index from "/dev/videoN"
        try:
            idx = int(node.replace("/dev/video", ""))
        except ValueError:
            continue  # skip anything unexpected

        # 2) Read device name from sysfs, if available
        sysfs_path = f"/sys/class/video4linux/video{idx}/name"
        if os.path.isfile(sysfs_path):
            try:
                with open(sysfs_path, "r") as f:
                    cam_name = f.read().strip()
            except Exception:
                cam_name = "Unknown"
        else:
            cam_name = "Unknown"

        # 3) Skip blacklisted devices
        if any(b in cam_name.lower() for b in blacklist):
            continue

        # 4) Try opening with the default backend to confirm it works
        cap = cv2.VideoCapture(idx)  # CAP_ANY
        if not cap.isOpened():
            cap.release()
            continue

        # Give the driver a short moment, then grab+retrieve
        time.sleep(0.1)
        if not cap.grab():
            cap.release()
            continue
        ret, _ = cap.retrieve()
        if not ret:
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()

        label = f"{idx}: {cam_name} ({w}×{h})"
        cams.append((idx, label))

    # Sort by index so dropdown is in ascending order
    cams.sort(key=lambda x: x[0])
    return cams


class LeapHandGUI:
    def __init__(self, master):
        self.live_containers = []  # initialize as an empty list, not None

        self.master = master
        master.title("LEAP Hand Real-Time Control GUI")

        # For delayed start of normal recording.
        self.start_delay_job = None

        # Calibration mapping: human 0°→robot 180°, human max→robot 270° (or corresponding end)
        self.calibration = {
            'Index': {
                'mcp': {'servo_open': 160.0, 'servo_closed': 260.0, 'h_open': 90.0, 'h_closed': 50.0},
                'pip': {'servo_open': 180.0, 'servo_closed': 280.0, 'h_open': 175.0, 'h_closed': 70.0},
                'dip': {'servo_open': 180.0, 'servo_closed': 260.0, 'h_open': 175.0, 'h_closed': 120.0},
                'abd': {'servo_open': 150.0, 'servo_closed': 190.0, 'h_open': 110.0, 'h_closed': 80.0},
            },
            'Middle': {
                'mcp': {'servo_open': 160.0, 'servo_closed': 200.0, 'h_open': 20.0, 'h_closed': 25.0},
                'pip': {'servo_open': 180.0, 'servo_closed': 260.0, 'h_open': 175.0, 'h_closed': 70.0},
                'dip': {'servo_open': 180.0, 'servo_closed': 260.0, 'h_open': 175.0, 'h_closed': 120.0},
                'abd': {'servo_open': 195.0, 'servo_closed': 170.0, 'h_open': 100.0, 'h_closed': 120.0},
            },
            'Ring': {
                'mcp': {'servo_open': 160.0, 'servo_closed': 200.0, 'h_open': 15.0, 'h_closed': 44.0},
                'pip': {'servo_open': 160.0, 'servo_closed': 240.0, 'h_open': 175.0, 'h_closed': 40.0},
                'dip': {'servo_open': 180.0, 'servo_closed': 240.0, 'h_open': 175.0, 'h_closed': 156.0},
                'abd': {'servo_open': 170.0, 'servo_closed': 190.0, 'h_open': 72.0, 'h_closed': 85.0},
            },
            'Thumb': {
                'mcp': {'servo_open': 180.0, 'servo_closed': 260.0, 'h_open': 150.0, 'h_closed': 100.0},
                'pip': {'servo_open': 180.0, 'servo_closed': 260.0, 'h_open': 163.0, 'h_closed': 130.0},
                'dip': {'servo_open': 180.0, 'servo_closed': 250.0, 'h_open': 173.0, 'h_closed': 120.0},
                'abd': {'servo_open': 180.0, 'servo_closed': 260.0, 'h_open': 85.0, 'h_closed': 10.0},
                'twist': {
                    # when the human thumb is fully “opposed” (rotated in toward palm at ~+30°),
                    # send the servo to its “closed” twist limit:
                    'h_open': 20.0,  # human twist angle at which servo should be at its “open” limit
                    'h_closed': 60.0,  # human twist angle at which servo should be at its “closed” limit
                    'servo_open': 150.0,  # choose based on your hardware’s neutral/open twist position
                    'servo_closed': 180.0,  # choose based on your hardware’s max twist position
                },
            },
        }

        self.previous_angles = {}  # Store previous angles for each finger
        self.angle_update_threshold = 2  # degrees
        self.camera_id = 0
        # --- Random Play Option Variables ---
        self.rand_thumb_var = tk.BooleanVar(value=False)
        self.rand_index_var = tk.BooleanVar(value=False)
        self.rand_middle_var = tk.BooleanVar(value=False)
        self.rand_ring_var = tk.BooleanVar(value=False)
        self.rand_duration_var = tk.DoubleVar(value=1.0)  # in minutes

        # New: Effort limit and threshold variables.
        self.effort_limit_var = tk.IntVar(value=550)
        self.threshold_var = tk.IntVar(value=300)

        # Use smaller default dimensions.
        self.stream_width_var = tk.IntVar(value=320)
        self.stream_height_var = tk.IntVar(value=240)
        self.playing_paused = False

        # Main layout frames.
        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y, expand=False)
        # self.left_frame.config(width=250)
        self.right_frame = tk.Frame(master)
        # self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        # --- Mediapipe Teleop Stream Toggle ---
        self.mediapipe_enabled = tk.BooleanVar(value=False)
        self.mediapipe_checkbox = tk.Checkbutton(
            self.left_frame,
            text="Enable Mediapipe Teleop Stream",
            variable=self.mediapipe_enabled,
            command=self.toggle_mediapipe_stream
        )
        self.mediapipe_checkbox.pack(padx=5, pady=5, fill=tk.X)

        # In LeapHandGUI.__init__, after loading URDF & before building joint_map:
        self.human_cal = {
            "Thumb": {
                "abd": {"h_open": 93.0, "h_closed": 50.0},
                "mcp": {"h_open": 172.0, "h_closed": 155.0},
                "pip": {"h_open": 159.0, "h_closed": 112.0},
                "twist": {"h_open": 53.0, "h_closed": 127.0},
                # we’ll skip twist in joint_map for now
            },
            "Index": {
                "abd": {"h_open": 79.0, "h_closed": 142.0},
                "mcp": {"h_open": 105.0, "h_closed": 28.0},
                "pip": {"h_open": 175.0, "h_closed": 102.0},
                "dip": {"h_open": 176.0, "h_closed": 104.0},
            },
            "Middle": {
                "abd": {"h_open": 98.0, "h_closed": 110.0},
                "mcp": {"h_open": 18.0, "h_closed": 14.0},
                "pip": {"h_open": 174.0, "h_closed": 116.0},
                "dip": {"h_open": 177.0, "h_closed": 67.0},
            },
            "Ring": {
                "abd": {"h_open": 109.0, "h_closed": 95.0},
                "mcp": {"h_open": 17.0, "h_closed": 25.0},
                "pip": {"h_open": 176.0, "h_closed": 113.0},
                "dip": {"h_open": 175.0, "h_closed": 74.0},
            },
        }

        # Button: record “hand open” pose
        btn_open = tk.Button(self.left_frame, text="Calibrate Open Pose",
                             command=lambda: self.record_calibration("open"))
        btn_open.pack(fill=tk.X, padx=5, pady=(10, 2))

        # Button: record “fist closed” pose
        btn_closed = tk.Button(self.left_frame, text="Calibrate Closed Pose",
                               command=lambda: self.record_calibration("closed"))
        btn_closed.pack(fill=tk.X, padx=5, pady=(2, 10))

        self.available_cameras = find_available_cameras()
        camera_labels = [label for (_, label) in self.available_cameras]
        if not camera_labels:
            camera_labels = ["(no cameras found)"]

        tk.Label(self.left_frame, text="Select Camera:", font=("Arial", 10)).pack(anchor="w", padx=5)
        self.camera_combo = ttk.Combobox(
            self.left_frame,
            values=camera_labels,
            state="readonly",
            width=30
        )
        self.camera_combo.pack(anchor="w", padx=5, pady=(0, 10))

        # Default to the first available index (if any); otherwise -1
        if self.available_cameras:
            self.camera_combo.current(0)
            self.camera_id = self.available_cameras[0][0]
        else:
            self.camera_combo.current(0)
            self.camera_id = -1

        def on_camera_selection(event):
            sel = self.camera_combo.current()
            if sel < len(self.available_cameras):
                idx, _ = self.available_cameras[sel]
                self.camera_id = idx
                print(f"[GUI] camera_id updated to: {self.camera_id}")
            else:
                self.camera_id = -1
                print("[GUI] no valid camera selected")

        self.camera_combo.bind("<<ComboboxSelected>>", on_camera_selection)

        self.hand_active_var = tk.BooleanVar(value=False)
        self.activate_chk = tk.Checkbutton(
            self.master,
            text="Activate Hand",
            variable=self.hand_active_var,
            command=self.toggle_hand_activation
        )
        # You can choose where to pack it; here we just pack at the top.
        self.activate_chk.pack(anchor="nw", padx=5, pady=5)

        # Compute the “flat/safe” pose now, but do NOT move yet. Store it for later.
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        self.default_pose = np.deg2rad(flat_pose_deg)
        safe_pose = lhu.angle_safety_clip(self.default_pose)
        self.safe_pose = safe_pose.copy()
        self.safe_pose_deg = np.rad2deg(safe_pose)
        # self.current_command = np.copy(safe_pose)

        self.leap_node = None
        self.robot_initialized = False

        print("[GUI] LeapNode initialized but NOT active. Check-box must be ticked to move hand.")

        # ─────────────────────────────────────────────────────────────────
        # 2) Initialize PyBullet (DIRECT) & load the LEAP Hand URDF
        # ─────────────────────────────────────────────────────────────────

        # 2A) Start the physics server in headless mode
        self._pb_client = p.connect(p.GUI)
        # Optionally, add a small plane or use p.GUI if you want to visualize.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._pb_client)
        p.setGravity(0, 0, 0, physicsClientId=self._pb_client)

        # 2B) Specify the path to your URDF (the one you copied under urdfs/leap_robot)
        leap_urdf_path = "python/leap_hand_utils/urdfs/leap_hand/robot.urdf"

        # 2C) Load the URDF with self‐collision enabled; useFixedBase=1 so it doesn’t fall
        # Correct (positional arguments for basePosition and baseOrientation):
        base_position = [0, 0, 0]  # Must be a 3-element list :contentReference[oaicite:2]{index=2}
        base_orientation = [1, 0, 0, 0]  # Identity quaternion (x,y,z,w) :contentReference[oaicite:3]{index=3}
        use_fixed_base = 1  # 1 means the hand will stay fixed (no gravity effect)
        flags = p.URDF_USE_SELF_COLLISION  # Enable internal link–link collisions

        # If using default physicsClientId:
        self.hand_id = p.loadURDF(
            leap_urdf_path,
            base_position,
            base_orientation,
            use_fixed_base,
            flags
        )
        # ─── 1) Define the official home‐pose joint angles ───
        home = {
            0: 0.0, 1: 1.047, 2: 0.506, 3: 0.366,  # Index
            4: 0.0, 5: 1.047, 6: 0.506, 7: 0.366,  # Middle
            8: 0.0, 9: 1.047, 10: 0.506, 11: 0.366,  # Ring
            12: 0.349, 13: 2.443, 14: 1.900, 15: 1.880,  # Thumb
        }

        # ─── 2) Reset each joint kinematically to that home pose ───
        for ji, angle in home.items():
            p.resetJointState(self.hand_id, ji, angle, 0)

        # ─── 3) Step the sim so you see the home pose in GUI ───
        p.stepSimulation()

        # 2D) Cache all revolute/prismatic (DOF) joint indices and their limits
        num_joints = p.getNumJoints(self.hand_id)
        self._joint_names = []
        for ji in range(num_joints):
            info = p.getJointInfo(self.hand_id, ji)
            name = info[1].decode("utf-8")
            self._joint_names.append(name)

        # 2) Create a name → PyBullet index map
        self.urdf_name_to_pb_index = {
            self._joint_names[ji]: ji for ji in range(num_joints)
        }

        # 3) Build the explicit mapping from your 16-element cmd array to PB joint indices
        #    (Change these strings to match exactly what you saw in the printout!)
        self.cmd_to_pb_index = {
            # Index finger:
            0: self.urdf_name_to_pb_index["0"],  # “0” in URDF = Index_abd
            1: self.urdf_name_to_pb_index["1"],  # “1” in URDF = Index_mcp
            2: self.urdf_name_to_pb_index["2"],  # “2” in URDF = Index_pip
            3: self.urdf_name_to_pb_index["3"],  # “3” in URDF = Index_dip

            # Middle finger:
            4: self.urdf_name_to_pb_index["4"],  # “4” in URDF = Middle_abd
            5: self.urdf_name_to_pb_index["5"],  # “5” in URDF = Middle_mcp
            6: self.urdf_name_to_pb_index["6"],  # “6” in URDF = Middle_pip
            7: self.urdf_name_to_pb_index["7"],  # “7” in URDF = Middle_dip

            # Ring finger:
            8: self.urdf_name_to_pb_index["8"],  # “8” in URDF = Ring_abd
            9: self.urdf_name_to_pb_index["9"],  # “9” in URDF = Ring_mcp
            10: self.urdf_name_to_pb_index["10"],  # “10” in URDF = Ring_pip
            11: self.urdf_name_to_pb_index["11"],  # “11” in URDF = Ring_dip

            # Thumb:
            12: self.urdf_name_to_pb_index["12"],  # “12” in URDF = Thumb_abd
            13: self.urdf_name_to_pb_index["13"],  # “13” in URDF = Thumb_mcp
            14: self.urdf_name_to_pb_index["14"],  # “14” in URDF = Thumb_pip
            15: self.urdf_name_to_pb_index["15"],  # “15” in URDF = Thumb_dip
        }
        # Build a set of link‐index pairs that you WANT to ignore (adjacent links)
        self._allowed_self_collisions = set()

        # For each revolute joint, we know `parent` and `child` link indices:
        for ji in range(p.getNumJoints(self.hand_id)):
            info = p.getJointInfo(self.hand_id, ji)
            parent_link = info[16]  # index of the parent link
            child_link = info[0]  # joint index is also the child link index
            # Allow contacts between parent_link and child_link
            self._allowed_self_collisions.add((parent_link, child_link))
            self._allowed_self_collisions.add((child_link, parent_link))

        # 4) Cache pb_joint_limits for every DOF joint
        self.pb_joint_limits = {}
        for pb_idx in self.cmd_to_pb_index.values():
            info = p.getJointInfo(self.hand_id, pb_idx)
            lo, hi = info[8], info[9]
            self.pb_joint_limits[pb_idx] = (lo, hi)

        # 5) Build linear maps A,B for every cmd_idx including Thumb DIP
        # Build per‐joint linear maps: sim_angle = A * human_rad + B
        self.joint_map = {}

        for finger, base_idx in [("Index", 0), ("Middle", 4), ("Ring", 8), ("Thumb", 12)]:
            for i, joint_key in enumerate(["abd", "mcp", "pip", "dip"]):
                # Skip Thumb DIP if your model doesn’t use it:
                if finger == "Thumb" and joint_key == "dip":
                    continue

                # Human endpoints in degrees
                h_open = self.human_cal[finger][joint_key]["h_open"]
                h_closed = self.human_cal[finger][joint_key]["h_closed"]
                ro, rc = np.deg2rad(h_open), np.deg2rad(h_closed)

                cmd_idx = base_idx + i
                pb_idx = self.cmd_to_pb_index[cmd_idx]
                lo, hi = self.pb_joint_limits[pb_idx]

                # Solve A·ro + B = hi  and  A·rc + B = lo
                A = (lo - hi) / (rc - ro)
                B = hi - A * ro

                self.joint_map[cmd_idx] = (A, B)
                print(f"{finger}.{joint_key}: map {h_open}°→{np.degrees(hi):.1f}°, "
                      f"{h_closed}°→{np.degrees(lo):.1f}°")

        # Create the contact detection panel.
        self.create_contact_status_panel()
        self.update_contact_status()
        self.mp_gain = 0.4
        # -------------------------
        # LEFT SIDE – Controls and Configuration
        # -------------------------

        self.pos_slider_frame = tk.Frame(self.left_frame)
        self.pos_slider_frame.pack(padx=5, pady=5, fill=tk.X)

        tk.Label(self.pos_slider_frame, text="Position Control", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=5, pady=5)

        finger_names = ["Index", "Middle", "Ring", "Thumb"]
        joint_names = ["MCP Side", "MCP Forward", "PIP", "DIP"]
        self.sliders = [None] * 16  # Flat list for 16 sliders.

        for col, finger in enumerate(finger_names):
            header = tk.Label(self.pos_slider_frame, text=finger, font=("Arial", 10, "bold"))
            header.grid(row=1, column=col + 1, padx=5, pady=5)

        self.index_deg_limits = {
            i: (np.degrees(lo), np.degrees(hi))
            for i, (lo, hi) in self.pb_joint_limits.items()
            if i < 4
        }
        for row, joint in enumerate(joint_names):
            joint_label = tk.Label(self.pos_slider_frame, text=joint, width=12)
            joint_label.grid(row=row + 2, column=0, padx=5, pady=5)
            for col, finger in enumerate(finger_names):
                joint_index = col * 4 + row
                s = tk.Scale(self.pos_slider_frame, from_=0, to=360, orient=tk.HORIZONTAL,
                             resolution=1, length=250,
                             command=lambda v, idx=joint_index: self.position_slider_changed(idx, v))
                s.set(self.safe_pose_deg[joint_index])
                s.grid(row=row + 2, column=col + 1, padx=5, pady=5)
                self.sliders[joint_index] = s

        # --- Finger Teach Manager ---
        self.fingers = {"Index": [0, 1, 2, 3], "Middle": [4, 5, 6, 7],
                        "Ring": [8, 9, 10, 11], "Thumb": [12, 13, 14, 15]}
        self.finger_teach_frame = tk.Frame(self.left_frame)
        self.finger_teach_frame.pack(padx=5, pady=5, fill=tk.X)
        self.finger_teach_buttons = {}
        self.finger_teach_status = {}
        for finger in self.fingers:
            f_frame = tk.Frame(self.finger_teach_frame)
            f_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            btn = tk.Button(f_frame, text=f"{finger}: Enter Teach Mode")
            btn.pack(side=tk.LEFT, padx=5)
            lbl = tk.Label(f_frame, text="Normal")
            lbl.pack(side=tk.LEFT, padx=5)
            self.finger_teach_buttons[finger] = btn
            self.finger_teach_status[finger] = lbl
        self.finger_teach_manager = FingerTeachManager(self.master, self.fingers, self.leap_node,
                                                       self.sliders, self.move_to_pose)
        self.finger_teach_manager.register_ui_refs(self.finger_teach_buttons, self.finger_teach_status)
        for finger in self.fingers:
            self.finger_teach_buttons[finger].config(
                command=lambda f=finger: self.finger_teach_manager.toggle_finger_teach_mode(
                    f, self.finger_teach_buttons[f], self.finger_teach_status[f])
            )

        # --- Configuration Manager ---
        self.config_frame = tk.Frame(self.left_frame)
        self.config_frame.pack(padx=5, pady=5, fill=tk.X)
        self.save_config_button = tk.Button(self.config_frame, text="Save Current Configuration")
        self.save_config_button.grid(row=0, column=0, padx=5, pady=5)
        self.config_listbox = tk.Listbox(self.config_frame, height=6, width=40)
        self.config_listbox.grid(row=1, column=0, padx=5, pady=5)
        self.load_config_button = tk.Button(self.config_frame, text="Load Selected Configuration")
        self.load_config_button.grid(row=2, column=0, padx=5, pady=5)
        self.delete_config_button = tk.Button(self.config_frame, text="Delete Selected Configuration")
        self.delete_config_button.grid(row=3, column=0, padx=5, pady=5)
        self.rename_config_button = tk.Button(self.config_frame, text="Rename Selected Configuration")
        self.rename_config_button.grid(row=4, column=0, padx=5, pady=5)
        from configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager(self.config_listbox, self.sliders, self.leap_node, self.move_to_pose)
        self.save_config_button.config(command=self.config_manager.save_current_configuration)
        self.load_config_button.config(command=self.config_manager.load_selected_configuration)
        self.delete_config_button.config(command=self.config_manager.delete_selected_configuration)
        self.rename_config_button.config(command=self.config_manager.rename_selected_configuration)

        # --- Effort Settings ---
        control_frame = tk.Frame(self.left_frame)
        control_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(control_frame, text="Effort Limit").grid(row=0, column=0, padx=5, sticky="w")
        self.effort_slider = tk.Scale(control_frame, from_=100, to=1000, resolution=10, orient=tk.HORIZONTAL,
                                      length=300, variable=self.effort_limit_var, command=self.effort_limit_changed)
        self.effort_slider.grid(row=0, column=1, padx=5)
        tk.Label(control_frame, text="Effort Threshold").grid(row=1, column=0, padx=5, sticky="w")
        self.threshold_slider = tk.Scale(control_frame, from_=50, to=500, resolution=10, orient=tk.HORIZONTAL,
                                         length=300, variable=self.threshold_var, command=self.threshold_changed)
        self.threshold_slider.grid(row=1, column=1, padx=5)

        # --- Playback Speed Control ---
        self.playback_speed_var = tk.DoubleVar(value=1.0)
        self.playback_speed_slider = tk.Scale(self.left_frame, from_=0.5, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
                                              label="Playback Speed (x)", length=300,
                                              variable=self.playback_speed_var, command=self.playback_speed_changed)
        self.playback_speed_slider.pack(padx=5, pady=5)

        # --- Random Play Options ---
        self.random_frame = tk.Frame(self.left_frame, relief=tk.RIDGE, borderwidth=2)
        self.random_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(self.random_frame, text="Random Play Options", font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Checkbutton(self.random_frame, text="Thumb", variable=self.rand_thumb_var).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(self.random_frame, text="Index", variable=self.rand_index_var).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(self.random_frame, text="Middle", variable=self.rand_middle_var).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(self.random_frame, text="Ring", variable=self.rand_ring_var).pack(side=tk.LEFT, padx=2)
        tk.Label(self.random_frame, text="Duration (min):").pack(side=tk.LEFT, padx=5)
        self.rand_duration_entry = tk.Entry(self.random_frame, textvariable=self.rand_duration_var, width=5)
        self.rand_duration_entry.pack(side=tk.LEFT, padx=5)
        self.start_rand_button = tk.Button(self.random_frame, text="Start Random Play", command=self.start_random_play)
        self.start_rand_button.pack(side=tk.LEFT, padx=5)
        self.stop_rand_button = tk.Button(self.random_frame, text="Stop Random Play", command=self.stop_random_play,
                                          state=tk.DISABLED)
        self.stop_rand_button.pack(side=tk.LEFT, padx=5)
        self.rand_time_remaining_label = tk.Label(self.random_frame, text="Time Remaining: N/A", font=("Arial", 10))
        self.rand_time_remaining_label.pack(side=tk.LEFT, padx=5)

        # --- Trajectory Controls & Library ---
        traj_ctrl_frame = tk.Frame(self.left_frame)
        traj_ctrl_frame.pack(padx=5, pady=5)
        self.start_rec_button = tk.Button(traj_ctrl_frame, text="Start Recording")
        self.start_rec_button.pack(side=tk.LEFT, padx=5)
        self.stop_rec_button = tk.Button(traj_ctrl_frame, text="Stop Recording", state=tk.DISABLED)
        self.stop_rec_button.pack(side=tk.LEFT, padx=5)
        self.refresh_traj_button = tk.Button(traj_ctrl_frame, text="Refresh Trajectories",
                                             command=lambda: self.trajectory_manager.refresh_trajectory_library())
        self.refresh_traj_button.pack(side=tk.LEFT, padx=5)
        self.stop_playback_button = tk.Button(traj_ctrl_frame, text="Stop Playback", command=self.stop_playback)
        self.stop_playback_button.pack(side=tk.LEFT, padx=5)
        traj_lib_frame = tk.Frame(self.left_frame)
        traj_lib_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(traj_lib_frame, text="Trajectory Library:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.traj_listbox = tk.Listbox(traj_lib_frame, height=4, width=40)
        self.traj_listbox.pack(side=tk.LEFT, padx=5, pady=5)
        traj_btn_frame = tk.Frame(traj_lib_frame)
        traj_btn_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.play_traj_button = tk.Button(traj_btn_frame, text="Play Selected Trajectory",
                                          command=lambda: threading.Thread(
                                              target=self.trajectory_manager.play_trajectory,
                                              args=(self.traj_listbox.curselection()[0]
                                                    if self.traj_listbox.curselection() else None,),
                                              daemon=True).start())
        self.play_traj_button.pack(pady=2)
        self.pause_traj_button = tk.Button(traj_btn_frame, text="Pause", state=tk.DISABLED,
                                           command=self.toggle_pause)
        self.pause_traj_button.pack(pady=2)
        self.delete_traj_button = tk.Button(traj_btn_frame, text="Delete Selected Trajectory",
                                            command=lambda: self.trajectory_manager.delete_trajectory(
                                                self.traj_listbox.curselection()[0]
                                                if self.traj_listbox.curselection() else None))
        self.delete_traj_button.pack(pady=2)

        # --- Progress Bar ---
        self.progress_bar = ttk.Progressbar(self.left_frame, orient="horizontal", length=300, mode="determinate",
                                            maximum=100)
        self.progress_bar.pack(padx=5, pady=5, fill=tk.X)

        # -------------------------
        # RIGHT SIDE – Live & Playback Streams
        # -------------------------
        self.live_frame = tk.Frame(self.right_frame)
        self.live_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.playback_frame = tk.Frame(self.right_frame)
        self.playback_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.stream_names = ["Camera", "Thumb", "Index", "Middle", "Ring"]
        self.live_labels = {}
        self.playback_labels = {}
        for i, stream in enumerate(self.stream_names):
            # Create a plain tk.Frame container (so we can size it later)
            container = tk.Frame(self.live_frame, borderwidth=2, relief=tk.GROOVE)
            container.grid(row=0, column=i, padx=5, sticky="n")
            container.pack_propagate(True)  # disable auto‐resize
            self.live_containers.append(container)  # collect for size‐locking
            container.configure(width = self.stream_width_var.get(),  # or hard-code 640
                                height = self.stream_height_var.get()  # or hard-code 480
            )
            # Title label stays the same
            tk.Label(container, text=stream, font=("Arial", 12, "bold")).pack(side=tk.TOP, pady=2)

            # The actual image Label has no fixed width/height
            lbl = tk.Label(container, text="LIVE", bg="black", fg="red")
            lbl.pack(fill=tk.BOTH, expand=True, pady=5)
            self.live_labels[stream] = lbl

        for i, stream in enumerate(self.stream_names):
            playback_col_frame = tk.Frame(self.playback_frame, borderwidth=2, relief=tk.GROOVE)
            playback_col_frame.grid(row=0, column=i, padx=5, sticky="n")
            tk.Label(playback_col_frame, text=stream, font=("Arial", 12, "bold")).pack(side=tk.TOP, pady=2)
            label = tk.Label(playback_col_frame, bg="gray",
                             width=self.stream_width_var.get(), height=self.stream_height_var.get())
            self.playback_labels[stream] = label

        def update_live_image(stream, image):
            try:
                img = Image.fromarray(image)
                img = img.resize((self.stream_width_var.get(), self.stream_height_var.get()),
                                 Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.live_labels[stream].config(image=photo)
                self.live_labels[stream].image = photo
            except Exception as e:
                print(f"Error updating live image for {stream}: {e}")

        def update_playback_image(stream, image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((self.stream_width_var.get(), self.stream_height_var.get()),
                                 Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = self.playback_labels[stream]
                if not label.winfo_ismapped():
                    label.pack(side=tk.TOP, pady=5)
                label.config(image=photo)
                label.image = photo
            except Exception as e:
                print(f"Error updating playback image for {stream}: {e}")

        self.update_live_image = update_live_image
        self.update_playback_image = update_playback_image

        self.safe_pose = safe_pose.copy()  # preserve this as your “rest” pose
        self.current_command = safe_pose.copy()
        self.mp_alpha = 0.3  # smoothing factor (0 < α < 1)

        def update_replay_camera_from_array(stream, img):
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img = pil_img.resize((self.stream_width_var.get(), self.stream_height_var.get()),
                                         Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
                label = self.playback_labels[stream]
                if not label.winfo_ismapped():
                    label.pack(side=tk.TOP, pady=5)
                label.config(image=photo)
                label.image = photo
            except Exception as e:
                print(f"Error updating replay image for {stream} from array: {e}")

        self.sensor_manager = SensorManager(self.master, self.stream_width_var, self.stream_height_var)
        self.sensor_manager.live_update_callback = lambda frame: self.update_live_image("Camera", frame)
        self.sensor_manager.tactile_live_update_callback = lambda finger, frame: self.update_live_image(finger, frame)
        # self.sensor_manager.setup_digit_sensors(self.right_frame)
        # self.sensor_manager.setup_realsense_stream(self.right_frame)
        # hook up sensors & camera (unchanged)
        self.sensor_manager.setup_digit_sensors(self.right_frame)
        self.sensor_manager.setup_realsense_stream(self.right_frame)

        # ── FORCE FIXED SIZE ON ALL STREAM PANES ──
        # pack the right panel (unchanged)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Now lock only the *tk.Frame* containers you’ll create below,
        # not the Label widgets themselves.
        for container in self.live_containers:
            container.config(
                width=self.stream_width_var.get(),
                height=self.stream_height_var.get()
            )

        self.size_frame = tk.Frame(self.right_frame)
        self.size_frame.pack(side=tk.TOP, padx=5, pady=5)

        tk.Label(self.size_frame, text="Stream Width:").pack(side=tk.LEFT)
        tk.Spinbox(self.size_frame, from_=160, to=1280, increment=16,
                   textvariable=self.stream_width_var, width=5, command=self.update_stream_dimensions).pack(
            side=tk.LEFT, padx=5)
        tk.Label(self.size_frame, text="Stream Height:").pack(side=tk.LEFT)
        tk.Spinbox(self.size_frame, from_=120, to=720, increment=16,
                   textvariable=self.stream_height_var, width=5, command=self.update_stream_dimensions).pack(
            side=tk.LEFT, padx=5)

        self.update_current_positions()

        self.trajectory_manager = TrajectoryManager(self.master, self.leap_node, self.stream_width_var,
                                                    self.stream_height_var, self.start_rec_button,
                                                    self.stop_rec_button, self.traj_listbox)
        self.trajectory_manager.set_hand_pose_callback = lambda pose: self.move_to_pose(np.array(pose))
        self.trajectory_manager.get_tactile_frame_callback = lambda finger: self.sensor_manager.sensors[
            finger].get_frame() if finger in self.sensor_manager.sensors else None
        self.trajectory_manager.get_camera_frame_callback = lambda: self.sensor_manager.get_realsense_frame()
        self.trajectory_manager.replay_panel_show_callback = lambda: None
        self.trajectory_manager.pause_traj_button_callback = lambda state, text: self.pause_traj_button.config(
            state=tk.NORMAL if state == "normal" else tk.DISABLED, text=text)
        self.trajectory_manager.update_replay_camera_callback = lambda cam_path: self.update_playback_image("Camera",
                                                                                                            cam_path)
        self.trajectory_manager.update_replay_camera_callback_from_array = lambda img: update_replay_camera_from_array(
            "Camera", img)
        self.trajectory_manager.update_replay_tactile_callback = lambda finger, path: self.update_playback_image(finger,
                                                                                                                 path)
        self.trajectory_manager.progress_update_callback = lambda progress: self.master.after(0,
                                                                                              lambda: self.progress_bar.config(
                                                                                                  value=progress))
        self.trajectory_manager.random_play_end_callback = self.reset_random_play_buttons
        self.trajectory_manager.return_to_initial_pose_callback = lambda: self.move_to_pose(safe_pose,
                                                                                            override_safety=True)

        self.start_rec_button.config(command=self.delayed_start_recording)
        self.stop_rec_button.config(command=self.cancel_recording)
        self.start_rand_button.config(command=self.start_random_play)
        self.stop_rand_button.config(command=self.stop_random_play)
        self.play_traj_button.config(command=lambda: threading.Thread(target=self.trajectory_manager.play_trajectory,
                                                                      args=(self.traj_listbox.curselection()[0]
                                                                            if self.traj_listbox.curselection() else None,),
                                                                      daemon=True).start())
        self.pause_traj_button.config(command=self.toggle_pause)
        self.stop_playback_button.config(command=self.stop_playback)
        self.delete_traj_button.config(command=lambda: self.trajectory_manager.delete_trajectory(
            self.traj_listbox.curselection()[0]
            if self.traj_listbox.curselection() else None))

        # DEBUG: Print joint info for debugging
        print("Joint limits (rad):")
        for pb_idx in sorted(self.cmd_to_pb_index.values()):
            lo, hi = self.pb_joint_limits[pb_idx]
            print(f"  Joint {pb_idx}: [{lo:.2f}, {hi:.2f}] rad "
                  f"({np.degrees(lo):.1f}° → {np.degrees(hi):.1f}°)")
        self.prev_cmd_deg = self.safe_pose_deg.copy()


    def toggle_hand_activation(self):
        """
        Called whenever the “Activate Hand” checkbox is toggled.
        If checked: initialize robot (if needed), move to safe pose, then resume loop.
        If unchecked: pause control loop (if robot exists).
        """
        if self.hand_active_var.get():
            # 1) If robot not yet initialized, do it now:
            if not self.robot_initialized:
                try:
                    self.leap_node = LeapNode()
                    self.leap_node.initialize_current_pose_from_motors()
                    self.robot_initialized = True
                    print("[GUI] LeapNode initialized. Now activating hand.")
                except Exception as e:
                    print("[GUI] Error initializing robot:", e)
                    # Uncheck the box automatically since robot isn't present
                    self.hand_active_var.set(False)
                    return

            # 2) Now that robot exists, move to safe pose and resume loop
            try:
                self.leap_node.pause_control_loop()
                self.move_to_pose(self.safe_pose, override_safety=True)
                self.leap_node.resume_control_loop()
                print("[GUI] Hand ACTIVATED. Control loop running.")
            except Exception as e:
                print("[GUI] Error activating hand:", e)

        else:
            # When user unchecks the box: pause control loop (if robot exists)
            if self.robot_initialized and self.leap_node:
                try:
                    self.leap_node.pause_control_loop()
                    print("[GUI] Hand DEACTIVATED. No further commands will be sent.")
                except Exception as e:
                    print("[GUI] Error deactivating hand:", e)
            else:
                # Robot was never initialized, nothing to do
                print("[GUI] Hand checkbox unchecked, but robot was never initialized.")

    def toggle_mediapipe_stream(self):
        if self.mediapipe_enabled.get():
            # Only start Mediapipe if a valid camera_id has been chosen
            if self.camera_id < 0:
                print("[GUI] Cannot start Mediapipe: no camera selected.")
                # Uncheck the box automatically
                self.mediapipe_enabled.set(False)
                return

            self.show_teleop_checkboxes()
            self.start_mediapipe_stream()
        else:
            self.hide_teleop_checkboxes()
            self.stop_mediapipe_stream()

    def hide_teleop_checkboxes(self):
        # Destroy the panel and vars when teleop is turned off
        if hasattr(self, 'teleop_select_frame'):
            self.teleop_select_frame.destroy()
            del self.teleop_select_frame
        if hasattr(self, 'teleop_vars'):
            del self.teleop_vars

    def show_teleop_checkboxes(self):
        # Create a panel of checkboxes for each finger
        self.teleop_select_frame = tk.Frame(self.left_frame, relief=tk.RIDGE, borderwidth=2)
        self.teleop_select_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(self.teleop_select_frame, text="Teleop Fingers:", font=("Arial", 10, "bold")).pack(anchor="w", padx=5)
        self.teleop_vars = {}
        for finger in ["Thumb", "Index", "Middle", "Ring"]:
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(self.teleop_select_frame, text=finger, variable=var)
            chk.pack(side=tk.LEFT, padx=5)
            self.teleop_vars[finger] = var

    def start_mediapipe_stream(self):
        # Create a frame to display the mediapipe stream.
        self.mediapipe_frame = tk.Frame(self.right_frame, borderwidth=2, relief=tk.GROOVE)
        # Pack without fill=tk.X so it can expand vertically to fit the image.
        self.mediapipe_frame.pack(side=tk.BOTTOM, padx=5, pady=5)

        # Title label
        tk.Label(
            self.mediapipe_frame,
            text="Mediapipe Teleop Stream",
            font=("Arial", 12, "bold")
        ).pack(side=tk.TOP, pady=2)

        # Stream label — no fixed width/height so it sizes to the PhotoImage
        self.mediapipe_label = tk.Label(self.mediapipe_frame, bg="black")
        self.mediapipe_label.pack(side=tk.TOP, pady=5)

        # Instantiate and start the MediaPipe stream thread
        self.mediapipe_stream = MediapipeStream(self.camera_id)
        self.mediapipe_thread = threading.Thread(
            target=self.mediapipe_stream_loop,
            daemon=True
        )
        self.mediapipe_thread.start()

    def stop_mediapipe_stream(self):
        if hasattr(self, 'mediapipe_stream'):
            self.mediapipe_stream.stop()
        if hasattr(self, 'mediapipe_thread') and self.mediapipe_thread.is_alive():
            self.mediapipe_thread.join(timeout=1)
        if hasattr(self, 'mediapipe_frame') and self.mediapipe_frame:
            self.mediapipe_frame.destroy()
            self.mediapipe_frame = None

    def mediapipe_stream_loop(self):
        """
        Runs in its own thread:
          - receives (image, angles) from Mediapipe
          - updates the GUI image
          - applies angles to robot if enabled
        """
        from PIL import Image, ImageTk

        def callback(img, angles):
            # 1) Display the image as before
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.mediapipe_label.after(0, lambda p=photo: self.update_mediapipe_image(p))

            # 2) If teleop is enabled and we have angles, print human vs commanded
            if self.mediapipe_enabled.get() and angles:
                self.last_human_angles = angles.copy()
                # Print human‐detected angles:
                print("\n[Human]  Detected joint angles (degrees):")
                for finger, sub in angles.items():
                    # e.g. sub = {'mcp': 63, 'pip': 159, ...}
                    print(f"  {finger}: ", ", ".join(f"{k}={v:.1f}°" for k, v in sub.items()))

                # Build your 16‐element cmd_deg array exactly as in apply_mediapipe_angles:
                cmd_deg = self.compute_cmd_deg(angles)

                print("[Command] Fresh servo commands (deg):")
                for i, finger in enumerate(["Index", "Middle", "Ring", "Thumb"]):
                    block = cmd_deg[i * 4:(i + 1) * 4]
                    print(f"  {finger}: {' , '.join(f'{ang:.1f}°' for ang in block)}")

                # 2) Now call apply, passing in both cmd_deg and angles
                self.master.after(0, lambda a=angles, c=cmd_deg: self.apply_mediapipe_angles(a, c))

        w = self.stream_width_var.get() * 3
        h = self.stream_height_var.get() * 3
        self.mediapipe_stream.stream_loop(callback, w, h)
        # Compute display size
        w = self.stream_width_var.get() * 3
        h = self.stream_height_var.get() * 3

        # Enter the Mediapipe‐capture loop
        self.mediapipe_stream.stream_loop(callback, w, h)

    def compute_cmd_deg(self, angles: dict[str, dict[str, float]]) -> np.ndarray:
        """
        From the Mediapipe `angles` dict (in degrees), compute
        the 16-element servo‐degree command array, before mapping.
        """
        cmd_deg = np.zeros(16, dtype=float)
        α = 0.1

        # Thumb (cmd indices 12–15)
        if 'Thumb' in angles:
            a = angles['Thumb']['abduction']
            m = angles['Thumb']['mcp']
            p1 = angles['Thumb']['pip']
            # twist ignored for now or map to cmd[15] if you like
            cmd_deg[12] = (1 - α) * self.prev_cmd_deg[12] + α * a
            cmd_deg[13] = (1 - α) * self.prev_cmd_deg[13] + α * m
            cmd_deg[14] = (1 - α) * self.prev_cmd_deg[14] + α * p1
            cmd_deg[15] = cmd_deg[15]  # or handle twist

        # Then Index (0–3), Middle (4–7), Ring (8–11) similarly:
        for i, finger in enumerate(['Index', 'Middle', 'Ring']):
            base = i * 4
            if finger in angles:
                # abduction only for Index/Middle/Ring at index 0 of each block
                abd = angles[finger]['abduction']
                mcp = angles[finger]['mcp']
                pip = angles[finger]['pip']
                dip = angles[finger]['dip']
                cmd_deg[base + 0] = (1 - α) * self.prev_cmd_deg[base + 0] + α * abd
                cmd_deg[base + 1] = (1 - α) * self.prev_cmd_deg[base + 1] + α * mcp
                cmd_deg[base + 2] = (1 - α) * self.prev_cmd_deg[base + 2] + α * pip
                cmd_deg[base + 3] = (1 - α) * self.prev_cmd_deg[base + 3] + α * dip

        # Store for smoothing next time
        self.prev_cmd_deg = cmd_deg.copy()
        return cmd_deg

    def apply_mediapipe_angles(self, angles,cmd_deg):
        """
        1) Build a 16-element 'cmd' array (in radians) from Mediapipe angles + calibration.
        2) Kinematically reset those joint angles in PyBullet.
        3) Step simulation and check for any link–link self-collisions.
        4) If collision→reject; else→send to real robot or store locally.
        """
        # ── Guard clauses ────────────────────────────────────────
        if not angles or not hasattr(self, "teleop_vars"):
            return
        if not any(var.get() for var in self.teleop_vars.values()):
            return

        # start from the pose we last sent to the robot
        cmd = np.copy(self.current_command)
        α = 0.1  # smoothing factor

        # ------------ Index / Middle / Ring ---------------------
        for finger in ["Index", "Middle", "Ring"]:
            # skip if the checkbox is not ticked
            if finger not in self.teleop_vars or not self.teleop_vars[finger].get():
                continue
            if finger not in angles:
                continue

            for joint_name, pos in [("abduction", 0), ("mcp", 1),
                                    ("pip", 2), ("dip", 3)]:
                h = float(angles[finger].get(joint_name, 0.0))
                prev = self.previous_angles.get((finger, joint_name))
                sm = h if prev is None else (1 - α) * prev + α * h
                self.previous_angles[(finger, joint_name)] = sm

                key = "abd" if joint_name == "abduction" else joint_name
                cal = self.calibration[finger][key]
                ratio = (sm - cal["h_open"]) / (cal["h_closed"] - cal["h_open"])
                sd = cal["servo_open"] + ratio * (cal["servo_closed"] - cal["servo_open"])
                sd = max(min(sd, max(cal["servo_open"], cal["servo_closed"])),
                         min(cal["servo_open"], cal["servo_closed"]))

                idx = self.fingers[finger][pos]
                cmd[idx] = np.deg2rad(sd)
                # print degrees and radians for debugging
                print(f"→ {finger} {joint_name} → motor {idx}: "
                      f"{np.rad2deg(cmd[idx]):.1f}° (clipped from {sd:.1f}°)")

        # --------------------- Thumb -----------------------------
        if self.teleop_vars.get("Thumb", tk.BooleanVar(value=False)).get() and "Thumb" in angles:
            tidx = self.fingers["Thumb"]  # [motor0,1,2,3]

            # Abduction  → motor 0
            raw = float(angles["Thumb"].get("abduction", 0.0))
            prev = self.previous_angles.get(("Thumb", "abduction"))
            sm = raw if prev is None else (1 - α) * prev + α * raw
            self.previous_angles[("Thumb", "abduction")] = sm
            cal = self.calibration["Thumb"]["abd"]
            ratio = (sm - cal["h_open"]) / (cal["h_closed"] - cal["h_open"])
            sd = cal["servo_open"] + ratio * (cal["servo_closed"] - cal["servo_open"])
            sd = max(min(sd, max(cal["servo_open"], cal["servo_closed"])),
                     min(cal["servo_open"], cal["servo_closed"]))
            cmd[tidx[0]] = np.deg2rad(sd)

            # Twist       → motor 1
            raw = float(angles["Thumb"].get("twist", 0.0))
            prev = self.previous_angles.get(("Thumb", "twist"))
            sm = raw if prev is None else (1 - α) * prev + α * raw
            self.previous_angles[("Thumb", "twist")] = sm
            cal = self.calibration["Thumb"]["twist"]
            ratio = (sm - cal["h_open"]) / (cal["h_closed"] - cal["h_open"])
            sd = cal["servo_open"] + ratio * (cal["servo_closed"] - cal["servo_open"])
            sd = max(min(sd, max(cal["servo_open"], cal["servo_closed"])),
                     min(cal["servo_open"], cal["servo_closed"]))
            cmd[tidx[1]] = np.deg2rad(sd)

            # MCP flexion → motor 2
            raw = float(angles["Thumb"].get("mcp", 0.0))
            prev = self.previous_angles.get(("Thumb", "mcp"))
            sm = raw if prev is None else (1 - α) * prev + α * raw
            self.previous_angles[("Thumb", "mcp")] = sm
            cal = self.calibration["Thumb"]["mcp"]
            ratio = (sm - cal["h_open"]) / (cal["h_closed"] - cal["h_open"])
            sd = cal["servo_open"] + ratio * (cal["servo_closed"] - cal["servo_open"])
            sd = max(min(sd, max(cal["servo_open"], cal["servo_closed"])),
                     min(cal["servo_open"], cal["servo_closed"]))
            cmd[tidx[2]] = np.deg2rad(sd)

            # IP flexion  → motor 3
            raw = float(angles["Thumb"].get("pip", 0.0))
            prev = self.previous_angles.get(("Thumb", "pip"))
            sm = raw if prev is None else (1 - α) * prev + α * raw
            self.previous_angles[("Thumb", "pip")] = sm
            cal = self.calibration["Thumb"]["pip"]
            ratio = (sm - cal["h_open"]) / (cal["h_closed"] - cal["h_open"])
            sd = cal["servo_open"] + ratio * (cal["servo_closed"] - cal["servo_open"])
            sd = max(min(sd, max(cal["servo_open"], cal["servo_closed"])),
                     min(cal["servo_open"], cal["servo_closed"]))
            cmd[tidx[3]] = np.deg2rad(sd)

        # 3) Reset each controlled joint in PyBullet (positional arguments)
        # cmd_deg is your human‐servo degrees array; if you only have cmd in radians,
        # convert back: sd_rad = cmd[cmd_idx]; sd_deg = np.rad2deg(sd_rad)
        # but better: compute sd_deg before mapping into cmd[]

        for cmd_idx, pb_idx in self.cmd_to_pb_index.items():
            if cmd_idx not in self.joint_map:
                continue
            # Convert the smoothed human‐servo deg → radians
            sd_rad = np.deg2rad(cmd_deg[cmd_idx])

            # Apply the new linear map
            A, B = self.joint_map[cmd_idx]
            sim_angle = A * sd_rad + B

            p.setJointMotorControl2(
                self.hand_id,
                pb_idx,
                p.POSITION_CONTROL,
                targetPosition=sim_angle,
                positionGain=0.8,
                velocityGain=1.0,
                force=100
            )
        p.stepSimulation()


        # 5) Query for self-collision contacts (positional)
        # getContactPoints(bodyUniqueIdA, bodyUniqueIdB=-1, linkIndexA=-1, linkIndexB=-1)
        contacts = p.getContactPoints(self.hand_id)
        real_collisions = []
        for c in contacts:
            linkA = c[3]  # linkIndexA
            linkB = c[4]  # linkIndexB

            # Skip any contact involving the base (link index -1)
            if linkA < 0 or linkB < 0:
                continue

            # Skip adjacent link contacts (allowed self‐contacts)
            if (linkA, linkB) in self._allowed_self_collisions:
                continue

            # Otherwise, record as a genuine collision
            real_collisions.append((linkA, linkB))

        # 3) If any remaining contacts exist, block the pose
        if real_collisions:
            print(f"[GUI][Collision] Real self-collision between links: {real_collisions}")
            try:
                self.mediapipe_label.config(highlightbackground="red", highlightthickness=2)
                self.master.after(200, lambda: self.mediapipe_label.config(highlightthickness=0))
            except Exception:
                pass
            return

        # 6) No collision → forward to real robot if active, else store locally
        if self.hand_active_var.get() and self.robot_initialized and self.leap_node:
            safe_cmd = lhu.angle_safety_clip(cmd)
            self.leap_node.set_leap(safe_cmd)
            self.current_command = safe_cmd
        else:
            self.current_command = cmd

    def update_mediapipe_image(self, photo):
        self.mediapipe_label.config(image=photo)
        self.mediapipe_label.image = photo

    def reset_random_play_buttons(self):
        self.start_rand_button.config(state=tk.NORMAL)
        self.stop_rand_button.config(state=tk.DISABLED)
        self.rand_time_remaining_label.config(text="Time Remaining: N/A")

    def create_contact_status_panel(self):
        self.contact_status_panel = tk.Frame(self.left_frame, relief=tk.RIDGE, borderwidth=2)
        self.contact_status_panel.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(self.contact_status_panel, text="Fingertip Touch Status", font=("Arial", 10, "bold")).pack(
            anchor="w", padx=5, pady=2)
        self.contact_status_labels = {}
        for finger in ["Thumb", "Index", "Middle", "Ring"]:
            lbl = tk.Label(self.contact_status_panel, text=f"{finger}: No Contact", fg="green")
            lbl.pack(anchor="w", padx=5, pady=2)
            self.contact_status_labels[finger] = lbl
        self.finger_tip_map = {
            "Thumb": 15,
            "Index": 3,
            "Middle": 7,
            "Ring": 11
        }

    def update_contact_status(self):
        """
        If the robot exists, read motor currents and update contact labels.
        Otherwise, show 'N/A'. Always reschedule itself after 200 ms.
        """
        # If the robot is not initialized yet, show “N/A” and reschedule
        if not (self.robot_initialized and self.leap_node):
            for finger in self.finger_tip_map:
                self.contact_status_labels[finger].config(text=f"{finger}: N/A", fg="gray")
            self.master.after(200, self.update_contact_status)
            return

        # At this point, leap_node exists:
        try:
            currents = self.leap_node.dxl_client.read_cur()
            threshold = 34
            for finger, tip_index in self.finger_tip_map.items():
                if currents is not None:
                    current_value = currents[tip_index]
                    if current_value > threshold:
                        text = f"{finger}: {current_value:.2f} (Contact)"
                        self.contact_status_labels[finger].config(text=text, fg="red")
                    else:
                        text = f"{finger}: {current_value:.2f} (No Contact)"
                        self.contact_status_labels[finger].config(text=text, fg="green")
                else:
                    # If `read_cur()` returned None for some reason
                    self.contact_status_labels[finger].config(text=f"{finger}: Unknown", fg="orange")
        except Exception as e:
            print("Error updating contact status:", e)
        finally:
            # Always reschedule
            self.master.after(200, self.update_contact_status)

    def position_slider_changed(self, idx, value):
        try:
            # 1) Read the slider command
            cmd_deg = float(value)
            cmd_rad = np.deg2rad(cmd_deg)

            # 2) Figure out this joint’s PB limits
            pb_idx = self.cmd_to_pb_index[idx]
            lo, hi = self.pb_joint_limits[pb_idx]

            # 3) Clamp the command to [lo, hi]
            clamped_rad = max(min(cmd_rad, hi), lo)
            clamped_deg = np.degrees(clamped_rad)
            self.current_command[idx] = clamped_rad

            # 4) Kinematically reset so no force‐limit jumps
            p.resetJointState(self.hand_id, pb_idx, clamped_rad, 0)
            p.stepSimulation()

            # 5) Read back actual (should equal clamped_rad)
            actual_rad = p.getJointState(self.hand_id, pb_idx)[0]
            actual_deg = np.degrees(actual_rad)

            # 6) Print for comparison
            print(f"[SliderCmd ] cmd_idx={idx} → commanded={cmd_deg:.1f}°  "
                  f"clamped={clamped_deg:.1f}°")
            print(f"[SimState ] cmd_idx={idx} → actual   ={actual_deg:.1f}°")

        except Exception as e:
            print(f"[Error] Slider {idx}: {e}")

    def effort_limit_changed(self, value):
        try:
            new_limit = int(value)
            self.leap_node.dxl_client.set_current_limit(self.leap_node.motors, new_limit)
            print(f"Effort (current limit) set to: {new_limit}")
        except Exception as e:
            print("Error setting effort limit:", e)

    def threshold_changed(self, value):
        try:
            threshold = int(value)
            self.leap_node.dxl_client.set_effort_threshold(self.leap_node.motors, threshold)
            print(f"Effort threshold set to: {threshold}")
        except Exception as e:
            print("Error setting effort threshold:", e)

    def playback_speed_changed(self, value):
        try:
            self.trajectory_manager.playback_speed = float(value)
            print(f"Playback speed set to: {value}x")
        except Exception as e:
            print("Error setting playback speed:", e)

    def update_stream_dimensions(self):
        width = self.stream_width_var.get()
        height = self.stream_height_var.get()
        for stream in self.stream_names:
            self.live_labels[stream].config(width=width, height=height)
            self.playback_labels[stream].config(width=width, height=height)
        print(f"[INFO] Stream dimensions updated: {width}x{height}")

    def update_slider_positions(self):
        try:
            current_pose_deg = np.rad2deg(self.leap_node.read_pos())
            for i, slider in enumerate(self.sliders):
                slider.set(current_pose_deg[i])
        except Exception as e:
            print(f"Slider update failed: {e}")
        self.master.after(500, self.update_slider_positions)

    def read_positions(self):
        try:
            return self.leap_node.read_pos()
        except Exception as e:
            print("Error reading positions:", e)
            return None

    def update_current_positions(self):
        """
        Called every second to read the robot’s current joint positions and update sliders.
        If the robot is not initialized, simply reschedule without doing anything.
        """
        # If the robot is not initialized, just reschedule
        if not (self.robot_initialized and self.leap_node):
            self.master.after(1000, self.update_current_positions)
            return

        # At this point, the robot exists
        try:
            current_pose_rad = self.leap_node.read_pos()
            if current_pose_rad is not None:
                current_pose_deg = np.rad2deg(current_pose_rad)
                for i, slider in enumerate(self.sliders):
                    slider.set(current_pose_deg[i])
        except Exception as e:
            print("Error reading positions:", e)
        finally:
            # Always reschedule
            self.master.after(1000, self.update_current_positions)

    def move_to_pose(self, target_pose, steps=None, delay=None, override_safety=False):
        # We assume this is only called when hand_active_var is True,
        # so you can either put a guard here, or rely on toggle_hand_activation
        # to be the only caller before activation.
        current_pose = self.leap_node.curr_pos.copy()
        if steps is None:
            steps = GLOBAL_SMOOTHING_STEPS
        if delay is None:
            delay = GLOBAL_SMOOTHING_DELAY

        self.leap_node.pause_control_loop()
        for i in range(1, steps + 1):
            t = i / steps
            s = 3 * (t ** 2) - 2 * (t ** 3)
            interp_pose = current_pose + (target_pose - current_pose) * s
            self.leap_node.set_leap(interp_pose, override_safety=override_safety)
            time.sleep(delay / 1000.0)
        self.leap_node.resume_control_loop()

    def toggle_pause(self):
        if self.trajectory_manager:
            self.trajectory_manager.toggle_pause()
        else:
            print("Trajectory manager not initialized.")

    def stop_playback(self):
        if self.trajectory_manager:
            self.trajectory_manager.stop_playback()
            print("Stop playback requested.")
        else:
            print("Trajectory manager not initialized.")

    def delayed_start_recording(self):
        print("Start Recording requested. Recording will begin in 3 seconds...")
        self.start_rec_button.config(state=tk.DISABLED)
        self.start_delay_job = self.master.after(3000, self.execute_start_recording)

    def execute_start_recording(self):
        self.start_delay_job = None
        self.trajectory_manager.start_recording()

    def cancel_recording(self):
        if self.start_delay_job is not None:
            self.master.after_cancel(self.start_delay_job)
            self.start_delay_job = None
            print("Delayed start canceled.")
            self.start_rec_button.config(state=tk.NORMAL)
        else:
            self.trajectory_manager.stop_recording()

    def start_random_play(self):
        if self.start_delay_job is not None:
            self.master.after_cancel(self.start_delay_job)
            self.start_delay_job = None
        fingers = []
        if self.rand_thumb_var.get():
            fingers.append("Thumb")
        if self.rand_index_var.get():
            fingers.append("Index")
        if self.rand_middle_var.get():
            fingers.append("Middle")
        if self.rand_ring_var.get():
            fingers.append("Ring")
        duration = self.rand_duration_var.get()
        if not fingers:
            print("No fingers selected for random play.")
            return
        self.start_rand_button.config(state=tk.DISABLED)
        self.stop_rand_button.config(state=tk.NORMAL)
        self.trajectory_manager.record_random_trajectory(duration, fingers)
        self.update_random_time_remaining()

    def stop_random_play(self):
        self.trajectory_manager.stop_recording()
        self.start_rand_button.config(state=tk.NORMAL)
        self.stop_rand_button.config(state=tk.DISABLED)

    def stop_playback_flag(self):
        self.stop_playback()

    def update_random_time_remaining(self):
        if self.trajectory_manager.recording:
            elapsed = time.time() - self.trajectory_manager.record_start_time
            total = self.trajectory_manager.random_duration_seconds if hasattr(self.trajectory_manager,
                                                                               "random_duration_seconds") else 0
            remaining = total - elapsed
            if remaining < 0:
                remaining = 0
            self.rand_time_remaining_label.config(text=f"Time Remaining: {int(remaining)} sec")
            self.master.after(500, self.update_random_time_remaining)
        else:
            self.rand_time_remaining_label.config(text="Random Play Complete")
            self.start_rand_button.config(state=tk.NORMAL)
            self.stop_rand_button.config(state=tk.DISABLED)

    def on_closing(self):
        # Only move to the safe pose if the hand was active
        if self.hand_active_var.get():
            flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
            flat_pose_deg = np.rad2deg(flat_pose_rad)
            for idx in [1, 5, 9]:
                flat_pose_deg[idx] = 94
            safe_pose = np.deg2rad(flat_pose_deg)
            try:
                self.move_to_pose(safe_pose, override_safety=True)
            except Exception as e:
                print("Error moving to safe pose on exit:", e)

        # Disable torque regardless of whether the hand was active
        try:
            self.leap_node.dxl_client.set_torque_enabled(self.leap_node.motors, False)
        except Exception as e:
            print("Error disabling torque:", e)

        # Disconnect any sensors
        try:
            self.sensor_manager.disconnect_all()
        except Exception as e:
            print("Error disconnecting sensors:", e)

        self.master.destroy()

    def record_calibration(self, phase: str):
        """
        phase = "open" or "closed".
        Grabs the most recent Mediapipe angles and prints/stores them.
        """
        if not hasattr(self, "last_human_angles") or self.last_human_angles is None:
            print("[Calib] No hand detected yet.")
            return

        ang = self.last_human_angles
        print(f"\n=== Calibration ({phase}) ===")
        for finger, joints in ang.items():
            # Build a line only for available keys
            parts = []
            for joint_key in ["abduction", "mcp", "pip", "dip", "twist"]:
                if joint_key in joints and joints[joint_key] is not None:
                    parts.append(f"{joint_key}={joints[joint_key]:.1f}°")
                    # Store it
                    self.human_cal[finger].setdefault(phase, {})[joint_key] = joints[joint_key]
            print(f"{finger}:  " + ", ".join(parts))
        print("=== End Calibration ===\n")


def main():
    root = tk.Tk()
    gui = LeapHandGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
