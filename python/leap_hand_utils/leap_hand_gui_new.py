# leap_hand_gui_new.py

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


class LeapHandGUI:
    def __init__(self, master):
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
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- Mediapipe Teleop Stream Toggle ---
        self.mediapipe_enabled = tk.BooleanVar(value=False)
        self.mediapipe_checkbox = tk.Checkbutton(
            self.left_frame,
            text="Enable Mediapipe Teleop Stream",
            variable=self.mediapipe_enabled,
            command=self.toggle_mediapipe_stream
        )
        self.mediapipe_checkbox.pack(padx=5, pady=5, fill=tk.X)

        # Create LeapNode instance.
        self.leap_node = LeapNode()
        self.leap_node.initialize_current_pose_from_motors()

        # Compute safe pose.
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        self.default_pose = np.deg2rad(flat_pose_deg)
        safe_pose = lhu.angle_safety_clip(self.default_pose)
        # Compute safe_pose in degrees for use in the sliders.
        safe_pose_deg = np.rad2deg(safe_pose)

        # Initialize a persistent command vector to hold desired positions.
        self.current_command = np.copy(safe_pose)

        self.leap_node.pause_control_loop()
        self.move_to_pose(safe_pose, override_safety=True)
        print("[GUI] Initialized LeapNode with default pose:", self.default_pose)
        self.leap_node.resume_control_loop()

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

        for row, joint in enumerate(joint_names):
            joint_label = tk.Label(self.pos_slider_frame, text=joint, width=12)
            joint_label.grid(row=row + 2, column=0, padx=5, pady=5)
            for col, finger in enumerate(finger_names):
                joint_index = col * 4 + row
                s = tk.Scale(self.pos_slider_frame, from_=0, to=360, orient=tk.HORIZONTAL,
                             resolution=1, length=250,
                             command=lambda v, idx=joint_index: self.position_slider_changed(idx, v))
                s.set(safe_pose_deg[joint_index])
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
            live_col_frame = tk.Frame(self.live_frame, borderwidth=2, relief=tk.GROOVE)
            live_col_frame.grid(row=0, column=i, padx=5, sticky="n")
            tk.Label(live_col_frame, text=stream, font=("Arial", 12, "bold")).pack(side=tk.TOP, pady=2)
            self.live_labels[stream] = tk.Label(live_col_frame, text="LIVE", bg="black",
                                                width=self.stream_width_var.get(), height=self.stream_height_var.get(),
                                                fg="red")
            self.live_labels[stream].pack(side=tk.TOP, pady=5)
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
        self.sensor_manager.setup_digit_sensors(self.right_frame)
        self.sensor_manager.setup_realsense_stream(self.right_frame)

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

    def toggle_mediapipe_stream(self):
        if self.mediapipe_enabled.get():
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
        self.mediapipe_stream = MediapipeStream()
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
            # display
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.mediapipe_label.after(0, lambda p=photo: self.update_mediapipe_image(p))

            # only apply if teleop is on and we actually detected a hand
            if self.mediapipe_enabled.get() and angles:
                print("[GUI] Human hand angles:", angles)
                default_pose_deg = np.round(np.rad2deg(self.default_pose))
                print("[GUI] Robot hand angles:", default_pose_deg)
                self.master.after(0, lambda a=angles: self.apply_mediapipe_angles(a))

        w, h = self.stream_width_var.get(), self.stream_height_var.get()
        self.mediapipe_stream.stream_loop(callback, w, h)

    def apply_mediapipe_angles(self, angles):
        """
        Per-finger tele-operation gated by check-boxes.

        Only fingers whose BooleanVar is checked ( .get() == True )
        are updated from Mediapipe; all others stay exactly where
        they were (self.current_command).

        If no hand, or no check-boxes ticked, we do nothing.
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

        # ---------- send & remember pose -------------------------
        self.leap_node.set_leap(lhu.angle_safety_clip(cmd))
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
        try:
            currents = self.leap_node.dxl_client.read_cur()
            threshold = 34
            for finger, tip_index in self.finger_tip_map.items():
                if currents is not None:
                    current_value = currents[tip_index]
                    if current_value > threshold:
                        text = f"{finger}: {current_value:.2f} (Contact)"
                        self.contact_status_labels[finger].config(text=text, fg="red")
                        print(f"Contact detected on {finger}: {current_value:.2f} (above {threshold})")
                    else:
                        text = f"{finger}: {current_value:.2f} (No Contact)"
                        self.contact_status_labels[finger].config(text=text, fg="green")
                else:
                    self.contact_status_labels[finger].config(text=f"{finger}: Unknown", fg="orange")
                    print(f"{finger} current: Unknown")
        except Exception as e:
            print("Error updating contact status:", e)
        self.master.after(200, self.update_contact_status)

    def position_slider_changed(self, idx, value):
        try:
            angle_deg = float(value)
            self.current_command[idx] = np.deg2rad(angle_deg)
            self.leap_node.set_leap(self.current_command)
        except Exception as e:
            print(f"Error in position_slider_changed for joint {idx}: {e}")

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
        self.read_positions()
        self.master.after(1000, self.update_current_positions)

    def move_to_pose(self, target_pose, steps=None, delay=None, override_safety=False):
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
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        safe_pose = np.deg2rad(flat_pose_deg)
        self.move_to_pose(safe_pose, override_safety=True)
        try:
            self.leap_node.dxl_client.set_torque_enabled(self.leap_node.motors, False)
        except Exception as e:
            print("Error disabling torque:", e)
        try:
            self.sensor_manager.disconnect_all()
        except Exception as e:
            print("Error disconnecting sensors:", e)
        self.master.destroy()


def main():
    root = tk.Tk()
    gui = LeapHandGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
