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

# Global smoothing parameters.
GLOBAL_SMOOTHING_STEPS = 30
GLOBAL_SMOOTHING_DELAY = 20


class LeapHandGUI:
    def __init__(self, master):
        self.master = master
        master.title("LEAP Hand Real-Time Control GUI")

        # For delayed start of normal recording.
        self.start_delay_job = None

        # --- Random Play Option Variables ---
        self.rand_thumb_var = tk.BooleanVar(value=False)
        self.rand_index_var = tk.BooleanVar(value=False)
        self.rand_middle_var = tk.BooleanVar(value=False)
        self.rand_ring_var = tk.BooleanVar(value=False)
        self.rand_duration_var = tk.DoubleVar(value=1.0)  # in minutes

        # --- Control Mode Variable ---
        # Options: "position" or "velocity"
        self.control_mode_var = tk.StringVar(value="position")

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

        # Create LeapNode instance.
        self.leap_node = LeapNode()
        self.leap_node.initialize_current_pose_from_motors()

        # Compute safe pose.
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        default_pose = np.deg2rad(flat_pose_deg)
        safe_pose = lhu.angle_safety_clip(default_pose)
        # Compute safe_pose in degrees for use in position sliders.
        safe_pose_deg = np.rad2deg(safe_pose)

        self.leap_node.pause_control_loop()
        self.move_to_pose(safe_pose, override_safety=True)
        self.leap_node.resume_control_loop()

        # -------------------------
        # LEFT SIDE – Controls and Configuration
        # -------------------------

        # --- Control Mode Selection ---
        mode_frame = tk.Frame(self.left_frame)
        mode_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(mode_frame, text="Control Mode:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="Position", variable=self.control_mode_var,
                       value="position", command=self.switch_control_mode).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="Velocity", variable=self.control_mode_var,
                       value="velocity", command=self.switch_control_mode).pack(side=tk.LEFT, padx=5)

        # --- Create two separate slider frames: one for Position, one for Velocity ---
        self.pos_slider_frame = tk.Frame(self.left_frame)
        self.vel_slider_frame = tk.Frame(self.left_frame)

        # Populate Position Sliders (Range: 0–360°)
        self.pos_sliders = []
        tk.Label(self.pos_slider_frame, text="Position Control", font=("Arial", 10, "bold")).pack()
        for i in range(16):
            f = tk.Frame(self.pos_slider_frame)
            f.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(f, text=f"Joint {i + 1} (°):", width=12).pack(side=tk.LEFT)
            s = tk.Scale(f, from_=0, to=360, orient=tk.HORIZONTAL, resolution=1, length=250,
                         command=lambda v, idx=i: self.position_slider_changed(idx, v))
            s.set(safe_pose_deg[i])
            s.pack(side=tk.LEFT)
            self.pos_sliders.append(s)

        # Populate Velocity Sliders (Range: -50 to 50; adjust range as needed)
        self.vel_sliders = []
        tk.Label(self.vel_slider_frame, text="Velocity Control", font=("Arial", 10, "bold")).pack()
        for i in range(16):
            f = tk.Frame(self.vel_slider_frame)
            f.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(f, text=f"Joint {i + 1} (rad/s):", width=15).pack(side=tk.LEFT)
            s = tk.Scale(f, from_=-50, to=50, orient=tk.HORIZONTAL, resolution=1, length=250,
                         command=lambda v, idx=i: self.velocity_slider_changed(idx, v))
            s.set(0)  # Neutral corresponds to zero offset
            s.pack(side=tk.LEFT)
            self.vel_sliders.append(s)

        # Initially show position sliders.
        self.pos_slider_frame.pack(padx=5, pady=5, fill=tk.X)
        self.vel_slider_frame.forget()

        # --- Finger Teach Manager (remains unchanged) ---
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
                                                       self.pos_sliders, self.move_to_pose)
        self.finger_teach_manager.register_ui_refs(self.finger_teach_buttons, self.finger_teach_status)

        def make_teach_command(finger, button, label):
            return lambda: self.finger_teach_manager.toggle_finger_teach_mode(finger, button, label)

        for finger in self.fingers:
            self.finger_teach_buttons[finger].config(
                command=make_teach_command(finger, self.finger_teach_buttons[finger], self.finger_teach_status[finger]))

        # --- Configuration Manager (as before) ---
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
        self.config_manager = ConfigurationManager(self.config_listbox, self.pos_sliders, self.leap_node,
                                                   self.move_to_pose)
        self.save_config_button.config(command=self.config_manager.save_current_configuration)
        self.load_config_button.config(command=self.config_manager.load_selected_configuration)
        self.delete_config_button.config(command=self.config_manager.delete_selected_configuration)
        self.rename_config_button.config(command=self.config_manager.rename_selected_configuration)

        # --- Velocity Control & Effort Settings ---
        control_frame = tk.Frame(self.left_frame)
        control_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(control_frame, text="Set Velocity (rad/s) [top control]").grid(row=0, column=0, padx=5, sticky="w")
        self.vel_slider_top = tk.Scale(control_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=300,
                                       command=self.velocity_changed)
        self.vel_slider_top.set(0)
        self.vel_slider_top.grid(row=0, column=1, padx=5)
        tk.Label(control_frame, text="Effort Limit").grid(row=1, column=0, padx=5, sticky="w")
        self.effort_slider = tk.Scale(control_frame, from_=100, to=1000, resolution=10, orient=tk.HORIZONTAL,
                                      length=300,
                                      variable=self.effort_limit_var, command=self.effort_limit_changed)
        self.effort_slider.grid(row=1, column=1, padx=5)
        tk.Label(control_frame, text="Effort Threshold").grid(row=2, column=0, padx=5, sticky="w")
        self.threshold_slider = tk.Scale(control_frame, from_=50, to=500, resolution=10, orient=tk.HORIZONTAL,
                                         length=300,
                                         variable=self.threshold_var, command=self.threshold_changed)
        self.threshold_slider.grid(row=2, column=1, padx=5)

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
                                                self.traj_listbox.curselection()[
                                                    0] if self.traj_listbox.curselection() else None))
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
                img = img.resize((self.stream_width_var.get(), self.stream_height_var.get()), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.live_labels[stream].config(image=photo)
                self.live_labels[stream].image = photo
            except Exception as e:
                print(f"Error updating live image for {stream}: {e}")

        def update_playback_image(stream, image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((self.stream_width_var.get(), self.stream_height_var.get()), Image.Resampling.LANCZOS)
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

        self.start_rec_button.config(command=self.delayed_start_recording)
        self.stop_rec_button.config(command=self.cancel_recording)
        self.start_rand_button.config(command=self.start_random_play)
        self.stop_rand_button.config(command=self.stop_random_play)
        self.play_traj_button.config(command=lambda: threading.Thread(target=self.trajectory_manager.play_trajectory,
                                                                      args=(self.traj_listbox.curselection()[
                                                                                0] if self.traj_listbox.curselection() else None,),
                                                                      daemon=True).start())
        self.pause_traj_button.config(command=self.toggle_pause)
        self.stop_playback_button.config(command=self.stop_playback)
        self.delete_traj_button.config(command=lambda: self.trajectory_manager.delete_trajectory(
            self.traj_listbox.curselection()[0] if self.traj_listbox.curselection() else None))

        self.create_touch_detection_panel()

    def switch_control_mode(self):
        mode = self.control_mode_var.get()
        if mode == "position":
            self.vel_slider_frame.forget()
            self.pos_slider_frame.pack(padx=5, pady=5, fill=tk.X)
        elif mode == "velocity":
            # Capture current positions as baseline.
            self.velocity_baseline = self.leap_node.read_pos().copy()
            self.pos_slider_frame.forget()
            self.vel_slider_frame.pack(padx=5, pady=5, fill=tk.X)
        print(f"Switched control mode to: {mode}")

    def position_slider_changed(self, idx, value):
        try:
            angle_deg = float(value)
            angle_rad = np.deg2rad(angle_deg)
            current_pos_deg = np.rad2deg(self.leap_node.read_pos())
            current_pos_deg[idx] = angle_deg
            new_pos = np.deg2rad(current_pos_deg)
            self.leap_node.set_leap(new_pos)
        except Exception as e:
            print(f"Error in position_slider_changed for joint {idx}: {e}")

    def velocity_slider_changed(self, idx, value):
        try:
            vel_command = float(value)
            # Gain factor determines how much the slider value changes the position offset.
            gain = 0.05  # Adjust this gain to suit your system.
            offset = vel_command * gain
            if not hasattr(self, "velocity_baseline"):
                self.velocity_baseline = self.leap_node.read_pos().copy()
            desired_position = self.velocity_baseline.copy()
            desired_position[idx] += offset
            # Command the new position while keeping safe baseline for other joints.
            self.leap_node.set_leap(desired_position, override_safety=True)
        except Exception as e:
            print(f"Error in velocity_slider_changed for joint {idx}: {e}")

    def velocity_changed(self, value):
        try:
            vel = float(value)
            velocities = np.full(16, np.deg2rad(vel))
            if self.control_mode_var.get() == "velocity":
                self.leap_node.set_velocity(velocities)
            print(f"Top velocity slider set to: {vel} (converted to rad/s)")
        except Exception as e:
            print("Error setting velocity:", e)

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

    def stop_random_play(self):
        self.trajectory_manager.stop_recording()
        self.start_rand_button.config(state=tk.NORMAL)
        self.stop_rand_button.config(state=tk.DISABLED)

    def stop_playback_flag(self):
        self.stop_playback()

    def create_touch_detection_panel(self):
        """Create a panel that displays current (torque) measurements for each finger."""
        self.touch_panel = tk.Frame(self.left_frame, relief=tk.RIDGE, borderwidth=2)
        self.touch_panel.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(self.touch_panel, text="Touch Detection (Current Levels)", font=("Arial", 10, "bold")).pack(anchor="w",
                                                                                                             padx=5,
                                                                                                             pady=2)

        # Mapping of fingers to motor indices (modify if needed).
        self.finger_motor_map = {
            "Thumb": [12, 13, 14, 15],
            "Index": [0, 1, 2, 3],
            "Middle": [4, 5, 6, 7],
            "Ring": [8, 9, 10, 11]
        }

        self.touch_bars = {}
        # Create a progress bar for each finger.
        for finger in self.finger_motor_map:
            frame = tk.Frame(self.touch_panel)
            frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(frame, text=f"{finger} Current:").pack(side=tk.LEFT)
            # The maximum value here is arbitrary. Calibrate it according to your motors.
            bar = ttk.Progressbar(frame, orient="horizontal", length=200, mode="determinate", maximum=1000)
            bar.pack(side=tk.LEFT, padx=5)
            self.touch_bars[finger] = bar

        # Schedule periodic update of the touch panel.
        self.update_touch_panel()

    def update_touch_panel(self):
        """Periodically update the touch detection panel with current motor loads."""
        try:
            # Read currents from the Dynamixel client.
            # It is assumed that self.leap_node.dxl_client.read_cur() returns a NumPy array
            # with current values for each motor.
            currents = self.leap_node.dxl_client.read_cur()
            if currents is not None:
                for finger, motor_ids in self.finger_motor_map.items():
                    # Get the currents for the motors that belong to this finger.
                    finger_currents = [currents[i] for i in motor_ids]
                    # Compute an average (or max) value.
                    avg_current = np.mean(finger_currents)
                    # Update the corresponding progress bar.
                    self.touch_bars[finger]['value'] = avg_current
        except Exception as e:
            print("Error updating touch panel:", e)
        # Schedule next update (e.g., every 200 ms).
        self.master.after(200, self.update_touch_panel)

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
