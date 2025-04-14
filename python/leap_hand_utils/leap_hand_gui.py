import threading
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import time
import cv2  # needed for image conversion in replay
from sensor_manager import SensorManager
from trajectory_manager import TrajectoryManager
from configuration_manager import ConfigurationManager
from finger_teach_manager import FingerTeachManager
from leap_node import LeapNode
import leap_hand_utils as lhu
from PIL import Image, ImageTk

# Global smoothing parameters for smooth movements.
GLOBAL_SMOOTHING_STEPS = 30  # Default interpolation steps for smooth transitions.
GLOBAL_SMOOTHING_DELAY = 20  # Default delay in ms between steps.


class LeapHandGUI:
    def __init__(self, master):
        self.master = master
        master.title("LEAP Hand Real-Time Control GUI")

        # For delayed start: store the after job id (if any).
        self.start_delay_job = None

        # Use smaller default dimensions for streams.
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

        # Compute the safe (flat) pose.
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        default_pose = np.deg2rad(flat_pose_deg)
        safe_pose = lhu.angle_safety_clip(default_pose)
        # Pause control loop, move to safe pose and resume.
        self.leap_node.pause_control_loop()
        self.move_to_pose(safe_pose, override_safety=True)
        self.leap_node.resume_control_loop()

        # -------------------------
        # LEFT SIDE – Manual Controls, Configuration, etc.
        # -------------------------
        finger_names = ["Index", "Middle", "Ring", "Thumb"]
        joint_names = ["MCP Side", "MCP Forward", "PIP", "DIP"]
        self.joint_labels = [f"{f} {j}" for f in finger_names for j in joint_names]

        self.slider_frame = tk.Frame(self.left_frame)
        self.slider_frame.pack(padx=5, pady=5)
        self.sliders = []
        safe_pose_deg = np.rad2deg(safe_pose)
        for i in range(16):
            tk.Label(self.slider_frame, text=self.joint_labels[i]).grid(row=i, column=0, sticky="w", padx=5, pady=3)
            s = tk.Scale(self.slider_frame, from_=0, to=360, orient=tk.HORIZONTAL, resolution=1, length=300,
                         command=self.slider_changed)
            s.set(safe_pose_deg[i])
            s.grid(row=i, column=1, padx=5, pady=3)
            self.sliders.append(s)

        # --- Finger Teach Manager ---
        self.fingers = {"Index": [0, 1, 2, 3],
                        "Middle": [4, 5, 6, 7],
                        "Ring": [8, 9, 10, 11],
                        "Thumb": [12, 13, 14, 15]}
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

        self.finger_teach_manager = FingerTeachManager(self.master, self.fingers, self.leap_node, self.sliders,
                                                       self.move_to_pose)
        self.finger_teach_manager.register_ui_refs(self.finger_teach_buttons, self.finger_teach_status)

        def make_teach_command(finger, button, label):
            return lambda: self.finger_teach_manager.toggle_finger_teach_mode(finger, button, label)

        for finger in self.fingers:
            self.finger_teach_buttons[finger].config(
                command=make_teach_command(finger, self.finger_teach_buttons[finger], self.finger_teach_status[finger]))

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

        # --- Velocity Control ---
        self.vel_frame = tk.Frame(self.left_frame)
        self.vel_frame.pack(padx=5, pady=5)
        tk.Label(self.vel_frame, text="Set Velocity (rad/s)").pack(side=tk.LEFT, padx=5)
        self.vel_slider = tk.Scale(self.vel_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=300,
                                   command=self.velocity_changed)
        self.vel_slider.set(0)
        self.vel_slider.pack(side=tk.LEFT, padx=5)

        # --- Playback Speed Control ---
        self.playback_speed_var = tk.DoubleVar(value=1.0)
        self.playback_speed_slider = tk.Scale(self.left_frame, from_=0.5, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
                                              label="Playback Speed (x)", length=300, variable=self.playback_speed_var,
                                              command=self.playback_speed_changed)
        self.playback_speed_slider.pack(padx=5, pady=5)

        # --- Trajectory Controls & Library ---
        traj_ctrl_frame = tk.Frame(self.left_frame)
        traj_ctrl_frame.pack(padx=5, pady=5)
        self.start_rec_button = tk.Button(traj_ctrl_frame, text="Start Recording")
        self.start_rec_button.pack(side=tk.LEFT, padx=5)
        self.stop_rec_button = tk.Button(traj_ctrl_frame, text="Stop Recording", state=tk.DISABLED)
        self.stop_rec_button.pack(side=tk.LEFT, padx=5)
        traj_lib_frame = tk.Frame(self.left_frame)
        traj_lib_frame.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(traj_lib_frame, text="Trajectory Library:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.traj_listbox = tk.Listbox(traj_lib_frame, height=4, width=40)
        self.traj_listbox.pack(side=tk.LEFT, padx=5, pady=5)
        traj_btn_frame = tk.Frame(traj_lib_frame)
        traj_btn_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.play_traj_button = tk.Button(traj_btn_frame, text="Play Selected Trajectory")
        self.play_traj_button.pack(pady=2)
        self.pause_traj_button = tk.Button(traj_btn_frame, text="Pause", state=tk.DISABLED)
        self.pause_traj_button.pack(pady=2)
        self.delete_traj_button = tk.Button(traj_btn_frame, text="Delete Selected Trajectory")
        self.delete_traj_button.pack(pady=2)

        # --- Add a progress bar for playback indicator ---
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
            label = tk.Label(playback_col_frame, bg="gray", width=self.stream_width_var.get(),
                             height=self.stream_height_var.get())
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

        # New function: update replay camera using an image array.
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

        # ----- Sensor Manager Setup -----
        self.sensor_manager = SensorManager(self.master, self.stream_width_var, self.stream_height_var)
        self.sensor_manager.live_update_callback = lambda frame: self.update_live_image("Camera", frame)
        self.sensor_manager.tactile_live_update_callback = lambda finger, frame: self.update_live_image(finger, frame)
        self.sensor_manager.setup_digit_sensors(self.right_frame)
        self.sensor_manager.setup_realsense_stream(self.right_frame)

        # Spinboxes for resizing streams.
        self.size_frame = tk.Frame(self.right_frame)
        self.size_frame.pack(side=tk.TOP, padx=5, pady=5)
        tk.Label(self.size_frame, text="Stream Width:").pack(side=tk.LEFT)
        tk.Spinbox(self.size_frame, from_=160, to=1280, increment=16, textvariable=self.stream_width_var, width=5,
                   command=self.update_stream_dimensions).pack(side=tk.LEFT, padx=5)
        tk.Label(self.size_frame, text="Stream Height:").pack(side=tk.LEFT)
        tk.Spinbox(self.size_frame, from_=120, to=720, increment=16, textvariable=self.stream_height_var, width=5,
                   command=self.update_stream_dimensions).pack(side=tk.LEFT, padx=5)

        self.update_current_positions()

        # --- Trajectory Manager Integration ---
        self.trajectory_manager = TrajectoryManager(
            master=self.master,
            leap_node=self.leap_node,
            stream_width_var=self.stream_width_var,
            stream_height_var=self.stream_height_var,
            start_rec_button=self.start_rec_button,
            stop_rec_button=self.stop_rec_button,
            traj_listbox=self.traj_listbox
        )
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

        # --- Bind Start/Stop Recording with delay and cancellation ---
        # Bind Start button to delayed start.
        self.start_rec_button.config(command=self.delayed_start_recording)
        # Bind Stop button to cancel delayed start (if any) and stop recording.
        self.stop_rec_button.config(command=self.cancel_recording)

        self.play_traj_button.config(
            command=lambda: threading.Thread(
                target=self.trajectory_manager.play_trajectory,
                args=(self.traj_listbox.curselection()[0] if self.traj_listbox.curselection() else None,),
                daemon=True
            ).start()
        )
        self.pause_traj_button.config(command=self.trajectory_manager.toggle_pause)
        self.delete_traj_button.config(
            command=lambda: self.trajectory_manager.delete_trajectory(
                self.traj_listbox.curselection()[0] if self.traj_listbox.curselection() else None
            )
        )

    def delayed_start_recording(self):
        # When the start button is pressed, wait 3 seconds before starting recording.
        print("Start Recording requested. Recording will begin in 1 second...")
        self.start_rec_button.config(state=tk.DISABLED)
        # Schedule the actual start after 3000ms.
        self.start_delay_job = self.master.after(1000, self.execute_start_recording)

    def execute_start_recording(self):
        self.start_delay_job = None
        self.trajectory_manager.start_recording()

    def cancel_recording(self):
        # If a delayed start is scheduled, cancel it.
        if self.start_delay_job is not None:
            self.master.after_cancel(self.start_delay_job)
            self.start_delay_job = None
            print("Delayed start canceled.")
            # Re-enable the start button.
            self.start_rec_button.config(state=tk.NORMAL)
        else:
            # Otherwise, if recording is active, stop it.
            self.trajectory_manager.stop_recording()

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

    def slider_changed(self, _):
        if hasattr(self, "_last_slider_time") and time.time() - self._last_slider_time < 0.05:
            return
        self._last_slider_time = time.time()
        angles_deg = np.array([slider.get() for slider in self.sliders])
        angles_rad = np.deg2rad(angles_deg)
        self.leap_node.set_leap(angles_rad)

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

    def velocity_changed(self, value):
        try:
            vel = float(value)
            velocities = np.full(16, vel)
            self.leap_node.set_velocity(velocities)
        except Exception as e:
            print("Error setting velocity:", e)

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
