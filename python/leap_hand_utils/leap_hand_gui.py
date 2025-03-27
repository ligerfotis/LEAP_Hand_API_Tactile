import tkinter as tk
import tkinter.simpledialog as simpledialog
import tkinter.messagebox as messagebox
import numpy as np
import time
import json
import os
import threading
from PIL import Image, ImageTk
import cv2
import pyrealsense2 as rs

import leap_hand_utils as lhu
from python.leap_hand_utils.leap_node import LeapNode

# Import the Digit class from digit-interface package.
from digit_interface import Digit

class LeapHandGUI:
    def __init__(self, master):
        self.master = master
        master.title("LEAP Hand Real-Time Control GUI")

        # Variables to control stream dimensions (default values)
        self.stream_width_var = tk.IntVar(value=640)
        self.stream_height_var = tk.IntVar(value=480)

        # Create two main frames: left for controls, right for sensor streams.
        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create an instance of LeapNode.
        self.leap_node = LeapNode()

        # Compute the target flat hand pose.
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        target_pose = np.deg2rad(flat_pose_deg)
        self.move_to_pose(target_pose, steps=50, delay=50)

        # --------------------------
        # Left Frame: Controls
        # --------------------------
        # --- GUI for manual control (sliders) ---
        finger_names = ["Index", "Middle", "Ring", "Thumb"]
        joint_names = ["MCP Side", "MCP Forward", "PIP", "DIP"]
        self.joint_labels = [f"{finger} {joint}" for finger in finger_names for joint in joint_names]

        self.slider_frame = tk.Frame(self.left_frame)
        self.slider_frame.pack(padx=5, pady=5)

        self.sliders = []
        for i in range(16):
            label = tk.Label(self.slider_frame, text=self.joint_labels[i])
            label.grid(row=i, column=0, padx=5, pady=3, sticky="w")
            slider = tk.Scale(self.slider_frame, from_=0, to=360, orient=tk.HORIZONTAL,
                              resolution=1, length=300, command=self.slider_changed)
            slider.set(flat_pose_deg[i])
            slider.grid(row=i, column=1, padx=5, pady=3)
            self.sliders.append(slider)

        self.read_button = tk.Button(self.left_frame, text="Read Current Positions", command=self.read_positions)
        self.read_button.pack(pady=5)
        self.current_positions_label = tk.Label(self.left_frame, text="Current Positions: ")
        self.current_positions_label.pack(pady=5)

        # --- GUI for Finger Teach Mode ---
        self.fingers = {
            "Index": [0, 1, 2, 3],
            "Middle": [4, 5, 6, 7],
            "Ring": [8, 9, 10, 11],
            "Thumb": [12, 13, 14, 15]
        }
        self.finger_teach_modes = {}
        self.finger_teach_buttons = {}
        self.finger_teach_status = {}

        self.finger_teach_frame = tk.Frame(self.left_frame)
        self.finger_teach_frame.pack(padx=5, pady=5, fill=tk.X)
        for finger in self.fingers.keys():
            self.finger_teach_modes[finger] = False
            finger_frame = tk.Frame(self.finger_teach_frame)
            finger_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            btn = tk.Button(finger_frame, text=f"{finger}: Enter Teach Mode",
                            command=lambda f=finger: self.toggle_finger_teach_mode(f))
            btn.pack(side=tk.LEFT, padx=5)
            self.finger_teach_buttons[finger] = btn
            status_label = tk.Label(finger_frame, text="Normal")
            status_label.pack(side=tk.LEFT, padx=5)
            self.finger_teach_status[finger] = status_label

        # --- GUI for Configuration Library ---
        self.config_frame = tk.Frame(self.left_frame)
        self.config_frame.pack(padx=5, pady=5, fill=tk.X)
        self.save_config_button = tk.Button(self.config_frame, text="Save Current Configuration",
                                            command=self.save_current_configuration)
        self.save_config_button.grid(row=0, column=0, padx=5, pady=5)
        self.config_listbox = tk.Listbox(self.config_frame, height=6, width=40)
        self.config_listbox.grid(row=1, column=0, padx=5, pady=5)
        self.load_config_button = tk.Button(self.config_frame, text="Load Selected Configuration",
                                            command=self.load_selected_configuration)
        self.load_config_button.grid(row=2, column=0, padx=5, pady=5)
        self.delete_config_button = tk.Button(self.config_frame, text="Delete Selected Configuration",
                                              command=self.delete_selected_configuration)
        self.delete_config_button.grid(row=3, column=0, padx=5, pady=5)
        self.rename_config_button = tk.Button(self.config_frame, text="Rename Selected Configuration",
                                              command=self.rename_selected_configuration)
        self.rename_config_button.grid(row=4, column=0, padx=5, pady=5)

        self.load_configurations()
        self.update_config_listbox()

        # --- GUI for Velocity Control ---
        self.vel_frame = tk.Frame(self.left_frame)
        self.vel_frame.pack(padx=5, pady=5)
        vel_label = tk.Label(self.vel_frame, text="Set Velocity (rad/s)")
        vel_label.pack(side=tk.LEFT, padx=5)
        self.vel_slider = tk.Scale(self.vel_frame, from_=0, to=5, resolution=0.1,
                                   orient=tk.HORIZONTAL, length=300, command=self.velocity_changed)
        self.vel_slider.set(0)
        self.vel_slider.pack(side=tk.LEFT, padx=5)

        self.update_current_positions()

        # --------------------------
        # Right Frame: Sensor Streams
        # --------------------------
        # Create a subframe at the top of the right frame to let the user choose stream sizes.
        self.size_frame = tk.Frame(self.right_frame)
        self.size_frame.pack(side=tk.TOP, padx=5, pady=5)
        tk.Label(self.size_frame, text="Stream Width:").pack(side=tk.LEFT)
        self.stream_width_spin = tk.Spinbox(self.size_frame, from_=160, to=1280, increment=16, textvariable=self.stream_width_var, width=5, command=self.update_stream_dimensions)
        self.stream_width_spin.pack(side=tk.LEFT, padx=5)
        tk.Label(self.size_frame, text="Stream Height:").pack(side=tk.LEFT)
        self.stream_height_spin = tk.Spinbox(self.size_frame, from_=120, to=720, increment=16, textvariable=self.stream_height_var, width=5, command=self.update_stream_dimensions)
        self.stream_height_spin.pack(side=tk.LEFT, padx=5)

        # Create a frame that holds two subframes: one for DIGIT sensors and one for the RealSense camera.
        self.sensor_right_frame = tk.Frame(self.right_frame)
        self.sensor_right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- DIGIT Sensor Streams ---
        # Create a frame that will contain the two tactile columns.
        self.digit_frame = tk.Frame(self.right_frame)
        self.digit_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        digit_title = tk.Label(self.digit_frame, text="DIGIT Sensor Streams", font=("Arial", 14, "bold"))
        digit_title.pack(pady=5)

        # Create two subframes for two columns.
        self.tactile_left_frame = tk.Frame(self.digit_frame)
        self.tactile_left_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.tactile_right_frame = tk.Frame(self.digit_frame)
        self.tactile_right_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Hardcoded sensor serial numbers for each finger.
        finger_sensor_serials = {
            "Thumb": "D21133",
            "Index": "D21117",
            "Middle": "D21131",
            "Ring": "D20844"
        }
        self.sensors = {}  # key: finger name, value: Digit object
        self.sensor_labels = {}  # key: finger name, value: tk.Label for image display

        from digit_interface import Digit
        for finger, serial in finger_sensor_serials.items():
            try:
                sensor = Digit(serial)
                sensor.connect()
                self.sensors[finger] = sensor

                # Choose the column based on finger name.
                container_parent = self.tactile_left_frame if finger in ["Index",
                                                                         "Middle"] else self.tactile_right_frame

                container = tk.Frame(container_parent, bd=2, relief=tk.RIDGE)
                container.pack(padx=5, pady=5)
                # Create a static label with the finger name.
                name_lbl = tk.Label(container, text=finger, font=("Arial", 12, "bold"))
                name_lbl.pack(side=tk.TOP, pady=(2, 0))
                # Create the image label with fixed dimensions.
                img_lbl = tk.Label(container, bg="black", width=self.stream_width_var.get(),
                                   height=self.stream_height_var.get())
                img_lbl.pack(side=tk.TOP, pady=(0, 2))
                # Optional: a "Show View" button for the sensor.
                view_btn = tk.Button(container, text="Show View",
                                     command=lambda f=finger: threading.Thread(target=self.show_sensor_view, args=(f,),
                                                                               daemon=True).start())
                view_btn.pack(side=tk.TOP, pady=(0, 2))
                self.sensor_labels[finger] = img_lbl
            except Exception as e:
                err_lbl = tk.Label(self.digit_frame, text=f"Error connecting {finger} sensor ({serial}): {e}")
                err_lbl.pack(pady=2)

        # --- RealSense Camera Stream ---
        self.camera_frame = tk.Frame(self.sensor_right_frame, bd=2, relief=tk.RIDGE)
        self.camera_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        camera_title = tk.Label(self.camera_frame, text="RealSense D435 RGB Stream", font=("Arial", 14, "bold"))
        camera_title.pack(pady=5)
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(padx=5, pady=5)

        # Start update loops in separate threads.
        threading.Thread(target=self.update_sensor_images, daemon=True).start()
        threading.Thread(target=self.update_camera_stream, daemon=True).start()

    def update_stream_dimensions(self):
        # This method is called when the user changes the stream dimensions.
        # It does not need to update the labels directly; the update loops will use the new dimensions.
        pass  # We use the spinbox variable values in the update functions.

    def move_to_pose(self, target_pose, steps=50, delay=50):
        current_pose = self.leap_node.curr_pos.copy()
        for i in range(1, steps + 1):
            interp_pose = current_pose + (target_pose - current_pose) * (i / steps)
            self.leap_node.set_leap(interp_pose)
            time.sleep(delay / 1000.0)

    def slider_changed(self, value):
        angles_deg = np.array([slider.get() for slider in self.sliders])
        angles_rad = np.deg2rad(angles_deg)
        self.leap_node.set_leap(angles_rad)

    def read_positions(self):
        try:
            pos_rad = self.leap_node.read_pos()
            pos_deg = np.rad2deg(pos_rad)
            pos_text = ", ".join(f"{p:.1f}" for p in pos_deg)
            self.current_positions_label.config(text="Current Positions: " + pos_text)
        except Exception as e:
            self.current_positions_label.config(text="Error reading positions: " + str(e))

    def update_current_positions(self):
        self.read_positions()
        self.master.after(1000, self.update_current_positions)

    # ---------- Finger Teach Mode Methods ----------
    def toggle_finger_teach_mode(self, finger):
        if not self.finger_teach_modes[finger]:
            self.finger_teach_modes[finger] = True
            self.finger_teach_buttons[finger].config(text=f"{finger}: Exit Teach Mode & Capture")
            self.finger_teach_status[finger].config(text="Teach Mode Active")
            self.update_finger_teach_mode(finger)
        else:
            self.finger_teach_modes[finger] = False
            captured_pose = self.leap_node.read_pos()
            captured_pose_deg = np.rad2deg(captured_pose)
            for i in self.fingers[finger]:
                self.sliders[i].set(captured_pose_deg[i])
            self.finger_teach_buttons[finger].config(text=f"{finger}: Enter Teach Mode")
            self.finger_teach_status[finger].config(text="Normal")

    def update_finger_teach_mode(self, finger):
        if self.finger_teach_modes[finger]:
            current_pose = self.leap_node.read_pos()
            new_pose = self.leap_node.curr_pos.copy()
            for i in self.fingers[finger]:
                new_pose[i] = current_pose[i]
            self.leap_node.set_leap(new_pose)
            self.master.after(50, lambda: self.update_finger_teach_mode(finger))

    # ---------- Configuration Library Methods ----------
    def load_configurations(self):
        if os.path.exists("saved_configs.json"):
            try:
                with open("saved_configs.json", "r") as f:
                    self.configurations = json.load(f)
            except Exception as e:
                print("Error loading configurations:", e)
                self.configurations = []
        else:
            self.configurations = []

    def save_configurations(self):
        try:
            with open("saved_configs.json", "w") as f:
                json.dump(self.configurations, f, indent=4)
        except Exception as e:
            print("Error saving configurations:", e)

    def update_config_listbox(self):
        self.config_listbox.delete(0, tk.END)
        for config in self.configurations:
            self.config_listbox.insert(tk.END, config["name"])

    def save_current_configuration(self):
        name = simpledialog.askstring("Save Configuration", "Enter configuration name:")
        if name:
            pose_rad = self.leap_node.read_pos()
            pose_deg = np.rad2deg(pose_rad).tolist()
            config = {"name": name, "pose": pose_deg}
            self.configurations.append(config)
            self.save_configurations()
            self.update_config_listbox()

    def load_selected_configuration(self):
        selected_index = self.config_listbox.curselection()
        if selected_index:
            config = self.configurations[selected_index[0]]
            target_pose = np.deg2rad(np.array(config["pose"]))
            self.move_to_pose(target_pose, steps=50, delay=50)
            loaded_pose_deg = np.rad2deg(target_pose)
            for i, slider in enumerate(self.sliders):
                slider.set(loaded_pose_deg[i])

    def delete_selected_configuration(self):
        selected_index = self.config_listbox.curselection()
        if selected_index:
            confirm = messagebox.askyesno("Delete Configuration",
                                          "Are you sure you want to delete the selected configuration?")
            if confirm:
                del self.configurations[selected_index[0]]
                self.save_configurations()
                self.update_config_listbox()

    def rename_selected_configuration(self):
        selected_index = self.config_listbox.curselection()
        if selected_index:
            new_name = simpledialog.askstring("Rename Configuration", "Enter the new configuration name:")
            if new_name:
                self.configurations[selected_index[0]]["name"] = new_name
                self.save_configurations()
                self.update_config_listbox()

    # ---------- New: Velocity Control Method ----------
    def velocity_changed(self, value):
        try:
            vel = float(value)
            velocities = np.full(16, vel)
            self.leap_node.set_velocity(velocities)
        except Exception as e:
            print("Error setting velocity:", e)

    # ---------- New: Sensor Image Update for DIGIT ----------
    def update_sensor_images(self):
        """
        Continuously update the DIGIT sensor images.
        For each sensor, get the current frame, resize it according to the current
        stream dimensions, convert it to a PhotoImage, and update its corresponding label.
        """
        while True:
            width = self.stream_width_var.get()
            height = self.stream_height_var.get()
            for finger, sensor in self.sensors.items():
                try:
                    frame = sensor.get_frame()  # Expected to return a NumPy array (H x W x 3)
                    img = Image.fromarray(frame)
                    img = img.resize((width, height))
                    photo = ImageTk.PhotoImage(img)
                    self.master.after(0, lambda p=photo, f=finger: self.sensor_labels[f].config(image=p))
                    self.sensor_labels[finger].image = photo
                except Exception as e:
                    print(f"Error streaming {finger} sensor: {e}")
            time.sleep(0.03)

    # ---------- New: RealSense Camera Stream ----------
    def update_camera_stream(self):
        pipeline = rs.pipeline()
        config = rs.config()
        width = self.stream_width_var.get()
        height = self.stream_height_var.get()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        pipeline.start(config)
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(img))
                self.master.after(0, lambda p=photo: self.camera_label.config(image=p))
                self.camera_label.image = photo
                time.sleep(0.03)
        finally:
            pipeline.stop()

    # ---------- New: Show View Button Callback for DIGIT ----------
    def show_sensor_view(self, finger):
        try:
            self.sensors[finger].show_view()
        except Exception as e:
            print(f"Error showing view for {finger} sensor: {e}")

    # ---------- Safe Shutdown ----------
    def on_closing(self):
        flat_pose_rad = lhu.allegro_to_LEAPhand(np.zeros(16))
        flat_pose_deg = np.rad2deg(flat_pose_rad)
        for idx in [1, 5, 9]:
            flat_pose_deg[idx] = 94
        target_pose = np.deg2rad(flat_pose_deg)
        self.move_to_pose(target_pose, steps=50, delay=50)
        self.leap_node.dxl_client.set_torque_enabled(self.leap_node.motors, False)
        for finger, sensor in self.sensors.items():
            try:
                sensor.disconnect()
            except Exception as e:
                print(f"Error disconnecting {finger} sensor: {e}")
        self.master.destroy()

def main():
    root = tk.Tk()
    gui = LeapHandGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
