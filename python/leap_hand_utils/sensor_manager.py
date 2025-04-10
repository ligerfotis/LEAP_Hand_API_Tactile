# sensor_manager.py
import os
import time
import threading
import numpy as np
from PIL import Image, ImageTk
import cv2
import pyrealsense2 as rs
from digit_interface import Digit
import tkinter as tk

class SensorManager:
    def __init__(self, master, stream_width_var, stream_height_var):
        self.master = master
        self.stream_width_var = stream_width_var
        self.stream_height_var = stream_height_var

        self.realsense_pipeline = rs.pipeline()
        self.realsense_config = rs.config()
        self.realsense_started = False

        # Dictionary to hold DIGIT sensor objects.
        self.sensors = {}
        # Callbacks to update your custom UI (set these from your LeapHandGUI).
        self.live_update_callback = None          # For RealSense camera live updates.
        self.tactile_live_update_callback = None   # For DIGIT sensor live updates.

        # Hardcoded serial numbers for each finger sensor.
        self.finger_sensor_serials = {
            "Thumb": "D21133",
            "Index": "D21117",
            "Middle": "D21131",
            "Ring": "D20844"
        }

    def start_realsense(self):
        if not self.realsense_started:
            width = self.stream_width_var.get()
            height = self.stream_height_var.get()
            self.realsense_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
            try:
                self.realsense_pipeline.start(self.realsense_config)
                self.realsense_started = True
                print("[RealSense] Pipeline started.")
            except Exception as e:
                print(f"[RealSense] Failed to start pipeline: {e}")

    def setup_digit_sensors(self, _unused_container):
        # Initialize DIGIT sensors without creating any UI elements.
        for finger, serial in self.finger_sensor_serials.items():
            try:
                sensor = Digit(serial)
                sensor.connect()
                sensor.set_fps(Digit.STREAMS["VGA"]["fps"]["15fps"])
                self.sensors[finger] = sensor
            except Exception as e:
                print(f"[SensorManager] Error connecting {finger} sensor ({serial}): {e}")
        threading.Thread(target=self.update_sensor_images, daemon=True).start()

    def setup_realsense_stream(self, _unused_container):
        # Start the RealSense stream without creating a camera label.
        self.start_realsense()
        threading.Thread(target=self.update_camera_stream, daemon=True).start()

    def update_sensor_images(self):
        # Continuously update sensor images.
        while True:
            for finger, sensor in self.sensors.items():
                try:
                    frame = sensor.get_frame()  # Get the current frame from the sensor.
                    if self.tactile_live_update_callback:
                        self.tactile_live_update_callback(finger, frame)
                        print(f"[DEBUG] Tactile live update callback called for {finger}")
                except Exception as e:
                    print(f"[SensorManager] Error streaming {finger} sensor: {e}")
            time.sleep(0.03)

    def update_camera_stream(self):
        # Continuously update the camera stream.
        while True:
            try:
                frames = self.realsense_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.live_update_callback:
                    self.live_update_callback(img)
                    print("[DEBUG] Live update callback called for Camera")
            except Exception as e:
                print(f"[RealSense] Streaming error: {e}")
            time.sleep(0.03)

    def get_realsense_frame(self):
        if not self.realsense_started:
            self.start_realsense()
        try:
            frames = self.realsense_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"[RealSense] Failed to get frame: {e}")
        return None

    def show_sensor_view(self, finger):
        try:
            self.sensors[finger].show_view()
        except Exception as e:
            print(f"[SensorManager] Error showing view for {finger} sensor: {e}")

    def disconnect_all(self):
        for finger, sensor in self.sensors.items():
            try:
                sensor.disconnect()
            except Exception as e:
                print(f"[SensorManager] Error disconnecting {finger} sensor: {e}")

    def update_replay_camera(self, cam_path, camera_label):
        try:
            img = Image.open(cam_path).resize(
                (int(640 * 0.8), int(420 * 0.8)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            camera_label.config(image=photo)
            camera_label.image = photo
        except Exception as e:
            print(f"[SensorManager] Failed to update replay camera image: {e}")

    def update_replay_tactile(self, finger, tactile_path, label_widget):
        try:
            img = Image.open(tactile_path).resize(
                (int(640 * 0.8), int(420 * 0.8)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo)
            label_widget.image = photo
        except Exception as e:
            print(f"[SensorManager] Failed to update {finger} tactile replay image: {e}")

    def update_stream_size(self, width, height):
        # Update internal stream dimensions.
        self.stream_width_var.set(width)
        self.stream_height_var.set(height)
