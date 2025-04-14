import os
import time
import json
import shutil
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.simpledialog as simpledialog

class TrajectoryManager:
    def __init__(self, master, leap_node, stream_width_var, stream_height_var,
                 start_rec_button, stop_rec_button, traj_listbox):
        """
        Initialize the TrajectoryManager.
        Parameters:
            master: The TK root window.
            leap_node: The LEAP hand controller.
            stream_width_var, stream_height_var: tkinter IntVars for image dimensions.
            start_rec_button, stop_rec_button: Buttons for recording controls.
            traj_listbox: Listbox widget for showing saved trajectories.
        """
        self.master = master
        self.leap_node = leap_node
        self.stream_width_var = stream_width_var
        self.stream_height_var = stream_height_var
        self.start_rec_button = start_rec_button
        self.stop_rec_button = stop_rec_button
        self.traj_listbox = traj_listbox

        # Recording state and folder management.
        self.recording = False
        self.trajectory = []
        self.record_start_time = None
        self.current_traj_folder = None
        self.camera_folder = None
        self.thumb_folder = None
        self.index_folder = None
        self.middle_folder = None
        self.ring_folder = None

        self.trajectory_library = []
        self.trajectory_library_file = "trajectory_library.json"
        self.playing_paused = False

        self.recording_thread = None
        self.playback_speed = 1.0

        # Callback for updating playback progress (0 to 100)
        self.progress_update_callback = None

        # Callback to get the camera frame from the live stream.
        self.get_camera_frame_callback = None

        # Callback for tactile frame acquisition.
        self.get_tactile_frame_callback = None

        # Callbacks for other functionality.
        self.set_hand_pose_callback = None
        self.replay_panel_show_callback = None
        self.pause_traj_button_callback = None
        self.update_replay_camera_callback = None
        self.update_replay_tactile_callback = None

        # Load saved trajectories.
        self.load_trajectory_library()
        self.update_traj_listbox()

    def load_trajectory_library(self):
        if os.path.exists(self.trajectory_library_file):
            try:
                with open(self.trajectory_library_file, "r") as f:
                    self.trajectory_library = json.load(f)
                # Keep only trajectories whose folders exist.
                self.trajectory_library = [traj for traj in self.trajectory_library
                                           if os.path.exists(traj.get("folder", ""))]
                print("Loaded trajectory library from disk.")
            except Exception as e:
                print("Error loading trajectory library:", e)
                self.trajectory_library = []
        else:
            self.trajectory_library = []

    def save_trajectory_library(self):
        try:
            with open(self.trajectory_library_file, "w") as f:
                json.dump(self.trajectory_library, f, indent=4)
            print("Trajectory library saved to disk.")
        except Exception as e:
            print("Error saving trajectory library:", e)

    def update_traj_listbox(self):
        self.traj_listbox.delete(0, tk.END)
        for traj in self.trajectory_library:
            self.traj_listbox.insert(tk.END, traj["name"])

    def delete_trajectory(self, selected_index):
        if selected_index is not None:
            try:
                del self.trajectory_library[selected_index]
                self.update_traj_listbox()
                self.save_trajectory_library()
                print("Trajectory deleted.")
            except Exception as e:
                print("Error deleting trajectory:", e)

    def start_recording(self):
        if self.recording:
            return
        temp_name = "traj_" + time.strftime("%Y%m%d_%H%M%S")
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        self.current_traj_folder = os.path.join(datasets_dir, temp_name)
        os.makedirs(self.current_traj_folder, exist_ok=True)
        self.camera_folder = os.path.join(self.current_traj_folder, "camera")
        self.thumb_folder = os.path.join(self.current_traj_folder, "thumb")
        self.index_folder = os.path.join(self.current_traj_folder, "index")
        self.middle_folder = os.path.join(self.current_traj_folder, "middle")
        self.ring_folder = os.path.join(self.current_traj_folder, "ring")
        for folder in [self.camera_folder, self.thumb_folder, self.index_folder,
                       self.middle_folder, self.ring_folder]:
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {os.path.abspath(folder)}")
        self.trajectory = []
        self.record_start_time = time.time()
        self.recording = True
        self.start_rec_button.config(state=tk.DISABLED)
        self.stop_rec_button.config(state=tk.NORMAL)
        print(f"Started recording trajectory in folder: {os.path.abspath(self.current_traj_folder)}")
        self.recording_thread = threading.Thread(target=self.record_trajectory, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        if not self.recording:
            return
        # Signal to stop further data recording
        self.recording = False
        # Immediately update the UI buttons
        self.start_rec_button.config(state=tk.NORMAL)
        self.stop_rec_button.config(state=tk.DISABLED)
        print("Recording stopped; waiting for data capture to finish...")

        # Poll until the recording thread has finished, then prompt for a name.
        def wait_for_thread():
            if self.recording_thread.is_alive():
                self.master.after(50, wait_for_thread)
            else:
                # Now that no more data is being recorded, prompt for a trajectory name.
                self.post_recording_process()

        wait_for_thread()

    def post_recording_process(self):
        print("Recording thread finished. Processing recorded data.")
        if not self.trajectory:
            try:
                shutil.rmtree(self.current_traj_folder)
                print(f"Trajectory discarded and folder {os.path.abspath(self.current_traj_folder)} deleted.")
            except Exception as e:
                print("Error deleting temporary folder:", e)
            return
        final_name = simpledialog.askstring("Save Trajectory", "Enter a name for this trajectory:")
        if final_name:
            new_folder = os.path.join(os.path.dirname(self.current_traj_folder), final_name)
            base_new_folder = new_folder
            count = 1
            while os.path.exists(new_folder):
                new_folder = f"{base_new_folder}_{count}"
                count += 1
            try:
                os.rename(self.current_traj_folder, new_folder)
                self.current_traj_folder = new_folder
                print(f"Renamed trajectory folder to: {os.path.abspath(new_folder)}")
            except Exception as e:
                print(f"Error renaming folder: {e}")
                final_name = os.path.basename(self.current_traj_folder)
            traj_entry = {"name": final_name, "folder": self.current_traj_folder, "trajectory": self.trajectory}
            self.trajectory_library.append(traj_entry)
            self.update_traj_listbox()
            self.save_trajectory_library()
            print(f"Trajectory '{final_name}' saved.")
        else:
            try:
                shutil.rmtree(self.current_traj_folder)
                print(f"Trajectory discarded and folder {os.path.abspath(self.current_traj_folder)} deleted.")
            except Exception as e:
                print(f"Error deleting temporary folder: {e}")

    def record_trajectory(self):
        sample_index = 0
        try:
            while self.recording:
                elapsed = time.time() - self.record_start_time
                config_list = self.leap_node.read_pos().tolist()
                # Get the latest camera frame via the callback.
                if self.get_camera_frame_callback:
                    cam_img = self.get_camera_frame_callback()
                    if cam_img is not None:
                        cam_filename = f"{sample_index:05d}.jpg"
                        cam_path = os.path.join(self.camera_folder, cam_filename)
                        cv2.imwrite(cam_path, cam_img)
                    else:
                        cam_filename = ""
                else:
                    cam_filename = ""
                tactile_filenames = {}
                for finger, folder in zip(["Thumb", "Index", "Middle", "Ring"],
                                          [self.thumb_folder, self.index_folder,
                                           self.middle_folder, self.ring_folder]):
                    try:
                        if self.get_tactile_frame_callback:
                            sensor_img = self.get_tactile_frame_callback(finger)
                            tactile_filename = f"{sample_index:05d}.jpg"
                            tactile_path = os.path.join(folder, tactile_filename)
                            cv2.imwrite(tactile_path, sensor_img)
                            tactile_filenames[finger] = tactile_filename
                        else:
                            tactile_filenames[finger] = ""
                    except Exception as e:
                        print(f"Error capturing {finger} sensor image: {e}")
                        tactile_filenames[finger] = ""
                sample_data = {
                    "timestamp": elapsed,
                    "configuration": config_list,
                    "camera_frame": cam_filename,
                    "tactile_frames": tactile_filenames
                }
                self.trajectory.append(sample_data)
                print(f"Recorded sample {sample_index} at {elapsed:.2f} sec")
                sample_index += 1
                time.sleep(0.1)
        except Exception as e:
            print("Error during recording:", e)
        finally:
            json_path = os.path.join(self.current_traj_folder, "trajectory_data.json")
            try:
                with open(json_path, "w") as f:
                    json.dump(self.trajectory, f, indent=4)
                print(f"Saved trajectory JSON: {os.path.abspath(json_path)}")
            except Exception as e:
                print("Error saving trajectory JSON:", e)
            print("Trajectory recording finished.")

    def sleep_with_pause(self, duration):
        remaining = duration
        while remaining > 0:
            if self.playing_paused:
                time.sleep(0.1)
            else:
                dt = min(0.01, remaining)
                time.sleep(dt)
                remaining -= dt

    def play_trajectory(self, selected_index=None):
        if selected_index is None or selected_index < 0 or selected_index >= len(self.trajectory_library):
            print("No valid trajectory selected.")
            return
        traj_entry = self.trajectory_library[selected_index]
        traj_folder = traj_entry["folder"]
        json_path = os.path.join(traj_folder, "trajectory_data.json")
        try:
            with open(json_path, "r") as f:
                traj = json.load(f)
        except Exception as e:
            print("Error loading trajectory JSON:", e)
            return
        if not traj or len(traj) < 2:
            print("Trajectory does not have enough points.")
            return

        if self.replay_panel_show_callback:
            self.replay_panel_show_callback()

        print(f"Playing trajectory '{traj_entry['name']}' at speed factor {self.playback_speed}...")
        # Adjust the timeline: compute adjusted timestamps.
        start_ts = traj[0]["timestamp"]
        for sample in traj:
            sample["adjusted_ts"] = (sample["timestamp"] - start_ts) / self.playback_speed

        start_playback = time.time()
        if self.pause_traj_button_callback:
            self.pause_traj_button_callback(state="normal", text="Pause")
        total_samples = len(traj)
        for i, sample in enumerate(traj):
            target_time = sample["adjusted_ts"]
            now = time.time() - start_playback
            delay = target_time - now
            # Enforce a minimum delay (e.g., 50 ms) even if playback speed is high.
            delay = max(delay, 0.05)
            if delay > 0:
                self.sleep_with_pause(delay)
            config = sample["configuration"]
            if self.set_hand_pose_callback:
                self.set_hand_pose_callback(config)
            cam_filename = sample["camera_frame"]
            if cam_filename:
                cam_path = os.path.join(traj_folder, "camera", cam_filename)
                if os.path.exists(cam_path) and self.update_replay_camera_callback:
                    self.update_replay_camera_callback(cam_path)
            tactile_frames = sample.get("tactile_frames", {})
            for finger, fname in tactile_frames.items():
                if fname:
                    tactile_path = os.path.join(traj_folder, finger.lower(), fname)
                    if os.path.exists(tactile_path) and self.update_replay_tactile_callback:
                        self.update_replay_tactile_callback(finger, tactile_path)
            if self.progress_update_callback:
                progress_value = int((i + 1) / total_samples * 100)
                self.progress_update_callback(progress_value)
            while self.playing_paused:
                time.sleep(0.1)
        print("Trajectory playback finished.")
        if self.pause_traj_button_callback:
            self.pause_traj_button_callback(state="disabled", text="Pause")

    def toggle_pause(self):
        self.playing_paused = not self.playing_paused
        state_text = "Resume" if self.playing_paused else "Pause"
        if self.pause_traj_button_callback:
            self.pause_traj_button_callback(state="normal", text=state_text)
        print(f"Playback {'paused' if self.playing_paused else 'resumed'}.")
