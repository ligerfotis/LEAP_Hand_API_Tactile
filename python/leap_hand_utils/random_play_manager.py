import os
import time
import json
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
from random_play_utils import generate_random_middle_pose


class RandomPlayManager:
    def __init__(self, leap_node, sensors, stream_width_var, stream_height_var):
        """
        Manages random play data collection.

        Args:
            leap_node: Object for hand control.
            sensors (dict): DIGIT tactile sensor objects.
            stream_width_var, stream_height_var: Tk variables for camera stream dimensions.
        """
        self.leap_node = leap_node
        self.sensors = sensors
        self.stream_width_var = stream_width_var
        self.stream_height_var = stream_height_var

        self.random_trajectory = []
        self.random_record_start_time = None
        self.random_traj_folder = None
        self.random_play_name = ""
        self.random_play_active = False

    def start_random_play_collection(self, name):
        if not name:
            raise ValueError("Name cannot be empty")
        self.random_play_name = name
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        new_folder = os.path.join(datasets_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(new_folder, exist_ok=True)
        self.random_traj_folder = new_folder
        # Create subfolders.
        self.camera_folder = os.path.join(new_folder, "camera")
        self.thumb_folder = os.path.join(new_folder, "thumb")
        self.index_folder = os.path.join(new_folder, "index")
        self.middle_folder = os.path.join(new_folder, "middle")
        self.ring_folder = os.path.join(new_folder, "ring")
        for folder in [self.camera_folder, self.thumb_folder, self.index_folder,
                       self.middle_folder, self.ring_folder]:
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {os.path.abspath(folder)}")
        self.random_trajectory = []
        self.random_record_start_time = time.time()
        self.random_play_active = True
        threading.Thread(target=self.random_play_mode, daemon=True).start()

    def stop_random_play(self):
        self.random_play_active = False

    def check_current_limit(self, threshold=600):
        try:
            currents = self.leap_node.read_cur()  # Should return an array of current measurements.
            if np.any(np.array(currents) > threshold):
                print("Force/Torque limit exceeded! Currents:", currents)
                return True
        except Exception as e:
            print("Error reading current:", e)
        return False

    def random_play_mode(self):
        pipeline = rs.pipeline()
        config = rs.config()
        width = self.stream_width_var.get()
        height = self.stream_height_var.get()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        pipeline.start(config)
        sample_index = 0
        try:
            while self.random_play_active:
                if self.check_current_limit(threshold=600):
                    print("Current above threshold. Stopping random play mode.")
                    self.stop_random_play()
                    break
                elapsed = time.time() - self.random_record_start_time
                current_pose = self.leap_node.curr_pos.copy()
                # For full-range motion, here we set bounds from 0 to π for joints 4–7.
                target_pose = generate_random_middle_pose(current_pose,
                                                          safe_bounds={4: (0, np.pi), 5: (0, np.pi), 6: (0, np.pi),
                                                                       7: (0, np.pi)})
                # Interpolate to the target pose smoothly.
                steps = 20
                delay = 50
                for i in range(1, steps + 1):
                    interp_pose = current_pose + (target_pose - current_pose) * (i / steps)
                    self.leap_node.set_leap(interp_pose)
                    time.sleep(delay / 1000.0)
                config_list = self.leap_node.curr_pos.tolist()
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    cam_img = np.asanyarray(color_frame.get_data())
                    cam_filename = f"{sample_index:05d}.jpg"
                    cam_path = os.path.join(self.camera_folder, cam_filename)
                    cv2.imwrite(cam_path, cam_img)
                else:
                    cam_filename = ""
                tactile_filenames = {}
                for finger, folder in zip(["Thumb", "Index", "Middle", "Ring"],
                                          [self.thumb_folder, self.index_folder, self.middle_folder, self.ring_folder]):
                    try:
                        sensor_img = self.sensors[finger].get_frame()
                        tactile_filename = f"{sample_index:05d}.jpg"
                        tactile_path = os.path.join(folder, tactile_filename)
                        cv2.imwrite(tactile_path, sensor_img)
                        tactile_filenames[finger] = tactile_filename
                    except Exception as e:
                        print(f"Error capturing {finger} sensor image: {e}")
                        tactile_filenames[finger] = ""
                sample_data = {
                    "timestamp": elapsed,
                    "configuration": config_list,
                    "camera_frame": cam_filename,
                    "tactile_frames": tactile_filenames
                }
                self.random_trajectory.append(sample_data)
                print(f"Recorded random sample {sample_index} at {elapsed:.2f} sec")
                sample_index += 1
                time.sleep(0.1)
        finally:
            pipeline.stop()
            json_path = os.path.join(self.random_traj_folder, "trajectory_data.json")
            with open(json_path, "w") as f:
                json.dump(self.random_trajectory, f, indent=4)
            print("Random play data collection finished.")
