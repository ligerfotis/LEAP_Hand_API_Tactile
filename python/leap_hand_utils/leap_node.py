import threading
import time
from dynamixel_client import DynamixelClient
import leap_hand_utils as lhu
import numpy as np

class LeapNode:
    def __init__(self):
        self.kP = 1500
        self.kI = 0
        self.kD = 200
        self.curr_lim = 550
        self.prev_pos = self.pos = self.curr_pos = None  # Will be initialized later

        self.motors = list(range(16))
        self.write_lock = threading.Lock()

        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(self.motors, 'COM13', 4000000)
                self.dxl_client.connect()

        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        # self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        # self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD), 80, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

        self.control_loop_running = False

    def initialize_current_pose_from_motors(self):
        pos = self.read_pos()
        self.prev_pos = self.pos = self.curr_pos = np.array(pos)

    def start_control_loop(self):
        if not self.control_loop_running:
            self.control_loop_running = True
            threading.Thread(target=self._control_loop, daemon=True).start()

    def pause_control_loop(self):
        self.control_loop_running = False

    def resume_control_loop(self):
        if not self.control_loop_running:
            self.start_control_loop()

    def _control_loop(self):
        while self.control_loop_running:
            with self.write_lock:
                if self.curr_pos is not None:
                    self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
            time.sleep(0.05)

    def set_leap(self, pose, override_safety=False):
        safe_pose = pose if override_safety else lhu.angle_safety_clip(pose)
        with self.write_lock:
            self.prev_pos = self.curr_pos
            self.curr_pos = np.array(safe_pose)
            self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        with self.write_lock:
            self.prev_pos = self.curr_pos
            self.curr_pos = np.array(pose)
            self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        with self.write_lock:
            self.prev_pos = self.curr_pos
            self.curr_pos = np.array(pose)
            self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_vel(self):
        return self.dxl_client.read_vel()

    def read_cur(self):
        return self.dxl_client.read_cur()

    def pos_vel(self):
        return self.dxl_client.read_pos_vel()

    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()

    def set_compliant_mode(self):
        teach_kP = self.kP // 2
        teach_kD = self.kD // 2
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * teach_kP, 84, 2)
        # self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (teach_kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * teach_kD, 80, 2)
        # self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (teach_kD * 0.75), 80, 2)

    def restore_normal_mode(self):
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        # self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        # self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
