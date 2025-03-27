from python.leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils as lhu
import numpy as np

class LeapNode:
    def __init__(self):
        # PID and current limit parameters.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 550
        # Initialize the hand pose using the allegro conversion.
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.motors = motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        # Set up motor control parameters.
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        # Command the initial pose.
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
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
        # Lower the stiffness for gravity compensation in Teach Mode.
        # Here we reduce the P and D gains by half.
        teach_kP = self.kP // 2   # e.g., half of normal P gain
        teach_kD = self.kD // 2   # e.g., half of normal D gain
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * teach_kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (teach_kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * teach_kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (teach_kD * 0.75), 80, 2)

    def restore_normal_mode(self):
        # Restore the original PID gains.
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)

    def set_velocity(self, velocities):
        """Sets the desired velocities for all motors.

        Args:
             velocities: A numpy array of desired velocities (rad/s) for each motor.
        """
        self.dxl_client.write_desired_vel(self.motors, velocities)


