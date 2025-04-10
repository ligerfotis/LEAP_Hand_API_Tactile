import numpy as np


class FingerTeachManager:
    def __init__(self, master, fingers, leap_node, sliders, update_fn):
        self.master = master
        self.fingers = fingers
        self.leap_node = leap_node
        self.sliders = sliders
        self.update_fn = update_fn
        self.finger_teach_modes = {finger: False for finger in fingers}
        self.button_refs = {}
        self.status_refs = {}

    def register_ui_refs(self, buttons, statuses):
        self.button_refs = buttons
        self.status_refs = statuses

    def toggle_finger_teach_mode(self, finger, button, status_label):
        if not self.finger_teach_modes[finger]:
            self.finger_teach_modes[finger] = True
            button.config(text=f"{finger}: Exit Teach Mode & Capture")
            status_label.config(text="Teach Mode Active")
            self.leap_node.set_compliant_mode()
            self._update_finger_teach_mode(finger)
        else:
            self.finger_teach_modes[finger] = False
            self.leap_node.restore_normal_mode()
            captured_pose = self.leap_node.read_pos()
            captured_pose_deg = np.rad2deg(captured_pose)
            for i in self.fingers[finger]:
                self.sliders[i].set(captured_pose_deg[i])
            button.config(text=f"{finger}: Enter Teach Mode")
            status_label.config(text="Normal")

    def _update_finger_teach_mode(self, finger):
        if self.finger_teach_modes[finger]:
            current_pose = self.leap_node.read_pos()
            new_pose = self.leap_node.curr_pos.copy()
            for i in self.fingers[finger]:
                new_pose[i] = current_pose[i]
            self.leap_node.set_leap(new_pose)
            self.master.after(50, lambda: self._update_finger_teach_mode(finger))
