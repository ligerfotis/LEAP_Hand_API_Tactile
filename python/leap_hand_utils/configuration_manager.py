import os
import json
import tkinter.simpledialog as simpledialog
import tkinter.messagebox as messagebox
import numpy as np

class ConfigurationManager:
    def __init__(self, listbox, sliders, leap_node, move_callback):
        self.config_listbox = listbox
        self.sliders = sliders
        self.leap_node = leap_node
        self.move_callback = move_callback  # New: a callback to smoothly move the robot
        self.configurations = []
        self.load_configurations()
        self.update_config_listbox()

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
        self.config_listbox.delete(0, "end")
        for config in self.configurations:
            self.config_listbox.insert("end", config["name"])

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
            # Instead of applying the pose instantly, use the move_callback for a smooth transition.
            self.move_callback(target_pose)
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
