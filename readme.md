# LEAP Hand API with GUI for Trajectory Collection and Teleoperation

This fork of the original LEAP Hand API integrates a new graphical user interface (GUI) that enables users to both teleoperate the LEAP Hand in real time and capture hand trajectories for later analysis or playback. This added functionality makes it easier to experiment, calibrate, and record various hand movements.

## Features

- **Trajectory Recording:**  
  Capture hand trajectories with timestamps, sensor data, and current readings. Recorded data is organized by folder and saved as JSON alongside image sequences.

- **Teleoperation:**  
  Command the hand in real time using slider controls and other GUI elements, allowing smooth transitions and precise adjustments.

- **Finger Teach Mode:**  
  Engage a special mode for individual fingers that allows fine-tuned calibration and “teach mode” capture.

- **Live Streams:**  
  Integrated support for the RealSense camera and DIGIT tactile sensors. The GUI displays live video and tactile sensor streams for immediate visual feedback.

- **Configuration Management:**  
  Save, load, rename, and delete hand configurations from within the GUI for repeated use or rapid adjustments.

- **Underlying Robust Control:**  
  Maintains all the standard functionalities of the original LEAP Hand API (PID control, current/velocity commands, etc.) while introducing new, higher level features via the GUI.

## Software Setup

### Conda Environment

It is recommended to use a Conda environment to manage dependencies. Create and activate a new environment with Python 3.10 as follows:

```bash
conda create -n LEAP_Hand_API_Tactile python=3.10
conda activate LEAP_Hand_API_Tactile
```

Then install the required dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` covers the following primary libraries:
- `dynamixel_sdk` – for communicating with the Dynamixel motors.
- `numpy` – for array operations and numerical computations.
- `Pillow` – for image handling in the GUI.
- `opencv-python` – for image processing and frame capture.
- `pyrealsense2` – for supporting the RealSense camera (if used).
- Optionally, if using DIGIT sensors:  
  `digit-interface @ git+https://github.com/facebookresearch/digit-interface.git`

### Building (Optional)

For the C++ components (if you plan on compiling or modifying the SDK), navigate to the `cpp` folder and follow the instructions in the original C++ README file. However, the primary focus of this fork is the Python API with the new GUI.

## Running the GUI

To start the GUI for teleoperating the hand and collecting trajectories, simply run the `leap_hand_gui_new.py` file. For example:

```bash
python python/leap_hand_utils/leap_hand_gui_new.py
```

The GUI window provides:
- **Real-Time Control:** Use sliders to adjust individual joint angles.
- **Recording Controls:** Start/stop trajectory recording to capture both sensor data and images.
- **Configuration Management:** Save and load custom hand configurations.
- **Sensor Displays:** Live updates from the camera and tactile sensors.

## Hardware Setup

- **Power:** Connect a 5V power source to the hand. You should see the Dynamixel motors light up on boot.
- **Connection:** Attach the Micro USB cable (avoid excessive USB extensions) and ensure the correct serial port is specified in the code (e.g., `/dev/ttyUSB0` or `COM13`).
- **Additional Configuration:**  
  Use tools such as Dynamixel Wizard to verify motor settings. Make sure no other process (like Dynamixel Wizard) is occupying the USB port when running the API.

## Additional Information

- **Documentation:** For further details on the original API features, check the [original documentation](http://leaphand.com/) and the [Dynamixel SDK manual](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/).
- **Troubleshooting:**  
  - Verify serial port permissions if motors are not detected.
  - Adjust PID settings as needed based on your specific hand setup.
  - Consult the original README sections on hardware setup and troubleshooting if issues occur.

## License

This project is distributed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Support and Contributions

For support, suggestions, or contributions, please contact the project maintainer or submit a pull request. Contributions that help enhance the GUI, sensor integration, or trajectory analysis are welcome.