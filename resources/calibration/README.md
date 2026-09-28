# Eye-on-hand calibration

`eye_on_hand.yaml` is generated locally by `scripts/calibrate_eye_on_hand.py` and
must not be copied from another robot/camera mount. It stores `T_tcp_camera`, a
transform from the RealSense color optical frame to the **currently configured**
UR TCP frame.

The Intel RealSense D435i must be rigidly fixed to the tool. Fix the DFVision
Q12-240-15 checkerboard in the workspace, move the robot manually with the teach
pendant, and capture diverse stationary views. The target has 12x9 printed
squares at 15 mm, so OpenCV uses 11x8 inner corners. Its pattern area is
180x135 mm and its overall plate is 240x200 mm. The script is RTDE receive-only
and never sends a motion command.

Changing the camera mount or the active TCP in the UR controller invalidates the
calibration. Run the calibration again after either change.
