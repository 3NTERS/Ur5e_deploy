# Eye-on-base calibration

`eye_on_base.yaml` is generated locally by `scripts/calibrate_eye_on_base.py` and
must not be copied from another cell. It stores `T_base_camera`, i.e. a transform
that maps a point expressed in the optical camera frame (metres) to the UR base
frame.

Before calibration, measure the printed QR centre pose in the UR base frame and
replace `calibration.base_to_qr` in `resources/config/ur5e_deploy.yaml`. Keep the
QR fixed and fully visible for all collected samples.
