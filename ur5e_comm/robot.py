from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field
from time import monotonic, sleep
import numpy as np


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError("{} must contain {} finite values".format(name, size))
    return result


@dataclass(frozen=True)
class RobotSnapshot:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    tcp_pose: np.ndarray
    tcp_speed: np.ndarray
    gripper_position: int
    timestamp: float
    joint_current: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    target_moment: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    robot_mode: int = -1
    safety_mode: int = -1
    protective_stopped: bool = False
    emergency_stopped: bool = False

    def __post_init__(self):
        _vector(self.joint_position, 6, "joint_position")
        _vector(self.joint_velocity, 6, "joint_velocity")
        _vector(self.tcp_pose, 6, "tcp_pose")
        _vector(self.tcp_speed, 6, "tcp_speed")
        _vector(self.joint_current, 6, "joint_current")
        _vector(self.target_moment, 6, "target_moment")
        if not 0 <= int(self.gripper_position) <= 255:
            raise RuntimeError("gripper_position must be in [0, 255]")
        if not np.isfinite(self.timestamp):
            raise RuntimeError("snapshot timestamp must be finite")


class RobotiqSocket:
    """Minimal Robotiq 2F socket client for the URCap server on port 63352."""

    def __init__(self, host, port=63352, timeout=1.0):
        self.socket = socket.create_connection((str(host), int(port)), timeout=float(timeout))
        self.socket.settimeout(float(timeout))

    def _exchange(self, command):
        self.socket.sendall((command.rstrip() + "\n").encode("ascii"))
        payload = self.socket.recv(1024).decode("ascii", errors="replace").strip()
        if not payload:
            raise RuntimeError("Robotiq server returned an empty response to {!r}".format(command))
        return payload

    def get(self, variable):
        response = self._exchange("GET {}".format(variable))
        fields = response.split()
        if len(fields) < 2 or fields[0] != variable:
            raise RuntimeError("Unexpected Robotiq response: {!r}".format(response))
        return int(fields[1])

    def set(self, **variables):
        command = "SET " + " ".join("{} {}".format(key, int(value)) for key, value in variables.items())
        response = self._exchange(command)
        if response != "ack":
            raise RuntimeError("Robotiq command was not acknowledged: {!r}".format(response))

    def activate(self, timeout=10.0):
        if self.get("STA") == 3:
            return
        self.set(ACT=0)
        sleep(0.2)
        self.set(ACT=1, GTO=1, SPE=255, FOR=100)
        deadline = monotonic() + float(timeout)
        while monotonic() < deadline:
            if self.get("STA") == 3:
                return
            sleep(0.1)
        raise TimeoutError("Robotiq activation timed out")

    def move(self, position, speed=255, force=100):
        position = int(np.clip(round(position), 0, 255))
        self.set(POS=position, SPE=int(np.clip(speed, 0, 255)), FOR=int(np.clip(force, 0, 255)), GTO=1)

    def stop(self):
        self.set(GTO=0)

    def close(self):
        try:
            self.socket.close()
        except OSError:
            pass


class UR5eHardware:
    """Synchronous RTDE arm plus Robotiq gripper interface."""

    def __init__(
        self,
        host,
        gripper_port=63352,
        gripper_timeout=1.0,
        activate_gripper=True,
        servo_speed=0.5,
        servo_acceleration=1.0,
        servo_period=0.002,
        servo_lookahead=0.1,
        servo_gain=300,
        gripper_speed=255,
        gripper_force=100,
        allow_motion=False,
    ):
        try:
            import rtde_receive
        except ImportError as error:
            raise RuntimeError("Install ur-rtde to connect to the UR5e") from error
        self.receive = rtde_receive.RTDEReceiveInterface(str(host))
        self.allow_motion = bool(allow_motion)
        self.control = None
        if self.allow_motion:
            try:
                import rtde_control
            except ImportError as error:
                raise RuntimeError("Install ur-rtde control support to command the UR5e") from error
            self.control = rtde_control.RTDEControlInterface(str(host))
        self.gripper = RobotiqSocket(host, gripper_port, gripper_timeout)
        if activate_gripper and self.allow_motion:
            self.gripper.activate()
        self.servo_speed = float(servo_speed)
        self.servo_acceleration = float(servo_acceleration)
        self.servo_period = float(servo_period)
        self.servo_lookahead = float(servo_lookahead)
        self.servo_gain = int(servo_gain)
        self.gripper_speed = int(gripper_speed)
        self.gripper_force = int(gripper_force)
        self.closed = False
        self._target_lock = threading.Lock()
        self._servo_stop_event = threading.Event()
        self._servo_error = None
        self._motion_active = False
        self._last_gripper_target = None
        self._last_gripper_command_time = -np.inf
        self._arm_target = None
        self._servo_thread = None

    def start_motion(self):
        if not self.allow_motion or self.control is None:
            raise RuntimeError("Hardware interface is read-only; motion was not armed")
        if self._motion_active:
            return
        self._servo_stop_event = threading.Event()
        self._servo_error = None
        self._last_gripper_target = None
        self._last_gripper_command_time = -np.inf
        self._arm_target = _vector(self.receive.getActualQ(), 6, "initial actual q")
        self._motion_active = True
        self._servo_thread = threading.Thread(
            target=self._servo_loop,
            name="ur5e-servoj",
            daemon=True,
        )
        self._servo_thread.start()

    def _servo_loop(self):
        try:
            while not self._servo_stop_event.is_set():
                started = self.control.initPeriod() if hasattr(self.control, "initPeriod") else None
                with self._target_lock:
                    target = self._arm_target.copy()
                ok = self.control.servoJ(
                    target.tolist(),
                    self.servo_speed,
                    self.servo_acceleration,
                    self.servo_period,
                    self.servo_lookahead,
                    self.servo_gain,
                )
                if ok is False:
                    raise RuntimeError("UR RTDE servoJ rejected the target")
                if started is not None and hasattr(self.control, "waitPeriod"):
                    self.control.waitPeriod(started)
        except Exception as error:
            self._servo_error = error
            self._servo_stop_event.set()

    def _raise_servo_error(self):
        if self._servo_error is not None:
            raise RuntimeError("UR servo loop failed: {}".format(self._servo_error))

    def _optional(self, name, default):
        method = getattr(self.receive, name, None)
        return default if method is None else method()

    def read(self) -> RobotSnapshot:
        self._raise_servo_error()
        return RobotSnapshot(
            joint_position=_vector(self.receive.getActualQ(), 6, "actual q"),
            joint_velocity=_vector(self.receive.getActualQd(), 6, "actual qd"),
            tcp_pose=_vector(self.receive.getActualTCPPose(), 6, "tcp pose"),
            tcp_speed=_vector(self.receive.getActualTCPSpeed(), 6, "tcp speed"),
            gripper_position=self.gripper.get("POS"),
            joint_current=_vector(self._optional("getActualCurrent", np.zeros(6)), 6, "joint current"),
            target_moment=_vector(self._optional("getTargetMoment", np.zeros(6)), 6, "target moment"),
            robot_mode=int(self._optional("getRobotMode", -1)),
            safety_mode=int(self._optional("getSafetyMode", -1)),
            protective_stopped=bool(self._optional("isProtectiveStopped", False)),
            emergency_stopped=bool(self._optional("isEmergencyStopped", False)),
            timestamp=monotonic(),
        )

    def command(self, arm_target, gripper_position):
        if not self.allow_motion or self.control is None:
            raise RuntimeError("Hardware interface is read-only; motion was not armed")
        if not self._motion_active:
            raise RuntimeError("Motion interface has not been started for this episode")
        self._raise_servo_error()
        target = _vector(arm_target, 6, "arm target")
        with self._target_lock:
            self._arm_target = target.copy()
        now = monotonic()
        # The Robotiq URCap socket is much slower than the 500 Hz arm servo.
        # Send the newest absolute target at at most 20 Hz and ignore 1-byte noise.
        if (
            self._last_gripper_target is None
            or abs(int(gripper_position) - self._last_gripper_target) >= 2
        ) and now - self._last_gripper_command_time >= 0.05:
            self.gripper.move(gripper_position, self.gripper_speed, self.gripper_force)
            self._last_gripper_target = int(gripper_position)
            self._last_gripper_command_time = now

    def stop_motion(self):
        if not self.allow_motion or self.control is None:
            return
        if not self._motion_active:
            return
        self._motion_active = False
        self._servo_stop_event.set()
        errors = []
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=max(0.05, 5.0 * self.servo_period))
        try:
            self.control.servoStop()
        except Exception as error:  # safety cleanup must attempt both devices
            errors.append(error)
        try:
            self.gripper.stop()
        except Exception as error:
            errors.append(error)
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=2.0)
            if self._servo_thread.is_alive():
                errors.append(TimeoutError("UR servo thread did not terminate"))
        if errors:
            raise RuntimeError("Failed to stop all devices: {}".format(errors))

    def close(self):
        if self.closed:
            return
        try:
            self.stop_motion()
        finally:
            try:
                if self.control is not None:
                    self.control.stopScript()
            finally:
                self.gripper.close()
                disconnect = getattr(self.receive, "disconnect", None)
                if disconnect is not None:
                    disconnect()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
