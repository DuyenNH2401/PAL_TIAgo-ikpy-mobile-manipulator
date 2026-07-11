# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Manipulation BT nodes — object recognition, arm control, grasping, lifting, placing."""

import math
import numpy as np
import py_trees

from robot.base import STARTING_POSITION, LIFT_POSITION, PLACE_POSITION, ANGLE_TOLERANCE


def _camera_to_world(robot, cam_pos):
    """Convert camera-frame position to world frame using GPS + compass.

    Input:
        robot   : TiagoFull instance with gps, compass, and sensors attributes.
        cam_pos (list): [x, y, z] position reported by camera recognition (camera frame).
    Output:
        list: [world_x, world_y, world_z] position in world frame (metres).
    """
    gps_val      = robot.gps.getValues()
    compass_val  = robot.compass.getValues()
    robot_angle  = np.arctan2(compass_val[0], compass_val[1])
    cos_t, sin_t = np.cos(robot_angle), np.sin(robot_angle)

    torso_h = robot.sensors["torso_lift_joint"].getValue() if "torso_lift_joint" in robot.sensors else 0.0
    cam_h   = gps_val[2] + 0.891 + torso_h
    cam_fwd = 0.25

    world_x = gps_val[0] + cos_t * (cam_pos[0] + cam_fwd) + (-sin_t) * cam_pos[1]
    world_y = gps_val[1] + sin_t * (cam_pos[0] + cam_fwd) +   cos_t  * cam_pos[1]

    ref_torso  = 0.2
    z_correct  = -1.87 * (torso_h - ref_torso) if torso_h > ref_torso else 0.0
    world_z    = cam_h + cam_pos[2] + z_correct

    return [world_x, world_y, world_z]


def _approach_offsets(robot_pos_2d, target_pos_2d):
    """Return (offset_x, offset_y) vector pointing from the target back toward the robot.

    Input:
        robot_pos_2d  (list): [x, y] robot position in world frame.
        target_pos_2d (list): [x, y] target object position in world frame.
    Output:
        tuple:
            offset_x (float): X component of the approach offset (metres).
            offset_y (float): Y component of the approach offset (metres).
    """
    dx = target_pos_2d[0] - robot_pos_2d[0]
    dy = target_pos_2d[1] - robot_pos_2d[1]
    approach_angle = math.atan2(dy, dx) + math.pi
    mag = 0.045
    return mag * math.cos(approach_angle), mag * math.sin(approach_angle)


def _solve_ik(robot, target_pos, offset_x=0.0, offset_y=0.0):
    """Solve inverse kinematics for target_pos plus optional offsets.

    Input:
        robot      : TiagoFull instance with ik_chain, sensors, and joint_limits.
        target_pos (list): [x, y, z] target end-effector position in world frame.
        offset_x   (float): Additional x offset added to the IK target.
        offset_y   (float): Additional y offset added to the IK target.
    Output:
        dict or None: Mapping of {joint_name: angle_rad} if IK converged,
                      or None if the solver raised a ValueError.
    """
    final = [target_pos[0] + offset_x, target_pos[1] + offset_y, target_pos[2]]
    initial = [
        robot.sensors[link.name].getValue() if link.name in robot.sensors else 0.0
        for link in robot.ik_chain.links
    ]
    for i, link in enumerate(robot.ik_chain.links):
        if link.name in robot.joint_limits:
            lo = robot.joint_limits[link.name]["lower"]
            hi = robot.joint_limits[link.name]["upper"]
            initial[i] = float(np.clip(initial[i], lo, hi))
    try:
        result = robot.ik_chain.inverse_kinematics(
            target_position=final,
            initial_position=initial,
            target_orientation=[0, 0, 1],
            orientation_mode="Y",
        )
        from robot.base import PART_NAMES
        return {
            link.name: result[i]
            for i, link in enumerate(robot.ik_chain.links)
            if link.name in PART_NAMES
        }
    except ValueError as e:
        print(f"IK error: {e}")
        return None


class CheckHardware(py_trees.behaviour.Behaviour):
    """Verify all critical hardware components are present.

    Checks that required motors, sensors, camera, GPS, and compass are available
    on the robot. Returns SUCCESS if all components are found, FAILURE otherwise.
    """

    _REQUIRED_MOTORS  = ["torso_lift_joint", "arm_1_joint", "gripper_left_finger_joint"]
    _REQUIRED_SENSORS = ["torso_lift_joint", "arm_1_joint", "gripper_left_finger_joint"]

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.blackboard = blackboard

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        for m in self._REQUIRED_MOTORS:
            if m not in robot.motors:
                print(f"CheckHardware: motor '{m}' missing")
                return py_trees.common.Status.FAILURE
        for s in self._REQUIRED_SENSORS:
            if s not in robot.sensors:
                print(f"CheckHardware: sensor '{s}' missing")
                return py_trees.common.Status.FAILURE
        if not robot.camera or not robot.gps or not robot.compass:
            print("CheckHardware: camera/GPS/compass missing")
            return py_trees.common.Status.FAILURE

        print("CheckHardware: all OK")
        return py_trees.common.Status.SUCCESS


class MoveToPosition(py_trees.behaviour.Behaviour):
    """Command a set of joints to target positions and wait until reached.

    Args:
        joint_targets: dict {joint_name: target_rad}
        tolerance:     acceptable position error (rad)
        timeout:       max seconds before forcing SUCCESS
    """

    def __init__(self, name, blackboard, joint_targets, tolerance=0.02, timeout=10.0):
        super().__init__(name)
        self.blackboard    = blackboard
        self.joint_targets = joint_targets
        self.tolerance     = tolerance
        self.timeout       = timeout
        self._start_t      = None
        self._prog_t       = None
        self._last_errors  = {}
        self._done         = False

    def initialise(self):
        self._done = False
        robot = self.blackboard.read("robot")
        if robot is None:
            return
        self._start_t = robot.robot.getTime()
        self._prog_t  = self._start_t
        self._last_errors = {}
        for joint, target in self.joint_targets.items():
            if joint in robot.motors:
                robot.motors[joint].setPosition(target)
            if joint in robot.sensors:
                self._last_errors[joint] = abs(target - robot.sensors[joint].getValue())

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        now = robot.robot.getTime()
        if now - self._start_t > self.timeout:
            print(f"{self.name}: timeout → SUCCESS")
            self._done = True
            return py_trees.common.Status.SUCCESS

        all_done   = True
        progressed = False
        for joint, target in self.joint_targets.items():
            if joint not in robot.sensors:
                continue
            current = robot.sensors[joint].getValue()
            error   = abs(target - current)
            if joint in self._last_errors:
                if abs(self._last_errors[joint] - error) > 0.005:
                    progressed = True
            self._last_errors[joint] = error
            if error > self.tolerance:
                robot.motors[joint].setPosition(target)
                all_done = False

        if progressed:
            self._prog_t = now
        elif now - self._prog_t > 3.0:
            print(f"{self.name}: no progress → SUCCESS")
            self._done = True
            return py_trees.common.Status.SUCCESS

        if all_done:
            print(f"{self.name}: completed")
            self._done = True
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class ObjectRecognizer(py_trees.behaviour.Behaviour):
    """Detect objects via camera recognition and store position on blackboard.

    Args:
        z_offset: vertical correction applied after coordinate conversion
        samples:  number of recognition samples averaged per tick
        timeout:  seconds before FAILURE
    """

    _TARGET_MODELS = {"jam jar", "honey jar"}

    def __init__(self, name, blackboard, z_offset=0.0, samples=5, timeout=3.0):
        super().__init__(name)
        self.blackboard = blackboard
        self.z_offset   = z_offset
        self.samples    = samples
        self.timeout    = timeout
        self._start_t   = None

    def initialise(self):
        self._start_t = None

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        if self._start_t is None:
            self._start_t = robot.robot.getTime()

        if robot.robot.getTime() - self._start_t > self.timeout:
            return py_trees.common.Status.FAILURE

        positions = []
        n = 1 if "After Scan" in self.name else self.samples
        for _ in range(n):
            for obj in robot.camera.getRecognitionObjects():
                try:
                    if obj.getModel() not in self._TARGET_MODELS:
                        continue
                    wp = _camera_to_world(robot, list(obj.getPosition()))
                    if not (0.0 <= wp[2] <= 2.0):
                        continue
                    positions.append((wp, obj.getModel()))
                except Exception as e:
                    print(f"{self.name}: recognition error: {e}")

        if not positions:
            return py_trees.common.Status.RUNNING

        # Pick closest to robot
        rx, ry = robot.xw, robot.yw
        positions.sort(key=lambda item: (item[0][0]-rx)**2 + (item[0][1]-ry)**2)
        pos, model = positions[0]

        self.blackboard.write("target_position", pos)
        self.blackboard.write("object_name", model)
        print(f"{self.name}: found '{model}' at {pos}")
        return py_trees.common.Status.SUCCESS


class ComprehensiveScanner(py_trees.behaviour.Behaviour):
    """Rotate robot through N angles, trying object recognition at each stop.

    Rotates in increments of angle_increment degrees, pausing at each stop
    to attempt object recognition. Returns SUCCESS as soon as a target is found,
    or SUCCESS after all angles have been scanned.
    """

    def __init__(self, name, blackboard, total_angles=8, angle_increment=45,
                 rotation_speed=1.0):
        super().__init__(name)
        self.blackboard     = blackboard
        self.total_angles   = total_angles
        self.rotation_speed = rotation_speed
        self.rot_duration   = abs(math.radians(angle_increment) / rotation_speed)

        self._angle_idx    = 0
        self._start_t      = None
        self._rot_complete = False

    def initialise(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return
        self._angle_idx    = 0
        self._start_t      = robot.robot.getTime()
        self._rot_complete = False

        robot.motors["torso_lift_joint"].setPosition(0.35)
        robot.motors["head_1_joint"].setPosition(0.0)
        robot.motors["head_2_joint"].setPosition(-0.2)

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        now     = robot.robot.getTime()
        elapsed = now - self._start_t

        if self._rot_complete:
            if elapsed > self.rot_duration + 0.3:
                if self._angle_idx >= self.total_angles - 1:
                    return py_trees.common.Status.SUCCESS
                self._angle_idx   += 1
                self._start_t      = now
                self._rot_complete = False
            else:
                robot.set_wheel_velocity(0.0, 0.0)
            return py_trees.common.Status.RUNNING

        if elapsed >= self.rot_duration:
            robot.set_wheel_velocity(0.0, 0.0)
            self._rot_complete = True

            # Quick recognition attempt
            rec = ObjectRecognizer(f"Scan@{self._angle_idx}", self.blackboard,
                                   timeout=2.0)
            rec.initialise()
            if rec.update() == py_trees.common.Status.SUCCESS:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

        robot.set_wheel_velocity(self.rotation_speed, -self.rotation_speed)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        robot = self.blackboard.read("robot")
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)


class MoveArmIK(py_trees.behaviour.Behaviour):
    """Compute IK for target_position from blackboard and drive arm joints.

    Args:
        offset_x, offset_y: extra safety margin added to IK target
        tolerance:          joint angle tolerance (rad)
        timeout:            max seconds before forcing SUCCESS
    """

    _PRE_GRASP = {
        "torso_lift_joint": 0.3,
        "arm_1_joint":      0.7,
        "arm_2_joint":      0.4,
        "arm_3_joint":     -1.5,
        "arm_4_joint":      1.7,
        "arm_5_joint":     -1.5,
        "arm_6_joint":      0.0,
        "arm_7_joint":      0.0,
    }

    def __init__(self, name, blackboard, offset_x=0.0, offset_y=0.0,
                 tolerance=0.015, timeout=5.0):
        super().__init__(name)
        self.blackboard  = blackboard
        self.offset_x    = offset_x
        self.offset_y    = offset_y
        self.tolerance   = tolerance
        self.timeout     = timeout
        self._started    = False
        self._done       = False
        self._targets    = None
        self._start_t    = None

    def initialise(self):
        self._started = False
        self._done    = False
        self._targets = None
        self._start_t = None

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        if not self._started:
            self._start_t = robot.robot.getTime()
            target = self.blackboard.read("target_position")
            if not target:
                print(f"{self.name}: no target_position on blackboard")
                return py_trees.common.Status.FAILURE

            # Pre-grasp posture
            for joint, pos in self._PRE_GRASP.items():
                if joint in robot.motors:
                    robot.motors[joint].setPosition(pos)
            robot.robot.step(robot.timestep * 5)

            ox, oy = _approach_offsets([robot.xw, robot.yw], target[:2])
            self._targets = _solve_ik(robot, target,
                                      ox + self.offset_x,
                                      oy + self.offset_y)
            if not self._targets:
                return py_trees.common.Status.FAILURE

            for joint, angle in self._targets.items():
                if joint in robot.motors:
                    robot.motors[joint].setPosition(angle)

            self._started = True
            return py_trees.common.Status.RUNNING

        now = robot.robot.getTime()
        if now - self._start_t > self.timeout:
            self._done = True
            return py_trees.common.Status.SUCCESS

        for joint, target_angle in self._targets.items():
            if joint in robot.sensors:
                if abs(target_angle - robot.sensors[joint].getValue()) > self.tolerance:
                    return py_trees.common.Status.RUNNING

        self._done = True
        return py_trees.common.Status.SUCCESS


class GraspController(py_trees.behaviour.Behaviour):
    """Close gripper until force threshold is met, then verify grip.

    States: APPROACHING → VERIFYING
    """

    def __init__(self, name, blackboard, force_threshold=-10.0):
        super().__init__(name)
        self.blackboard      = blackboard
        self.force_threshold = force_threshold
        self._state          = "APPROACHING"
        self._grip_w         = 0.045
        self._verify_t       = None

    def initialise(self):
        self._state  = "APPROACHING"
        self._grip_w = 0.045
        self._verify_t = None
        self.blackboard.write("grasp_success", False)

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        lf = robot.motors["gripper_left_finger_joint"].getForceFeedback()
        rf = robot.motors["gripper_right_finger_joint"].getForceFeedback()
        lp = robot.sensors["gripper_left_finger_joint"].getValue()
        rp = robot.sensors["gripper_right_finger_joint"].getValue()
        ft = abs(self.force_threshold)

        if self._state == "APPROACHING":
            self._grip_w = max(0.0, self._grip_w - 0.001)
            robot.motors["gripper_left_finger_joint"].setPosition(self._grip_w)
            robot.motors["gripper_right_finger_joint"].setPosition(self._grip_w)
            if abs(lf) >= ft and abs(rf) >= ft:
                self._state    = "VERIFYING"
                self._verify_t = robot.robot.getTime()

        elif self._state == "VERIFYING":
            w = max(0.0, self._grip_w - 0.001)
            robot.motors["gripper_left_finger_joint"].setPosition(w)
            robot.motors["gripper_right_finger_joint"].setPosition(w)
            if robot.robot.getTime() - self._verify_t >= 0.5:
                if abs(lf) >= ft and abs(rf) >= ft:
                    self.blackboard.write("grasp_success", True)
                    print(f"{self.name}: grasp SUCCESS (L={lf:.2f}, R={rf:.2f})")
                    return py_trees.common.Status.SUCCESS
                self._state = "APPROACHING"

        if lp < 0.005 and rp < 0.005 and abs(lf) < ft:
            print(f"{self.name}: grasp FAILED — fingers closed, no force")
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        robot = self.blackboard.read("robot")
        if robot and new_status == py_trees.common.Status.FAILURE:
            robot.motors["gripper_left_finger_joint"].setPosition(0.045)
            robot.motors["gripper_right_finger_joint"].setPosition(0.045)


class LiftAndVerify(py_trees.behaviour.Behaviour):
    """Move arm to lift position while verifying grip force is maintained.

    Returns FAILURE immediately if the force drops below the threshold,
    indicating the object was dropped.
    """

    def __init__(self, name, blackboard, lift_positions, timeout=2.0, force_threshold=-5.0):
        super().__init__(name)
        self.blackboard      = blackboard
        self.lift_positions  = lift_positions
        self.timeout         = timeout
        self.force_threshold = force_threshold
        self._start_t        = None
        self._started        = False

    def initialise(self):
        self._start_t = None
        self._started = False
        robot = self.blackboard.read("robot")
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        if self._start_t is None:
            self._start_t = robot.robot.getTime()

        lf = robot.motors["gripper_left_finger_joint"].getForceFeedback()
        rf = robot.motors["gripper_right_finger_joint"].getForceFeedback()
        ft = abs(self.force_threshold)

        if abs(lf) < ft and abs(rf) < ft:
            print(f"{self.name}: object dropped!")
            self.blackboard.write("grasp_success", False)
            return py_trees.common.Status.FAILURE

        if not self._started:
            for joint, pos in self.lift_positions.items():
                if joint in robot.motors:
                    robot.motors[joint].setPosition(pos)
            self._started = True

        now = robot.robot.getTime()
        if now - self._start_t > self.timeout:
            if abs(lf) >= ft or abs(rf) >= ft:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING


class BackupAfterGrasp(py_trees.behaviour.Behaviour):
    """Reverse a short distance after grasping to create clearance.

    States: INIT (brief pause + torso lift check) → BACKUP (reverse until
    backup_distance is reached or duration expires).
    """

    def __init__(self, name, blackboard, backup_distance=0.12, duration=3.0):
        super().__init__(name)
        self.blackboard      = blackboard
        self.backup_distance = backup_distance
        self.duration        = duration
        self._start_t        = None
        self._start_pos      = None
        self._state          = "INIT"

    def initialise(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return
        self._start_t   = robot.robot.getTime()
        self._start_pos = [robot.xw, robot.yw]
        self._state     = "INIT"

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        now  = robot.robot.getTime()
        dx   = robot.xw - self._start_pos[0]
        dy   = robot.yw - self._start_pos[1]
        dist = math.sqrt(dx**2 + dy**2)

        if self._state == "INIT":
            if now - self._start_t > 1.0:
                torso_h = robot.sensors["torso_lift_joint"].getValue() if "torso_lift_joint" in robot.sensors else 0.3
                if torso_h < 0.25:
                    robot.motors["torso_lift_joint"].setPosition(0.30)
                    robot.set_wheel_velocity(0.0, 0.0)
                    return py_trees.common.Status.RUNNING
                self._state = "BACKUP"
            else:
                robot.set_wheel_velocity(0.0, 0.0)
            return py_trees.common.Status.RUNNING

        elif self._state == "BACKUP":
            if dist >= self.backup_distance or now - self._start_t > self.duration:
                robot.set_wheel_velocity(0.0, 0.0)
                return py_trees.common.Status.SUCCESS
            robot.set_wheel_velocity(-1.5, -1.5)

        return py_trees.common.Status.RUNNING


class OpenGripper(py_trees.behaviour.Behaviour):
    """Open gripper fingers to release grasped object.

    Sends the open position command in initialise() then waits until fingers
    reach the target or timeout expires.
    """

    def __init__(self, name, blackboard, open_position=0.045, timeout=2.0):
        super().__init__(name)
        self.blackboard    = blackboard
        self.open_position = open_position
        self.timeout       = timeout
        self._start_t      = None
        self._opened       = False

    def initialise(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return
        self._start_t = robot.robot.getTime()
        self._opened  = False
        robot.motors["gripper_left_finger_joint"].setPosition(self.open_position)
        robot.motors["gripper_right_finger_joint"].setPosition(self.open_position)

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        now = robot.robot.getTime()
        if now - self._start_t > self.timeout:
            return py_trees.common.Status.SUCCESS

        lp = robot.sensors["gripper_left_finger_joint"].getValue()
        rp = robot.sensors["gripper_right_finger_joint"].getValue()
        if abs(lp - self.open_position) < 0.005 and abs(rp - self.open_position) < 0.005:
            if not self._opened:
                self._opened  = True
                self._start_t = now - self.timeout + 0.5
        if self._opened and now - self._start_t > 0.5:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
