# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Robot base — hardware initialisation: motors, sensors, IK chain, and wheel control."""

import math
import warnings
import numpy as np
from controller import Supervisor
from ikpy.chain import Chain
import urdf_parser_py.urdf as urdf_model


MAX_MOTOR_SPEED = 6.0
MIN_MOTOR_SPEED = 0.5
ANGLE_TOLERANCE = 0.05  # ~3.5 degrees

# Arm joint names (order matters for IK chain)
PART_NAMES = [
    "head_2_joint",
    "head_1_joint",
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
    "wheel_left_joint",
    "wheel_right_joint",
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
]

SPECIAL_SENSOR_NAMES = {
    "gripper_left_finger_joint": "gripper_left_sensor_finger_joint",
    "gripper_right_finger_joint": "gripper_right_sensor_finger_joint",
}

# IK chain base elements
_IK_BASE_ELEMENTS = [
    "base_link",
    "base_link_Torso_joint",
    "Torso",
    "torso_lift_joint",
    "torso_lift_link",
    "torso_lift_link_TIAGo front arm_joint",
    "TIAGo front arm_3",
    "arm_1_joint",
    "TIAGo front arm_3",
    "arm_2_joint",
    "arm_2_link",
    "arm_3_joint",
    "arm_3_link",
    "arm_4_joint",
    "arm_4_link",
    "arm_5_joint",
    "arm_5_link",
    "arm_6_joint",
    "arm_6_link",
    "arm_7_joint",
    "arm_7_link",
    "arm_7_link_wrist_ft_tool_link_joint",
    "wrist_ft_tool_link",
    "wrist_ft_tool_link_front_joint",
]

# Predefined arm positions
STARTING_POSITION = {
    "torso_lift_joint": 0.3,
    "arm_1_joint": 0.71,
    "arm_2_joint": 1.02,
    "arm_3_joint": -2.815,
    "arm_4_joint": 1.011,
    "arm_5_joint": 0.0,
    "arm_6_joint": 0.0,
    "arm_7_joint": 0.0,
    "gripper_left_finger_joint": 0.045,
    "gripper_right_finger_joint": 0.045,
    "head_1_joint": 0.0,
    "head_2_joint": 0.0,
}

LIFT_POSITION = {
    "torso_lift_joint": 0.3,
    "arm_1_joint": 0.7,
    "arm_2_joint": 0.4,
    "arm_3_joint": -1.5,
    "arm_4_joint": 1.7,
    "arm_5_joint": -1.5,
    "arm_6_joint": 0.0,
    "arm_7_joint": 0.0,
}

PLACE_POSITION = {
    "torso_lift_joint": 0.15,
    "arm_1_joint": 1.6,
    "arm_2_joint": 1.5,  # 1.02
    "arm_3_joint": 0.0,
    "arm_4_joint": 1.2,
    "arm_5_joint": 0.5,
    "arm_6_joint": 0.0,
    "arm_7_joint": -2.07,
}


class RobotBase:
    """Core hardware: Supervisor, motors, sensors, camera, GPS, compass, lidar, IK chain.

    Initialises all robot devices and moves the arm to a safe starting position.
    Odometry state (xw, yw, alpha) and navigation constants are set up here and
    shared with OdometryMixin and TrajectoryMixin.
    """

    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Odometry state (shared with OdometryMixin / TrajectoryMixin)
        self.xw = 0.0
        self.yw = 0.0
        self.alpha = 0.0
        self.state = "forward"  # used by TrajectoryMixin
        self.is_finished = False

        # Navigation constants (used by TrajectoryMixin)
        self.max_velocity = MAX_MOTOR_SPEED
        self.p_alpha = 3.0
        self.p_rho = math.pi

        # Occupancy map (used by MappingMixin)
        self.map = np.zeros((300, 300))

        # Mapping waypoints — robot traces around the kitchen counter
        self.waypoints = np.array(
            [
                [0.8, -0.2],
                [0.8, -0.45],
                [0.57, -1.83],
                [0.14, -3.14],
                [-0.9, -3.15],
                [-1.60, -2.92],
                [-1.60, -1.6],
                [-1.60, -0.09],
                [-0.74, 0.50],
            ]
        )

        self._init_devices()

    def _init_devices(self):
        """Initialise all robot devices: motors, sensors, camera, GPS, compass, lidar, IK chain.

        Input:
            None (reads URDF directly from the Supervisor).
        Output:
            None (populates self.motors, self.sensors, self.camera, self.gps,
                  self.compass, self.lidar, self.ik_chain, self.joint_limits).
        """
        # Save / parse URDF for IK
        urdf_path = "Robot.urdf"
        with open(urdf_path, "w") as f:
            f.write(self.robot.getUrdf())
        urdf_root = urdf_model.URDF.from_xml_file(urdf_path)

        self.joint_limits = {
            joint.name: {
                "lower": joint.limit.lower,
                "upper": joint.limit.upper,
                "velocity": joint.limit.velocity,
            }
            for joint in urdf_root.joint_map.values()
            if joint.limit is not None
        }

        # Motors & sensors
        self.motors = {}
        self.sensors = {}
        for name in PART_NAMES:
            try:
                motor = self.robot.getDevice(name)
                limit = self.joint_limits.get(name)
                motor.setVelocity(limit["velocity"] * 0.3 if limit else 1.0)

                sensor_name = SPECIAL_SENSOR_NAMES.get(name, f"{name}_sensor")
                sensor = self.robot.getDevice(sensor_name)
                if sensor:
                    sensor.enable(self.timestep)
                    self.sensors[name] = sensor

                self.motors[name] = motor
            except Exception as e:
                print(f"Warning: could not init '{name}': {e}")

        # Wheel motors
        self.left_wheel_motor = self.motors["wheel_left_joint"]
        self.right_wheel_motor = self.motors["wheel_right_joint"]
        self.left_wheel_motor.setPosition(float("inf"))
        self.right_wheel_motor.setPosition(float("inf"))
        self.left_wheel_motor.setVelocity(0.0)
        self.right_wheel_motor.setVelocity(0.0)

        # Gripper force feedback
        self.motors["gripper_left_finger_joint"].enableForceFeedback(self.timestep)
        self.motors["gripper_right_finger_joint"].enableForceFeedback(self.timestep)

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # Display
        self.display = self.robot.getDevice("display")
        self.display.attachCamera(self.camera)

        # Marker (used by TrajectoryMixin)
        marker_node = self.robot.getFromDef("marker")
        self.marker = marker_node.getField("translation") if marker_node else None

        # GPS & Compass
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        # Lidar
        self.lidar = self.robot.getDevice("Hokuyo URG-04LX-UG01")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # IK chain
        self.ik_chain = self._create_ik_chain(urdf_path)
        self.urdf_path = urdf_path

        # Move arm to safe start position
        for joint, pos in STARTING_POSITION.items():
            if joint in self.motors:
                self.motors[joint].setPosition(pos)

        print(
            f"RobotBase: {len(self.motors)} motors, {len(self.sensors)} sensors initialised."
        )

    def _create_ik_chain(self, urdf_path):
        """Create an IK chain from the URDF file, filtering for revolute joints.

        Input:
            urdf_path (str): Path to the URDF file.
        Output:
            Chain: An ikpy Chain object representing the robot's arm.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=_IK_BASE_ELEMENTS,
                last_link_vector=[0.016, 0, 0],
                name="tiago_arm",
            )
        mask = []
        for i, link in enumerate(chain.links):
            if i == 0:
                mask.append(False)
            elif getattr(link, "joint_type", None) == "revolute":
                mask.append(True)
            else:
                mask.append(False)
        print(f"IK chain: {len(chain.links)} links")
        return Chain(links=chain.links, active_links_mask=mask, name="tiago_arm")

    def step(self):
        """Advance the simulation by one timestep.

        Input:
            None.
        Output:
            bool: True if the simulation is still running, False if it should stop.
        """
        result = self.robot.step(self.timestep)
        if result == -1:
            return False
        return not self.is_finished

    def set_wheel_velocity(self, left_vel, right_vel):
        """Set left and right wheel velocities, clamped to max_velocity.

        Input:
            left_vel  (float): Desired left wheel velocity (rad/s).
            right_vel (float): Desired right wheel velocity (rad/s).
        Output:
            None.
        """
        left_vel = float(np.clip(left_vel, -self.max_velocity, self.max_velocity))
        right_vel = float(np.clip(right_vel, -self.max_velocity, self.max_velocity))
        self.left_wheel_motor.setVelocity(left_vel)
        self.right_wheel_motor.setVelocity(right_vel)
