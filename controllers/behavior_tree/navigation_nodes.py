# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Navigation BT nodes — ReturnHome, ComputePath, MoveTo, MoveToWaypoint, MoveToObject."""

import math
import numpy as np
import py_trees

from navigation.pathfinding import load_cspace, astar, world_to_pixel
from robot.base import ANGLE_TOLERANCE


class ReturnHome(py_trees.behaviour.Behaviour):
    """Drive robot back to (0, 0) facing 0° before pick-and-place begins.

    State machine: ORIENTING → MOVING → ALIGNING → DONE
    """

    HOME_X = 0.0
    HOME_Y = 0.0
    HOME_ANGLE = 0.0  # radians
    DIST_THRESHOLD = 0.15  # metres
    ANGLE_THRESHOLD = 0.05  # radians

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.blackboard = blackboard
        self._state = "ORIENTING"

    def initialise(self):
        self._state = "ORIENTING"
        robot = self.blackboard.read("robot")
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)
        print("ReturnHome: returning to (0,0) facing 0°")

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        xw, yw, alpha = robot.xw, robot.yw, robot.alpha

        dx = self.HOME_X - xw
        dy = self.HOME_Y - yw
        rho = math.sqrt(dx**2 + dy**2)

        if self._state == "ORIENTING":
            if rho < self.DIST_THRESHOLD:
                self._state = "ALIGNING"
                return py_trees.common.Status.RUNNING

            target_angle = math.atan2(dy, dx)
            angle_err = _norm_angle(target_angle - alpha)
            if abs(angle_err) < self.ANGLE_THRESHOLD:
                self._state = "MOVING"
                return py_trees.common.Status.RUNNING

            turn = np.clip(2.0 * angle_err, -2.0, 2.0)
            robot.set_wheel_velocity(-turn, turn)
            return py_trees.common.Status.RUNNING

        elif self._state == "MOVING":
            if rho < self.DIST_THRESHOLD:
                robot.set_wheel_velocity(0.0, 0.0)
                self._state = "ALIGNING"
                return py_trees.common.Status.RUNNING

            target_angle = math.atan2(dy, dx)
            angle_err = _norm_angle(target_angle - alpha)

            p1, p2 = 3.0, math.pi
            vL = -p1 * angle_err + p2 * rho
            vR = p1 * angle_err + p2 * rho
            robot.set_wheel_velocity(vL, vR)
            return py_trees.common.Status.RUNNING

        elif self._state == "ALIGNING":
            angle_err = _norm_angle(self.HOME_ANGLE - alpha)
            if abs(angle_err) < self.ANGLE_THRESHOLD:
                robot.set_wheel_velocity(0.0, 0.0)
                print("ReturnHome: at home position, facing 0°")
                return py_trees.common.Status.SUCCESS

            turn = np.clip(1.5 * angle_err, -1.5, 1.5)
            robot.set_wheel_velocity(-turn, turn)
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        robot = self.blackboard.read("robot")
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)


class ComputePath(py_trees.behaviour.Behaviour):
    """Compute A* path from current robot position to goal point."""

    def __init__(self, name, point, blackboard):
        super().__init__(name)
        self.point = point
        self.blackboard = blackboard

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        cspace = load_cspace()
        if cspace is None:
            self.feedback_message = "cspace.npy not found"
            return py_trees.common.Status.FAILURE

        rows, cols = cspace.shape
        start = (robot.xw, robot.yw)
        start_px = world_to_pixel(start[0], start[1], rows, cols)
        end_px = world_to_pixel(self.point[0], self.point[1], rows, cols)

        # Clear start cell and neighbours to avoid blocked-start failures
        cspace[start_px] = False
        cspace[end_px] = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = start_px[0] + dx, start_px[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    cspace[nx, ny] = False

        path = astar(cspace, start, self.point)
        if path is None:
            self.feedback_message = f"No path from {start} to {self.point}"
            return py_trees.common.Status.FAILURE

        self.blackboard.write("path", path)
        self.blackboard.write("path_index", 0)
        self.feedback_message = f"Path: {len(path)} waypoints"
        return py_trees.common.Status.SUCCESS


class MoveTo(py_trees.behaviour.Behaviour):
    """Follow the A* path stored on the blackboard."""

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.blackboard = blackboard

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        path = self.blackboard.read("path")
        if path is None:
            return py_trees.common.Status.FAILURE

        index = self.blackboard.read("path_index") or 0
        index, reached = robot.follow_path(index, path)
        self.blackboard.write("path_index", index)

        if reached:
            self.feedback_message = "arrived"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = f"waypoint {index}/{len(path)}"
        return py_trees.common.Status.RUNNING


class MoveToWaypoint(py_trees.behaviour.Behaviour):
    """Navigate through a fixed list of waypoints using a P-controller.

    Args:
        waypoints: list of (x, y, z) or (x, y) tuples
        timeout:   max seconds before declaring success anyway
    """

    def __init__(self, name, blackboard, waypoints, timeout=45.0):
        super().__init__(name)
        self.blackboard = blackboard
        self.waypoints = waypoints
        self.timeout = timeout
        self._index = 0
        self._start_t = None

        self.p1 = 4.0  # angular gain
        self.p2 = 2.0  # linear gain
        self.max_speed = 6.28
        self.dist_thresh = 0.15

    def initialise(self):
        self._index = 0
        robot = self.blackboard.read("robot")
        self._start_t = robot.robot.getTime() if robot else 0.0
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)
        print(f"{self.name}: navigating {len(self.waypoints)} waypoints")

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        now = robot.robot.getTime()
        if now - self._start_t > self.timeout:
            print(f"{self.name}: timeout — declaring success")
            robot.set_wheel_velocity(0.0, 0.0)
            return py_trees.common.Status.SUCCESS

        wp = self.waypoints[self._index]
        xw, yw = robot.xw, robot.yw
        alpha = robot.alpha

        rho = math.sqrt((xw - wp[0]) ** 2 + (yw - wp[1]) ** 2)
        angle = math.atan2(wp[1] - yw, wp[0] - xw) - alpha
        angle = _norm_angle(angle)

        vL = np.clip(-self.p1 * angle + self.p2 * rho, -self.max_speed, self.max_speed)
        vR = np.clip(self.p1 * angle + self.p2 * rho, -self.max_speed, self.max_speed)
        robot.set_wheel_velocity(vL, vR)

        if rho < self.dist_thresh:
            self._index += 1
            if self._index >= len(self.waypoints):
                robot.set_wheel_velocity(0.0, 0.0)
                print(f"{self.name}: all waypoints reached")
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        robot = self.blackboard.read("robot")
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)


class MoveToObject(py_trees.behaviour.Behaviour):
    """Navigate to the detected object using GPS + compass.

    Reads target_position from the blackboard. Triggers arm pre-positioning
    when within arm_adjustment_distance of the target.

    States: ORIENTING → APPROACHING → STABILIZING → ADJUSTING_ARM → FINAL_APPROACH
    """

    def __init__(self, name, blackboard, move_arm_behaviour=None):
        super().__init__(name)
        self.blackboard = blackboard
        self.move_arm_behaviour = move_arm_behaviour

        self.arm_adjustment_distance = 1.28
        self.very_close_distance = 1.0

        self.Kp_linear = 0.9
        self.Kp_angular = 1.0
        self.Kd_angular = 1.0
        self.max_speed = 3.0
        self.stabilization_duration = 0.5

        self._state = "ORIENTING"
        self._start_t = None
        self._stab_t = None
        self._arm_adj_t = None
        self._target = None
        self._prev_alpha = 0.0
        self._last_t = None

    def initialise(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return

        robot.set_wheel_velocity(0.0, 0.0)
        robot.robot.step(robot.timestep * 5)

        self._target = self.blackboard.read("target_position")
        if self._target is None:
            print(f"{self.name}: no target on blackboard!")
            return

        self._state = "ORIENTING"
        self._start_t = robot.robot.getTime()
        self._last_t = self._start_t
        self._prev_alpha = 0.0
        print(f"{self.name}: moving to {self._target[:2]}")

    def update(self):
        if self._target is None:
            return py_trees.common.Status.FAILURE

        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        xw, yw = robot.xw, robot.yw
        alpha = robot.alpha
        now = robot.robot.getTime()
        dt = max(now - self._last_t, robot.timestep / 1000.0)
        self._last_t = now

        tx, ty = self._target[0], self._target[1]
        dx, dy = tx - xw, ty - yw
        rho = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        err_alpha = _norm_angle(target_angle - alpha)
        alpha_rate = (err_alpha - self._prev_alpha) / dt
        self._prev_alpha = err_alpha

        if self._state == "ORIENTING":
            if abs(err_alpha) < ANGLE_TOLERANCE:
                robot.set_wheel_velocity(0.0, 0.0)
                self._state = "APPROACHING"
                return py_trees.common.Status.RUNNING
            turn = np.clip(1.0 * err_alpha - 0.5 * alpha_rate, -1.8, 1.8)
            robot.set_wheel_velocity(-turn, turn)
            return py_trees.common.Status.RUNNING

        elif self._state == "APPROACHING":
            if rho < self.arm_adjustment_distance:
                robot.set_wheel_velocity(0.0, 0.0)
                self._stab_t = now
                self._state = "STABILIZING"
                return py_trees.common.Status.RUNNING
            lin = self.Kp_linear * rho * (0.5 + rho / 3.0 if rho < 1.5 else 1.0)
            ang = self.Kp_angular * err_alpha - self.Kd_angular * alpha_rate
            robot.set_wheel_velocity(
                np.clip(lin - ang, -self.max_speed, self.max_speed),
                np.clip(lin + ang, -self.max_speed, self.max_speed),
            )
            return py_trees.common.Status.RUNNING

        elif self._state == "STABILIZING":
            robot.set_wheel_velocity(0.0, 0.0)
            if now - self._stab_t >= self.stabilization_duration:
                self._state = "ADJUSTING_ARM"
                self._arm_adj_t = now
                if self.move_arm_behaviour:
                    self.move_arm_behaviour.initialise()
            return py_trees.common.Status.RUNNING

        elif self._state == "ADJUSTING_ARM":
            robot.set_wheel_velocity(0.0, 0.0)
            elapsed = now - self._arm_adj_t
            arm_done = True
            if self.move_arm_behaviour:
                status = self.move_arm_behaviour.update()
                arm_done = status == py_trees.common.Status.SUCCESS or elapsed > 2.5
            if arm_done:
                if rho < self.very_close_distance:
                    return py_trees.common.Status.SUCCESS
                self._state = "FINAL_APPROACH"
            return py_trees.common.Status.RUNNING

        elif self._state == "FINAL_APPROACH":
            if rho < self.very_close_distance:
                robot.set_wheel_velocity(0.0, 0.0)
                return py_trees.common.Status.SUCCESS
            lin = 0.5 * self.Kp_linear * rho
            ang = self.Kp_angular * err_alpha - self.Kd_angular * alpha_rate
            max_s = 1.5
            robot.set_wheel_velocity(
                np.clip(lin - ang, -max_s, max_s),
                np.clip(lin + ang, -max_s, max_s),
            )
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        robot = self.blackboard.read("robot")
        if robot:
            robot.set_wheel_velocity(0.0, 0.0)


def _norm_angle(a):
    """Normalize an angle to the range [-π, π].

    Input:
        a (float): Angle in radians (any value).
    Output:
        float: Equivalent angle wrapped to [-π, π].
    """
    return np.arctan2(np.sin(a), np.cos(a))
