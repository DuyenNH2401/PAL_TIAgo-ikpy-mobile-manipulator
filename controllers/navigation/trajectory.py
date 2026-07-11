# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Trajectory following — waypoint-based navigation with forward/backward pass."""

import numpy as np
import math


class TrajectoryMixin:
    """Follows a list of waypoints, switching direction when the end is reached."""

    def trajectory_following(self, index, waypoints):
        """Drive toward waypoints[index] and advance the index when close enough.

        Switches from forward to backward pass when the last waypoint is reached,
        and sets self.is_finished when the backward pass is complete.

        Input:
            index     (int):        Current waypoint index to target.
            waypoints (np.ndarray): Array of (x, y) world-frame waypoints.
        Output:
            int: Updated waypoint index after the current step.
        """
        self._place_marker(waypoints, index)
        rho, alpha = self._compute_error(waypoints[index])

        phil = -self.p_alpha * alpha + self.p_rho * rho
        phir =  self.p_alpha * alpha + self.p_rho * rho
        self.set_wheel_velocity(phil, phir)

        if rho < 0.3 and self.state == "forward":
            index += 1
            if index >= len(waypoints):
                self.state = "backward"
                index = len(waypoints) - 2
        elif rho < 0.3 and self.state == "backward":
            index -= 1
            if index < 0:
                self.is_finished = True

        return index

    def follow_path(self, index, waypoints):
        """Follow a pre-computed path (e.g., A* output) one waypoint at a time.

        Input:
            index     (int):  Current index into the waypoints list.
            waypoints (list): List of (xw, yw) world-frame positions.
        Output:
            tuple:
                int:  Updated index after this step.
                bool: True if the final waypoint has been reached.
        """
        if index >= len(waypoints):
            return index, True

        self._place_marker(waypoints, index)
        rho, alpha = self._compute_error(waypoints[index])

        phil = -self.p_alpha * alpha + self.p_rho * rho
        phir =  self.p_alpha * alpha + self.p_rho * rho
        self.set_wheel_velocity(phil, phir)

        reached = False
        if rho < 0.3:
            index += 1
            if index >= len(waypoints):
                self.set_wheel_velocity(0.0, 0.0)
                reached = True

        return index, reached

    def _place_marker(self, waypoints, index):
        """Move the Webots marker node to the current target waypoint.

        Input:
            waypoints (np.ndarray): Array of (x, y) waypoints.
            index     (int):        Index of the current target waypoint.
        Output:
            None.
        """
        if self.marker is not None:
            self.marker.setSFVec3f([*waypoints[index], 0.0])

    def _compute_error(self, point):
        """Compute distance and heading error to a target point.

        Input:
            point (tuple): Target (x, y) position in world frame.
        Output:
            tuple:
                rho   (float): Euclidean distance to the target (metres).
                alpha (float): Heading error to the target (radians, wrapped to ±π).
        """
        dx  = point[0] - self.xw
        dy  = point[1] - self.yw
        rho = math.sqrt(dx**2 + dy**2)
        alpha = math.atan2(dy, dx) - self.alpha
        if alpha >  math.pi: alpha -= 2 * math.pi
        if alpha < -math.pi: alpha += 2 * math.pi
        return rho, alpha
