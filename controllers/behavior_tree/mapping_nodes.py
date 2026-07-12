# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Mapping BT nodes — MapExist, RunMapping, MoveTable."""

import py_trees
from mapping.mapping_utils import mapping_run, save_cspace, CSPACE_PATH


class MapExist(py_trees.behaviour.Behaviour):
    """Return SUCCESS if a saved cspace.npy already exists on disk, FAILURE otherwise."""

    def __init__(self, name="MapExist"):
        super().__init__(name)

    def update(self):
        if CSPACE_PATH.exists():
            print("Mapping: cspace found — skipping mapping phase.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class RunMapping(py_trees.behaviour.Behaviour):
    """Update probabilistic map from LiDAR each tick. Always returns RUNNING."""

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.blackboard = blackboard

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE
        mapping_run(robot)
        return py_trees.common.Status.RUNNING


class MoveTable(py_trees.behaviour.Behaviour):
    """Drive robot around the kitchen counter for mapping coverage.

    Returns SUCCESS when the forward+backward waypoint pass is complete.
    The robot traces the waypoint list forward then backward, calling
    save_cspace() once the backward pass finishes.
    """

    def __init__(self, name, blackboard):
        super().__init__(name)
        self.blackboard = blackboard
        self.index = 1
        self._initialised = False

    def initialise(self):
        if not self._initialised:
            robot = self.blackboard.read("robot")
            if robot is not None:
                self.index = 1
                self.waypoints = robot.waypoints
                self._initialised = True

    def update(self):
        robot = self.blackboard.read("robot")
        if robot is None:
            return py_trees.common.Status.FAILURE

        self.index = robot.trajectory_following(self.index, self.waypoints)

        if robot.state == "backward" and self.index == 0:
            save_cspace(robot)
            robot.set_wheel_velocity(0.0, 0.0)
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
