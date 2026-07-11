# Copyright (c) 2026 Nguyen Huu Duyen. All Rights Reserved.
# Author : Nguyen Huu Duyen
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""TiagoFull — assembled robot controller."""

from robot.base import RobotBase
from robot.odometry import OdometryMixin
from robot.perception import PerceptionMixin
from mapping.grid_map import MappingMixin
from navigation.trajectory import TrajectoryMixin


class TiagoFull(RobotBase, OdometryMixin, PerceptionMixin, MappingMixin, TrajectoryMixin):
    """TIAGo robot — hardware, odometry, perception, mapping, navigation, and manipulation."""
    pass
