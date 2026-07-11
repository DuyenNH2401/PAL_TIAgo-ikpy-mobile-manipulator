# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Odometry — GPS + compass fusion for robot pose estimation."""

import numpy as np


class OdometryMixin:
    """Updates xw, yw, alpha from GPS and compass each tick."""

    def update_odometry(self):
        """Read GPS and compass to refresh the robot's world-frame pose.

        Input:
            None (reads self.gps and self.compass directly).
        Output:
            None (updates self.xw, self.yw, self.alpha in place).
        """
        self.xw    = self.gps.getValues()[0]
        self.yw    = self.gps.getValues()[1]
        self.alpha = np.arctan2(
            self.compass.getValues()[0],
            self.compass.getValues()[1],
        )
