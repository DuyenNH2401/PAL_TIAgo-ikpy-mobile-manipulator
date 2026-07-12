# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

import numpy as np


class PerceptionMixin:
    """Converts raw LiDAR range image to robot-frame and world-frame point clouds."""

    def lidar2cartesian(self):
        """Convert the LiDAR range image to Cartesian point clouds.

        Trims 80 samples on each side (robot body in FOV), filters invalid
        readings, and transforms points into both robot frame and world frame.

        Input:
            None (reads self.lidar, self.alpha, self.xw, self.yw directly).
        Output:
            tuple:
                X_world (np.ndarray, shape 3×N): Points in world frame [x, y, 1].
                X_robot (np.ndarray, shape 3×N): Points in robot frame [x, y, 1].
        """
        ranges = np.array(self.lidar.getRangeImage())
        ranges[ranges == np.inf] = 100.0

        angles = np.linspace(2 / 3 * np.pi, -2 / 3 * np.pi, len(ranges))

        # Trim 80 samples on each side (robot body in FOV)
        ranges = ranges[80:-80]
        angles = angles[80:-80]

        valid = (ranges != np.inf) & (ranges < 5.0)
        ranges = ranges[valid]
        angles = angles[valid]

        X_lidar = np.array([
            ranges * np.cos(angles),
            ranges * np.sin(angles),
            np.ones_like(ranges),
        ])

        # Lidar offset from robot centre (0.23 m forward)
        r_X_l = np.array([[1, 0, 0.23], [0, 1, 0], [0, 0, 1]])

        w_T_r = np.array([
            [np.cos(self.alpha), -np.sin(self.alpha), self.xw],
            [np.sin(self.alpha),  np.cos(self.alpha), self.yw],
            [0,                   0,                  1      ],
        ])

        X_robot = r_X_l @ X_lidar
        X_world = w_T_r @ X_robot
        return X_world, X_robot
