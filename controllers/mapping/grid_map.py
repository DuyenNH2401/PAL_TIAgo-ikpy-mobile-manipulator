# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Grid mapping — world↔pixel transforms and probabilistic occupancy map."""

import numpy as np

# World boundaries (must match navigation/pathfinding.py)
X_MIN, X_MAX = -2.25, +2.25
Y_MIN, Y_MAX = -3.92, +1.75
MAP_SIZE = 300


class MappingMixin:
    """World-to-pixel coordinate transforms and probabilistic map updates."""

    def mapping(self, xw, yw):
        """Convert world coordinates (xw, yw) to pixel coordinates (x_px, y_px).

        Input:
            xw (float or np.ndarray): World x position(s) in metres.
            yw (float or np.ndarray): World y position(s) in metres.
        Output:
            tuple:
                x_pixel (int or np.ndarray): Pixel column index, clamped to [0, MAP_SIZE-1].
                y_pixel (int or np.ndarray): Pixel row index, clamped to [0, MAP_SIZE-1].
        """
        x_val = (xw - X_MIN) / (X_MAX - X_MIN) * MAP_SIZE
        y_val = (yw - Y_MIN) / (Y_MAX - Y_MIN) * MAP_SIZE

        if isinstance(xw, np.ndarray):
            x_pixel = np.clip(x_val.astype(int), 0, MAP_SIZE - 1)
            y_pixel = np.clip(y_val.astype(int), 0, MAP_SIZE - 1)
        else:
            x_pixel = max(0, min(MAP_SIZE - 1, int(x_val)))
            y_pixel = max(0, min(MAP_SIZE - 1, int(y_val)))

        return x_pixel, y_pixel

    def inverse_mapping(self, x_pixel, y_pixel):
        """Convert pixel coordinates back to world coordinates.

        Input:
            x_pixel (int): Pixel column index.
            y_pixel (int): Pixel row index.
        Output:
            tuple:
                xw (float): World x position in metres.
                yw (float): World y position in metres.
        """
        xw = (x_pixel / MAP_SIZE) * (X_MAX - X_MIN) + X_MIN
        yw = (y_pixel / MAP_SIZE) * (Y_MAX - Y_MIN) + Y_MIN
        return xw, yw

    def probabilistic_mapping(self, X_world):
        """Increment occupancy probability for all LiDAR hit pixels.

        Input:
            X_world (np.ndarray, shape 3×N): World-frame point cloud [x, y, 1].
        Output:
            None (updates self.map in place, values clamped to [0, 1]).
        """
        x_pixel, y_pixel = self.mapping(X_world[0], X_world[1])
        np.add.at(self.map, (x_pixel, y_pixel), 0.005)
        self.map = np.clip(self.map, 0.0, 1.0)
