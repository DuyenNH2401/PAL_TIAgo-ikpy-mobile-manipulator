# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""Mapping utilities — per-step map update and cspace save."""

import numpy as np
import scipy.signal as signal
from pathlib import Path

_MAP_DIR = Path(__file__).resolve().parent.parent / "map_save"
_MAP_DIR.mkdir(parents=True, exist_ok=True)

CSPACE_PATH       = _MAP_DIR / "cspace.npy"
CSPACE_ARRAY_PATH = _MAP_DIR / "cspace_array.npy"


def mapping_run(robot):
    """Run one tick of mapping: process LiDAR data and update the display.

    Input:
        robot: TiagoFull instance with lidar, probabilistic_mapping, mapping,
               xw, yw, display attributes populated.
    Output:
        None (updates robot.map and draws pixels on robot.display in place).
    """
    X_world, _ = robot.lidar2cartesian()
    robot.probabilistic_mapping(X_world)
    x_pixel, y_pixel = robot.mapping(X_world[0], X_world[1])

    for x, y in zip(x_pixel, y_pixel):
        prob = robot.map[x, y]
        if prob > 0.1:
            v = int(prob * 255)
            color = v * 256 ** 2 + v * 256 + v
            robot.display.setColor(color)
            robot.display.drawPixel(x, y)

    rx, ry = robot.mapping(robot.xw, robot.yw)
    robot.display.setColor(0xFF0000)
    robot.display.drawPixel(rx, ry)


def save_cspace(robot):
    """Convolve the occupancy map to configuration space and save both arrays to disk.

    Input:
        robot: TiagoFull instance with a populated robot.map (300×300 float array).
    Output:
        None (writes cspace.npy and cspace_array.npy to the map_save directory).
    """
    kernel = np.ones((21, 21))
    cmap   = signal.convolve2d(robot.map, kernel, mode="same")
    cspace = cmap > 0.5
    np.save(CSPACE_PATH, cspace)
    np.save(CSPACE_ARRAY_PATH, robot.map)
    print(f"Mapping: cspace saved to {CSPACE_PATH}")
