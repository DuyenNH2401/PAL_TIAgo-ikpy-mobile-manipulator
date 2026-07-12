# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""A* pathfinding on configuration-space grid maps with path simplification."""

import numpy as np
import heapq
from pathlib import Path

# World boundaries — must match mapping/grid_map.py
X_MIN, X_MAX = -2.25, +2.25
Y_MIN, Y_MAX = -3.92, +1.75

CSPACE_PATH = Path(__file__).resolve().parent.parent / "map_save" / "cspace.npy"


def load_cspace(path=None):
    """Load configuration-space map from file.

    Input:
        path (str or Path, optional): Custom path to the cspace .npy file.
            Defaults to the standard map_save/cspace.npy location.
    Output:
        np.ndarray or None: Boolean occupancy grid, or None if the file does not exist.
    """
    p = Path(path) if path else CSPACE_PATH
    if not p.exists():
        return None
    return np.load(p)


def world_to_pixel(xw, yw, rows, cols):
    """Convert world coordinates to pixel indices.

    Input:
        xw   (float): World x position in metres.
        yw   (float): World y position in metres.
        rows (int):   Number of rows in the grid.
        cols (int):   Number of columns in the grid.
    Output:
        tuple: (x_px, y_px) integer pixel indices clamped to grid bounds.
    """
    x_px = int(np.clip((xw - X_MIN) / (X_MAX - X_MIN) * cols, 0, cols - 1))
    y_px = int(np.clip((yw - Y_MIN) / (Y_MAX - Y_MIN) * rows, 0, rows - 1))
    return x_px, y_px


def pixel_to_world(x_px, y_px, rows, cols):
    """Convert pixel indices back to world coordinates.

    Input:
        x_px (int): Pixel column index.
        y_px (int): Pixel row index.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    Output:
        tuple: (xw, yw) world position in metres.
    """
    xw = (x_px / cols) * (X_MAX - X_MIN) + X_MIN
    yw = (y_px / rows) * (Y_MAX - Y_MIN) + Y_MIN
    return xw, yw


def astar(cspace, start_world, goal_world):
    """Run A* search on a configuration-space grid and return a simplified path.

    Input:
        cspace      (np.ndarray): Boolean occupancy grid (True = obstacle).
        start_world (tuple):      Start position (xw, yw) in world coordinates.
        goal_world  (tuple):      Goal position  (xw, yw) in world coordinates.
    Output:
        list of (xw, yw) tuples representing waypoints from start to goal,
        or None if no path exists.
    """
    rows, cols = cspace.shape
    start_px = world_to_pixel(*start_world, rows, cols)
    goal_px = world_to_pixel(*goal_world, rows, cols)

    start_xy = (start_px[0], start_px[1])
    goal_xy = (goal_px[0], goal_px[1])

    if not (0 <= start_xy[0] < rows and 0 <= start_xy[1] < cols):
        return None
    if not (0 <= goal_xy[0] < rows and 0 <= goal_xy[1] < cols):
        return None
    if cspace[start_xy] or cspace[goal_xy]:
        return None
    if start_xy == goal_xy:
        return [start_world, goal_world]

    _NEIGHBOURS = [
        (-1, -1, np.sqrt(2)),
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, np.sqrt(2)),
        (1, 0, 1.0),
        (1, 1, np.sqrt(2)),
    ]

    def _h(xy):
        return np.sqrt((xy[0] - goal_xy[0]) ** 2 + (xy[1] - goal_xy[1]) ** 2)

    heap = [(0.0, start_xy)]
    came_from = {}
    g_score = {start_xy: 0.0}

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal_xy:
            path_px = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path_px.append(cur)
            path_px.reverse()
            simple_px = simplify_path(path_px, cspace)
            return [pixel_to_world(px, py, rows, cols) for px, py in simple_px]

        for dx, dy, cost in _NEIGHBOURS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            if cspace[nb]:
                continue
            tent_g = g_score[cur] + cost
            if tent_g < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb] = tent_g
                heapq.heappush(heap, (tent_g + _h(nb), nb))

    return None


def simplify_path(path_px, cspace):
    """Remove redundant intermediate waypoints using line-of-sight checks.

    Input:
        path_px (list): List of (x, y) pixel tuples from A*.
        cspace  (np.ndarray): Boolean occupancy grid.
    Output:
        list: Reduced list of (x, y) pixel tuples with only necessary waypoints.
    """
    if len(path_px) <= 2:
        return path_px
    kept = [path_px[0]]
    anchor = 0
    for i in range(1, len(path_px) - 1):
        if not _line_of_sight(path_px[anchor], path_px[i + 1], cspace):
            kept.append(path_px[i])
            anchor = i
    kept.append(path_px[-1])
    return kept


def _line_of_sight(p1, p2, cspace):
    """Check whether a straight line between two pixels is obstacle-free.

    Input:
        p1     (tuple): Start pixel (x, y).
        p2     (tuple): End pixel (x, y).
        cspace (np.ndarray): Boolean occupancy grid.
    Output:
        bool: True if the line is clear of obstacles, False otherwise.
    """
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    rows, cols = cspace.shape
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        if not (0 <= x1 < rows and 0 <= y1 < cols):
            return False
        if cspace[x1, y1]:
            return False
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return True
