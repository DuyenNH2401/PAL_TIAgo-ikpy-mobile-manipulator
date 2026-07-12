# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

class Blackboard:
    """Simple key-value store shared across all BT nodes.

    Keys used by the system:
        robot           – TiagoFull instance
        path            – list of (xw, yw) waypoints (A* output)
        path_index      – current waypoint index for MoveTo
        target_position – [x, y, z] of detected object (world frame)
        object_name     – model name of detected object
        grasp_success   – bool
    """

    def __init__(self):
        self._data = {}

    def write(self, key, value):
        """Store a value under the given key.

        Input:
            key   (str): Blackboard key.
            value (any): Value to store.
        Output:
            None.
        """
        self._data[key] = value

    def read(self, key):
        """Retrieve a value by key, returning None if not found.

        Input:
            key (str): Blackboard key.
        Output:
            any: Stored value, or None if the key does not exist.
        """
        return self._data.get(key, None)
