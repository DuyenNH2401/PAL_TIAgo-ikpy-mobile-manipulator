# Copyright (c) 2026 DuyenNH2401. All Rights Reserved.
# Author : DuyenNH2401
# Email  : duyennhce200017@gmail.com
# Project: TIAgoController

"""
TIAGo Robot Controller — Mapping + Pick-and-Place
"""

import py_trees
from py_trees.composites import Parallel, Selector, Sequence

from robot import TiagoFull
from behavior_tree.blackboard import Blackboard
from behavior_tree.mapping_nodes import MapExist, RunMapping, MoveTable
from behavior_tree.navigation_nodes import ReturnHome, MoveToWaypoint, MoveToObject
from behavior_tree.manipulation_nodes import (
    CheckHardware,
    MoveToPosition,
    ObjectRecognizer,
    ComprehensiveScanner,
    MoveArmIK,
    GraspController,
    LiftAndVerify,
    BackupAfterGrasp,
    OpenGripper,
)
from robot.base import STARTING_POSITION, LIFT_POSITION, PLACE_POSITION

# Waypoints used for transport (world frame)
TABLE_WAYPOINTS = [(1.0, -0.9, 0.095), (0.2, -1.5, 0.095)]
HOME_WAYPOINT = [(0.3, 0.0, 0.095)]

# Per-jar Y-axis offset applied to IK target (fine-tuned for counter layout)
Y_OFFSETS = [0.13, -0.80, -0.6]


def create_behavior_tree(blackboard: Blackboard) -> py_trees.trees.BehaviourTree:
    """Build and return the full mission behaviour tree.

    Input:
        blackboard (Blackboard): Shared key-value store pre-loaded with the
                                 'robot' key pointing to a TiagoFull instance.
    Output:
        py_trees.trees.BehaviourTree: Configured tree ready for setup() and tick().
    """
    root = Sequence(name="Root", memory=True)

    # 1. Initialization
    init_seq = Sequence(name="Initialization", memory=True)
    init_seq.add_children(
        [
            CheckHardware("Check Hardware", blackboard),
            MoveToPosition("Safe Start Position", blackboard, STARTING_POSITION),
        ]
    )

    # 2. Mapping Phase
    # Selector: if cspace.npy exists → skip; else run mapping in parallel
    mapping_selector = Selector(name="Mapping Phase", memory=False)
    mapping_selector.add_child(MapExist("Map already exists?"))

    mapping_parallel = Parallel(
        name="Build Map",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
    )
    mapping_parallel.add_children(
        [
            RunMapping("Update Map", blackboard),
            MoveTable("Drive Around Counter", blackboard),
        ]
    )
    mapping_selector.add_child(mapping_parallel)

    # 3. Return Home
    return_home = ReturnHome("Return to Origin", blackboard)

    # 4. Handle Jars
    root.add_children([init_seq, mapping_selector, return_home])

    for i in range(3):
        jar_seq = Sequence(name=f"Handle Jar {i + 1}", memory=True)

        # 4a. Find object
        find_selector = Selector(name=f"Find Object {i + 1}", memory=True)
        find_selector.add_children(
            [
                ObjectRecognizer(f"Recognize {i + 1}", blackboard, timeout=3.0),
                ComprehensiveScanner(
                    f"Scan {i + 1}", blackboard, total_angles=8, angle_increment=45
                ),
            ]
        )

        # 4b. Approach
        approach_seq = Sequence(name=f"Approach {i + 1}", memory=True)
        prepare_arm = MoveToPosition(f"Prepare Arm {i + 1}", blackboard, LIFT_POSITION)
        move_arm_ik = MoveArmIK(f"MoveArmIK {i + 1}", blackboard, offset_y=Y_OFFSETS[i])
        move_to_obj = MoveToObject(
            f"MoveToObject {i + 1}", blackboard, move_arm_behaviour=move_arm_ik
        )
        approach_seq.add_children([prepare_arm, move_to_obj])

        # 4c. Grasp
        grasp = GraspController(f"Grasp {i + 1}", blackboard, force_threshold=-10.0)

        # 4d. Transport & Place
        transport_seq = Sequence(name=f"Transport {i + 1}", memory=True)
        transport_seq.add_children(
            [
                LiftAndVerify(f"Lift {i + 1}", blackboard, LIFT_POSITION),
                BackupAfterGrasp(f"Backup {i + 1}", blackboard),
                MoveToWaypoint(f"To Table {i + 1}", blackboard, TABLE_WAYPOINTS),
                MoveToPosition(
                    f"Place Arm {i + 1}", blackboard, PLACE_POSITION, timeout=8.0
                ),
                OpenGripper(f"Release {i + 1}", blackboard),
                MoveToPosition(
                    f"Reset Arm {i + 1}", blackboard, STARTING_POSITION, timeout=4.0
                ),
                MoveToWaypoint(f"To Home {i + 1}", blackboard, HOME_WAYPOINT),
            ]
        )

        jar_seq.add_children([find_selector, approach_seq, grasp, transport_seq])
        root.add_child(jar_seq)

    tree = py_trees.trees.BehaviourTree(root)
    tree.visitors.append(py_trees.visitors.DebugVisitor())
    return tree


def main():
    """Entry point: initialise robot, build the behaviour tree, and run the main loop."""
    robot = TiagoFull()
    blackboard = Blackboard()
    blackboard.write("robot", robot)

    tree = create_behavior_tree(blackboard)
    tree.setup(timeout=15)

    # Run a few steps to let sensors stabilise before the first BT tick
    for _ in range(10):
        robot.step()
    robot.update_odometry()

    gps_pos = robot.gps.getValues()
    import math as _math

    heading = _math.degrees(
        _math.atan2(robot.compass.getValues()[0], robot.compass.getValues()[1])
    )
    print(f"Start GPS : ({gps_pos[0]:.3f}, {gps_pos[1]:.3f}, {gps_pos[2]:.3f})")
    print(f"Start heading: {heading:.1f}°")

    last_print = robot.robot.getTime()
    PRINT_INTERVAL = 10.0

    while robot.step():
        robot.update_odometry()
        tree.tick()

        status = tree.root.status
        now = robot.robot.getTime()

        if now - last_print >= PRINT_INTERVAL:
            print("\n" + py_trees.display.ascii_tree(tree.root))
            last_print = now

        if status == py_trees.common.Status.SUCCESS:
            print("Mission complete — all jars placed!")
            break
        if status == py_trees.common.Status.FAILURE:
            print("Mission FAILED.")
            break

    robot.set_wheel_velocity(0.0, 0.0)


if __name__ == "__main__":
    main()
