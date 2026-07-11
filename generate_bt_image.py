"""
Generate one PNG per sub-tree of the TIAGo mission behaviour tree.
Output files go to docs/bt_parts/.
"""

import os
import pydot
import py_trees
from py_trees.composites import Parallel, Selector, Sequence

OUT_DIR = "/home/duyennh/Werobots_project/Final_project/docs/bt_parts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────

GRAPH_ATTRS = dict(
    bgcolor="white",
    fontname="Helvetica",
    fontsize="15",
    rankdir="TB",
    ranksep="0.55",
    nodesep="0.45",
    pad="0.5",
)
NODE_ATTRS = dict(fontname="Helvetica", fontsize="14")
EDGE_ATTRS = dict(color="#444444", penwidth="1.4")


def render(root_node, filename):
    """Render a subtree rooted at root_node to docs/bt_parts/<filename>.png"""
    graph = py_trees.display.dot_tree(
        root_node,
        visibility_level=py_trees.common.VisibilityLevel.ALL,
    )
    graph.set_graph_defaults(**GRAPH_ATTRS)
    graph.set_node_defaults(**NODE_ATTRS)
    graph.set_edge_defaults(**EDGE_ATTRS)
    path = os.path.join(OUT_DIR, filename)
    graph.write_png(path)
    print(f"  ✓  {path}")


def leaf(label):
    return py_trees.behaviours.Dummy(name=label)


# ── 1. Initialization ─────────────────────────────────────────────────────────
init_seq = Sequence(name="Initialization", memory=True)
init_seq.add_children([
    leaf("Check Hardware"),
    leaf("Safe Start Position"),
])
render(init_seq, "01_initialization.png")

# ── 2. Mapping Phase ──────────────────────────────────────────────────────────
mapping_sel = Selector(name="Mapping Phase", memory=False)
mapping_sel.add_child(leaf("Map already exists?"))
build_map = Parallel(
    name="Build Map",
    policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
)
build_map.add_children([
    leaf("Update Map\n(LiDAR probabilistic)"),
    leaf("Drive Around Counter\n(forward + backward waypoints)"),
])
mapping_sel.add_child(build_map)
render(mapping_sel, "02_mapping_phase.png")

# ── 3. Return to Origin ───────────────────────────────────────────────────────
rh = Sequence(name="Return to Origin", memory=True)
rh.add_children([
    leaf("Orient toward (0,0)"),
    leaf("Drive to (0,0)"),
    leaf("Align to 0°"),
])
render(rh, "03_return_home.png")

# ── 4-6. Handle Jar 1-3 ──────────────────────────────────────────────────────
Y_OFFSETS = [0.13, -0.80, -0.6]

for i in range(3):
    n = i + 1
    jar_seq = Sequence(name=f"Handle Jar {n}", memory=True)

    find_sel = Selector(name=f"Find Object {n}", memory=True)
    find_sel.add_children([
        leaf(f"Camera Recognize {n}"),
        leaf(f"360° Rotation Scan {n}\n(8 × 45°)"),
    ])

    approach_seq = Sequence(name=f"Approach {n}", memory=True)
    approach_seq.add_children([
        leaf(f"Prepare Arm {n}\n(lift position)"),
        leaf(f"Move to Object {n}\n(Nav + IK pre-position)"),
    ])

    grasp = leaf(f"Grasp {n}\n(close gripper, verify force)")

    transport_seq = Sequence(name=f"Transport {n}", memory=True)
    transport_seq.add_children([
        leaf(f"Lift & Verify {n}\n(check grip force)"),
        leaf(f"Backup {n}\n(reverse 12 cm)"),
        leaf(f"Navigate to Table {n}\n(waypoints)"),
        leaf(f"Place Arm {n}\n(place position)"),
        leaf(f"Release {n}\n(open gripper)"),
        leaf(f"Reset Arm {n}\n(start position)"),
        leaf(f"Return to Home {n}\n(home waypoint)"),
    ])

    jar_seq.add_children([find_sel, approach_seq, grasp, transport_seq])
    render(jar_seq, f"0{3+n}_handle_jar_{n}.png")

# ── 7. Full overview (root only, collapsed) ───────────────────────────────────
overview = Sequence(name="TIAGo Mission", memory=True)
overview.add_children([
    leaf("① Initialization"),
    leaf("② Mapping Phase"),
    leaf("③ Return to Origin"),
    leaf("④ Handle Jar 1"),
    leaf("⑤ Handle Jar 2"),
    leaf("⑥ Handle Jar 3"),
])
render(overview, "00_overview.png")

print("\nAll images generated in", OUT_DIR)
