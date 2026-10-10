"""Sonar point-cloud and trajectory textures (Script TOP helpers). Every array is exported by scripts/build_map.py and
memory-mapped by data_bridge. Slot order is identical in the DR / SLAM / attribute textures, so the vertex shader can
blend corrected and uncorrected positions of the same sounding."""
import numpy as np

BASE = "/project1/DEEP_OCEAN_ROBOT_VISION_4D"
KINDS = {"points_dr": "map_points_dr", "points_slam": "map_points_slam", "attr": "map_attr",
         "traj_dr": "traj_dr", "traj_slam": "traj_slam", "traj_truth": "traj_truth"}


def texture(base, kind):
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    return np.ascontiguousarray(np.asarray(st.array(KINDS[kind]), dtype=np.float32))


def cook(scriptOp, kind):
    """Script TOP onCook body. Keeps the upload cheap: the same array is only copied again if the dataset changed."""
    import td
    base = td.op(BASE)
    arr = texture(base, kind)
    scriptOp.par.outputresolution = "custom"
    scriptOp.par.resolutionw = arr.shape[1]
    scriptOp.par.resolutionh = arr.shape[0]
    scriptOp.copyNumpyArray(arr)
    return arr.shape


def point_counts(base):
    st = base.op("DATA_INPUT/data_bridge").module.get_store(base)
    return int(st.meta.get("display_points", 0))
