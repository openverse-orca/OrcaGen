from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from orcagen.core.types import Pose


def get_body_pose(env, body_name: str) -> Pose:
    xpos, xmat, xquat = env.get_body_xpos_xmat_xquat([body_name])
    pos = np.asarray(xpos, dtype=np.float64).reshape(3)
    mat = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    quat = np.asarray(xquat, dtype=np.float64).reshape(4)
    return Pose(pos=pos, mat=mat, quat=quat)


def get_site_pose(env, site_name: str) -> Pose:
    site_dict = env.query_site_pos_and_mat([site_name])
    if site_name not in site_dict:
        raise RuntimeError(f"query_site_pos_and_mat 未返回 {site_name}")
    pos = np.asarray(site_dict[site_name]["xpos"], dtype=np.float64).reshape(3)
    mat = np.asarray(site_dict[site_name]["xmat"], dtype=np.float64).reshape(3, 3)
    quat = SciRot.from_matrix(mat).as_quat()  # x,y,z,w
    quat = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
    return Pose(pos=pos, mat=mat, quat=quat)


def worldsim_to_world0(p_worldsim: np.ndarray, R0: np.ndarray, t0: np.ndarray) -> np.ndarray:
    return R0.T @ (p_worldsim - t0)


def mat_to_euler_xyz_deg(m: np.ndarray) -> Tuple[float, float, float]:
    e = SciRot.from_matrix(m).as_euler("xyz", degrees=True)
    return float(e[0]), float(e[1]), float(e[2])


def norm_angle_deg_to_unit(a_deg: float) -> float:
    v = a_deg / 180.0
    if v >= 1.0:
        return 0.999
    if v <= -1.0:
        return -0.999
    return float(v)

