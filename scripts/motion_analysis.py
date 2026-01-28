#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 OrcaGen 的 metadata.jsonl 进行运动模式推断（1~2 个物体）。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np

from orcagen.enums.motion import MotionType


@dataclass
class MotionStats:
    speed_med: float
    speed_p90: float
    speed_std: float
    omega_med: float
    omega_p90: float
    contact_ratio: float
    z_trend: float
    sign_changes_xy: int
    angle_sign_changes: int
    angle_range_deg: float


def _load_tracks(metadata_path: str, object_id: str) -> Tuple[List[float], List[np.ndarray], List[np.ndarray], List[bool]]:
    ts: List[float] = []
    pos: List[np.ndarray] = []
    ang: List[np.ndarray] = []
    contact: List[bool] = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ann = None
            for a in rec.get("annotations", []):
                if a.get("object_id") == object_id:
                    ann = a
                    break
            if ann is None:
                continue
            bbox = ann.get("bbox3d") or {}
            pos.append(np.array([bbox.get("x_center", 0.0), bbox.get("y_center", 0.0), bbox.get("z_center", 0.0)], dtype=np.float64))
            ang.append(np.array([bbox.get("pitch", 0.0), bbox.get("yaw", 0.0), bbox.get("roll", 0.0)], dtype=np.float64) * 180.0)
            ts.append(float(rec.get("timestamp_s", 0.0)))

            c = rec.get("contacts", {})
            has_contact = bool(c.get("has_contact", False))
            if isinstance(c.get("pairs"), list):
                for p in c["pairs"]:
                    if isinstance(p, list) and object_id in p:
                        has_contact = True
                        break
            contact.append(has_contact)

    return ts, pos, ang, contact


def _compute_stats(ts: List[float], pos: List[np.ndarray], ang: List[np.ndarray], contact: List[bool]) -> MotionStats:
    if len(ts) < 3:
        return MotionStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    speeds = []
    omegas = []
    z_vals = [p[2] for p in pos]
    z_trend = float(z_vals[-1] - z_vals[0])
    dx_sign_changes = 0
    dy_sign_changes = 0
    last_dx = 0.0
    last_dy = 0.0
    ang_sign_changes = 0
    last_da = np.zeros(3, dtype=np.float64)

    for i in range(1, len(ts)):
        dt = max(1e-6, ts[i] - ts[i - 1])
        dp = pos[i] - pos[i - 1]
        da = ang[i] - ang[i - 1]
        speeds.append(float(np.linalg.norm(dp) / dt))
        omegas.append(float(np.linalg.norm(da) / dt))

        dx, dy = dp[0], dp[1]
        if i >= 2:
            if dx * last_dx < 0:
                dx_sign_changes += 1
            if dy * last_dy < 0:
                dy_sign_changes += 1
        last_dx, last_dy = dx, dy

        if i >= 2:
            for k in range(3):
                if da[k] * last_da[k] < 0:
                    ang_sign_changes += 1
        last_da = da

    contact_ratio = float(np.mean(contact)) if contact else 0.0
    sign_changes_xy = int(dx_sign_changes + dy_sign_changes)
    angle_range_deg = float(np.max(ang, axis=0).max() - np.min(ang, axis=0).min())

    speeds_np = np.array(speeds, dtype=np.float64)
    omegas_np = np.array(omegas, dtype=np.float64)
    return MotionStats(
        speed_med=float(np.median(speeds_np)),
        speed_p90=float(np.percentile(speeds_np, 90)),
        speed_std=float(np.std(speeds_np)) if len(speeds_np) > 0 else 0.0,
        omega_med=float(np.median(omegas_np)),
        omega_p90=float(np.percentile(omegas_np, 90)),
        contact_ratio=contact_ratio,
        z_trend=z_trend,
        sign_changes_xy=sign_changes_xy,
        angle_sign_changes=ang_sign_changes,
        angle_range_deg=angle_range_deg,
    )


def infer_motion_type(metadata_path: str, object_id: str) -> MotionType:
    ts, pos, ang, contact = _load_tracks(metadata_path, object_id)
    stats = _compute_stats(ts, pos, ang, contact)

    speed_small = 0.01
    speed_move = 0.03
    omega_small = 5.0
    omega_move = 15.0
    pendulum_angle_range = 15.0

    if stats.speed_med < speed_small and stats.omega_med < omega_small:
        return MotionType.STATIC

    # 匀速直线运动判断：速度较大且稳定，方向变化少，角速度小
    # 速度一致性：速度标准差相对于中位数较小（变异系数 < 0.3）
    speed_cv = stats.speed_std / max(stats.speed_med, 1e-6)  # 变异系数
    is_uniform_speed = speed_cv < 0.3 and stats.speed_med > speed_move
    is_linear_direction = stats.sign_changes_xy <= 2  # 方向变化很少
    is_low_rotation = stats.omega_med < omega_small
    
    if is_uniform_speed and is_linear_direction and is_low_rotation:
        return MotionType.UNIFORM_LINEAR

    if stats.sign_changes_xy >= 4 and stats.speed_med > speed_small:
        return MotionType.OSCILLATING

    if stats.speed_med < speed_move and stats.omega_med > omega_move:
        if stats.angle_sign_changes >= 6 and stats.angle_range_deg >= pendulum_angle_range:
            return MotionType.PENDULUM

    if stats.contact_ratio < 0.2 and stats.z_trend < -0.05:
        return MotionType.FREE_FALL

    if stats.contact_ratio >= 0.2:
        if stats.speed_med > speed_move and stats.omega_med > omega_move:
            return MotionType.ROLLING
        if stats.speed_med > speed_move and stats.omega_med <= omega_move:
            return MotionType.SLIDING
        if stats.speed_med <= speed_move and stats.omega_med > omega_move:
            return MotionType.ROTATING_IN_PLACE

    return MotionType.UNKNOWN


def infer_motion_types(metadata_path: str, object_ids: Iterable[str]) -> Dict[str, MotionType]:
    out: Dict[str, MotionType] = {}
    for oid in object_ids:
        out[oid] = infer_motion_type(metadata_path, oid)
    return out

