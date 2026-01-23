#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 metadata.jsonl 生成运动曲线图（位置/姿态随时间变化）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

try:
    from scipy.spatial.transform import Rotation as SciRot
except Exception:
    SciRot = None


def _load_series(metadata_path: str) -> Dict[str, Dict[str, List[float]]]:
    series: Dict[str, Dict[str, List[float]]] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = float(rec.get("timestamp_s", 0.0))
            for ann in rec.get("annotations", []):
                oid = ann.get("object_id", "object")
                bbox = ann.get("bbox3d") or {}
                if oid not in series:
                    series[oid] = {
                        "t": [],
                        "x": [],
                        "y": [],
                        "z": [],
                        "pitch": [],
                        "yaw": [],
                        "roll": [],
                        "sx": [],
                        "sy": [],
                        "sz": [],
                    }
                s = series[oid]
                s["t"].append(t)
                s["x"].append(float(bbox.get("x_center", 0.0)))
                s["y"].append(float(bbox.get("y_center", 0.0)))
                s["z"].append(float(bbox.get("z_center", 0.0)))
                s["pitch"].append(float(bbox.get("pitch", 0.0)) * 180.0)
                s["yaw"].append(float(bbox.get("yaw", 0.0)) * 180.0)
                s["roll"].append(float(bbox.get("roll", 0.0)) * 180.0)
                s["sx"].append(float(bbox.get("x_size", 0.0)))
                s["sy"].append(float(bbox.get("y_size", 0.0)))
                s["sz"].append(float(bbox.get("z_size", 0.0)))
    return series


def _plot_one(oid: str, s: Dict[str, List[float]], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(s["t"], dtype=np.float64)
    if t.size == 0:
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, s["x"], label="x")
    ax.plot(t, s["y"], label="y")
    ax.plot(t, s["z"], label="z")
    ax.set_title(f"Position vs Time ({oid})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{oid}_position.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, s["pitch"], label="pitch")
    ax.plot(t, s["yaw"], label="yaw")
    ax.plot(t, s["roll"], label="roll")
    ax.set_title(f"Euler Angles vs Time ({oid})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{oid}_euler.png"), dpi=150)
    plt.close(fig)

    if SciRot is None:
        return
    try:
        sizes = np.stack([s["sx"], s["sy"], s["sz"]], axis=1)
        axis_idx = int(np.argmax(np.mean(sizes, axis=0)))
        half_len = 0.5 * np.mean(sizes[:, axis_idx])
        axis_local = np.zeros(3, dtype=np.float64)
        axis_local[axis_idx] = 1.0

        tips = []
        for i in range(len(t)):
            R = SciRot.from_euler("xyz", [s["pitch"][i], s["yaw"][i], s["roll"][i]], degrees=True).as_matrix()
            center = np.array([s["x"][i], s["y"][i], s["z"][i]], dtype=np.float64)
            tip = center + R @ (axis_local * half_len)
            tips.append(tip)
        tips = np.array(tips, dtype=np.float64)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, tips[:, 0], label="tip_x")
        ax.plot(t, tips[:, 1], label="tip_y")
        ax.plot(t, tips[:, 2], label="tip_z")
        ax.set_title(f"Tip Position vs Time ({oid})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (m)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{oid}_tip_position.png"), dpi=150)
        plt.close(fig)
    except Exception:
        return


def plot_motion_curves(metadata_jsonl: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    series = _load_series(metadata_jsonl)
    for oid, s in series.items():
        _plot_one(oid, s, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    plot_motion_curves(args.metadata_jsonl, args.out_dir)
    print(f"[OrcaGen] motion curves saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

