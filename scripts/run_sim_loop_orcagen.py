#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaGen 定制版 run_sim_loop（负责节拍）。

目标：
- 由该脚本成为唯一“节拍源”（类似 OrcaManipulation 的 DataCollectionManager：step + render + sleep 对齐）。
- OrcaStudio/OrcaSim 端建议：不要再锁步进/锁视频帧（避免双时钟节流导致时长翻倍/进度滞后）。

用法示例：
  python -m scripts.run_sim_loop_orcagen --orcagym_addr localhost:50051
  python -m scripts.run_sim_loop_orcagen --time_step 0.001 --frame_skip 20 --target_fps 30
  python -m scripts.run_sim_loop_orcagen --no_render
  python -m scripts.run_sim_loop_orcagen --render_every 2   # 每 2 步尝试 render 一次（仍受 render_fps 节流）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np


ENV_ENTRY_POINT = {
    "SimulationLoop": "orca_gym.scripts.sim_env:SimEnv",
}


def _register_env(
    *,
    orcagym_addr: str,
    env_name: str,
    env_index: int,
    agent_name: str,
    max_episode_steps: int,
    time_step: float,
    frame_skip: int,
) -> str:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {
        "frame_skip": int(frame_skip),
        "orcagym_addr": orcagym_addr,
        "agent_names": agent_names,
        "time_step": float(time_step),
    }
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id


def _configure_render_throttle(env, render_fps: Optional[float], sync_render: Optional[bool]) -> None:
    # Best-effort: tune OrcaGymLocalEnv internal throttling fields.
    if render_fps is not None:
        try:
            env.metadata["render_fps"] = int(render_fps)
        except Exception:
            pass
        try:
            rfps = float(render_fps)
            if hasattr(env, "_render_interval"):
                env._render_interval = 1.0 / max(rfps, 1e-6)
            if hasattr(env, "_render_count_interval") and hasattr(env, "realtime_step"):
                env._render_count_interval = float(env.realtime_step) * rfps
        except Exception:
            pass
    if sync_render is not None:
        try:
            env._sync_render = bool(sync_render)
        except Exception:
            pass


def run() -> None:
    ap = argparse.ArgumentParser(description="OrcaGen 定制 run_sim_loop（负责节拍）")
    ap.add_argument("--orcagym_addr", default="localhost:50051")
    ap.add_argument("--env_name", default="SimulationLoop", choices=list(ENV_ENTRY_POINT.keys()))
    ap.add_argument("--env_index", type=int, default=0)
    ap.add_argument("--agent_name", default="NoRobot")
    ap.add_argument("--time_step", type=float, default=0.001)
    ap.add_argument("--frame_skip", type=int, default=20)
    ap.add_argument("--target_fps", type=float, default=None, help="若指定则按该 fps 节拍 sleep；否则用 time_step*frame_skip")
    ap.add_argument("--render", dest="render", action="store_true", default=True, help="启用 env.render（默认启用）")
    ap.add_argument("--no_render", dest="render", action="store_false", help="禁用 env.render（不产生相机帧时可用）")
    ap.add_argument("--render_every", type=int, default=1, help="每 N 步尝试 render 一次（默认 1）")
    ap.add_argument("--render_fps", type=float, default=30.0, help="用于 OrcaGymLocalEnv 的 render 节流（默认 30）")
    ap.add_argument("--sync_render", action="store_true", default=None, help="强制同步渲染节流模式（可选）")
    ap.add_argument("--async_render", action="store_true", default=None, help="强制异步渲染节流模式（可选）")
    ap.add_argument("--max_episode_steps", type=int, default=2**31 - 1)
    ap.add_argument("--log_every_s", type=float, default=5.0)
    args = ap.parse_args()

    # Resolve sync_render flag
    sync_render: Optional[bool] = None
    if args.sync_render:
        sync_render = True
    if args.async_render:
        sync_render = False

    env_id = _register_env(
        orcagym_addr=args.orcagym_addr,
        env_name=args.env_name,
        env_index=int(args.env_index),
        agent_name=args.agent_name,
        max_episode_steps=int(args.max_episode_steps),
        time_step=float(args.time_step),
        frame_skip=int(args.frame_skip),
    )

    # Ensure render_mode is enabled when args.render=True
    render_mode = "human" if args.render else "none"
    env = gym.make(env_id, render_mode=render_mode)
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    _configure_render_throttle(env, args.render_fps if args.render else None, sync_render)

    # Step pacing
    realtime_step = float(args.time_step) * int(args.frame_skip)
    target_dt = (1.0 / float(args.target_fps)) if args.target_fps else realtime_step
    target_dt = max(target_dt, 1e-6)
    print("[OrcaGen] run_sim_loop_orcagen")
    print(" - orcagym_addr:", args.orcagym_addr)
    print(" - time_step:", args.time_step, "frame_skip:", args.frame_skip, "realtime_step:", realtime_step)
    print(" - target_dt:", target_dt, "render:", bool(args.render), "render_every:", int(args.render_every))
    print(" - 建议：OrcaStudio 不锁步进/不锁视频帧（避免双时钟）")

    obs, _ = env.reset()
    t_last_log = time.time()
    steps = 0
    try:
        while True:
            t0 = time.time()
            # Minimal action: zeros if possible
            try:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            except Exception:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if args.render and (steps % max(int(args.render_every), 1) == 0):
                try:
                    env.render()
                except Exception:
                    pass

            elapsed = time.time() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            now = time.time()
            if args.log_every_s > 0 and now - t_last_log >= float(args.log_every_s):
                hz = steps / max(1e-6, (now - (t_last_log - float(args.log_every_s))))
                print(f"[OrcaGen] sim_loop steps={steps}, last_step_elapsed={elapsed*1000:.1f}ms")
                t_last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    run()

