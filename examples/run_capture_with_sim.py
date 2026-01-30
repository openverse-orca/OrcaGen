#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双进程采集：仿真与采集分两个进程，避免单进程下 step 与采集同进程导致的运动顿挫。

- 无外部仿真时：本脚本先启动 run_sim_loop，等 gRPC 就绪后再启动 run_capture --external_drive。
- 有外部仿真时：加 --external_sim，不启动 run_sim_loop，只执行 run_capture --external_drive。

用法:
  python examples/run_capture_with_sim.py --object_sites_groups "Bin1|Bin2|Bin3|Bin4" --duration_s 20 --auto_frame_skip
  python examples/run_capture_with_sim.py --external_sim --object_sites_groups "..." --duration_s 20  # 仿真已由外部启动
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

# 本脚本专用参数：解析后不传给 run_capture
LAUNCHER_OPTS = {
    "--sim_loop_cmd": True,   # 带值
    "--sim_loop_wait_s": True,
    "--no_kill_sim_on_exit": False,
    "--orcagym_addr": True,
    "--external_sim": False,
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_for_grpc(addr: str, timeout_s: float, interval_s: float = 0.5) -> bool:
    import socket
    host, port_str = addr.split(":")
    port = int(port_str)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            s = socket.socket()
            s.settimeout(1.0)
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            time.sleep(interval_s)
    return False


def _build_capture_argv() -> list[str]:
    """从 sys.argv 中去掉本脚本专用参数，得到传给 run_capture 的 argv（已含 --external_drive）。"""
    argv = [sys.executable, "-m", "examples.run_capture", "--external_drive"]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in LAUNCHER_OPTS:
            takes_value = LAUNCHER_OPTS[arg]
            i += 2 if takes_value and i + 1 < len(sys.argv) else 1
            continue
        # 兼容 --sim_loop_xxx / --no_kill_sim 等变体
        if arg.startswith("--sim_loop") or arg.startswith("--no_kill_sim"):
            i += 2 if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-") else 1
            continue
        argv.append(arg)
        i += 1
    return argv


def main() -> None:
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    parser = argparse.ArgumentParser(
        description="双进程采集：先启仿真（或 --external_sim），再启采集 --external_drive",
    )
    parser.add_argument(
        "--sim_loop_cmd",
        default=os.getenv("ORCAGEN_SIM_LOOP_CMD", f"{sys.executable} -m scripts.run_sim_loop_orcagen"),
        help="仿真循环命令（默认: <当前python> -m scripts.run_sim_loop_orcagen；该定制版负责节拍，建议 OrcaStudio 解锁步进/视频帧）",
    )
    parser.add_argument(
        "--sim_loop_wait_s",
        type=float,
        default=float(os.getenv("ORCAGEN_SIM_LOOP_WAIT_S", "15")),
        help="等待 gRPC 就绪的最长时间（秒）",
    )
    parser.add_argument(
        "--no_kill_sim_on_exit",
        action="store_true",
        help="采集结束后不结束仿真进程",
    )
    parser.add_argument(
        "--orcagym_addr",
        default="localhost:50051",
        help="OrcaGym gRPC 地址",
    )
    parser.add_argument(
        "--external_sim",
        action="store_true",
        help="仿真已由外部启动，不启动 run_sim_loop，仅执行 run_capture --external_drive",
    )
    args, _ = parser.parse_known_args()

    capture_argv = _build_capture_argv()
    cwd = _project_root()
    env = os.environ.copy()
    env["ORCAGEN_NO_INTERACTIVE"] = "1"

    # 外部仿真：只跑采集
    if args.external_sim:
        print("[OrcaGen] --external_sim：不启动 run_sim_loop，直接启动采集")
        ret = subprocess.run(capture_argv, cwd=cwd, env=env)
        sys.exit(ret.returncode)

    # 先启仿真，等 gRPC 就绪后再启采集
    sim_proc = None
    try:
        cmd = shlex.split(args.sim_loop_cmd) if isinstance(args.sim_loop_cmd, str) else args.sim_loop_cmd
        print("[OrcaGen] 启动仿真:", " ".join(cmd))
        sim_proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        print("[OrcaGen] 等待 gRPC:", args.orcagym_addr, f"(最多 {args.sim_loop_wait_s}s)")
        if not _wait_for_grpc(args.orcagym_addr, args.sim_loop_wait_s):
            err = sim_proc.stderr.read().decode("utf-8", errors="replace") if sim_proc.stderr else ""
            print("[OrcaGen] 错误：gRPC 未就绪")
            if err:
                print("[OrcaGen] 仿真 stderr:", err[:400])
            try:
                sim_proc.terminate()
                sim_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                sim_proc.kill()
            sys.exit(1)
        print("[OrcaGen] 启动采集 (external_drive)")
        ret = subprocess.run(capture_argv, cwd=cwd, env=env)
        sys.exit(ret.returncode)
    finally:
        if sim_proc is not None and not args.no_kill_sim_on_exit:
            try:
                sim_proc.terminate()
                sim_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                sim_proc.kill()
            except Exception:
                pass
            print("[OrcaGen] 已结束仿真进程")


if __name__ == "__main__":
    main()
