#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 OrcaEditor/OrcaSim 的相机 WebSocket 流录制视频（默认端口 7070）。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import websockets


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _h264_to_mp4(h264_path: str, mp4_path: str, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-r",
        str(fps),
        "-i",
        h264_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]
    subprocess.check_call(cmd)


@dataclass(frozen=True)
class RecordStats:
    packets: int
    bytes_video: int
    duration_s: float


async def record_ws_to_files(
    uri: str,
    ts_path: str,
    h264_path: str,
    duration_s: float,
    *,
    recv_timeout_s: float = 1.0,
    wait_first_packet_s: float = 5.0,
) -> RecordStats:
    start = time.time()
    packets = 0
    bytes_video = 0

    async with websockets.connect(uri, max_size=None) as ws:
        with open(ts_path, "wb") as fts, open(h264_path, "wb") as fvid:
            have_sps = False
            have_pps = False

            def _scan_sps_pps(payload: bytes) -> tuple[bool, bool]:
                sps = b"\x00\x00\x00\x01\x67" in payload or b"\x00\x00\x01\x67" in payload
                pps = b"\x00\x00\x00\x01\x68" in payload or b"\x00\x00\x01\x68" in payload
                return sps, pps

            t0 = time.time()
            while True:
                if time.time() - t0 >= wait_first_packet_s:
                    return RecordStats(packets=0, bytes_video=0, duration_s=0.0)
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=min(0.5, wait_first_packet_s))
                except asyncio.TimeoutError:
                    continue
                if not isinstance(data, (bytes, bytearray)) or len(data) <= 8:
                    continue
                payload = bytes(data[8:])
                sps, pps = _scan_sps_pps(payload)
                have_sps = have_sps or sps
                have_pps = have_pps or pps
                if have_sps and have_pps:
                    fts.write(data[:8])
                    fvid.write(payload)
                    packets += 1
                    bytes_video += len(payload)
                    start = time.time()
                    break

            while True:
                elapsed = time.time() - start
                if elapsed >= duration_s:
                    break
                timeout = min(recv_timeout_s, max(0.0, duration_s - elapsed))
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                if not isinstance(data, (bytes, bytearray)):
                    continue
                if len(data) <= 8:
                    continue
                fts.write(data[:8])
                fvid.write(data[8:])
                packets += 1
                bytes_video += len(data) - 8

    return RecordStats(packets=packets, bytes_video=bytes_video, duration_s=min(duration_s, time.time() - start))


def record_ws_to_files_sync(
    uri: str,
    ts_path: str,
    h264_path: str,
    duration_s: float,
    *,
    recv_timeout_s: float = 1.0,
    wait_first_packet_s: float = 5.0,
) -> RecordStats:
    return asyncio.run(
        record_ws_to_files(
            uri,
            ts_path,
            h264_path,
            duration_s,
            recv_timeout_s=recv_timeout_s,
            wait_first_packet_s=wait_first_packet_s,
        )
    )


def write_ws_video_in_sequence(
    *,
    sequence_dir: str,
    name: str,
    host: str = "localhost",
    port: int = 7070,
    duration_s: float = 30.0,
    fps: int = 30,
    to_mp4: bool = True,
    recv_timeout_s: float = 1.0,
    wait_first_packet_s: float = 5.0,
) -> dict:
    video_dir = os.path.join(sequence_dir, "video")
    _mkdirp(video_dir)
    base = os.path.join(video_dir, name)
    ts_path = base + "_ts.bin"
    h264_path = base + "_video.h264"
    mp4_path = os.path.join(video_dir, f"{name}.mp4")
    uri = f"ws://{host}:{port}"

    stats = record_ws_to_files_sync(
        uri,
        ts_path,
        h264_path,
        duration_s,
        recv_timeout_s=recv_timeout_s,
        wait_first_packet_s=wait_first_packet_s,
    )

    out = {
        "uri": uri,
        "ts_path": ts_path,
        "h264_path": h264_path,
        "mp4_path": mp4_path if to_mp4 else None,
        "packets": stats.packets,
        "bytes_video": stats.bytes_video,
        "duration_s": stats.duration_s,
    }

    if to_mp4:
        if not _ffmpeg_exists():
            raise SystemExit("[OrcaGen] ffmpeg 不存在，无法转 mp4。")
        _h264_to_mp4(h264_path, mp4_path, fps=fps)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7070)
    ap.add_argument("--duration_s", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--name", default="Camera")
    ap.add_argument("--output_root", default="/home/guojiatao/OrcaWorkStation/OrcaGen")
    ap.add_argument("--sequence_id", default=None)
    ap.add_argument("--to_mp4", action="store_true")
    ap.add_argument("--wait_first_packet_s", type=float, default=5.0)
    ap.add_argument("--recv_timeout_s", type=float, default=1.0)
    ap.add_argument("--min_packets_to_mp4", type=int, default=10)
    args = ap.parse_args()

    seq_id = args.sequence_id or f"sequence_ws_video_{args.name}_{_now_id()}"
    seq_dir = os.path.join(args.output_root, seq_id)
    video_dir = os.path.join(seq_dir, "video")
    _mkdirp(video_dir)

    base = os.path.join(video_dir, args.name)
    ts_path = base + "_ts.bin"
    h264_path = base + "_video.h264"
    mp4_path = os.path.join(video_dir, f"{args.name}.mp4")

    uri = f"ws://{args.host}:{args.port}"
    print("[OrcaGen] recording ws video:", uri)
    print("[OrcaGen] duration_s:", args.duration_s)
    print("[OrcaGen] output:")
    print(" -", ts_path)
    print(" -", h264_path)

    stats = record_ws_to_files_sync(
        uri,
        ts_path,
        h264_path,
        args.duration_s,
        recv_timeout_s=args.recv_timeout_s,
        wait_first_packet_s=args.wait_first_packet_s,
    )
    print(f"[OrcaGen] packets={stats.packets}, bytes_video={stats.bytes_video}, recorded_s={stats.duration_s:.3f}")
    if stats.packets <= 1:
        print("[OrcaGen] 警告：只收到很少的数据包，mp4 可能只有 1 帧，看起来会像“空视频”。")

    if args.to_mp4:
        if not _ffmpeg_exists():
            raise SystemExit("[OrcaGen] ffmpeg 不存在，无法 --to_mp4。请先安装 ffmpeg 或取消该参数。")
        if stats.packets < args.min_packets_to_mp4:
            print(f"[OrcaGen] 跳过 mp4 转码：packets={stats.packets} < min_packets_to_mp4={args.min_packets_to_mp4}")
        else:
            _h264_to_mp4(h264_path, mp4_path, fps=args.fps)
            print("[OrcaGen] mp4:", mp4_path)

    print("[OrcaGen] OK. sequence_dir:", seq_dir)


if __name__ == "__main__":
    main()

