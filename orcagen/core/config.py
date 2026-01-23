from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class CaptureConfig:
    orcagym_addr: str = "localhost:50051"
    agent_name: str = "NoRobot"
    duration_s: float = 30.0
    fps: int = 30
    resolution: str = "2560x1440"
    render_style: str = "PBR"
    time_step_s: float = 0.001
    frame_skip: int = 20
    auto_frame_skip: bool = False
    sync_time_step_with_server: bool = True
    object_source: str = "site"
    object_body: str = "Sphere1"
    object_bodies: Optional[str] = None
    object_site: Optional[str] = None
    object_sites: Optional[str] = None
    object_ids: Optional[str] = None
    camera_body: Optional[str] = None
    camera_name: str = "Camera"
    capture_mode: str = "ASYNC"
    save_video: bool = False
    no_drive_sim: bool = False
    no_reset: bool = False
    no_render: bool = True
    render_fps: int = 30
    use_realtime_loop: bool = True
    contacts_mode: str = "assume_ground"
    ws_video: bool = False
    ws_video_port: int = 7070
    ws_video_name: str = "Camera"
    ws_video_to_mp4: bool = False
    ws_video_wait_first_packet_s: float = 5.0
    ws_video_recv_timeout_s: float = 1.0
    ws_video_min_packets_to_mp4: int = 10
    sequence_id: Optional[str] = None
    output_root: str = "/home/guojiatao/OrcaWorkStation/OrcaGen"
    infer_motion: bool = True
    plot_motion: bool = True
    normalize_video: bool = True
    external_drive: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CaptureConfig":
        values = vars(args)
        # 只保留 CaptureConfig 支持的字段，避免 argparse 扩展字段报错
        allowed = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        cfg = cls(**allowed)
        if getattr(args, "no_sync_time_step_with_server", False):
            cfg.sync_time_step_with_server = False
        if getattr(args, "no_infer_motion", False):
            cfg.infer_motion = False
        if getattr(args, "no_normalize_video", False):
            cfg.normalize_video = False
        if getattr(args, "no_use_realtime_loop", False):
            cfg.use_realtime_loop = False
        return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcagym_addr", default="localhost:50051")
    ap.add_argument("--agent_name", default="NoRobot")
    ap.add_argument("--duration_s", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", default="2560x1440")
    ap.add_argument("--render_style", default="PBR")

    ap.add_argument("--time_step_s", type=float, default=0.001)
    ap.add_argument("--frame_skip", type=int, default=20)
    ap.add_argument("--auto_frame_skip", action="store_true")
    ap.add_argument("--sync_time_step_with_server", action="store_true", default=True)
    ap.add_argument("--no_sync_time_step_with_server", action="store_true")

    ap.add_argument("--object_source", choices=["site", "body"], default="site")
    ap.add_argument("--object_site", default=None)
    ap.add_argument("--object_sites", default=None)
    ap.add_argument("--object_ids", default=None)
    ap.add_argument("--camera_body", default=None)
    ap.add_argument("--camera_name", default="Camera")
    ap.add_argument("--capture_mode", choices=["SYNC", "ASYNC"], default="ASYNC")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--no_drive_sim", action="store_true")
    ap.add_argument("--no_reset", action="store_true")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--render_fps", type=int, default=30)
    ap.add_argument("--use_realtime_loop", action="store_true", default=True)
    ap.add_argument("--no_use_realtime_loop", action="store_true")
    ap.add_argument("--contacts_mode", choices=["assume_ground", "remote", "none"], default="assume_ground")

    ap.add_argument("--ws_video", action="store_true")
    ap.add_argument("--ws_video_port", type=int, default=7070)
    ap.add_argument("--ws_video_name", default="Camera")
    ap.add_argument("--ws_video_to_mp4", action="store_true")
    ap.add_argument("--ws_video_wait_first_packet_s", type=float, default=5.0)
    ap.add_argument("--ws_video_recv_timeout_s", type=float, default=1.0)
    ap.add_argument("--ws_video_min_packets_to_mp4", type=int, default=10)

    ap.add_argument("--sequence_id", default=None)
    ap.add_argument("--output_root", default="/home/guojiatao/OrcaWorkStation/OrcaGen")
    ap.add_argument("--infer_motion", action="store_true", default=True)
    ap.add_argument("--no_infer_motion", action="store_true")
    ap.add_argument("--plot_motion", action="store_true", default=True)
    ap.add_argument("--normalize_video", action="store_true", default=True)
    ap.add_argument("--no_normalize_video", action="store_true")
    ap.add_argument("--external_drive", action="store_true", help="外部程序驱动仿真：本程序仅采集/录像，不 step，不 reset")
    return ap

