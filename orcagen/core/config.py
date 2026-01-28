from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

# 视频/录制默认相对路径（相对于项目根目录，与代码同级），会解析为绝对路径传递
RECORD_RELATIVE_PATH: str = "record"


def _project_root() -> str:
    """项目根目录（包含 orcagen 包的目录）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_output_root(output_root: str) -> str:
    """
    将 output_root 解析为绝对路径。
    - 若为 "." 则使用 RECORD_RELATIVE_PATH（OrcaGen/record）相对于项目根解析；
    - 否则对 output_root 做 os.path.abspath。
    """
    if output_root == ".":
        return os.path.abspath(os.path.join(_project_root(), RECORD_RELATIVE_PATH))
    return os.path.abspath(output_root)


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
    object_sites_groups: Optional[str] = None
    object_ids: Optional[str] = None
    camera_body: Optional[str] = None
    camera_name: str = "Camera"
    capture_mode: str = "ASYNC"
    save_video: bool = True  # 默认启用视频录制
    # 以下参数由 external_drive 自动管理，不应直接设置
    no_drive_sim: bool = False  # 由 external_drive 自动设置
    no_reset: bool = False  # 由 external_drive 自动设置
    no_render: bool = True  # 默认不渲染，可通过 --render 启用
    render_fps: int = 30
    use_realtime_loop: bool = True  # 默认启用实时循环
    contacts_mode: str = "assume_ground"
    ws_video: bool = False
    ws_video_port: int = 7070
    ws_video_name: str = "Camera"
    ws_video_to_mp4: bool = False
    ws_video_wait_first_packet_s: float = 5.0
    ws_video_recv_timeout_s: float = 1.0
    ws_video_min_packets_to_mp4: int = 10
    video_subdir: str = "rgb_main"
    sequence_id: Optional[str] = None
    sequence_prefix: Optional[str] = None
    output_root: str = "."
    infer_motion: bool = True
    plot_motion: bool = True
    normalize_video: bool = True
    external_drive: bool = False

    def apply_external_drive_mode(self) -> None:
        """规范化处理 external_drive 模式：自动设置相关参数"""
        if self.external_drive:
            self.no_drive_sim = True
            self.no_reset = True
            # external_drive 模式下默认不渲染（除非用户显式指定 --render）
            if not hasattr(self, '_render_explicitly_set'):
                self.no_render = True

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
        # use_realtime_loop 现在通过 --no_use_realtime_loop 的 dest 自动处理
        # 标记是否显式设置了 render（必须在 apply_external_drive_mode 之前）
        if hasattr(args, "render") and args.render:
            cfg._render_explicitly_set = True
            cfg.no_render = False
        elif hasattr(args, "no_render") and args.no_render:
            cfg._render_explicitly_set = True
        # 应用 external_drive 模式（会检查 _render_explicitly_set）
        cfg.apply_external_drive_mode()
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
    ap.add_argument(
        "--object_sites_groups",
        default=None,
        help='多子环境物体组，使用 "|" 分割，例如: "Bin1|Bin2"',
    )
    ap.add_argument("--object_ids", default=None)
    ap.add_argument("--camera_body", default=None)
    ap.add_argument("--camera_name", default="Camera")
    ap.add_argument("--capture_mode", choices=["SYNC", "ASYNC"], default="ASYNC")
    ap.add_argument("--save_video", action="store_true", default=True, help="保存视频（默认启用）")
    ap.add_argument("--no_save_video", action="store_false", dest="save_video", help="禁用视频录制")
    # no_drive_sim 和 no_reset 由 external_drive 自动管理，不暴露为独立参数
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--render", action="store_true", help="显式启用渲染（覆盖 external_drive 的默认行为）")
    ap.add_argument("--render_fps", type=int, default=30)
    # use_realtime_loop 默认启用，通常不需要修改
    ap.add_argument("--no_use_realtime_loop", action="store_false", dest="use_realtime_loop", help="禁用实时循环")
    ap.add_argument("--contacts_mode", choices=["assume_ground", "remote", "none"], default="assume_ground")

    ap.add_argument("--ws_video", action="store_true")
    ap.add_argument("--ws_video_port", type=int, default=7070)
    ap.add_argument("--ws_video_name", default="Camera")
    ap.add_argument("--ws_video_to_mp4", action="store_true")
    ap.add_argument("--ws_video_wait_first_packet_s", type=float, default=5.0)
    ap.add_argument("--ws_video_recv_timeout_s", type=float, default=1.0)
    ap.add_argument("--ws_video_min_packets_to_mp4", type=int, default=10)
    ap.add_argument("--video_subdir", default="rgb_main")

    ap.add_argument("--sequence_id", default=None)
    ap.add_argument("--sequence_prefix", default=None)
    ap.add_argument("--output_root", default=".", help="输出根目录（默认 . 时解析为项目根下的 record 绝对路径，与代码同级）")
    ap.add_argument("--infer_motion", action="store_true", default=True)
    ap.add_argument("--no_infer_motion", action="store_true")
    ap.add_argument("--plot_motion", action="store_true", default=True)
    ap.add_argument("--normalize_video", action="store_true", default=True)
    ap.add_argument("--no_normalize_video", action="store_true")
    ap.add_argument("--external_drive", action="store_true", help="外部程序驱动仿真：本程序仅采集/录像，不 step，不 reset")
    return ap

