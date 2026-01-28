from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import shutil
import glob
from typing import Dict, List, Optional

import numpy as np

from orca_gym.core.orca_gym_local import CaptureMode
from scripts.motion_analysis import infer_motion_types
from orcagen.core.base import BaseRunner
from orcagen.core.config import CaptureConfig, resolve_output_root
from orcagen.enums.motion import MotionType
from envs.orcagym_env import OrcaGymEnvAdapter
from orcagen.utils.fs import mkdirp
from orcagen.utils.ids import now_id, default_object_id
from orcagen.utils.mujoco import (
    find_body_name,
    find_site_name,
    ensure_site_dict,
    ensure_geom_dict,
    infer_bbox_size_for_body,
    infer_bbox_size_for_site,
)
from orcagen.utils.pose import (
    mat_to_euler_xyz_deg,
    norm_angle_deg_to_unit,
    worldsim_to_world0,
)


class CaptureRunner(BaseRunner):
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config

    def run(self) -> None:
        self._check_connectivity()
        seq_id = self._get_sequence_id()
        seq_dir, video_dir, project_dir, meta_dir = self._prepare_dirs(seq_id)
        ws_proc = None

        # external_drive 模式已通过 config.apply_external_drive_mode() 规范化处理

        env_adapter = OrcaGymEnvAdapter(self.config)
        env_adapter.initialize()
        env = env_adapter.env
        base_env = env_adapter.base_env

        if self.config.auto_frame_skip:
            target = 1.0 / float(self.config.fps)
            self.config.frame_skip = max(1, int(round(target / float(self.config.time_step_s))))
            print(f"[OrcaGen] auto_frame_skip: time_step_s={self.config.time_step_s}, frame_skip={self.config.frame_skip}, realtime_step≈{self.config.time_step_s*self.config.frame_skip:.6f}s")

        realtime_step = float(self.config.time_step_s) * int(self.config.frame_skip)
        try:
            base_env.metadata["render_fps"] = int(self.config.render_fps)
        except Exception:
            pass

        obj_ids, obj_body_by_id, obj_site_by_id, obj_id_by_body = self._resolve_objects(base_env)
        cam_body = self._find_camera_body(base_env)
        cam_pose_worldsim_pos, cam_pose_worldsim_mat = self._get_camera_pose(env_adapter, cam_body, seq_id)

        R0 = cam_pose_worldsim_mat
        t0 = cam_pose_worldsim_pos

        obj_sizes = self._infer_object_sizes(base_env, obj_ids, obj_body_by_id, obj_site_by_id)

        video_save_dir = os.path.join(video_dir, "rgb_main")
        mkdirp(video_save_dir)
        cap_mode = CaptureMode.SYNC if self.config.capture_mode == "SYNC" else CaptureMode.ASYNC
        video_started = False
        if self.config.save_video:
            if self.config.capture_mode == "SYNC":
                print("[OrcaGen] 提示：你当前使用 --capture_mode SYNC；若 Editor 端出现 CameraSyncManager WaitingLastFrame Timeout，建议改为 ASYNC。")
            env_adapter.begin_save_video(os.path.abspath(video_save_dir), cap_mode)
            video_started = True

        total_frames = int(round(self.config.duration_s * self.config.fps))
        meta_path = os.path.join(meta_dir, "metadata.jsonl")
        collision_times: Dict[str, Optional[int]] = {oid: None for oid in obj_ids}

        # 用于友好打印：每隔3秒简述物体运动状态
        last_status_print_time = time.time()
        status_print_interval = 3.0
        recent_positions: Dict[str, List[np.ndarray]] = {oid: [] for oid in obj_ids}

        def _start_ws_recorder() -> Optional[subprocess.Popen]:
            if not self.config.ws_video:
                return None
            cmd = [
                sys.executable,
                "-m",
                "scripts.record_ws_video",
                "--host",
                "localhost",
                "--port",
                str(int(self.config.ws_video_port)),
                "--duration_s",
                str(float(self.config.duration_s)),
                "--fps",
                str(int(self.config.fps)),
                "--name",
                self.config.ws_video_name,
                "--output_root",
                resolve_output_root(self.config.output_root),
                "--sequence_id",
                seq_id,
                "--wait_first_packet_s",
                str(float(self.config.ws_video_wait_first_packet_s)),
                "--recv_timeout_s",
                str(float(self.config.ws_video_recv_timeout_s)),
                "--min_packets_to_mp4",
                str(int(self.config.ws_video_min_packets_to_mp4)),
            ]
            if self.config.ws_video_to_mp4:
                cmd.append("--to_mp4")
            return subprocess.Popen(cmd)

        do_render = not self.config.no_render
        ws_started = False
        render_err_count = 0
        camera_frame_seen = False
        camera_frame_count = 0
        camera_first_idx: Optional[int] = None
        camera_last_idx: Optional[int] = None

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                recorded = 0
                last_frame_idx = -1
                while recorded < total_frames:
                    loop_start = time.time()

                    if not self.config.no_drive_sim:
                        _ = env.step(env.action_space.sample() if env.action_space.shape != (0,) else np.array([]))
                    else:
                        # external_drive 模式下，通过 gRPC 从服务器拉取最新状态
                        # 然后更新本地 _mjData，这样 query_body_xpos_xmat_xquat 才能获取到最新坐标
                        try:
                            # 通过 gRPC 直接查询状态
                            if hasattr(base_env, 'gym') and hasattr(base_env.gym, 'stub'):
                                # 动态导入 mjc_message_pb2（OrcaGym 的 protos 路径）
                                try:
                                    import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2
                                    request = mjc_message_pb2.QueryAllQposQvelQaccRequest()
                                except ImportError:
                                    request = None
                                
                                if request is not None:
                                    # 使用同步调用（因为我们在同步代码中）
                                    response = base_env.loop.run_until_complete(
                                        base_env.gym.stub.QueryAllQposQvelQacc(request)
                                    )
                                    # 更新本地 _mjData
                                    if hasattr(base_env.gym, '_mjData') and base_env.gym._mjData is not None:
                                        base_env.gym._mjData.qpos[:] = np.array(response.qpos)
                                        base_env.gym._mjData.qvel[:] = np.array(response.qvel)
                                        base_env.gym._mjData.qacc[:] = np.array(response.qacc)
                                        base_env.gym._mjData.time = float(response.time) if hasattr(response, 'time') else base_env.gym._mjData.time
                                        # 执行前向计算，更新 xpos, xmat, xquat 等
                                        base_env.gym.mj_forward()
                                        # 同步到封装的 data 对象
                                        base_env.gym.update_data()
                        except Exception:
                            # 如果 gRPC 查询失败，静默处理（避免影响采集）
                            pass

                    if do_render:
                        try:
                            env.render()
                            if self.config.ws_video and not ws_started:
                                ws_proc = _start_ws_recorder()
                                ws_started = True
                            render_err_count = 0  # 成功时重置计数
                        except Exception:
                            render_err_count += 1
                            # external_drive 模式下减少警告打印
                            if not self.config.external_drive:
                                if render_err_count <= 5 or (render_err_count % 60 == 0):
                                    print(f"[OrcaGen] 警告：env.render 失败（count={render_err_count}），将继续采集。")
                            elif render_err_count == 5:
                                do_render = False
                                print("[OrcaGen] external_drive 模式下连续 render 失败，后续将跳过 render，仅采集/录像。")
                    if (not do_render) and self.config.ws_video and (not ws_started):
                        try:
                            if base_env.get_current_frame() >= 0:
                                ws_proc = _start_ws_recorder()
                                ws_started = True
                        except Exception:
                            pass

                    try:
                        cur_frame = base_env.get_next_frame()
                    except Exception:
                        cur_frame = -1
                    if cur_frame >= 0:
                        camera_frame_seen = True
                        if cur_frame == last_frame_idx:
                            time.sleep(0.001)
                            continue
                        if camera_first_idx is None:
                            camera_first_idx = cur_frame
                        camera_last_idx = cur_frame
                        camera_frame_count += 1
                        last_frame_idx = cur_frame

                    cam_pos_ws = cam_pose_worldsim_pos if cam_body is None else env_adapter.get_body_pose(cam_body).pos
                    cam_mat_ws = cam_pose_worldsim_mat if cam_body is None else env_adapter.get_body_pose(cam_body).mat
                    cam_pos_w0 = worldsim_to_world0(cam_pos_ws, R0, t0)
                    cam_mat_w0 = R0.T @ cam_mat_ws

                    if self.config.contacts_mode == "assume_ground":
                        has_contact = True
                        contact_pairs = [[oid, "ground"] for oid in obj_ids]
                        # 额外补充物体-物体碰撞（不改变 ground 标注）
                        try:
                            contacts_simple = base_env.query_contact_simple()
                        except Exception:
                            contacts_simple = []
                        obj_pairs = self._get_contact_pairs(base_env, contacts_simple, obj_id_by_body)
                        obj_set = set(obj_ids)
                        for p in obj_pairs:
                            if len(p) == 2 and p[0] in obj_set and p[1] in obj_set:
                                if p not in contact_pairs:
                                    contact_pairs.append(p)
                                for oid in p:
                                    if collision_times[oid] is None or collision_times[oid] == 0:
                                        collision_times[oid] = recorded
                    elif self.config.contacts_mode == "remote":
                        try:
                            contacts_simple = base_env.query_contact_simple()
                        except Exception:
                            contacts_simple = []
                        contact_pairs = self._get_contact_pairs(base_env, contacts_simple, obj_id_by_body)
                        has_contact = len(contact_pairs) > 0
                        for p in contact_pairs:
                            for oid in obj_ids:
                                if oid in p and collision_times[oid] is None:
                                    collision_times[oid] = recorded
                    else:
                        has_contact = False
                        contact_pairs = None

                    annotations = []
                    for oid in obj_ids:
                        if self.config.object_source == "site":
                            site_name = obj_site_by_id[oid]
                            obj = env_adapter.get_site_pose(site_name)
                            body_name = obj_body_by_id.get(oid)
                        else:
                            body_name = obj_body_by_id[oid]
                            obj = env_adapter.get_body_pose(body_name)
                        p_cam = cam_mat_ws.T @ (obj.pos - cam_pos_ws)
                        R_cam_obj = cam_mat_ws.T @ obj.mat
                        pitch_deg, yaw_deg, roll_deg = mat_to_euler_xyz_deg(R_cam_obj)
                        pitch = norm_angle_deg_to_unit(pitch_deg)
                        yaw = norm_angle_deg_to_unit(yaw_deg)
                        roll = norm_angle_deg_to_unit(roll_deg)
                        x_size, y_size, z_size = obj_sizes[oid]
                        annotations.append(
                            {
                                "object_id": oid,
                                "body_name": body_name,
                                "site_name": obj_site_by_id.get(oid),
                                "bbox3d": {
                                    "x_center": float(p_cam[0]),
                                    "y_center": float(p_cam[1]),
                                    "z_center": float(p_cam[2]),
                                    "x_size": float(x_size),
                                    "y_size": float(y_size),
                                    "z_size": float(z_size),
                                    "pitch": pitch,
                                    "yaw": yaw,
                                    "roll": roll,
                                },
                                "bbox3d_array": [
                                    float(p_cam[0]),
                                    float(p_cam[1]),
                                    float(p_cam[2]),
                                    float(x_size),
                                    float(y_size),
                                    float(z_size),
                                    pitch,
                                    yaw,
                                    roll,
                                ],
                            }
                        )

                    rec = {
                        "frame_index": recorded,
                        "timestamp_s": round(recorded / float(self.config.fps), 10),
                        "sim_time_s": float(getattr(base_env.data, "time", 0.0)),
                        "camera": {
                            "camera_id": "main",
                            "body_name": cam_body,
                            "camera_name": self.config.camera_name if cam_body is None else None,
                            "pose_world0": {
                                "pos": [float(x) for x in cam_pos_w0.tolist()],
                                "mat3x3_rowmajor": [float(x) for x in cam_mat_w0.reshape(-1).tolist()],
                            },
                        },
                        "annotations": annotations,
                        "contacts": {"has_contact": bool(has_contact), "pairs": contact_pairs},
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    recorded += 1

                    # 收集最近的位置用于状态打印
                    for ann in annotations:
                        oid = ann["object_id"]
                        bbox = ann["bbox3d"]
                        pos = np.array([bbox["x_center"], bbox["y_center"], bbox["z_center"]], dtype=np.float64)
                        recent_positions[oid].append(pos)
                        # 只保留最近30帧
                        if len(recent_positions[oid]) > 30:
                            recent_positions[oid].pop(0)

                    # 每隔3秒打印一次物体运动状态
                    current_time = time.time()
                    if current_time - last_status_print_time >= status_print_interval:
                        elapsed_s = recorded / float(self.config.fps)
                        progress_pct = (recorded / max(total_frames, 1)) * 100.0
                        status_lines = [f"[OrcaGen] 采集进度: {elapsed_s:.1f}s / {self.config.duration_s:.1f}s ({progress_pct:.1f}%)"]
                        for oid in obj_ids:
                            if len(recent_positions[oid]) >= 2:
                                pos_list = recent_positions[oid]
                                recent_dp = pos_list[-1] - pos_list[0]
                                recent_speed = np.linalg.norm(recent_dp) / max((len(pos_list) - 1) / float(self.config.fps), 1e-6)
                                status_lines.append(f"  {oid}: 位置=({pos_list[-1][0]:.3f}, {pos_list[-1][1]:.3f}, {pos_list[-1][2]:.3f})m, 速度≈{recent_speed:.3f}m/s")
                            else:
                                status_lines.append(f"  {oid}: 位置数据不足")
                        print("\n".join(status_lines))
                        last_status_print_time = current_time

                    if self.config.use_realtime_loop:
                        elapsed = time.time() - loop_start
                        if elapsed < realtime_step:
                            time.sleep(realtime_step - elapsed)
        finally:
            if video_started:
                try:
                    env_adapter.stop_save_video()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"[OrcaGen] 警告：stop_save_video 失败（可能导致视频未 finalize）：{e}")

        motion_map = infer_motion_types(meta_path, obj_ids) if self.config.infer_motion else {oid: MotionType.UNKNOWN for oid in obj_ids}
        video_artifacts, camera_ids = self._collect_video_artifacts(
            video_save_dir=video_save_dir,
            seq_dir=seq_dir,
            camera_name_filter=self.config.camera_name,
        ) if self.config.save_video else ([], [])
        if not video_artifacts:
            video_artifacts = [{"path": "video/rgb_main.mp4", "camera_id": "main"}]
        index_path = os.path.join(meta_dir, "index.json")
        size_source = "site_size" if self.config.object_source == "site" else "geom_size"
        index = {
            "version": "orcagen-metadata-0.1",
            "sequence_id": seq_id,
            "object": {"object_id": obj_ids[0], "name": obj_ids[0], "part": "main"},
            "motion_mode": motion_map.get(obj_ids[0], MotionType.UNKNOWN).value,
            "motion_modes": {oid: motion_map.get(oid, MotionType.UNKNOWN).value for oid in obj_ids},
            "motion_start": 0,
            "motion_end": total_frames - 1,
            "collision_time": collision_times.get(obj_ids[0]),
            "collision_times": collision_times,
            "coordinate_system": {
                "world_definition": "world0 == camera(main) frame0",
                "axes": {"x": "right", "y": "down", "z": "forward"},
                "origin": "camera(main) position at frame0",
                "units": {"position": "meter", "angles": "deg", "angle_norm": "(-1,1)*180"},
            },
            "capture": {
                "fps": int(self.config.fps),
                "resolution": self._parse_resolution(),
                "duration_s": float(self.config.duration_s),
                "render_style": self.config.render_style,
                "capture_mode": self.config.capture_mode,
                "time_step_s": float(self.config.time_step_s),
                "frame_skip": int(self.config.frame_skip),
            },
            "artifacts": {
                "video": video_artifacts,
                "project": {"path": "project/scene.usd", "format": "USD", "contains": "keyframes+dyn_cache"},
            },
            "objects": [
                {
                    "object_id": oid,
                    "body_name": obj_body_by_id.get(oid),
                    "site_name": obj_site_by_id.get(oid),
                    "semantic": {"class": "rigid", "category": "rigid"},
                    "canonical_size_m": [float(obj_sizes[oid][0]), float(obj_sizes[oid][1]), float(obj_sizes[oid][2])],
                    "motion_mode": motion_map.get(oid, MotionType.UNKNOWN).value,
                    "size_source": size_source,
                }
                for oid in obj_ids
            ],
            "cameras": self._build_camera_entries(
                cam_body=cam_body,
                camera_ids=camera_ids,
            ),
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        env_adapter.close()

        if ws_proc is not None:
            try:
                ws_proc.wait(timeout=float(self.config.duration_s) + 15.0)
            except subprocess.TimeoutExpired:
                ws_proc.terminate()

        print("OK")
        print("sequence_dir:", seq_dir)
        print("index.json:", index_path)
        print("metadata.jsonl:", meta_path)
        if self.config.save_video:
            print("video_dir:", video_save_dir)
            try:
                if os.path.exists(video_save_dir):
                    total = 0
                    for root, _, files in os.walk(video_save_dir):
                        for fn in files:
                            fp = os.path.join(root, fn)
                            try:
                                total += os.path.getsize(fp)
                            except Exception:
                                pass
                    print("[OrcaGen] video_dir_total_bytes:", total)
            except Exception:
                pass
        if self.config.ws_video:
            print("[OrcaGen] ws_video_dir:", video_dir)

        if self.config.save_video and (not camera_frame_seen):
            print("[OrcaGen] 警告：采集期间相机帧索引一直不可用（get_current_frame/get_next_frame < 0），mp4 可能为空。请确认 OrcaEditor 中相机/Viewport 已启用并在渲染。")
        if camera_frame_count > 0:
            est_fps = camera_frame_count / max(1e-6, float(self.config.duration_s))
            print(f"[OrcaGen] camera_frames={camera_frame_count}, camera_fps≈{est_fps:.2f}, first_idx={camera_first_idx}, last_idx={camera_last_idx}")
            if camera_frame_count < int(self.config.duration_s * self.config.fps * 0.5):
                print("[OrcaGen] 警告：相机帧数量明显偏少，视频可能提前结束或后半段静止。")

        # if self.config.save_video:
        #     self._wait_for_video_stable(video_save_dir)
        #     if self.config.repack_video:
        #         self._repack_videos(video_save_dir)
        # if self.config.save_video and self.config.normalize_video:
        #     self._normalize_videos(video_save_dir)

        if self.config.plot_motion:
            self._plot_motion(meta_path, meta_dir)

    def _check_connectivity(self) -> None:
        try:
            host, port_str = self.config.orcagym_addr.split(":")
            port = int(port_str)
            s = socket.socket()
            s.settimeout(0.5)
            s.connect((host, port))
            s.close()
        except Exception as e:
            raise SystemExit(
                f"[OrcaGen] 无法连接 OrcaGym gRPC: {self.config.orcagym_addr} ({e}).\n"
                f"- 请先确认 OrcaEditor/OrcaSim 已启动并在该端口监听（常见为 50051）。\n"
                f"- 注意：7070 通常是相机/视频流端口，不是 gRPC 端口。"
            )

    def _get_sequence_id(self) -> str:
        if self.config.sequence_id:
            return self.config.sequence_id
        if self.config.object_source == "site":
            if self.config.object_sites:
                first_name = [x.strip() for x in self.config.object_sites.split(",") if x.strip()][0]
            elif self.config.object_site:
                first_name = self.config.object_site
            else:
                first_name = self.config.object_body
        else:
            if self.config.object_bodies:
                first_name = [x.strip() for x in self.config.object_bodies.split(",") if x.strip()][0]
            else:
                first_name = self.config.object_body
        seq_label = default_object_id(first_name)
        return f"sequence_capture_{seq_label}_{now_id()}"

    def _prepare_dirs(self, seq_id: str):
        output_root_abs = resolve_output_root(self.config.output_root)
        seq_dir = os.path.join(output_root_abs, seq_id)
        video_dir = os.path.join(seq_dir, "video")
        project_dir = os.path.join(seq_dir, "project")
        meta_dir = os.path.join(seq_dir, "metadata")
        mkdirp(video_dir)
        mkdirp(project_dir)
        mkdirp(meta_dir)
        return seq_dir, video_dir, project_dir, meta_dir

    def _parse_resolution(self) -> List[int]:
        try:
            w, h = self.config.resolution.lower().split("x")
            return [int(w), int(h)]
        except Exception:
            return [1920, 1080]

    def _resolve_objects(self, base_env):
        if self.config.object_source == "site":
            if self.config.object_sites:
                obj_candidates = [x.strip() for x in self.config.object_sites.split(",") if x.strip()]
            elif self.config.object_site:
                obj_candidates = [self.config.object_site]
            else:
                obj_candidates = [self.config.object_body]
            if len(obj_candidates) == 0 or len(obj_candidates) > 2:
                raise SystemExit("[OrcaGen] 仅支持 1~2 个物体，请检查 --object_sites/--object_site 参数。")
            object_sites: List[str] = []
            for s in obj_candidates:
                object_sites.append(find_site_name(base_env, s, keywords=[s]))
            if self.config.object_ids:
                obj_ids = [x.strip() for x in self.config.object_ids.split(",") if x.strip()]
                if len(obj_ids) != len(object_sites):
                    raise SystemExit("[OrcaGen] --object_ids 数量需与 --object_sites 一致。")
            else:
                obj_ids = [default_object_id(s) for s in object_sites]
            obj_site_by_id = dict(zip(obj_ids, object_sites))
            site_dict = ensure_site_dict(base_env)
            obj_body_by_id: Dict[str, Optional[str]] = {}
            obj_id_by_body: Dict[str, str] = {}
            for oid, site_name in obj_site_by_id.items():
                body_name = None
                try:
                    body_id = site_dict[site_name]["BodyID"]
                    body_name = base_env.model.body_id2name(int(body_id))
                except Exception:
                    body_name = None
                obj_body_by_id[oid] = body_name
                if body_name:
                    obj_id_by_body[body_name] = oid
            return obj_ids, obj_body_by_id, obj_site_by_id, obj_id_by_body

        if self.config.object_bodies:
            obj_candidates = [x.strip() for x in self.config.object_bodies.split(",") if x.strip()]
        else:
            obj_candidates = [self.config.object_body]
        if len(obj_candidates) == 0 or len(obj_candidates) > 2:
            raise SystemExit("[OrcaGen] 仅支持 1~2 个物体，请检查 --object_bodies/--object_body 参数。")
        object_bodies: List[str] = []
        for b in obj_candidates:
            object_bodies.append(find_body_name(base_env.model, b, keywords=[b, "sphere", "cube", "box"]))
        if self.config.object_ids:
            obj_ids = [x.strip() for x in self.config.object_ids.split(",") if x.strip()]
            if len(obj_ids) != len(object_bodies):
                raise SystemExit("[OrcaGen] --object_ids 数量需与 --object_bodies 一致。")
        else:
            obj_ids = [default_object_id(b) for b in object_bodies]
        obj_body_by_id = dict(zip(obj_ids, object_bodies))
        obj_id_by_body = dict(zip(object_bodies, obj_ids))
        obj_site_by_id = {}
        return obj_ids, obj_body_by_id, obj_site_by_id, obj_id_by_body

    def _find_camera_body(self, base_env) -> Optional[str]:
        try:
            return find_body_name(base_env.model, self.config.camera_body, keywords=["CameraViewport", "camera"])
        except Exception:
            return None

    def _get_camera_pose(self, env_adapter: OrcaGymEnvAdapter, cam_body: Optional[str], seq_id: str):
        if cam_body is not None:
            cam0 = env_adapter.get_body_pose(cam_body)
            return cam0.pos, cam0.mat
        if not self.config.camprobe:
            print("[OrcaGen] 警告：未找到相机 body，且未开启 --camprobe；将使用单位矩阵作为相机位姿。")
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        probe_dir = os.path.join(resolve_output_root(self.config.output_root), ".orcagen_camprobe", seq_id)
        mkdirp(probe_dir)
        cam0 = env_adapter.get_camera_pose(None, self.config.camera_name, probe_dir)
        return cam0.pos, cam0.mat

    def _infer_object_sizes(self, base_env, obj_ids, obj_body_by_id, obj_site_by_id):
        obj_sizes: Dict[str, tuple] = {}
        for oid in obj_ids:
            if self.config.object_source == "site":
                site_name = obj_site_by_id[oid]
                obj_sizes[oid] = infer_bbox_size_for_site(base_env, site_name, fallback_xyz=(0.10, 0.10, 0.10))
            else:
                body = obj_body_by_id[oid]
                obj_sizes[oid] = infer_bbox_size_for_body(base_env, body, fallback_xyz=(0.10, 0.10, 0.10))
        return obj_sizes

    def _get_contact_pairs(self, base_env, contacts_simple, obj_id_by_body):
        geom_dict = ensure_geom_dict(base_env)
        pairs: List[List[str]] = []
        for c in contacts_simple:
            g1 = int(c.get("Geom1", -1))
            g2 = int(c.get("Geom2", -1))
            try:
                n1 = base_env.model.geom_id2name(g1)
                n2 = base_env.model.geom_id2name(g2)
            except Exception:
                continue
            b1 = geom_dict.get(n1, {}).get("BodyName")
            b2 = geom_dict.get(n2, {}).get("BodyName")
            o1 = obj_id_by_body.get(b1)
            o2 = obj_id_by_body.get(b2)
            if o1 is None and o2 is None:
                continue
            if o1 is None:
                o1 = "unknown"
            if o2 is None:
                o2 = "unknown"
            pairs.append([o1, o2])
        return pairs

    def _get_video_duration_s(self, path: str) -> Optional[float]:
        if not os.path.exists(path):
            return None
        if shutil.which("ffprobe"):
            try:
                out = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
                    text=True,
                ).strip()
                return float(out)
            except Exception:
                return None
        try:
            out = subprocess.check_output(["ffmpeg", "-hide_banner", "-i", path], stderr=subprocess.STDOUT, text=True)
            for line in out.splitlines():
                if "Duration:" in line:
                    hms = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = hms.split(":")
                    return float(h) * 3600 + float(m) * 60 + float(s)
        except Exception:
            pass
        return None

    # def _normalize_videos(self, video_save_dir: str) -> None:
    #     try:
    #         target = float(self.config.duration_s)
    #         for root, _, files in os.walk(video_save_dir):
    #             for fn in files:
    #                 if not fn.lower().endswith(".mp4"):
    #                     continue
    #                 src = os.path.join(root, fn)
    #                 dur = self._get_video_duration_s(src)
    #                 if dur is None or target <= 0:
    #                     continue
    #                 ratio = dur / target
    #                 if 0.95 <= ratio <= 1.05:
    #                     continue
    #                 out = src.replace(".mp4", "_normalized.mp4")
    #                 speed = ratio
    #                 subprocess.run(
    #                     [
    #                         "ffmpeg",
    #                         "-y",
    #                         "-hide_banner",
    #                         "-i",
    #                         src,
    #                         "-filter:v",
    #                         f"setpts=PTS/{speed:.6f}",
    #                         "-r",
    #                         str(int(self.config.fps)),
    #                         "-an",
    #                         out,
    #                     ],
    #                     check=True,
    #                 )
    #                 print(f"[OrcaGen] 已校正视频速度: {src} -> {out} (ratio={ratio:.3f})")
    #     except Exception as e:
    #         print(f"[OrcaGen] 警告：视频速度校正失败：{e}")

    def _wait_for_video_stable(self, video_save_dir: str, timeout_s: float = 3.0) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            changed = False
            for root, _, files in os.walk(video_save_dir):
                for fn in files:
                    if not fn.lower().endswith(".mp4"):
                        continue
                    fp = os.path.join(root, fn)
                    try:
                        s1 = os.path.getsize(fp)
                        time.sleep(0.3)
                        s2 = os.path.getsize(fp)
                        if s2 != s1:
                            changed = True
                    except Exception:
                        pass
            if not changed:
                return
    
    def _parse_camera_file(self, file_path: str) -> tuple[str, Optional[str]]:
        base = os.path.splitext(os.path.basename(file_path))[0]
        stream = None
        camera_name = base
        if base.endswith("_color"):
            camera_name = base[:-6]
            stream = "color"
        elif base.endswith("_depth"):
            camera_name = base[:-6]
            stream = "depth"
        elif base.endswith("_seg"):
            camera_name = base[:-4]
            stream = "seg"
        if stream is None:
            parent = os.path.basename(os.path.dirname(file_path)).lower()
            if parent in ("depth", "seg", "segmentation"):
                stream = parent
        return camera_name, stream

    def _collect_video_artifacts(
        self,
        video_save_dir: str,
        seq_dir: str,
        camera_name_filter: Optional[str] = None,
    ) -> tuple[list[dict], list[str]]:
        if not os.path.exists(video_save_dir):
            return [], []
        video_files = []
        for root, _, files in os.walk(video_save_dir):
            for fn in files:
                if fn.lower().endswith((".mp4", ".h264", ".mov", ".avi")):
                    video_files.append(os.path.join(root, fn))
        if not video_files:
            return [], []
        artifacts: list[dict] = []
        camera_ids: set[str] = set()
        for fp in sorted(set(video_files)):
            rel_path = os.path.relpath(fp, seq_dir).replace(os.sep, "/")
            camera_name, stream = self._parse_camera_file(fp)
            if camera_name_filter and camera_name != camera_name_filter:
                continue
            camera_id = "main" if camera_name == self.config.camera_name else camera_name
            entry = {"path": rel_path, "camera_id": camera_id}
            if stream:
                entry["stream"] = stream
            artifacts.append(entry)
            camera_ids.add(camera_id)
        return artifacts, sorted(camera_ids)
    
    def _build_camera_entries(self, cam_body: Optional[str], camera_ids: list[str]) -> list[dict]:
        entries = [
            {
                "camera_id": "main",
                "body_name": cam_body,
                "camera_name": self.config.camera_name if cam_body is None else None,
                "extrinsics_world0": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            }
        ]
        for cid in camera_ids:
            if cid == "main":
                continue
            entries.append(
                {
                    "camera_id": cid,
                    "body_name": None,
                    "camera_name": cid,
                    "extrinsics_world0": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
                }
            )
        return entries

    def _distribute_videos_by_camera(self, unified_video_dir: str, group_setups: List[Dict]) -> None:
        video_files = []
        max_retries = 3
        for retry in range(max_retries):
            video_files = []
            if os.path.exists(unified_video_dir):
                for root, _, files in os.walk(unified_video_dir):
                    for fn in files:
                        if fn.lower().endswith((".mp4", ".h264", ".mov", ".avi")):
                            video_files.append(os.path.join(root, fn))
            if video_files:
                break
            if retry < max_retries - 1:
                print(f"[OrcaGen] 未找到视频文件，等待中... (重试 {retry + 1}/{max_retries})")
                time.sleep(2.0)
        
        if not video_files:
            print(f"[OrcaGen] 警告：在 {unified_video_dir} 中未找到视频文件，无法分发到各组目录")
            print(f"[OrcaGen] 提示：请检查引擎日志，确认视频是否已成功保存")
            return

        camera_names = sorted({self._parse_camera_file(fp)[0] for fp in video_files})
        if camera_names and len(camera_names) == len(group_setups):
            for idx, name in enumerate(camera_names):
                group_setups[idx]["camera_name_override"] = name
            print(f"[OrcaGen] 多组录制：根据文件名推断 camera 映射 = {camera_names}")

        for idx, setup in enumerate(group_setups):
            target_dir = setup.get("target_video_dir", setup["video_save_dir"])
            if os.path.abspath(target_dir) == os.path.abspath(unified_video_dir):
                print(f"[OrcaGen] 组 {idx} camera={setup.get('camera_name_override', setup['config'].camera_name)} 目标目录与统一目录一致，跳过复制")
                continue
            camera_name = setup.get("camera_name_override", setup["config"].camera_name)
            matched = 0
            mkdirp(target_dir)
            for src_file in video_files:
                src_camera, _ = self._parse_camera_file(src_file)
                if src_camera != camera_name:
                    continue
                rel_path = os.path.relpath(src_file, unified_video_dir)
                target_file = os.path.join(target_dir, rel_path)
                target_parent = os.path.dirname(target_file)
                mkdirp(target_parent)
                try:
                    if os.path.exists(target_file):
                        os.remove(target_file)
                    shutil.move(src_file, target_file)
                    matched += 1
                except Exception as e:
                    print(f"[OrcaGen] 警告：移动视频到组 {idx} 失败：{e}")
            print(f"[OrcaGen] 组 {idx} camera={camera_name} 移动文件数={matched}")
    # def _repack_videos(self, video_save_dir: str) -> None:
    #     if shutil.which("ffmpeg") is None:
    #         print("[OrcaGen] 警告：未找到 ffmpeg，跳过视频重封装。")
    #         return
    #     for root, _, files in os.walk(video_save_dir):
    #         for fn in files:
    #             if not fn.lower().endswith(".mp4"):
    #                 continue
    #             src = os.path.join(root, fn)
    #             tmp = src.replace(".mp4", "_repack_tmp.mp4")
    #             try:
    #                 subprocess.run(
    #                     [
    #                         "ffmpeg",
    #                         "-y",
    #                         "-err_detect",
    #                         "ignore_err",
    #                         "-fflags",
    #                         "+discardcorrupt",
    #                         "-i",
    #                         src,
    #                         "-c:v",
    #                         "libx264",
    #                         "-pix_fmt",
    #                         "yuv420p",
    #                         tmp,
    #                     ],
    #                     check=True,
    #                 )
    #                 os.replace(tmp, src)
    #                 print(f"[OrcaGen] 已重封装视频: {src}")
    #             except Exception as e:
    #                 try:
    #                     if os.path.exists(tmp):
    #                         os.remove(tmp)
    #                 except Exception:
    #                     pass
    #                 print(f"[OrcaGen] 警告：视频重封装失败：{src} ({e})")


    def _plot_motion(self, meta_path: str, meta_dir: str) -> None:
        try:
            from scripts.plot_motion_curves import plot_motion_curves
            plots_dir = os.path.join(meta_dir, "plots")
            plot_motion_curves(meta_path, plots_dir)
        except Exception as e:
            print(f"[OrcaGen] 警告：生成运动曲线图失败：{e}")

class MultiEnvCaptureRunner(BaseRunner):
    """
    多子环境录制：为每个组创建独立 env 实例并分别录制
    用于确保每个子环境的 camera 录制到对应目录
    """
    def __init__(self, group_configs: List[CaptureConfig]) -> None:
        self.group_configs = group_configs
        if not group_configs:
            raise ValueError("group_configs 不能为空")

    def run(self) -> None:
        for idx, config in enumerate(self.group_configs):
            print(f"[OrcaGen] 启动子环境 {idx} 录制: seq_id={config.sequence_id}")
            runner = CaptureRunner(config)
            runner.run()


class MultiGroupCaptureRunner(BaseRunner):
    """
    多组并行录制：使用同一个 env 实例，所有组同时开始/结束录制
    参考 OrcaManipulation 的实现方式
    """
    def __init__(self, group_configs: List[CaptureConfig]) -> None:
        self.group_configs = group_configs
        if not group_configs:
            raise ValueError("group_configs 不能为空")
        # 使用第一个组的配置作为主配置（env 相关）
        self.config = group_configs[0]

    def run(self) -> None:
        # 检查连接（使用第一个 runner 实例）
        first_runner = CaptureRunner(self.config)
        first_runner._check_connectivity()
        
        # 准备所有组的目录和配置
        group_setups = []
        for idx, config in enumerate(self.group_configs):
            seq_id = config.sequence_id or first_runner._get_sequence_id()
            seq_dir, video_dir, project_dir, meta_dir = first_runner._prepare_dirs(seq_id)
            video_save_dir = os.path.join(video_dir, config.video_subdir)
            mkdirp(video_save_dir)
            group_setups.append({
                "config": config,
                "seq_id": seq_id,
                "seq_dir": seq_dir,
                "video_dir": video_dir,
                "video_save_dir": video_save_dir,
                "target_video_dir": video_save_dir,
                "meta_dir": meta_dir,
                "meta_path": os.path.join(meta_dir, "metadata.jsonl"),
            })

        # 多组共用一个录制目录（与 OrcaManipulation 一致：只录制一次，输出到一个目录）
        unified_video_dir = group_setups[0]["video_save_dir"]

        # external_drive 模式已通过 config.apply_external_drive_mode() 规范化处理

        # 使用第一个组的配置初始化 env（所有组共享同一个 env）
        env_adapter = OrcaGymEnvAdapter(self.config)
        env_adapter.initialize()
        env = env_adapter.env
        base_env = env_adapter.base_env

        if self.config.auto_frame_skip:
            target = 1.0 / float(self.config.fps)
            self.config.frame_skip = max(1, int(round(target / float(self.config.time_step_s))))
            print(f"[OrcaGen] auto_frame_skip: time_step_s={self.config.time_step_s}, frame_skip={self.config.frame_skip}, realtime_step≈{self.config.time_step_s*self.config.frame_skip:.6f}s")

        realtime_step = float(self.config.time_step_s) * int(self.config.frame_skip)
        try:
            base_env.metadata["render_fps"] = int(self.config.render_fps)
        except Exception:
            pass

        # 为每个组解析对象和相机（使用第一个组的 runner 实例来调用辅助方法）
        for setup in group_setups:
            config = setup["config"]
            # 临时设置 config 以便调用辅助方法
            original_config = first_runner.config
            first_runner.config = config
            try:
                obj_ids, obj_body_by_id, obj_site_by_id, obj_id_by_body = first_runner._resolve_objects(base_env)
                cam_body = first_runner._find_camera_body(base_env)
                cam_pose_worldsim_pos, cam_pose_worldsim_mat = first_runner._get_camera_pose(env_adapter, cam_body, setup["seq_id"])
                obj_sizes = first_runner._infer_object_sizes(base_env, obj_ids, obj_body_by_id, obj_site_by_id)
            finally:
                first_runner.config = original_config
            
            setup["obj_ids"] = obj_ids
            setup["obj_body_by_id"] = obj_body_by_id
            setup["obj_site_by_id"] = obj_site_by_id
            setup["obj_id_by_body"] = obj_id_by_body
            setup["cam_body"] = cam_body
            setup["cam_pose_worldsim_pos"] = cam_pose_worldsim_pos
            setup["cam_pose_worldsim_mat"] = cam_pose_worldsim_mat
            setup["obj_sizes"] = obj_sizes
            setup["collision_times"] = {oid: None for oid in obj_ids}
            # 使用第一帧的相机坐标系作为世界坐标系
            setup["R0"] = cam_pose_worldsim_mat
            setup["t0"] = cam_pose_worldsim_pos

        # 启动视频录制（照搬 OrcaManipulation 的逻辑：只调用一次 begin_save_video）
        # 多组共用一个录制目录，不做分发
        cap_mode = CaptureMode.SYNC if self.config.capture_mode == "SYNC" else CaptureMode.ASYNC
        video_started = False
        if self.config.save_video:
            if self.config.capture_mode == "SYNC":
                print("[OrcaGen] 提示：你当前使用 --capture_mode SYNC；若 Editor 端出现 CameraSyncManager WaitingLastFrame Timeout，建议改为 ASYNC。")
            # 照搬 OrcaManipulation：只调用一次 begin_save_video（传递绝对路径）
            # 使用第一个组的目录作为统一录制目录（引擎不支持同时录制到多个目录）
            print(f"[OrcaGen] 多组录制：统一录制目录 = {unified_video_dir}")
            env_adapter.begin_save_video(os.path.abspath(unified_video_dir), cap_mode)
            video_started = True

        total_frames = int(round(self.config.duration_s * self.config.fps))
        do_render = not self.config.no_render
        render_err_count = 0

        # 用于友好打印：每隔3秒简述物体运动状态
        last_status_print_time = time.time()
        status_print_interval = 3.0
        recent_positions: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for setup in group_setups:
            recent_positions[setup["seq_id"]] = {oid: [] for oid in setup["obj_ids"]}

        # 打开所有组的元数据文件
        meta_files = []
        try:
            for setup in group_setups:
                f = open(setup["meta_path"], "w", encoding="utf-8")
                meta_files.append((setup, f))
            
            recorded = 0
            last_frame_idx = -1
            while recorded < total_frames:
                loop_start = time.time()

                if not self.config.no_drive_sim:
                    _ = env.step(env.action_space.sample() if env.action_space.shape != (0,) else np.array([]))
                else:
                    # external_drive 模式下，通过 gRPC 从服务器拉取最新状态
                    # 然后更新本地 _mjData，这样 query_body_xpos_xmat_xquat 才能获取到最新坐标
                    try:
                        # 通过 gRPC 直接查询状态
                        if hasattr(base_env, 'gym') and hasattr(base_env.gym, 'stub'):
                            # 动态导入 mjc_message_pb2（OrcaGym 的 protos 路径）
                            try:
                                import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2
                                request = mjc_message_pb2.QueryAllQposQvelQaccRequest()
                            except ImportError:
                                request = None
                            if request is not None:
                                # 使用同步调用（因为我们在同步代码中）
                                response = base_env.loop.run_until_complete(
                                    base_env.gym.stub.QueryAllQposQvelQacc(request)
                                )
                            else:
                                response = None
                            # 更新本地 _mjData
                            if response is not None and hasattr(base_env.gym, '_mjData') and base_env.gym._mjData is not None:
                                base_env.gym._mjData.qpos[:] = np.array(response.qpos)
                                base_env.gym._mjData.qvel[:] = np.array(response.qvel)
                                base_env.gym._mjData.qacc[:] = np.array(response.qacc)
                                base_env.gym._mjData.time = float(response.time) if hasattr(response, 'time') else base_env.gym._mjData.time
                                # 执行前向计算，更新 xpos, xmat, xquat 等
                                base_env.gym.mj_forward()
                                # 同步到封装的 data 对象
                                base_env.gym.update_data()
                    except Exception as e:
                        # 如果 gRPC 查询失败，静默处理（避免影响采集）
                        pass

                if do_render:
                    try:
                        env.render()
                        render_err_count = 0  # 成功时重置计数
                    except Exception:
                        render_err_count += 1
                        # external_drive 模式下减少警告打印
                        if not self.config.external_drive:
                            if render_err_count <= 5 or (render_err_count % 60 == 0):
                                print(f"[OrcaGen] 警告：env.render 失败（count={render_err_count}），将继续采集。")
                        elif render_err_count == 5:
                            do_render = False
                            print("[OrcaGen] external_drive 模式下连续 render 失败，后续将跳过 render，仅采集/录像。")

                try:
                    cur_frame = base_env.get_next_frame()
                except Exception:
                    cur_frame = -1
                if cur_frame >= 0 and cur_frame == last_frame_idx:
                    time.sleep(0.001)
                    continue
                last_frame_idx = cur_frame

                # 为每个组采集元数据
                for setup, f in meta_files:
                    config = setup["config"]
                    obj_ids = setup["obj_ids"]
                    obj_body_by_id = setup["obj_body_by_id"]
                    obj_site_by_id = setup["obj_site_by_id"]
                    cam_body = setup["cam_body"]
                    R0 = setup["R0"]
                    t0 = setup["t0"]
                    obj_sizes = setup["obj_sizes"]
                    collision_times = setup["collision_times"]

                    cam_pos_ws = env_adapter.get_body_pose(cam_body).pos if cam_body is not None else t0
                    cam_mat_ws = env_adapter.get_body_pose(cam_body).mat if cam_body is not None else R0
                    cam_pos_w0 = worldsim_to_world0(cam_pos_ws, R0, t0)
                    cam_mat_w0 = R0.T @ cam_mat_ws

                    if config.contacts_mode == "assume_ground":
                        has_contact = True
                        contact_pairs = [[oid, "ground"] for oid in obj_ids]
                        try:
                            contacts_simple = base_env.query_contact_simple()
                        except Exception:
                            contacts_simple = []
                        obj_pairs = first_runner._get_contact_pairs(base_env, contacts_simple, setup["obj_id_by_body"])
                        obj_set = set(obj_ids)
                        for p in obj_pairs:
                            if len(p) == 2 and p[0] in obj_set and p[1] in obj_set:
                                if p not in contact_pairs:
                                    contact_pairs.append(p)
                                for oid in p:
                                    if collision_times[oid] is None or collision_times[oid] == 0:
                                        collision_times[oid] = recorded
                    elif config.contacts_mode == "remote":
                        try:
                            contacts_simple = base_env.query_contact_simple()
                        except Exception:
                            contacts_simple = []
                        contact_pairs = first_runner._get_contact_pairs(base_env, contacts_simple, setup["obj_id_by_body"])
                        has_contact = len(contact_pairs) > 0
                        for p in contact_pairs:
                            for oid in obj_ids:
                                if oid in p and collision_times[oid] is None:
                                    collision_times[oid] = recorded
                    else:
                        has_contact = False
                        contact_pairs = None

                    annotations = []
                    for oid in obj_ids:
                        if config.object_source == "site":
                            site_name = obj_site_by_id[oid]
                            obj = env_adapter.get_site_pose(site_name)
                            body_name = obj_body_by_id.get(oid)
                        else:
                            body_name = obj_body_by_id[oid]
                            obj = env_adapter.get_body_pose(body_name)
                        p_cam = cam_mat_ws.T @ (obj.pos - cam_pos_ws)
                        R_cam_obj = cam_mat_ws.T @ obj.mat
                        pitch_deg, yaw_deg, roll_deg = mat_to_euler_xyz_deg(R_cam_obj)
                        pitch = norm_angle_deg_to_unit(pitch_deg)
                        yaw = norm_angle_deg_to_unit(yaw_deg)
                        roll = norm_angle_deg_to_unit(roll_deg)
                        x_size, y_size, z_size = obj_sizes[oid]
                        annotations.append({
                            "object_id": oid,
                            "body_name": body_name,
                            "site_name": obj_site_by_id.get(oid),
                            "bbox3d": {
                                "x_center": float(p_cam[0]),
                                "y_center": float(p_cam[1]),
                                "z_center": float(p_cam[2]),
                                "x_size": float(x_size),
                                "y_size": float(y_size),
                                "z_size": float(z_size),
                                "pitch": pitch,
                                "yaw": yaw,
                                "roll": roll,
                            },
                            "bbox3d_array": [
                                float(p_cam[0]), float(p_cam[1]), float(p_cam[2]),
                                float(x_size), float(y_size), float(z_size),
                                pitch, yaw, roll,
                            ],
                        })

                    rec = {
                        "frame_index": recorded,
                        "timestamp_s": round(recorded / float(config.fps), 10),
                        "sim_time_s": float(getattr(base_env.data, "time", 0.0)),
                        "camera": {
                            "camera_id": "main",
                            "body_name": cam_body,
                            "camera_name": config.camera_name if cam_body is None else None,
                            "pose_world0": {
                                "pos": [float(x) for x in cam_pos_w0.tolist()],
                                "mat3x3_rowmajor": [float(x) for x in cam_mat_w0.reshape(-1).tolist()],
                            },
                        },
                        "annotations": annotations,
                        "contacts": {"has_contact": bool(has_contact), "pairs": contact_pairs},
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
                    # 收集最近的位置用于状态打印
                    for ann in annotations:
                        oid = ann["object_id"]
                        bbox = ann["bbox3d"]
                        pos = np.array([bbox["x_center"], bbox["y_center"], bbox["z_center"]], dtype=np.float64)
                        if oid in recent_positions[setup["seq_id"]]:
                            recent_positions[setup["seq_id"]][oid].append(pos)
                            # 只保留最近30帧
                            if len(recent_positions[setup["seq_id"]][oid]) > 30:
                                recent_positions[setup["seq_id"]][oid].pop(0)

                recorded += 1

                # 每隔3秒打印一次物体运动状态
                current_time = time.time()
                if current_time - last_status_print_time >= status_print_interval:
                    elapsed_s = recorded / float(self.config.fps)
                    progress_pct = (recorded / max(total_frames, 1)) * 100.0
                    status_lines = [f"[OrcaGen] 多组采集进度: {elapsed_s:.1f}s / {self.config.duration_s:.1f}s ({progress_pct:.1f}%)"]
                    for idx, (setup, _) in enumerate(meta_files):
                        # 组索引从1开始显示（组 1, 组 2, 组 3, 组 4）
                        group_display_idx = idx + 1
                        status_lines.append(f"  组 {group_display_idx} ({setup['seq_id']}):")
                        for oid in setup["obj_ids"]:
                            pos_list = recent_positions.get(setup["seq_id"], {}).get(oid, [])
                            if len(pos_list) >= 2:
                                recent_dp = pos_list[-1] - pos_list[0]
                                recent_speed = np.linalg.norm(recent_dp) / max((len(pos_list) - 1) / float(self.config.fps), 1e-6)
                                status_lines.append(f"    {oid}: 位置=({pos_list[-1][0]:.3f}, {pos_list[-1][1]:.3f}, {pos_list[-1][2]:.3f})m, 速度≈{recent_speed:.3f}m/s")
                            else:
                                status_lines.append(f"    {oid}: 位置数据不足")
                    print("\n".join(status_lines))
                    last_status_print_time = current_time

                if self.config.use_realtime_loop:
                    elapsed = time.time() - loop_start
                    if elapsed < realtime_step:
                        time.sleep(realtime_step - elapsed)
        finally:
            # 关闭所有元数据文件
            for _, f in meta_files:
                f.close()
            
            # 停止视频录制（照搬 OrcaManipulation 的逻辑：只调用一次 stop_save_video）
            if video_started:
                try:
                    env_adapter.stop_save_video()
                    # 等待视频稳定
                    first_runner._wait_for_video_stable(unified_video_dir, timeout_s=5.0)
                    if len(group_setups) > 1:
                        first_runner._distribute_videos_by_camera(unified_video_dir, group_setups)
                except Exception as e:
                    print(f"[OrcaGen] 警告：stop_save_video 失败（可能导致视频未 finalize）：{e}")

        # 为每个组生成 index.json 和运动分析
        for idx, setup in enumerate(group_setups):
            config = setup["config"]
            obj_ids = setup["obj_ids"]
            obj_body_by_id = setup["obj_body_by_id"]
            obj_site_by_id = setup["obj_site_by_id"]
            cam_body = setup["cam_body"]
            collision_times = setup["collision_times"]
            obj_sizes = setup["obj_sizes"]
            # 组索引从1开始显示（组 1, 组 2, 组 3, 组 4）
            group_display_idx = idx + 1
            
            motion_map = infer_motion_types(setup["meta_path"], obj_ids) if config.infer_motion else {oid: MotionType.UNKNOWN for oid in obj_ids}
            index_path = os.path.join(setup["meta_dir"], "index.json")
            size_source = "site_size" if config.object_source == "site" else "geom_size"
            camera_name_filter = setup.get("camera_name_override", config.camera_name)
            video_artifacts, camera_ids = first_runner._collect_video_artifacts(
                video_save_dir=setup.get("target_video_dir", setup["video_save_dir"]),
                seq_dir=setup["seq_dir"],
                camera_name_filter=camera_name_filter,
            ) if config.save_video else ([], [])
            if not video_artifacts:
                video_artifacts = [{"path": f"video/{config.video_subdir}.mp4", "camera_id": "main"}]
            index = {
                "version": "orcagen-metadata-0.1",
                "sequence_id": setup["seq_id"],
                "object": {"object_id": obj_ids[0], "name": obj_ids[0], "part": "main"},
                "motion_mode": motion_map.get(obj_ids[0], MotionType.UNKNOWN).value,
                "motion_modes": {oid: motion_map.get(oid, MotionType.UNKNOWN).value for oid in obj_ids},
                "motion_start": 0,
                "motion_end": total_frames - 1,
                "collision_time": collision_times.get(obj_ids[0]),
                "collision_times": collision_times,
                "coordinate_system": {
                    "world_definition": "world0 == camera(main) frame0",
                    "axes": {"x": "right", "y": "down", "z": "forward"},
                    "origin": "camera(main) position at frame0",
                    "units": {"position": "meter", "angles": "deg", "angle_norm": "(-1,1)*180"},
                },
                "capture": {
                    "fps": int(config.fps),
                    "resolution": first_runner._parse_resolution(),
                    "duration_s": float(config.duration_s),
                    "render_style": config.render_style,
                    "capture_mode": config.capture_mode,
                    "time_step_s": float(config.time_step_s),
                    "frame_skip": int(config.frame_skip),
                },
                "artifacts": {
                    "video": video_artifacts,
                    "project": {"path": "project/scene.usd", "format": "USD", "contains": "keyframes+dyn_cache"},
                },
                "objects": [
                    {
                        "object_id": oid,
                        "body_name": obj_body_by_id.get(oid),
                        "site_name": obj_site_by_id.get(oid),
                        "semantic": {"class": "rigid", "category": "rigid"},
                        "canonical_size_m": [float(obj_sizes[oid][0]), float(obj_sizes[oid][1]), float(obj_sizes[oid][2])],
                        "motion_mode": motion_map.get(oid, MotionType.UNKNOWN).value,
                        "size_source": size_source,
                    }
                    for oid in obj_ids
                ],
                "cameras": first_runner._build_camera_entries(
                    cam_body=cam_body,
                    camera_ids=camera_ids,
                ),
            }
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

            if config.plot_motion:
                first_runner._plot_motion(setup["meta_path"], setup["meta_dir"])
                # 打印运动判断结果和提示
                if config.infer_motion:
                    print(f"\n[OrcaGen] ========== 组 {group_display_idx} 运动模式判断结果 ==========")
                    for oid in obj_ids:
                        motion_type = motion_map.get(oid, MotionType.UNKNOWN)
                        print(f"  物体 {oid}: {motion_type.value}")
                    print("[OrcaGen] 提示：如果判断有误，请检查 scripts/motion_analysis.py 中的判断逻辑，")
                    print("          或调整阈值参数（speed_small, speed_move, omega_small 等）。")
                    print("=" * 50)

            print("OK")
            print("sequence_dir:", setup["seq_dir"])
            print("index.json:", index_path)
            print("metadata.jsonl:", setup["meta_path"])
            if config.save_video:
                print("video_dir:", setup.get("target_video_dir", setup["video_save_dir"]))

        env_adapter.close()
        print("[OrcaGen] 所有组录制完成")


