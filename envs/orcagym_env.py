from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Optional

from orcagen.core.base import BaseEnvAdapter
from orcagen.core.config import CaptureConfig
from orcagen.core.types import Pose
from orcagen.utils.ids import now_id
from orcagen.utils.pose import get_body_pose, get_site_pose


class OrcaGymEnvAdapter(BaseEnvAdapter):
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._env_id: Optional[str] = None
        self._env = None
        self._base_env = None

    @property
    def env(self) -> Any:
        return self._env

    @property
    def base_env(self) -> Any:
        return self._base_env

    def _register_sim_env(self) -> str:
        orcagym_addr_str = self.config.orcagym_addr.replace(":", "-")
        env_id = f"OrcaGenCapture-SimulationLoop-OrcaGym-{orcagym_addr_str}-{now_id()}"
        kwargs = {
            "frame_skip": self.config.frame_skip,
            "orcagym_addr": self.config.orcagym_addr,
            "agent_names": [self.config.agent_name],
            "time_step": self.config.time_step_s,
        }
        gym.register(
            id=env_id,
            entry_point="orca_gym.scripts.sim_env:SimEnv",
            kwargs=kwargs,
            max_episode_steps=10_000_000,
            reward_threshold=0.0,
        )
        return env_id

    def initialize(self) -> None:
        self._env_id = self._register_sim_env()
        self._env = gym.make(self._env_id)
        if not self.config.no_reset:
            self._env.reset()
        else:
            try:
                self._env.unwrapped.mj_forward()
            except Exception:
                pass
        self._base_env = self._env.unwrapped
        if self.config.sync_time_step_with_server:
            self._sync_time_step_with_server()

    def _sync_time_step_with_server(self) -> None:
        try:
            opt = self._base_env.gym.query_opt_config()
            server_ts = float(opt.get("timestep", self.config.time_step_s))
        except Exception:
            server_ts = self.config.time_step_s

        if abs(server_ts - float(self.config.time_step_s)) > 1e-9:
            print(f"[OrcaGen] server opt.timestep={server_ts} differs from args.time_step_s={self.config.time_step_s}, recreating env.")
            self._env.close()
            self.config.time_step_s = server_ts
            if self.config.auto_frame_skip:
                target = 1.0 / float(self.config.fps)
                self.config.frame_skip = max(1, int(round(target / float(self.config.time_step_s))))
            self._env_id = self._register_sim_env()
            self._env = gym.make(self._env_id)
            if not self.config.no_reset:
                self._env.reset()
            self._base_env = self._env.unwrapped

    def reset(self) -> None:
        self._env.reset()

    def step(self, action: Any) -> None:
        _ = self._env.step(action)

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()

    def get_body_pose(self, body_name: str) -> Pose:
        return get_body_pose(self._base_env, body_name)

    def get_site_pose(self, site_name: str) -> Pose:
        return get_site_pose(self._base_env, site_name)

    def get_camera_pose(self, camera_body: Optional[str], camera_name: str, probe_dir: str) -> Pose:
        if camera_body is not None:
            return get_body_pose(self._base_env, camera_body)
        cam_ret = self._base_env.get_frame_png(probe_dir)
        if camera_name not in cam_ret:
            raise RuntimeError(f"[OrcaGen] 未找到相机 '{camera_name}' 的位姿。get_frame_png 返回: {list(cam_ret.keys())}")
        pos = np.asarray(cam_ret[camera_name]["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(cam_ret[camera_name]["quat"], dtype=np.float64).reshape(4)
        mat = np.asarray(cam_ret[camera_name]["mat"], dtype=np.float64).reshape(3, 3) if "mat" in cam_ret[camera_name] else None
        if mat is None:
            from scipy.spatial.transform import Rotation as SciRot
            mat = SciRot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        return Pose(pos=pos, mat=mat, quat=quat)

    def get_current_frame(self) -> int:
        try:
            return int(self._base_env.get_current_frame())
        except Exception:
            return -1

    def get_next_frame(self) -> int:
        try:
            return int(self._base_env.get_next_frame())
        except Exception:
            return -1

    def begin_save_video(self, video_dir: str, capture_mode: Any) -> None:
        self._base_env.begin_save_video(video_dir, capture_mode)

    def stop_save_video(self) -> None:
        self._base_env.stop_save_video()

