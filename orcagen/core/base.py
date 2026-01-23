from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any

from orcagen.core.types import Pose


class BaseEnvAdapter(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action: Any) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def get_body_pose(self, body_name: str) -> Pose:
        pass

    @abstractmethod
    def get_site_pose(self, site_name: str) -> Pose:
        pass

    @abstractmethod
    def get_camera_pose(self, camera_body: Optional[str], camera_name: str, probe_dir: str) -> Pose:
        pass

    @abstractmethod
    def get_current_frame(self) -> int:
        pass

    @abstractmethod
    def get_next_frame(self) -> int:
        pass

    @abstractmethod
    def begin_save_video(self, video_dir: str, capture_mode: Any) -> None:
        pass

    @abstractmethod
    def stop_save_video(self) -> None:
        pass

    @property
    @abstractmethod
    def env(self) -> Any:
        pass

    @property
    @abstractmethod
    def base_env(self) -> Any:
        pass


class BaseRunner(ABC):
    @abstractmethod
    def run(self) -> None:
        pass

