# OrcaGen

基于 OrcaGym 的多模态仿真数据采集工具，输出视频与逐帧结构化元数据（JSON/JSONL）。

## 目录结构（对齐 OrcaGym 风格）
- `envs/`：环境适配
- `examples/`：运行入口示例（推荐）
- `scripts/`：工具脚本（曲线绘制、ws 录像、运动分析）
- `orcagen/`：核心配置/基础类/枚举/工具

## 快速开始

### 系统依赖
- OrcaStudio/OrcaSim（负责渲染与仿真）
- OrcaGym Server（gRPC 服务，默认端口 `50051`）
- ffmpeg（可选：用于 ws 录像转码/视频后处理）

### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 启动采集
```bash
python examples/run_capture.py \
  --orcagym_addr localhost:50051 \
  --object_site 待录制物体 \
  --duration_s 30 --fps 30 \
  --auto_frame_skip \
  --save_video --capture_mode ASYNC
```

说明：
- `--external_drive`：由外部程序驱动仿真，本程序仅采集/录像
- `--save_video`：服务端录制视频（gRPC）
- 默认 `no_render=True`，如需 render 可在配置里调整

### 双进程采集（推荐：无外部仿真时）
仿真与采集分两进程，避免单进程干扰导致运动顿挫。先启仿真（run_sim_loop），等 gRPC 就绪后再启采集。

```bash
python examples/run_capture_with_sim.py --object_sites_groups "Bin1|Bin2|Bin3|Bin4" --duration_s 20 --auto_frame_skip
```

- 仿真已由外部启动时加 `--external_sim`，不启 run_sim_loop，只跑采集。
- 默认仿真命令：`python -m orca_gym.scripts.run_sim_loop`；可用 `--sim_loop_cmd`、`--sim_loop_wait_s`、`--no_kill_sim_on_exit`、`--orcagym_addr`。

注意：
- 为确保视频时长正确，需要在 OrcaStudio 中将 MuJoCoCamera 锁定为 30 帧：
  Home 键唤起菜单 → Video Options → 选择 30

## 输出
```
sequence_xxx/
  video/
  project/
  metadata/
```

## 工具脚本
- 运动曲线：`python -m scripts.plot_motion_curves --metadata_jsonl ... --out_dir ...`
- WS 录像：`python -m scripts.record_ws_video --host localhost --port 7070 --duration_s 30 --to_mp4`


