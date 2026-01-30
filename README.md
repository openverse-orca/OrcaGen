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
- 默认仿真命令：`python -m scripts.run_sim_loop_orcagen`（OrcaGen 定制版，负责节拍）；可用 `--sim_loop_cmd`、`--sim_loop_wait_s`、`--no_kill_sim_on_exit`、`--orcagym_addr` 覆盖。

注意：
- 使用定制 `run_sim_loop_orcagen` 负责节拍时：**OrcaStudio 请解锁步进/解锁视频帧**（避免双时钟节流导致“时长翻倍/进度滞后”）。

### runsimloop 重构说明（OrcaGen 定制版）与视频帧相关

本项目新增了 **OrcaGen 定制版 runsimloop**：`scripts/run_sim_loop_orcagen.py`，用于让“仿真进程”成为唯一的节拍源（类似 OrcaManipulation 的 `DataCollectionManager`：每步计算耗时并 `sleep` 对齐节拍）。

#### 为什么需要“解锁视频帧/解锁步进”

当仿真由 Python 侧循环驱动（`run_sim_loop_orcagen` 或 OrcaGym 的 `run_sim_loop`）时，通常存在：

- **节拍源 A（Python）**：每步 `env.step()` + `env.render()`，并 `sleep(target_dt - elapsed)` 对齐节拍；
- **节拍源 B（OrcaStudio）**：若你在 OrcaStudio 里锁步进/锁视频帧，相当于又引入一套节拍/节流机制。

两套节拍叠加会导致：
- **有效相机帧率下降**（常见从 30fps 变成 15fps 或不稳定）
- **视频时长“翻倍”**（同样帧数按更低 fps 播放）
- **采集进度/保存进度滞后**（现实时间过了很久，采集进度只走了一点）

因此：**当使用 `run_sim_loop_orcagen` 作为节拍源时，请在 OrcaStudio 解锁步进/解锁视频帧**，避免双时钟节流。

#### 推荐用法

- **标准双进程采集（仿真由 OrcaGen 定制 runsimloop 驱动）**：

```bash
python examples/run_capture_with_sim.py --object_sites_groups "Bin1|Bin2|Bin3|Bin4" --duration_s 20 --auto_frame_skip
```

- **手动单独启动定制 runsimloop**（建议 OrcaStudio 解锁步进/视频帧）：

```bash
python -m scripts.run_sim_loop_orcagen --orcagym_addr localhost:50051
```

- **外部仿真已启动（不再起 runsimloop，仅采集）**：

```bash
python examples/run_capture_with_sim.py --external_sim --object_sites_groups "..." --duration_s 20 --auto_frame_skip
```

#### 常见问题排查

- **现象：mp4 时长明显变长/翻倍**  
  - **优先检查**：OrcaStudio 是否仍在锁步进/锁视频帧；使用 `run_sim_loop_orcagen` 时应解锁。

- **现象：采集进度明显落后于真实时间**  
  - 常见原因是 `render()` 阻塞或双时钟节流导致单步耗时长期大于目标步长。
  - 可尝试：降低渲染频率（见下一条）、或确保 OrcaStudio 解锁相关选项。

- **想降低渲染/相机压力（更稳）**  
  - `run_sim_loop_orcagen` 支持 `--no_render`（不渲染）以及 `--render_every N`（每 N 步 render 一次）等参数。

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


