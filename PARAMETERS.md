# OrcaGen 参数列表

本文档列出了 OrcaGen 的所有命令行参数及其默认值，供审查和优化。

## 连接与基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--orcagym_addr` | str | `localhost:50051` | OrcaGym gRPC 服务器地址 |
| `--agent_name` | str | `NoRobot` | 智能体名称 |
| `--output_root` | str | `.` | 输出根目录（默认：当前目录） |

## 录制配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--duration_s` | float | `30.0` | 录制时长（秒）**（可交互式输入）** |
| `--fps` | int | `30` | 目标帧率 |
| `--resolution` | str | `2560x1440` | 分辨率 |
| `--render_style` | str | `PBR` | 渲染风格 |
| `--save_video` | bool | `True` | 保存视频（默认启用） |
| `--no_save_video` | bool | - | 禁用视频录制 |
| `--capture_mode` | str | `ASYNC` | 捕获模式：`SYNC` 或 `ASYNC` |

## 仿真参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--time_step_s` | float | `0.001` | 仿真时间步长（秒） |
| `--frame_skip` | int | `20` | 帧跳过数 |
| `--auto_frame_skip` | bool | `False` | 自动计算 frame_skip 以匹配 fps |
| `--sync_time_step_with_server` | bool | `True` | 与服务器同步时间步长 |
| `--no_sync_time_step_with_server` | bool | `False` | 禁用时间步长同步 |

## 物体配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--object_source` | str | `site` | 物体来源：`site` 或 `body` |
| `--object_body` | str | `Sphere1` | 单个物体 body 名称 |
| `--object_bodies` | str | `None` | 多个物体 body 名称（逗号分隔） |
| `--object_site` | str | `None` | 单个物体 site 名称 |
| `--object_sites` | str | `None` | 多个物体 site 名称（逗号分隔） |
| `--object_sites_groups` | str | `None` | 多组物体（用 `\|` 分隔，例如：`Bin1\|Bin2`）**（可自动提取前缀）** |
| `--object_ids` | str | `None` | 自定义物体 ID（逗号分隔） |

## 相机配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--camera_body` | str | `None` | 相机 body 名称 |
| `--camera_name` | str | `Camera` | 相机名称 |

## 仿真控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--external_drive` | bool | `False` | 外部程序驱动仿真**（可交互式输入）**<br>自动设置 `no_drive_sim=True` 和 `no_reset=True` |
| `--no_render` | bool | `True` | 禁用渲染（默认不渲染） |
| `--render` | bool | `False` | 显式启用渲染（覆盖 external_drive 默认行为） |
| `--render_fps` | int | `30` | 渲染帧率 |
| `--no_use_realtime_loop` | bool | - | 禁用实时循环（默认启用） |

**注意**：
- `--no_drive_sim` 和 `--no_reset` 已移除，由 `--external_drive` 自动管理
- `--use_realtime_loop` 已移除，默认启用，使用 `--no_use_realtime_loop` 禁用

## 碰撞检测

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--contacts_mode` | str | `assume_ground` | 碰撞模式：`assume_ground`、`remote` 或 `none` |

## WebSocket 视频（高级）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ws_video` | bool | `False` | 启用 WebSocket 视频录制 |
| `--ws_video_port` | int | `7070` | WebSocket 端口 |
| `--ws_video_name` | str | `Camera` | WebSocket 相机名称 |
| `--ws_video_to_mp4` | bool | `False` | 转换为 MP4 |
| `--ws_video_wait_first_packet_s` | float | `5.0` | 等待第一个数据包的超时时间 |
| `--ws_video_recv_timeout_s` | float | `1.0` | 接收超时时间 |
| `--ws_video_min_packets_to_mp4` | int | `10` | 转换为 MP4 所需的最小数据包数 |

## 输出配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--video_subdir` | str | `rgb_main` | 视频子目录 |
| `--sequence_id` | str | `None` | 序列 ID（完整） |
| `--sequence_prefix` | str | `None` | 序列前缀**（可从 object_sites_groups 自动提取）** |

## 后处理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--infer_motion` | bool | `True` | 推断运动模式 |
| `--no_infer_motion` | bool | `False` | 禁用运动模式推断 |
| `--plot_motion` | bool | `True` | 绘制运动曲线 |
| `--normalize_video` | bool | `True` | 标准化视频速度 |
| `--no_normalize_video` | bool | `False` | 禁用视频标准化 |

## 交互式功能

以下参数支持交互式输入（可通过环境变量 `ORCAGEN_NO_INTERACTIVE=1` 禁用）：

1. **`--duration_s`**：如果使用默认值，会提示输入录制时长
2. **`--external_drive`**：如果未指定，会询问仿真是否已由外部程序启动
3. **`--sequence_prefix`**：如果提供了 `--object_sites_groups` 但未指定前缀，会自动提取并询问确认

## 自动功能

1. **自动提取前缀**：当提供 `--object_sites_groups` 时，如果未指定 `--sequence_prefix`，会自动从第一个组提取前缀（例如：`Bin1|Bin2` → `bin`）

## 参数优化历史

### 已完成的优化：

1. ✅ **`--output_root`**：从绝对路径改为相对路径（`.`）
2. ✅ **`--save_video`**：默认值从 `False` 改为 `True`，添加 `--no_save_video` 禁用
3. ✅ **仿真控制参数简化**：
   - 移除 `--no_drive_sim` 和 `--no_reset` 作为独立参数，由 `--external_drive` 自动管理
   - 移除 `--use_realtime_loop`，默认启用，使用 `--no_use_realtime_loop` 禁用
   - 保留 `--render` / `--no_render` 用于显式控制渲染

### 仍可优化的参数对：

1. **`--infer_motion` / `--no_infer_motion`**：可以用单一参数 `--no_infer_motion` 表示禁用
2. **`--normalize_video` / `--no_normalize_video`**：可以用单一参数 `--no_normalize_video` 表示禁用
3. **`--sync_time_step_with_server` / `--no_sync_time_step_with_server`**：可以用单一参数 `--no_sync_time_step_with_server` 表示禁用

### 未来优化建议：

1. **简化布尔参数**：统一使用 `--no_xxx` 的形式表示禁用（默认启用）
2. **分组相关参数**：将 WebSocket 相关参数合并为一个子命令或配置文件
3. **默认值优化**：根据实际使用场景持续调整默认值

## 使用示例

```bash
# 基本使用（交互式）
python examples/run_capture.py \
  --object_sites "CardBoxA_01" \
  --save_video

# 多组录制（自动提取前缀）
python examples/run_capture.py \
  --object_sites_groups "Bin1|Bin2|Bin3|Bin4" \
  --save_video

# 外部驱动模式（交互式确认）
python examples/run_capture.py \
  --object_sites "CardBoxA_01" \
  --external_drive \
  --save_video

# 禁用交互式输入
ORCAGEN_NO_INTERACTIVE=1 python examples/run_capture.py \
  --object_sites "CardBoxA_01" \
  --duration_s 60 \
  --save_video
```
