# OrcaGen 逐帧结构化元数据（JSON）设计规范 v0.1

> 目标：基于 OrcaGym 仿真生成的 **运动视频（交付物 A）** 与 **3D 原始工程文件（交付物 B）**，产出可规模化合成与训练使用的 **逐帧结构化元数据**。  
> 本规范优先覆盖近期刚需：**1-2 个物体运动 + 逐帧抓取 bbox/3D 坐标 + 语义信息 + 碰撞帧**。

---

## 1. 交付物与目录组织（推荐）

一个样本（一个运动片段 / 一个场景运行）建议落为一个 `sequence` 目录：

```
sequence_<uuid>/
  video/
    rgb_main.mp4
    rgb_main.meta.json          # 可选：ffprobe/编码信息等
  project/
    scene.usd | scene.blend     # 交付物B（待定其一）
    cache/                      # 动力学解算缓存 / baking 结果（若有）
  metadata/
    metadata.jsonl              # 推荐：JSON Lines（逐帧一行）
    index.json                  # 推荐：全局信息 + 统计 + 快速索引
```

说明：
- `index.json` 存 **全局信息/静态信息**，`metadata.jsonl` 存 **逐帧信息**。
- 若你强制要求“一个 JSON 串里包含全局 key + frames[]”，也可以用 `metadata.json`（见第 2.3 节），但 **900+ 帧**时文件会很大、编辑/追加不便。

---

## 2. 两种 JSON 组织方式（按 30 秒视频举例）

30 秒、30 FPS ⇒ **900 帧**（frame_index: 0..899）。

### 2.1 推荐：JSON Lines（`metadata/metadata.jsonl`）——逐帧一行
- 优点：易流式写入、断点续写、并行消费、对大规模数据更稳。
- 每行是一个 frame record（JSON object），天然包含 `frame_index`。

### 2.2 推荐：单独的 `index.json` ——承载全局 key
- 把你要求的全局 key（`object/motion_mode/motion_start/...`）放到 `index.json` 顶层。
- `index.json` 同时存放：坐标系定义、视频参数、对象静态属性（语义/尺寸/部件）、工程文件引用等。

### 2.3 可选：单文件 JSON（`metadata/metadata.json`）——顶层 + `frames[]`
适合短片段或调试：
- 文件结构：`{ global_keys..., frames:[...] }`
- 缺点：写入必须一次性落盘或频繁重写，900 帧以上会明显变慢。

---

## 3. 坐标系与 3D 框定义（强制）

### 3.1 坐标系（必须写入 `index.json.coordinate_system`）
- **世界坐标系 = 第一帧相机坐标系（camera@frame0）**
- 轴向：**X 向右、Y 向下、Z 向前**
- 原点：**相机位置（frame0）**

> 记为 `world0`。后续所有“世界坐标”都指 `world0`。

### 3.2 3D 框（bbox3d）格式（必须与训练侧一致）
你指定的格式：

`<3dbbox>x_center y_center z_center x_size y_size z_size pitch yaw roll</3dbbox>`

含义：
- `x_center,y_center,z_center`：目标中心在 **当前帧相机坐标系**（camera_t）中沿 XYZ 的位置（单位：米）
- `x_size,y_size,z_size`：旋转角为零时目标沿 XYZ 的尺寸（单位：米）
- `pitch,yaw,roll`：围绕 XYZ 轴的欧拉角，**归一化到 (-1,1)**，乘以 180 得到角度（deg）

规范建议：
- 在 JSON 中以对象存储，并可额外提供 `bbox3d_array`（严格数组）保证和 `<3dbbox>` 序一致。
- 角度归一化务必定义清楚：`angle_deg = angle_norm * 180`（因此 angle_norm 必须在 (-1,1)）。

---

## 4. `index.json`（全局/静态）字段设计

### 4.1 必选顶层字段
- `version`：规范版本，例如 `"orcagen-metadata-0.1"`
- `sequence_id`：uuid 或可追溯 ID
- `object`：主目标（你要求的全局 key）
- `motion_mode`：运动类别（你要求的全局 key）
- `motion_start`：运动开始帧（你要求的全局 key）
- `motion_end`：运动结束帧（你要求的全局 key）
- `collision_time`：碰撞帧（你要求的全局 key；无则为 `null`）

### 4.2 强烈建议字段
- `coordinate_system`：见 3.1
- `capture`：与交付物 A 对齐
  - `fps`（应为 30）
  - `resolution`（如 `[1920,1080]` 或更高）
  - `duration_s`（30~60）
  - `render_style`（`"PBR"` / `"NPR"` / `"MIXED"`，若同一序列中会切换，可用数组区间表示）
  - `codec`（可选，如 h264/h265）
  - `capture_mode`（建议记录 OrcaGym `CaptureMode.SYNC/ASYNC`）
  - `time_step_s`、`frame_skip`（仿真码率相关，便于重建对齐）
- `artifacts`
  - `video`：`{"path":"video/rgb_main.mp4","camera":"main"}`
  - `project`：`{"path":"project/scene.usd","format":"USD","contains":"keyframes+dyn_cache"}`
  - `cache`：缓存目录（若有）
- `objects[]`：场景中可标注对象列表（即使目前只有 1-2 个也建议用数组）
  - `object_id`（稳定 ID，用于 frame 里引用）
  - `name` / `semantic.class` / `semantic.category`
  - `parts[]`（可选）：部件级标注时启用（例如 `"pendulum_bob"`, `"pendulum_rod"`）
  - `canonical_size_m`：用于 `x_size,y_size,z_size` 的来源（例如从 geom extents、资产元数据）
  - `size_source`：`"asset_metadata" | "geom_extent" | "manual"`
- `cameras[]`：相机列表（至少一个 `main`）
  - `camera_id`、`name`
  - `intrinsics`（可选，但推荐：fx,fy,cx,cy 或 K）
  - `extrinsics_world0`（frame0 的外参，若 `world0==camera0`，则 main 相机为单位变换）

---

## 5. `metadata.jsonl`（逐帧）字段设计

每一行：一个 JSON object（frame record）。

### 5.1 必选字段（每帧）
- `frame_index`：0..N-1
- `timestamp_s`：视频时间（建议 `frame_index / fps`）
- `sim_time_s`：仿真时间（MuJoCo time；若可获得）
- `camera`：当前帧使用的相机（通常为 main）
  - `camera_id`
  - `pose_world0`：相机在 `world0` 下的位姿（pos+quat 或 4x4）
- `annotations[]`：该帧所有标注目标（1-2 个也用数组）
  - `object_id`
  - `part_id`（可选）
  - `bbox3d`
    - `x_center,y_center,z_center,x_size,y_size,z_size,pitch,yaw,roll`
  - `bbox3d_array`（可选）：严格数组 `[x_center,y_center,z_center,x_size,y_size,z_size,pitch,yaw,roll]`
  - `pose_cam`（推荐）：目标在 **camera_t** 下位姿（pos+quat），便于未来扩展更复杂标注
- `contacts`：用于碰撞检测/碰撞帧定位
  - `has_contact`：bool
  - `pairs`：可选，记录 `(object_id_a, object_id_b)` 或 `(geom1,geom2)` 等

### 5.2 `collision_time` 的推荐定义
- **collision_time = 第一次 `contacts.has_contact==true` 的帧**  
若你关心“与谁碰撞”，则在 `contacts.pairs` 记录并在 `index.json.events` 聚合统计。

---

## 6. 最小示例

### 6.1 `index.json` 示例（截断）
```json
{
  "version": "orcagen-metadata-0.1",
  "sequence_id": "8a3f2c1d-....",
  "object": {"object_id": "block_001", "name": "block", "part": "main"},
  "motion_mode": "conveyor_roll",
  "motion_start": 0,
  "motion_end": 899,
  "collision_time": null,
  "coordinate_system": {
    "world_definition": "world0 == camera(main) frame0",
    "axes": {"x": "right", "y": "down", "z": "forward"},
    "origin": "camera(main) position at frame0",
    "units": {"position": "meter", "angles": "deg", "angle_norm": "(-1,1)*180"}
  },
  "capture": {
    "fps": 30,
    "resolution": [1920, 1080],
    "duration_s": 30.0,
    "render_style": "PBR",
    "capture_mode": "SYNC",
    "time_step_s": 0.005,
    "frame_skip": 4
  },
  "artifacts": {
    "video": [{"path": "video/rgb_main.mp4", "camera_id": "main"}],
    "project": {"path": "project/scene.usd", "format": "USD", "contains": "keyframes+dyn_cache"}
  },
  "objects": [
    {
      "object_id": "block_001",
      "semantic": {"class": "block", "category": "rigid"},
      "canonical_size_m": [0.08, 0.05, 0.03],
      "size_source": "geom_extent"
    }
  ],
  "cameras": [
    {
      "camera_id": "main",
      "name": "default_camera",
      "extrinsics_world0": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}
    }
  ]
}
```

### 6.2 `metadata.jsonl` 示例（3 行）
```json
{"frame_index":0,"timestamp_s":0.0,"sim_time_s":0.0,"camera":{"camera_id":"main","pose_world0":{"pos":[0,0,0],"quat":[1,0,0,0]}},"annotations":[{"object_id":"block_001","bbox3d":{"x_center":0.12,"y_center":-0.03,"z_center":1.25,"x_size":0.08,"y_size":0.05,"z_size":0.03,"pitch":0.00,"yaw":-0.12,"roll":0.01},"bbox3d_array":[0.12,-0.03,1.25,0.08,0.05,0.03,0.00,-0.12,0.01]}],"contacts":{"has_contact":false,"pairs":[]}}
{"frame_index":1,"timestamp_s":0.0333333333,"sim_time_s":0.02,"camera":{"camera_id":"main","pose_world0":{"pos":[0,0,0],"quat":[1,0,0,0]}},"annotations":[{"object_id":"block_001","bbox3d":{"x_center":0.125,"y_center":-0.031,"z_center":1.252,"x_size":0.08,"y_size":0.05,"z_size":0.03,"pitch":0.00,"yaw":-0.11,"roll":0.02},"bbox3d_array":[0.125,-0.031,1.252,0.08,0.05,0.03,0.00,-0.11,0.02]}],"contacts":{"has_contact":false,"pairs":[]}}
{"frame_index":2,"timestamp_s":0.0666666667,"sim_time_s":0.04,"camera":{"camera_id":"main","pose_world0":{"pos":[0,0,0],"quat":[1,0,0,0]}},"annotations":[{"object_id":"block_001","bbox3d":{"x_center":0.131,"y_center":-0.032,"z_center":1.255,"x_size":0.08,"y_size":0.05,"z_size":0.03,"pitch":0.01,"yaw":-0.10,"roll":0.02},"bbox3d_array":[0.131,-0.032,1.255,0.08,0.05,0.03,0.01,-0.10,0.02]}],"contacts":{"has_contact":true,"pairs":[["block_001","conveyor_belt"]]}}
```

---

## 7. 关于“逐帧提取”与仿真码率/对齐的落盘建议

### 7.1 视频帧率 vs 仿真步长
常见配置：`fps=30`，而仿真 `time_step_s * frame_skip` 可能是 0.02（即 50Hz）或其它。
- 若你使用 OrcaGym 的 `CaptureMode.SYNC`，建议以 **视频帧索引**作为主索引，并把 `sim_time_s` 记录下来。
- 若是 `ASYNC`，建议同样以 `frame_index` 主索引，必要时记录 `camera_timestamp`（若可从接口拿到）用于对齐误差分析。

### 7.2 文件体量与压缩
900 帧、每帧 1-2 个对象：JSONL 通常数十 MB 以内；批量生成时建议：
- `metadata.jsonl.gz`（gzip）
- 或按秒分片：`metadata_0000_0899.jsonl` / `metadata_0900_1799.jsonl`（60 秒时 1800 帧）

---

## 8. 扩展（加分项预留，不影响刚需）
- **多模态**：depth/segmentation/normal/optical flow 等，可在 `artifacts` 下追加并在 frame 里记录对应帧文件索引。
- **多相机**：`frames[i].cameras[]` 或每帧按 camera_id 分组。
- **任务级轨迹抽象**：可在 `index.json.motion_descriptor` 中记录参数（加速度曲线/谐波参数/阻尼参数/贝塞尔控制点等）。

---

## 9. 术语与一致性检查（建议写入生成器的校验）
- `motion_end - motion_start + 1 == fps * duration_s`（允许误差 1 帧）
- `pitch/yaw/roll ∈ (-1,1)`（超出则截断或报错）
- `bbox3d.size > 0`
- `object_id` 必须在 `index.json.objects[]` 中定义


