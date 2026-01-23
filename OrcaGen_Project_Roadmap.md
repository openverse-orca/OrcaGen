# OrcaGen 项目路线图（初级阶段）

## 项目概述
基于 OrcaGym 的大规模多模态物理仿真数据合成系统，用于生成带有精确 3D 标注的视频轨迹数据。

---

## 一、数据格式设计 ✅（已完成 90%）

### 1.1 元数据规范（JSON/JSONL）
**状态：已实现**
- ✅ `index.json`：全局序列信息
  - 坐标系定义（世界坐标系 = 第一帧相机坐标系）
  - 物体语义信息（object_id, class, category）
  - 运动模式标注（motion_mode, motion_start, motion_end）
  - 碰撞时间记录（collision_time）
  - 采集参数（fps, resolution, duration_s, capture_mode）
- ✅ `metadata.jsonl`：逐帧结构化数据
  - 每帧时间戳（frame_index, timestamp_s, sim_time_s）
  - 相机位姿（pose_world0: pos + mat3x3）
  - 3D 边界框（bbox3d: x_center, y_center, z_center, x_size, y_size, z_size, pitch, yaw, roll）
  - 接触/碰撞信息（contacts: has_contact, pairs）

**待优化：**
- ⚠️ 多物体场景的 `collision_time` 需支持数组格式（记录多个碰撞时刻）
- ⚠️ 碰撞对信息需要更精细（碰撞力、碰撞点坐标、法向量）

### 1.2 视频输出
**状态：已实现**
- ✅ RGB 视频流（`video/rgb_main/video/Camera_color.mp4`）
- ✅ Depth 视频流（`video/rgb_main/depth/Camera_depth.mp4`）
- ✅ 支持自定义分辨率、帧率、时长

**待优化：**
- ⚠️ 深度流当前拖慢性能，需在 Editor 侧配置可选关闭
- ⚠️ 多相机视角支持（目前仅单相机）---> (多相机只能在引擎中手动开关，不能动态控制）

### 1.3 工程文件（交付物 B）
**状态：待实现**
- ❌ Blender `.blend` 或 USD 格式导出
- ❌ 包含完整动画 Keyframes 及动力学解算缓存

---

## 二、物理运动设计（核心功能）

### 2.1 刚体运动 ✅（已验证）
**状态：已实现基础功能**
- ✅ 自由落体（sphere 在重力下滚动）
- ✅ 滚动摩擦（粗糙平面上的球体）
- ✅ 碰撞检测（sphere-ground 接触对）

**待扩展：**
- 🔲 **运动模式库**：
  - 匀速直线运动
  - 匀加速/匀减速运动
  - 抛物线运动（斜抛、平抛）
  - 旋转运动（陀螺、滚轮）
  - 弹跳运动（弹性碰撞、非弹性碰撞）
  - 滑动摩擦（冰面、粗糙面）
- 🔲 **外力干预**：
  - 恒力（推力、拉力）
  - 冲量（瞬时碰撞）
  - 周期力（简谐振动、阻尼振动）
  - 扭矩（旋转驱动）

### 2.2 柔体/变形体
**状态：待实现**
- ❌ 布料模拟（旗帜飘动、窗帘）
- ❌ 软体碰撞（橡胶球、海绵）
- ❌ 流体交互（水面波动、液体倾倒）

### 2.3 运动曲线设计
**状态：待实现**
- ❌ **参数化轨迹**：
  - 贝塞尔曲线路径
  - 样条曲线插值
  - 关键帧动画（Keyframe-based）
- ❌ **物理约束**：
  - 轨道约束（圆周、椭圆）
  - 铰链约束（摆锤、门）
  - 绳索约束（钟摆、吊灯）

### 2.4 多物体交互（1-2 物体）
**状态：待实现**
- ❌ **两物体碰撞**：
  - 球-球碰撞（弹性/非弹性）
  - 球-立方体碰撞
  - 多米诺骨牌效应
- ❌ **传送带场景**：
  - 物块在传送带上滚动
  - 传送带速度变化
- ❌ **钟摆场景**：
  - 单摆/双摆
  - 阻尼衰减（幅度越来越小）
- ❌ **堆叠与倒塌**：
  - 积木堆叠
  - 重力倒塌

---

## 三、视觉多样性（随机化 API）

### 3.1 光照随机化
**状态：待实现**
- ❌ **光源类型**：
  - 平行光（太阳光）
  - 点光源（灯泡）
  - 聚光灯（手电筒）
  - 环境光（HDR 天空盒）
- ❌ **光照参数**：
  - 强度（Intensity）
  - 颜色温度（Color Temperature）
  - 方向（Azimuth, Elevation）
  - 阴影硬度（Shadow Softness）

### 3.2 相机视角随机化 ！！！！（相机录制目前不支持动态关闭spawn等）
**状态：待实现**
- ❌ **相机位置**：
  - 随机高度（俯视/平视/仰视）
  - 随机方位角（360° 环绕）
  - 随机距离（近景/远景）
- ❌ **相机参数**：
  - 焦距（FOV）
  - 景深（Depth of Field）
  - 运动模糊（Motion Blur）

### 3.3 材质与纹理随机化
**状态：待实现**
- ❌ **材质属性**：
  - 粗糙度（Roughness）
  - 金属度（Metallic）
  - 透明度（Opacity）
  - 自发光（Emission）
- ❌ **纹理贴图**：
  - 随机纹理库（木纹、石纹、金属）
  - 颜色随机化（HSV 抖动）

### 3.4 遮挡与干扰
**状态：待实现**
- ❌ **静态遮挡物**：
  - 随机放置障碍物（墙、柱子）
  - 部分遮挡目标物体
- ❌ **动态干扰**：
  - 移动遮挡物（行人、车辆）
  - 粒子系统（烟雾、雨雪）

### 3.5 背景与场景随机化
**状态：待实现**
- ❌ **地面纹理**：
  - 地板材质（瓷砖、木板、水泥）
  - 地面粗糙度（影响摩擦系数）
- ❌ **环境布局**：
  - 室内/室外场景切换
  - 天气条件（晴天、阴天、夜晚）

---

## 四、系统架构设计

### 4.1 模块化设计
**状态：待实现**
- ❌ **场景生成模块**（SceneGenerator）：
  - 物体生成器（ObjectSpawner）
  - 材质随机器（MaterialRandomizer）
  - 光照配置器（LightingConfigurator）
- ❌ **运动控制模块**（MotionController）：
  - 轨迹规划器（TrajectoryPlanner）
  - 物理参数配置（PhysicsConfig）
  - 外力施加器（ForceApplier）
- ❌ **数据采集模块**（DataCollector）：
  - 视频录制器（VideoRecorder） ✅
  - 元数据生成器（MetadataGenerator） ✅
  - 碰撞检测器（CollisionDetector） ✅

### 4.2 任务抽象接口
**状态：待实现**
- ❌ **运动任务接口**（MotionTask）：
  ```python
  class MotionTask:
      def setup_scene(self) -> Scene
      def define_trajectory(self) -> Trajectory
      def apply_forces(self, t: float) -> Forces
      def check_termination(self) -> bool
  ```
- ❌ **随机化接口**（RandomizationAPI）：
  ```python
  class RandomizationAPI:
      def randomize_lighting(self, params: LightingParams)
      def randomize_camera(self, params: CameraParams)
      def randomize_materials(self, objects: List[Object])
  ```

### 4.3 命令行工具
**状态：部分实现**
- ✅ `capture_sphere1.py`：单物体采集脚本
- ❌ `orcagen_generate.py`：通用数据生成工具
  ```bash
  orcagen_generate \
    --task rolling_sphere \
    --num_sequences 1000 \
    --duration 30 \
    --fps 30 \
    --randomize lighting,camera,material \
    --output_dir /data/orcagen
  ```

### 4.4 多机并行支持
**状态：待实现**
- ❌ **任务分发**：
  - 任务队列（Redis/RabbitMQ）
  - 任务分片（按 sequence_id 分配）
- ❌ **资源管理**：
  - GPU 资源调度
  - 仿真实例管理（多 OrcaEditor 进程）
- ❌ **结果聚合**：
  - 元数据合并
  - 质量检查（QA）

---

## 五、开发优先级（近期 7 天刚需）

### 已完成 ✅
1. ✅ 视频轨迹录制（指定帧率、分辨率、时长）
2. ✅ 物体识别与语义信息获取
3. ✅ 逐帧 3D 边界框与空间坐标生成（JSON 输出）

### 高优先级（P0）
4. 🔲 **多物体场景**（1-2 物体交互）
   - 实现 2 个刚体碰撞场景（球-球、球-立方体）
   - 碰撞时间点精确记录（collision_time 数组）
5. 🔲 **运动模式库**（至少 5 种）
   - 自由落体 ✅
   - 匀速滚动
   - 弹跳
   - 滑动
   - 旋转
6. 🔲 **基础随机化 API**
   - 光照强度/方向随机化
   - 相机位置随机化（3 个预设视角）

### 中优先级（P1）
7. 🔲 **外力干预接口**
   - 恒力施加（推力、拉力）
   - 冲量施加（瞬时碰撞）
8. 🔲 **运动曲线设计**
   - 贝塞尔曲线轨迹
   - 关键帧动画
9. 🔲 **命令行工具封装**
   - 通用数据生成脚本
   - 配置文件支持（YAML/JSON）

### 低优先级（P2，加分项）
10. 🔲 柔体模拟
11. 🔲 流体交互
12. 🔲 多机并行部署
13. 🔲 Blender/USD 工程文件导出

---

## 六、技术债务与已知问题

### 性能问题
- ⚠️ **Depth 流拖慢录制**：当前 RGB+Depth 双流录制导致帧率下降，需在 Editor 侧配置关闭 Depth
- ⚠️ **SYNC 模式超时**：`CameraSyncManager WaitingLastFrame Timeout`，已改用 ASYNC 模式缓解

### 数据质量
- ⚠️ **碰撞检测粒度**：当前仅检测是否接触，未记录碰撞力、碰撞点
- ⚠️ **相机标定**：当前假设相机静态，未记录内参（焦距、畸变）

### 工程化
- ⚠️ **缺乏单元测试**：元数据生成、碰撞检测等核心逻辑未覆盖测试
- ⚠️ **配置硬编码**：物理参数（重力、摩擦系数）写死在代码中，需提取到配置文件

---

## 七、下一步行动（本周计划）

### Day 1-2：多物体交互
- [ ] 实现球-球碰撞场景
- [ ] 记录碰撞时间点（collision_time 数组）
- [ ] 验证 2 物体的 3D 边界框标注准确性

### Day 3-4：运动模式扩展
- [ ] 实现匀速滚动场景
- [ ] 实现弹跳场景（弹性碰撞）
- [ ] 实现滑动场景（低摩擦系数）

### Day 5-6：随机化 API
- [ ] 实现光照随机化（强度、方向）
- [ ] 实现相机位置随机化（3 个预设视角）
- [ ] 验证随机化后的数据质量

### Day 7：工具封装与文档
- [ ] 封装通用数据生成脚本
- [ ] 编写使用文档（README）
- [ ] 准备演示 Demo

---

## 八、参考资料

### 已完成文档
- `OrcaGen_JSON_Metadata_Spec.md`：元数据格式规范
- `capture_sphere1.py`：单物体数据采集脚本
- `record_ws_video.py`：WebSocket 视频录制工具

### 依赖项目
- **OrcaGym**：物理仿真环境（gRPC 接口）
- **OrcaManipulation**：机器人数据采集框架（视频录制参考）

### 外部参考
- MuJoCo 文档：物理引擎 API
- USD 格式规范：工程文件导出标准
- Blender Python API：动画 Keyframes 导出

---

**最后更新**：2026-01-21  
**当前版本**：v0.1-alpha（初级阶段）

