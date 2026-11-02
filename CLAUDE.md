# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

本仓库是 LocalRF（局部辐射场）的分支，包含两条流水线，通过 TUM 轨迹文件衔接：

1. **位姿估计** `scripts/estimate_pose.py`：单文件脚本，只依赖 numpy、scipy、opencv、PyYAML，不需要 torch。读取矫正左目图像和未运动补偿的原始 LiDAR 扫描（`pcd/`），输出载体系轨迹。
2. **神经 BA** `localTensoRF/train.py`：以**显式指定**的 tracking 轨迹为初值，对渐进划分的局部辐射场和每帧位姿做光度与可选米制深度联合优化（可选位姿先验），输出米制轨迹 `trajectory.tum`。

用法看 `docs/neural-ba-usage.md`；规划与验收标准看 `docs/neural-ba-v1-design.md`（规划文档，不代表功能已实现）。上游 README 中的 DPT/RAFT 预处理流程已不适用：`scripts/promptda.py` 负责离线稠密深度，`.gitmodules` 中的 DPT/RAFT 子模块未检出，代码也未引用它们。风格与数据约定见 `AGENTS.md`，此处不重复。

## 常用命令

默认在仓库根目录运行。仓库自带 `.venv`（Python 3.11），若 shell 未激活，把 `python` 换成 `.venv/bin/python`。

```bash
# 依赖（requirements.txt 不完整，见“环境与陷阱”）
python -m pip install -r requirements.txt
python -m pip install configargparse tensorboard kornia plyfile   # PyTorch 需按平台单独安装

# 单元测试：CPU；必须带 -s tests
python -m unittest discover -s tests -v
# 单个测试：-k 按名字过滤，或使用点路径
python -m unittest discover -s tests -v -k test_round_trip_recovers_body_poses
python -m unittest tests.test_tracking_poses.DatasetAlignmentTests.test_subsequence_and_step_keep_metric_poses_aligned_with_images

# 训练入口：只解析参数并检查 import 链，不训练
python localTensoRF/train.py --help

# 位姿估计：只检查输入（--end 64 只看前缀）；不要把整序列估计当作日常检查
python scripts/estimate_pose.py --dataset-dir /path/to/dataprocess/211 --check-only
# 短前缀实验：输出到 <数据集>/output/pose_estimation_<run-name>/，目录已存在则拒绝
python scripts/estimate_pose.py --dataset-dir /path/to/dataprocess/211 --end 64 --run-name exp01
```

神经 BA 的完整参数（如 16 帧短跑）见 `docs/neural-ba-usage.md`，当前默认值见下文。仓库未配置 lint 或格式化工具（无 ruff、flake8、pyproject、pre-commit），不要臆造 lint 命令。

## 架构

### 数据与坐标（两条流水线共用的约定）

- 数据集只读；本地样本在 `211/`，完整数据可位于仓库外（`dataprocess/<序列>/`）：`images/{idx:05d}.png`（1920×1200 矫正左目）、`pcd/{idx:05d}.pcd`、`meta/pairs.csv`（`output_index,camera_utc_us`，从 0 连续）、`meta/calib.yaml`（`left_camera`、`camera_to_body.T_body_cam`、`time.origin_utc_us`）。
- `LocalRFDataset` 统一调用 `dataLoader/tracking_poses.py` 的几何辅助函数：`load_tracking` 读取 TUM，用整数微秒（`camera_utc_us − origin_utc_us`）与 `pairs.csv` 逐帧匹配（容差 1 µs，多对一或缺失即报错），并变换到 G 系；`camera_to_body`、`write_tum` 负责反向导出。
- 变换约定 `p_A = T_A_B @ p_B`：`T_world_cam_cv = T_world_body @ T_body_cam`，LocalRF 射线系为 `cv @ diag(1,-1,-1,1)`（`CV_TO_RF`）。G 是原始第 0 帧的相机系，因此轨迹首帧必须是单位阵，否则拒绝。
- 单位：位姿、先验、`max_drift`、`lr_t_init` 都以米计。场内部以半径 R（`--scene_radius_m`，默认 8）归一化，`LocalTensorfs.forward` 中射线原点除以 R。R 仅在 `--trajectory` 模式生效；`max_drift`、`lr_t_init` 的默认值随 R 换算，平移 Adam 的 eps 也按 1/R 缩放（见 `local_tensorfs.py`）。

### 神经 BA（`localTensoRF/train.py`）

`reconstruction()` 是唯一的训练循环。设计文档中计划的 `neural_ba.py`、`ba_geometry.py`、`ba_field.py`、`render_neural_ba.py`、独立评估入口和 `legacy/` 目前都不存在。

1. **数据**（`dataLoader/localrf_dataset.py`）：`--subsequence`、`--frame_step` 选帧；`--test_frame_every` 标记留出帧。图像按需分块读取（`activate_frames`），切场后丢弃旧帧（`deactivate_frames`）。`sample()` 每步随机选 16 个视图，`batch_size` 条射线平均分配给它们。
2. **模型**（`local_tensorfs.py: LocalTensorfs`）：
   - 每帧一组可学习位姿：6D 旋转 `r_c2w` 与平移 `t_c2w`，各自一个 Adam。`pose_linked_rf[i]` 记录帧 i 属于哪个场。
   - 多个局部场 `tensorfs`，每个是 `models/tensoRF.py: TensorVMSplit`（VM 分解的密度与外观体素；收缩采样与 alpha mask 在 `models/tensorBase.py`）。每个场只有平移偏移 `world2rf`，轴与 G 对齐。
   - `blending_weights`（帧 × 场）记录重叠。训练时只渲染最新场；混合权重只在 `is_train=False` 分支生效，目前只有测试在调用它。
3. **渐进调度**（`train.py` 主循环 + `LocalTensorfs.optimizer_step`）：
   - 先放入 `--n_init_frames` 帧，之后每 `--add_frames_every` 次迭代加入一帧。
   - 当最新帧到当前场原点的距离超过 `--max_drift`（米），或活动窗口帧数达到 `--n_max_frames`，并且已加入的帧数超过 `--n_overlap`，进入精修。精修迭代数 = `--n_iters_per_frame` × 当前场的训练帧数；学习率按步衰减到 `--lr_decay_target_ratio`。
   - 精修结束后 `append_rf` 在最新相机位置新建场。最后 `n_overlap` 帧同时属于新旧两场（线性权重），旧帧移出训练窗口。
   - 位姿只更新 `pose_linked_rf` 等于当前场的帧，因此重叠帧在旧场上冻结。首个选中帧的学习率置零，以固定整体刚体自由度。
   - 帧用完后，对末尾窗口做收尾精修，然后结束。
4. **损失**：
   - 光度项：`0.25·|rgb − 观测|` 的加权 L1。`--uniform_loss_weights 1` 时保留有效像素遮罩，关闭逐帧 Laplacian 清晰度权重。
   - 位姿先验 `pose_prior_loss`：`λ_t·Σ‖t − t_track‖² + λ_r·Σ‖R − R_track‖²_F`。只对当前场的帧计算，只在非留出位姿步中加入，需要 `--trajectory` 且 λ > 0。
   - 稠密深度为有效像素的米制 `|D_hat − D|` 均值，`--depth_weight` 开启后全程启用;uint16 PNG 必须指定 `--depth_scale`。TV、密度 L1 仍为可选场正则;flow 流程已删除。
5. **输出**（`--logdir` 必须是新目录）：`checkpoints.th`（`kwargs` + `state_dict`）、`checkpoints_tmp.th`（每 `--vis_every` 次迭代写一次，非原子写入）、TensorBoard `events.*`。给了 `--trajectory` 时，结束写 `trajectory.tum`（只含选中帧）和 `tracking_export.json`。非空输出目录会被拒绝。

### 位姿估计（`scripts/estimate_pose.py`）

- `load_sequence` 要求标定为 1920×1200，并要求 `pcd/` 声明 `pcd_motion_compensation: none` 与 `pcd_indexing: output_index`。
- 阶段一 `estimate_visual_trajectory`：LK 光流为主、SIFT 匹配补充，并做单应性过滤；深度来自原始 LiDAR 点，用于求解度量运动。中间结果为 `visual_trajectory.tum`。
- 阶段二 `estimate_lidar_trajectory`：扫描与局部地图配准。LiDAR 主导姿态与高度，视觉只给局部 XY 位移软约束。`accept_registration` 为准入门限，`scan_pose_to_image_time` 把扫描时刻换算到图像时刻。最终输出 `trajectory.tum`（9 位小数，首帧为单位阵）。
- 不读取 GT 或稠密深度。

## 当前默认值

内参自动从 `meta/calib.yaml` 或旧 `transforms.json` 读取,支持双焦距;不提供 FOV 或手动内参 CLI。默认 `depth_weight=0`、密度 L1/TV/曝光优化为 0、`test_frame_every=0`、`test_pose_opt=0`、`uniform_loss_weights=1`。启用稠密深度需显式设置权重;`211` 的 PNG 使用 `--depth_scale 0.01`。旧 flow、逆深度、未实现的 `--ckpt`、无效模型选择和采样数量 CLI 已移除。

切场参数 `--n_overlap`（默认 30）、`--add_frames_every`（100）、`--n_max_frames`（100）、`--n_iters_per_frame`（600）也是上游值，全序列运行需要重新选定。

## 环境与陷阱

- `plyfile` 不在 `requirements.txt` 中，但 `utils/utils.py` 在模块顶层导入它，缺失时 `import utils.utils` 直接失败，测试和训练都跑不起来。`configargparse`（`opt.py`）、`tensorboard`（`train.py`）同样是硬依赖；`requirements.txt` 只在注释里提到后两者，并引用了不存在的 `docs/running.md`。
- `.gitignore` 对 `localTensoRF/models/` 只放行 `__init__.py`、`tensoRF.py`、`tensorBase.py`，新增模块会被忽略（`git check-ignore` 已确认），需要修改 `.gitignore`。
- 训练默认 `--device cuda:0`；单元测试全部在 CPU 上运行。
- 随机种子：`--seed` 默认 20211202，`reconstruction()` 内同时设置 Python `random`、NumPy 和 PyTorch，命令行和直接调用均生效；写入 `tracking_export.json`。这不保证 GPU 算子逐位确定性。

## 已知限制与工作区状态

- **无断点续训**：未实现的 `--ckpt` 训练参数已移除。
- **无独立留出评估**：`render.py` 可渲染保存视图并导出 RGB、深度和不透明度，不能当作正式留出评估。
- **`--n_overlap` 必须 ≥ 1**：训练入口拒绝 0，以保持切场的重叠窗口。
- **`--batch_size` 必须是 16 的倍数**：`sample()` 用 `reshape(n_views, -1)` 分配射线。
- **统一权重保留遮罩**：`--uniform_loss_weights 1` 仅关闭清晰度加权；光度权重为 ones × mask，稠密深度也使用同一观测遮罩。
- **tracking 历史版本**：当前 `estimate_pose.py` 已改为 `(c+0.5)/3−0.5` 主点缩放，并修复非单位 LiDAR 外参旋转时的杆臂换算。已生成的轨迹不因此改变，也不自动重估。前端候选框已使用完整 `T_body_lidar` 转到载体系后检查 x/y/z；2～35 m 径向筛选仍在 LiDAR XY 平面。固定分辨率和候选框几何仍是数据集假设，不能据此宣称任意标定均已验证。
- **背景与深度**：RGB 沿用强制不透明的远端可学习颜色；不存在固定白背景开关。场内期望深度排除该远端，保留 `1e-8` 分母下限；alpha 使用 `-expm1(-σΔt)` 以减少低密度浮点消减。训练不按预测不透明度屏蔽像素，只记录有效观测中 opacity < 1e-4 的比例。
- **深度权重与梯度**：示例使用无深度基线，211 前 8 帧短预算可对照 `1e-4～1e-3`；`0.1` 在本次扫描中对密度及位姿的梯度占主导。`--grad_log_every N` 记录已加权 RGB/深度/先验对活动参数的 L2 和最大绝对梯度，默认关闭，不改变优化。保留 `1e-8` 下限，不默认启用裁剪或 opacity 门控；实际低密度压力测试不是通用稳定性证明。条件和结果见 `docs/depth-weight-study-2026-11-02.md`。
- **深度导出命名**：`render_depth_moment()` 返回未归一化的一阶矩 M 和不透明度 A，`expected_depth()` 再除以 A；tracking 的 M 单位为米，M 不是期望深度。
- **资源与采样**：远场初始权重偏置、位姿先验按帧求和、TensoRF 收缩区域和窗口拼接峰值内存仍保留原实现。实测条件与风险判断见 `docs/neural-ba-review-2026-11-02.md`。
- **工作区修改**：修改前检查 `git status`，保留已有未提交改动；不要恢复或覆盖其他工作。

## 测试约定

- `tests/test_tracking_poses.py` 使用临时目录中的合成数据，不需要真实数据集或 GPU。测试把 `localTensoRF/` 加入 `sys.path` 后导入模块；新测试沿用同样做法。代码内部使用裸模块名导入（如 `from utils.utils import ...`），依赖 `localTensoRF/` 在路径上。
- 新增几何、数据接口或优化行为时，增加小型合成用例（`tests/test_*.py`），验证坐标往返、时间戳和单位。
