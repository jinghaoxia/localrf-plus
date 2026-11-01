# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

本仓库是 LocalRF（局部辐射场）的分支，包含两条流水线，通过 TUM 轨迹文件衔接：

1. **位姿估计** `scripts/estimate_pose.py`：单文件脚本，只依赖 numpy、scipy、opencv、PyYAML，不需要 torch。读取矫正左目图像和未运动补偿的原始 LiDAR 扫描（`pcd/`），输出载体系轨迹。
2. **神经 BA** `localTensoRF/train.py`：以**显式指定**的 tracking 轨迹为初值，对渐进划分的局部辐射场和每帧位姿做光度联合优化（可选位姿先验），输出米制轨迹 `trajectory.tum`。

用法看 `docs/neural-ba-usage.md`；规划与验收标准看 `docs/neural-ba-v1-design.md`（规划文档，不代表功能已实现）。上游 README 中的 DPT/RAFT 预处理流程已不适用：`scripts/` 里只有 `estimate_pose.py`，`.gitmodules` 中的 DPT/RAFT 子模块未检出，代码也未引用它们。风格与数据约定见 `AGENTS.md`，此处不重复。

## 常用命令

默认在仓库根目录运行。仓库自带 `.venv`（Python 3.11），若 shell 未激活，把 `python` 换成 `.venv/bin/python`。

```bash
# 依赖（requirements.txt 不完整，见“环境与陷阱”）
python -m pip install -r requirements.txt
python -m pip install configargparse tensorboard kornia plyfile   # PyTorch 需按平台单独安装

# 单元测试：CPU，9 项，约 0.1 秒；必须带 -s tests，否则发现 0 项
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

神经 BA 的完整参数（如 16 帧短跑）见 `docs/neural-ba-usage.md`，必须覆盖的上游默认值见下文。仓库未配置 lint 或格式化工具（无 ruff、flake8、pyproject、pre-commit），不要臆造 lint 命令。

## 架构

### 数据与坐标（两条流水线共用的约定）

- 数据集位于仓库外（`dataprocess/<序列>/`，只读）：`images/{idx:05d}.png`（1920×1200 矫正左目）、`pcd/{idx:05d}.pcd`、`meta/pairs.csv`（`output_index,camera_utc_us`，从 0 连续）、`meta/calib.yaml`（`left_camera`、`camera_to_body.T_body_cam`、`time.origin_utc_us`）。
- `localTensoRF/dataLoader/tracking_poses.py` 是 tracking 接口：`load_tracking` 读取 TUM，用整数微秒（`camera_utc_us − origin_utc_us`）与 `pairs.csv` 逐帧匹配（容差 1 µs，多对一或缺失即报错），并变换到 G 系；`camera_to_body`、`write_tum` 负责反向导出。
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
   - 光度项：`0.25·|rgb − 观测|` 的加权 L1。`--uniform_loss_weights 1` 时权重全为 1，关闭逐帧 Laplacian 清晰度权重。
   - 位姿先验 `pose_prior_loss`：`λ_t·Σ‖t − t_track‖² + λ_r·Σ‖R − R_track‖²_F`。只对当前场的帧计算，只在非留出位姿步中加入，需要 `--trajectory` 且 λ > 0。
   - 可选正则（TV、密度 L1、深度、光流）由权重开关控制；深度需要 `depth/`，光流需要 `flow_ds/`。
5. **输出**（`--logdir` 必须是新目录）：`checkpoints.th`（`kwargs` + `state_dict`）、`checkpoints_tmp.th`（每 `--vis_every` 次迭代写一次，非原子写入）、TensorBoard `events.*`。给了 `--trajectory` 时，结束写 `trajectory.tum`（只含选中帧）和 `tracking_export.json`。代码只检查 `trajectory.tum` 是否存在，其它文件会被直接覆盖。

### 位姿估计（`scripts/estimate_pose.py`）

- `load_sequence` 要求标定为 1920×1200，并要求 `pcd/` 声明 `pcd_motion_compensation: none` 与 `pcd_indexing: output_index`。
- 阶段一 `estimate_visual_trajectory`：LK 光流为主、SIFT 匹配补充，并做单应性过滤；深度来自原始 LiDAR 点，用于求解度量运动。中间结果为 `visual_trajectory.tum`。
- 阶段二 `estimate_lidar_trajectory`：扫描与局部地图配准。LiDAR 主导姿态与高度，视觉只给局部 XY 位移软约束。`accept_registration` 为准入门限，`scan_pose_to_image_time` 把扫描时刻换算到图像时刻。最终输出 `trajectory.tum`（9 位小数，首帧为单位阵）。
- 不读取 GT 或稠密深度。

## 上游默认值陷阱

`localTensoRF/opt.py` 中损失、留出、曝光和切场参数的默认值沿用上游。直接运行会启用深度、光流、密度 L1 和曝光优化，并尝试读取 `depth/`、`flow_ds/`。神经 BA 运行必须显式覆盖：

| 参数 | 默认值 | 神经 BA 取值 | 原因 |
|---|---|---|---|
| `--loss_depth_weight_inital` | 0.1 | 0 | 不做深度监督 |
| `--loss_flow_weight_inital` | 1 | 0 | 不做光流监督 |
| `--L1_weight` | 1e-2 | 0 | 不做密度稀疏正则 |
| `--lr_exposure_init` | 1e-3 | 0 | 固定曝光 |
| `--test_frame_every` | 10 | 0 | 默认让留出帧参与位姿更新 |
| `--uniform_loss_weights` | 0 | 1 | 关闭逐帧清晰度权重 |
| `--downsampling` | -1（原分辨率） | 如 4 | 控制显存 |

切场参数 `--n_overlap`（默认 30）、`--add_frames_every`（100）、`--n_max_frames`（100）、`--n_iters_per_frame`（600）也是上游值，全序列运行需要重新选定。

## 环境与陷阱

- `plyfile` 不在 `requirements.txt` 中，但 `utils/utils.py` 在模块顶层导入它，缺失时 `import utils.utils` 直接失败，测试和训练都跑不起来。`kornia`（`utils/ray_utils.py`）、`configargparse`（`opt.py`）、`tensorboard`（`train.py`）同样是硬依赖；`requirements.txt` 只在注释里提到后两者，并引用了不存在的 `docs/running.md`。
- `.gitignore` 对 `localTensoRF/models/` 只放行 `__init__.py`、`tensoRF.py`、`tensorBase.py`，新增模块会被忽略（`git check-ignore` 已确认），需要修改 `.gitignore`。
- 训练默认 `--device cuda:0`；单元测试全部在 CPU 上运行。
- 随机种子：`train.py` 只设置了 torch 与 numpy 的种子（20211202）。留出帧选择（`localrf_dataset.py` 的 `sample`）用的是未设种子的 Python `random`，不同运行的留出步骤不完全可复现。

## 已知限制与工作区状态

- **无断点续训**：`--ckpt` 已解析但未使用（`train.py` 中是 TODO）。
- **无渲染与留出图像评估**：`train.py` 不再渲染测试视图；上游的 `renderer.py` 在工作区被删除，`--render_*` 参数也已移除。
- **`--n_overlap` 必须 ≥ 1**：`local_tensorfs.py` 中 `1 / n_overlap` 没有保护，为 0 时第一次切场即除零。
- **`--batch_size` 必须是 16 的倍数**：`sample()` 用 `reshape(n_views, -1)` 分配射线。
- **`--uniform_loss_weights 1` 也会关闭遮罩**：`masks/` 生成的像素权重与清晰度权重共用同一数组（`localrf_dataset.py`），开启后遮罩被忽略。估计脚本列出的数据集输入只有 `images/`、`pcd/`、`meta/`，通常不会触发此问题。
- **内参缩放偏差**：`estimate_pose.py` 把 `cx, cy` 直接除以 3（约第 130 行），设计文档 §3.5 记录其与 OpenCV 像素中心约定相差约 1/3 像素。尚未修复；使用历史轨迹前需确认其版本。
- **文档与代码不一致**：`AGENTS.md` 的示例命令使用 `--dataset-dir` 与 `--output`，但 `train.py` 实际参数为 `--datadir` 与 `--logdir`（以 `--help` 为准）；`AGENTS.md` 也把 `neural_ba.py` 等模块写成已存在。
- **工作区有大量未提交改动**（`git status` 可见），包括 `renderer.py` 的删除和 `train.py` 的重写。这些是用户的工作，修改前先确认，不要恢复或覆盖。

## 测试约定

- `tests/test_tracking_poses.py` 使用临时目录中的合成数据，不需要真实数据集或 GPU。测试把 `localTensoRF/` 加入 `sys.path` 后导入模块；新测试沿用同样做法。代码内部使用裸模块名导入（如 `from utils.utils import ...`），依赖 `localTensoRF/` 在路径上。
- 新增几何、数据接口或优化行为时，增加小型合成用例（`tests/test_*.py`），验证坐标往返、时间戳和单位。
