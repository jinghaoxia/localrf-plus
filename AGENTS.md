# 仓库协作指南

## 项目目的与范围

本项目在上游 LocalRF（Progressively Optimized Local Radiance Fields）的基础上，接入已有视觉—LiDAR 跟踪轨迹，利用图像进行渐进式局部神经束调整（BA），联合优化相机位姿和局部辐射场。目标是在保留米制坐标约定的前提下，改善轨迹与图像重建的一致性，最终输出整条序列的载体系轨迹和局部辐射场，支持后续视图合成与评估。

完整流程分为两阶段：

1. **跟踪前端**：`scripts/estimate_pose.py` 使用矫正左目图像、原始 LiDAR 点云、标定和时间配对表估计米制轨迹。视觉阶段也利用 LiDAR 深度；随后通过扫描与局部地图配准融合视觉运动，输出图像时刻的载体系轨迹。
2. **神经 BA 后端（当前开发主线）**：显式读取已计算的跟踪轨迹作为初始化与位姿先验，以图像光度误差和相对跟踪初值的位姿正则联合优化位姿与辐射场。此阶段不把原始点云、深度或光流作为优化观测。

局部场按时间窗口逐步建立，历史场和归属旧场的位姿停止更新。这是渐进式局部优化；当前范围不包含全轨迹全局 BA、回环、全局位姿图优化、实时系统或动态目标建模。目标数据的图像为三通道重复灰度，重建外观应解释为灰度，不宣称恢复真实颜色。训练与评估均不读取 GT。

## 阅读顺序与实际代码结构

先阅读 `docs/neural-ba-usage.md` 了解当前用法，再阅读 `docs/neural-ba-v1-design.md` 了解设计目标，并与代码核对。设计文档包含尚未实现的规划，README 的上游说明也不直接适用于当前训练入口；描述现有行为时以代码为准，不能把实现偏差自动视为设计目标已变更。

- `scripts/estimate_pose.py`：独立的两阶段跟踪脚本、输入检查与 TUM 导出。
- `localTensoRF/train.py`：训练入口、跟踪接入、光度与位姿先验损失、渐进式调度、模型保存和轨迹导出。
- `localTensoRF/opt.py`：实际命令行参数及默认值。
- `localTensoRF/local_tensorfs.py`：局部场管理、位姿参数与优化器、重叠帧权重、米制射线转换及体渲染调用。
- `localTensoRF/dataLoader/tracking_poses.py`：TUM 读取、时间匹配、标定与坐标转换、载体系轨迹写出。
- `localTensoRF/dataLoader/localrf_dataset.py`：图像加载、子序列与跳帧选择、位姿对齐、射线采样和训练帧窗口。
- `localTensoRF/models/tensoRF.py`、`localTensoRF/models/tensorBase.py`：TensoRF 场表示、空间收缩、采样与体渲染。
- `localTensoRF/utils/`：射线、几何、损失及可视化辅助函数。
- `tests/test_tracking_poses.py`：基于合成数据的 CPU 测试。

当前仓库没有 `neural_ba.py`、`ba_geometry.py` 或 `ba_field.py`，不要据此虚构入口或模块。传感器数据和实验输出位于仓库外的 `dataprocess/`；当前工作区的数据根目录为 `/media/jinghao/本地磁盘/dataprocess`。

## 当前实现与设计差距

- 已接入显式 TUM 轨迹，支持绝对位姿初始化、子序列与跳帧对齐、米制场景半径、多局部场调度及载体系轨迹导出。首个选中帧固定在 tracking 初值；新帧从自身的 tracking 绝对位姿初始化。
- 当前位姿采用六维旋转表示和独立平移参数；`pose_prior_loss` 使用平移差平方和及旋转矩阵差的 Frobenius 范数平方。它尚不是设计文档中的右乘 SE(3) 指数映射增量及切空间正则。两个先验权重默认均为 0，必须显式设置才能启用先验。
- 已有 checkpoint 保存与模型加载方法，但训练入口中的恢复逻辑仍是 TODO；不能宣称 `--ckpt` 已支持断点续训。
- 底层体渲染用于训练；独立视图渲染与留出图像评估流程尚未实现。当前工作区的上游 `localTensoRF/renderer.py` 已删除，不要直接沿用旧渲染命令。
- 使用说明记录了 16 帧短片段和两个局部场的验证，不能据此宣称整序列或 8 GB 可用显存预算已通过验证。实际设备、CUDA 与可用显存应在目标服务器核查。
- 当前 tracking 适配器要求矫正图像、零畸变及 `fx ≈ fy`，尚不支持设计要求中的独立双焦距。修改相关逻辑时同步补充投影测试。

## 环境与常用命令

所有命令从仓库根目录运行，并先选择具备依赖的 Python 环境。以下使用 `python`；若环境仅提供 `python3`，使用该解释器。`requirements.txt` 是依赖列表而非完整锁定文件；训练还需核对 `configargparse`、`tensorboard`、`kornia`、`plyfile` 及适配设备的 PyTorch。不要把导入依赖失败误判为算法失败。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

仅在需要检查跟踪输入时运行：

```bash
python scripts/estimate_pose.py --dataset-dir /path/to/dataprocess/211 --check-only --end 64
```

仅在任务涉及跟踪前端且确需重估时，使用新的实验名称运行短片段：

```bash
python scripts/estimate_pose.py --dataset-dir /path/to/dataprocess/211 --end 64 --run-name unique_experiment_name
```

神经 BA 使用已有、明确指定的轨迹。训练参数是 `--datadir` 与 `--logdir`，不是跟踪入口的 `--dataset-dir`，也没有 `--output`。短序列示例：

```bash
python localTensoRF/train.py \
  --datadir /path/to/dataprocess/211 \
  --trajectory /path/to/pose_estimation_RUN/trajectory.tum \
  --logdir /path/to/new_ba_run \
  --downsampling 4 --subsequence 0 16 --frame_step 1 \
  --n_init_frames 5 --n_overlap 3 \
  --n_iters_per_frame 40 --n_iters_reg 20 --add_frames_every 20 \
  --batch_size 2048 --test_frame_every 0 \
  --L1_weight 0 --TV_weight_density 0 --TV_weight_app 0 \
  --loss_depth_weight_inital 0 --loss_flow_weight_inital 0 \
  --lr_i_init 0 --lr_exposure_init 0 --uniform_loss_weights 1 \
  --scene_radius_m 8 --pose_prior_lambda_t 0.0156 --pose_prior_lambda_r 1 \
  --vis_every 1000000 --device cuda:0
```

这些是短片段示例参数，不是已经验证的整序列配置。不要把全序列估计或训练作为常规检查，也不要因执行 BA 自动重跑 tracking 或自动挑选历史轨迹。

## 配置与输出注意事项

- 上游默认启用深度、光流、密度 L1 和曝光优化；本项目两项损失的实验必须显式关闭它们及 TV，并固定内参。保留参数的实际拼写 `inital`。
- 全部选中帧参与 BA 时使用 `--test_frame_every 0`。留出实验需设置 `--test_pose_opt 0`，防止测试图像参与位姿更新；这一开关不能替代完整、独立的留出评估流程。
- `--uniform_loss_weights 1` 会把所有像素权重置为 1，也会取消共用权重数组中的遮罩效果。涉及 `masks/` 时必须单独处理有效像素规则。
- `--subsequence start end` 使用右端不包含的区间，之后应用 `--frame_step`；图像和位姿必须同步选择。当前位姿先验数据集要求至少选中两帧。
- `--batch_size` 应为默认采样视图数 16 的倍数；多场切换应保证实际重叠帧数非零，`--n_overlap` 至少为 1。
- tracking 模式的位姿、`--max_drift`、平移学习率及导出平移均使用米。`--scene_radius_m` 是显式场景收缩半径，默认 8 m；场内部使用除以该半径的坐标，不能因此重缩放导出轨迹。
- 输出包括 `trajectory.tum`、`tracking_export.json`、`checkpoints.th` 与 TensorBoard 事件。必须使用全新 `--logdir`；代码只对已有 `trajectory.tum` 做拒绝覆盖检查，不能依赖它保护目录内的其他已有文件。

## 数据、时间与坐标约定

输入图像、PCD、标定和已有实验结果均只读。跟踪输入包含 `images/`、未运动补偿的原始 `pcd/`、`meta/pairs.csv` 和 `meta/calib.yaml`；BA 接入只需图像、这两份元数据和显式指定的 tracking TUM，不加载原始点云。

- TUM 为 `timestamp tx ty tz qx qy qz qw`，时间单位秒、平移单位米、四元数顺序 xyzw；位姿表示图像时刻的 `T_world_body`，原始序列第 0 帧为单位阵。
- 通过 `pairs.csv` 的 `output_index` 关联图像，以整数微秒计算 `camera_utc_us - origin_utc_us`，与 TUM 一对一匹配，容限最多 1 微秒。拒绝缺失、重复和非单调时间。当前加载器先验证完整配对表与轨迹，再选择训练片段。
- 坚持 `p_A = T_A_B @ p_B`。定义 `C = diag(1,-1,-1,1)`，相机转换为 `T_world_cam_rf = T_world_body @ T_body_cam @ C`。
- `G` 始终是原始序列第 0 帧的 LocalRF 相机系：`T_world_G = T_world_cam_rf(0)`，`T_G_cam_rf = inv(T_world_G) @ T_world_cam_rf`。非零起始片段也不能重定义 G；其首个选中帧仅固定在自身初值。
- 导出执行 `T_world_body = T_world_G @ T_G_cam_rf @ C @ inv(T_body_cam)`，不能遗漏外参平移。tracking 路径禁止使用相邻位移中位数归一化；旧 `transforms.json` 路径仍保留上游归一化行为。
- 内参从全分辨率标定读取，并处理 OpenCV 整数像素中心与 LocalRF `i+0.5` 射线中心的差异；按实际缩放比例转换主点。跟踪前端还需显式处理图像与 LiDAR 扫描时刻差，不能把原始扫描当成已去畸变数据。

## 开发、验证与结论边界

Python 使用四空格缩进，函数和变量用 `snake_case`，类用 `PascalCase`；几何变换名称写明坐标方向。遵循所在模块风格，避免无关格式调整。内部导入依赖 `localTensoRF/` 在模块搜索路径中，新增测试沿用现有测试的导入方式。

新增几何、数据接口或优化行为时，添加小规模合成 CPU 测试，覆盖坐标往返、时间匹配、米制单位、主点与子序列对齐等实际风险。运行相关测试后，再按需执行独立命名的 GPU 短序列实验；纯文档修改核对路径、参数和实现描述即可。不要把文档中的历史验证结果当成本次运行结果。

评估应区分图像重建质量与真实轨迹精度。更低的训练光度误差、更小的先验偏差或更平滑的轨迹都不能证明位姿更准确。固定 tracking 位姿与联合 BA 的对照应使用相同数据、窗口、预算和有效像素规则，并记录随机种子、位姿偏差、尺度/路径长度变化、窗口边界、覆盖率及资源消耗；正式留出评估不得使用测试图像优化位姿或场。

修改前检查 `git status`，保留用户已有未提交修改和删除，不擅自恢复文件。`.gitignore` 当前仅放行 `localTensoRF/models/` 下的三个既有源码文件；新增模型模块时检查是否被忽略。提交标题简洁并说明动作，拉取请求记录变更目的、坐标与数据约定、验证命令和实际结果；渲染变化附可视化对比。
