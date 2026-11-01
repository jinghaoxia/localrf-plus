# 神经 BA 使用说明

神经 BA 是在原 LocalRF 训练入口 `localTensoRF/train.py` 上的增量改动,不另起训练流程。输入为图像、标定、时间配对表和**显式指定的** tracking 轨迹。

## 运行

从仓库根目录运行,`--logdir` 必须是新目录,已有 `trajectory.tum` 时会拒绝运行:

```bash
python localTensoRF/train.py \
  --datadir /path/to/dataprocess/211 \
  --trajectory /path/to/output/pose_estimation_RUN/trajectory.tum \
  --logdir /path/to/new_run \
  --downsampling 4 --subsequence 0 16 --frame_step 1 \
  --n_init_frames 5 --n_overlap 3 --n_iters_per_frame 40 --n_iters_reg 20 --add_frames_every 20 \
  --batch_size 2048 --test_frame_every 0 \
  --L1_weight 0 --TV_weight_density 0 --TV_weight_app 0 \
  --loss_depth_weight_inital 0 --loss_flow_weight_inital 0 --lr_exposure_init 0 \
  --uniform_loss_weights 1 \
  --scene_radius_m 8 --pose_prior_lambda_t 0.0156 --pose_prior_lambda_r 1 \
  --vis_every 1000000 --device cuda:0
```

上游默认值会加载深度和光流、启用密度 L1 和曝光优化,所以上面的 `0` 覆盖是必需的。`--subsequence` 指定原始输出索引的 `[start, end)`,去掉它则使用全部帧。

## 单位

所有位姿、先验、`--max_drift`、`--lr_t_init` 和导出结果都使用**米**。`--scene_radius_m`(默认 8 m)是收缩开始的半径,场在约这个半径内有完整分辨率。模型内部以 R 为单位:射线原点除以 R,采样范围与 `distance_scale`(25)保持上游含义,因此场的几何与上游归一化版本逐点一致。

从上游归一化版本换算:

| 参数 | 归一化版本的值 | 米制换算 |
|---|---|---|
| `--pose_prior_lambda_t` | w | w / R² |
| `--pose_prior_lambda_r` | w | w(无量纲,不变) |
| `--lr_t_init` | 5e-4 每步 | 5e-4 × R 米每步(默认已自动换算) |
| `--max_drift` | 1 | R 米(默认已自动换算) |

平移的 Adam 数值 ε 也按 1/R 缩放,否则梯度变小后 ε 的相对影响会改变优化动力学。这些换算在代码中自动完成,未显式指定时使用上表的默认值,并写入 `tracking_export.json`。

与旧的归一化版本不同,R 是固定的显式参数。归一化版本的缩放由所选帧的中位步长决定,因此 `--subsequence` 不同会改变同一帧的坐标系。

## 主要选项

- `--trajectory`:显式指定 TUM 轨迹(载体系、米、xyzw 四元数)。它替代 `transforms.json` 作为位姿先验。
- `--scene_radius_m`:收缩半径 R(米),默认 8。
- `--pose_prior_lambda_t`、`--pose_prior_lambda_r`:每帧位姿相对 tracking 的平移平方和(单位 m⁻²)、旋转平方和权重。0 表示不加先验,即纯光度 BA。
- `--uniform_loss_weights 1`:关闭上游逐帧 Laplacian 清晰度加权。
- `--test_pose_opt 0`:使留出帧不参与位姿更新(用于留出实验)。
- `--downsampling`:训练分辨率。原图 1920×1200,4 倍下采样为 480×300。

## 输出

- `trajectory.tum`:导出的载体系轨迹,米制,每个选中帧一行。
- `tracking_export.json`:轨迹路径、单位、`scene_radius_m`、实际生效的 `max_drift_m`、`lr_t_init_m_per_step` 和先验权重。
- `checkpoints.th`、`events.*`(TensorBoard)。

## 位姿先验强度

16 帧、每场 40 次迭代,211 数据,下采样 4 倍,绝对初始化,R = 7.49 m。偏差是相对 tracking 的每帧最大值。`s*` 是 BA 轨迹与 tracking 之间的 Sim(3) 最佳拟合尺度,不使用 GT。

| λ_t (m⁻²) | 等价归一化权重 | 最大平移偏差 | 最大旋转偏差 | 路径长度(tracking 1.753 m) | s* |
|---|---|---|---|---|---|
| 0 | 0 | 248 mm | 0.039° | 2.300 m | 1.158 |
| 0.0178 | 1 | 6.1 mm | 0.051° | 1.748 m | 1.0006 |
| 0.178 | 10 | 2.1 mm | 0.059° | 1.754 m | 1.0004 |

λ=0 时光度项约束不住尺度:最佳拟合尺度膨胀约 16%,路径长度增加约 31%。λ 达到上游归一化权重 1 对应的值后,尺度偏差在 0.1% 以内。这些是 16 帧短片段的结果,只说明先验的作用,不能证明精度提高。

## 验证

- `python -m unittest discover -s tests`:9 项通过。覆盖 TUM 读取与时间匹配、标定转换、首帧为 G、主点、子序列与跳帧对齐、绝对初始化、米制与归一化渲染逐像素等价、上游 `transforms.json` 路径的归一化不变。
- 米制模式与归一化模式的逐步优化对照:前约 30 步两者的损失和位姿一致到 1e-10。之后差异缓慢放大到约 1 mm。把一帧位姿加 1.2e-7 的扰动,也会产生同量级的发散。因此**小于约 1 mm 的运行间差异不可靠**。
- 多场运行(`--max_drift 0.75`,R = 8):创建两个局部场,导出 16 行有限轨迹,四元数范数为 1,首帧为单位位姿。

## 已知限制

- R 尚未实验选择。8 m 与归一化版本在 211 数据上的等效半径(7.5–7.9 m)一致,只是沿用,不是验证结果。
- 采样近端为 0.1R(约 0.8 m),更近的物体不被采样。这是上游行为。
- 多场:新场创建后、第一批新帧挂上之前,先验为零;重叠帧冻结在旧场上。接缝处的台阶等于重叠帧的修正量,尚未在整序列上测量。
- 留出帧在导出轨迹中是绝对 tracking 初值,未被优化。
- 首个选中帧的位姿固定在 tracking 值(规范自由度)。
- 渲染和留出图像评估尚未实现。
- 目前只在 16 帧短片段上验证过,未做整序列和 8 GB 显存测试。
