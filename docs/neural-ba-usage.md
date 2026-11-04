# 神经 BA 使用说明

神经 BA 是在原 LocalRF 训练入口 `localTensoRF/train.py` 上的增量改动,不另起训练流程。输入为图像、标定、时间配对表和**显式指定的** tracking 轨迹；可选读取已生成的稠密光轴 z 深度。

## 运行

从仓库根目录运行,新训练的 `--logdir` 必须是新目录或空目录；显式 `--resume` 才允许继续写入该存档所在的非空目录。下面以仓库内的 `211` 样本为例:

```bash
python localTensoRF/train.py \
  --datadir ./211 \
  --trajectory ./211/meta/estimate_pose.tum \
  --logdir /path/to/new_run \
  --downsampling 4 --subsequence 0 16 --frame_step 1 \
  --n_init_frames 5 --n_overlap 3 --n_iters_per_frame 40 --n_iters_reg 20 --add_frames_every 20 \
  --batch_size 2048 --test_frame_every 0 \
  --L1_weight 0 --TV_weight_density 0 --TV_weight_app 0 \
  --depth_weight 0 --lr_exposure_init 0 \
  --uniform_loss_weights 1 --seed 20211202 \
  --scene_radius_m 8 --pose_prior_lambda_t 0.0156 --pose_prior_lambda_r 1 \
  --checkpoint_every 1000 --pose_vis_every 2000 --device cuda:0
```

当前默认关闭深度监督、密度 L1、TV 和曝光优化,固定内参,所有选中帧参与训练,使用保留遮罩的统一光度权重。示例作为无深度基线。要检查深度约束,另建输出目录,将 `--depth_weight 0` 改为 `--depth_weight 0.0001 --depth_scale 0.01 --grad_log_every 20`,并与 `0.001` 对照。`1e-4～1e-3` 只是 211 短片段的试验范围,不是整序列已验证的权重;不再以 `0.1` 作为用法示例。`--subsequence` 使用右端不包含的 `[start, end)`,然后应用 `--frame_step`;全部样本为 `[0, 31)`。

## 断点续训（2026-11-04）

新训练默认在开始时、每 **1000 个完整训练步**、每次切换局部场和训练结束时写入续训存档。`--checkpoint_every` 调整步数间隔，必须大于 0；它独立于可选完整渲染快照 `--vis_every`（现默认 0，关闭）。保存位置是固定的 `resume.th`，通过同目录临时文件、fsync 和原子替换提交。写盘失败不会先清空上一份存档。

续训只需要指定新版存档；原始参数自动恢复，速度因子不会重复应用：

```bash
python localTensoRF/train.py --resume /path/to/new_run/resume.th
```

默认在存档所在目录继续，TensorBoard 以存档的下一步编号清除旧日志中已被回退的后续记录。也可以写入新的空目录，并在 OOM 后显式降低 batch（仍需为 16 的倍数）：

```bash
python localTensoRF/train.py \
  --resume /path/to/new_run/resume.th \
  --logdir /path/to/resumed_run \
  --device cuda:0 --batch_size 1024
```

恢复内容包括全部场/相机参数、Adam 状态、每帧归属、混合权重、细化与升分辨率/alpha mask 调度、活动/预读数据窗口、添加帧计数、下一步编号，以及 Python/NumPy/PyTorch CPU 和训练设备的 CUDA 随机状态。冻结场以 `resume_fields/rf*.th` 分片保存一次，周期性只更新活动场和其他训练状态。**迁移或备份时必须一起保留 `resume.th` 和完整 `resume_fields/` 目录**。恢复时历史场保留在 CPU，仅当前场放到训练设备；历史场仍占用主机内存。

Ctrl+C 或 SIGTERM 会请求完成当前步后保存并退出。CUDA OOM、强制杀进程、断电等使用上一次已提交存档；不保存可能已完成部分优化器更新的异常中间状态。存档间的训练步需要重做，默认最多约 1000 步。若最后一个存档已标记训练结束，重启会补做最终 checkpoint 和轨迹导出。

- `--device`、`--batch_size`、日志/保存频率可以显式调整；batch 或计算设备改变后不承诺逐步数值一致。
- `--datadir`、`--trajectory` 可用于迁移输入位置，但必须通过一致性校验：选中图像名/内参/留出划分、元数据与轨迹内容、图像/深度/实际使用遮罩的大小和纳秒 mtime。复制数据需保留文件时间。观测大文件没有逐字节内容哈希，保留大小和时间的内容篡改不在此校验范围。
- 损失权重、选帧、窗口、场结构和学习率/调度不允许在续训时更改。CLI 和 config 文件的显式冲突均会报错。
- `run_history.jsonl` 追加记录原始参数、有效参数、数据指纹、恢复来源和起始步，便于追溯。
- `--pose_vis_every` 默认 2000，设为 0 关闭位姿绘图；`--progress_refresh_rate` 只控制控制台进度及资源日志。场边界每步只记录当前场，切场时补记刚完成场的最终边界。CUDA 资源日志包括 allocated/reserved/间隔内峰值和设备剩余显存。
- 旧 `checkpoints.th/checkpoints_tmp.th` 只包含渲染权重；旧 TensorBoard 日志也缺少训练状态，不能作为 `--resume` 输入。新版训练结束仍生成兼容渲染的 `checkpoints.th`。目前 `render.py` 使用完整渲染 checkpoint，不直接接收分片续训存档。

验证覆盖合成 CPU 多场训练、非零起始片段、深度、曝光/内参优化、非默认速度因子、升分辨率、alpha mask、同目录及新目录续训、SIGTERM 和模拟 OOM/磁盘写入失败。没有在本机验证 CUDA 数值复现、真实 GPU OOM 或完整 211/223 序列；详见 [中断与日志排查记录](training-resume-investigation-2026-11-04.md)。

## 数据与内参

`LocalRFDataset` 负责 tracking TUM、标定、时间匹配、选帧和轨迹导出。训练入口只接收数据集提供的内参、位姿先验和采样数据。跟踪转换的几何辅助函数仍位于 `dataLoader/tracking_poses.py`。

内参从 `meta/calib.yaml` 的 `left_camera.K` 自动读取,支持独立 `fx`、`fy`。主点先从 OpenCV 整数像素中心转为 LocalRF 的 `i+0.5` 约定,再按实际输出宽高分别缩放。原图尺寸必须与标定一致。没有标定文件的旧数据可以从 `transforms.json` 的 `fl_x/fl_y/cx/cy/w/h` 读取;缺少 `fl_y` 时保留旧数据的方形像素假设,缺少主点时使用图像中心。`--with_preprocessed_poses 1` 仍用于启用旧 JSON 位姿及其归一化。

已移除 `--fov` 和全景射线分支,也不提供手动 `--fx/--fy/--cx/--cy`。已移除 flow 参数、加载、损失及其辅助函数,以及未实现的 `--ckpt` 续训参数、无效的 `--model_name`/`--nSamples` 选项。旧参数不会再被静默接受。

## 稠密深度约束

`--depth_weight > 0` 时,按选中图像名读取 `depth/NNNNN.png` 或 `depth/NNNNN.npy`;同一帧两种格式同时存在会报错。深度必须与图像覆盖同一视场,可以是较低分辨率,以最近邻缩放到训练尺寸。

- PNG 必须是单通道 uint16,显式提供 `--depth_scale`（每个整数单位对应的米数）。`211` 样本是 `0.01`,不能按毫米读取。
- `.npy` 必须是二维浮点米制 z 深度,无需单位换算。
- 只使用有限、正深度及 `masks/` 允许的像素。0、负值、NaN、Inf 不参与监督。`--uniform_loss_weights 1` 保留遮罩,仅关闭清晰度加权。
- 米制深度需要显式 tracking 轨迹,不与旧 JSON 归一化位姿混用。`--depth_weight 0` 完全跳过深度文件加载。

损失为 `L = 0.25 L_RGB + depth_weight × mean_valid(|D_hat − D|) + L_pose`,另可显式开启原有场正则。`D_hat` 是 `Σ(w·z)/max(Σw, 1e-8)` 的场景内期望光轴 z 深度,排除强制不透明的远端边界采样,从场单位换算到米。训练与深度导出共享该计算,梯度传到密度和位姿。

深度项在渐进添加帧和细化阶段都启用,不随 `n_iters_reg` 关闭或学习率衰减,不做逆深度、尺度/偏移拟合或分位数裁剪。有效像素只由观测决定,不会通过渲染不透明度筛掉高误差像素;全部观测无效时该批深度项为 0。TensorBoard 记录 `depth_l1_m`、有效比例和不透明度均值。PromptDA 输出是伪深度,不等于 GT。

## 损失尺度与梯度诊断

`train/rgb_loss` 已包含 0.25 系数,`train/depth_l1_m` 是原始米制误差,`train/depth_loss` 是乘过 `depth_weight` 的项。按审查中的 `rgb_loss=0.0118`、`depth_l1_m=8.65 m`,若只让两个标量同量级,λ≈0.00136;让深度贡献约为 RGB 的 10%～100% 对应 λ≈1.36e-4～1.36e-3。这只是该步的量纲检查,不代表梯度相当或最佳收敛参数。

本次 211 前 8 帧、两个种子、每组 321 步的 CPU 对照中,λ=0.001 的训练视图深度 L1 均值为 0.112 m,λ=0.1 为 0.089 m;后者的深度/RGB 梯度 L2 比值中位数（再对种子平均）在密度/旋转/平移组分别约为 65/56/30,前者约为 0.69/0.65/0.31。加入深度后训练光度项均高于无深度基线。完整条件、逐种子结果和限制见 [深度权重实验记录](depth-weight-study-2026-11-02.md)。不能按最终标量损失较小就断言各项梯度平衡。

设置 `--grad_log_every N` 可每 N 步在相同前向样本上记录各**已加权**损失对实际活动参数的梯度:

- `grad/{rgb,depth,pose_prior}/{density,appearance,rotation,translation}_l2` 和 `_max` 分别为组内 L2 范数和最大绝对分量;相应损失不存在时不记录该项。曝光/内参仅在参与更新时记录。
- `grad/parameter_count/<group>` 记录组内参数数量;首帧零学习率、旧场与旧场所属位姿不计入。不同单位的参数组不合并比较。
- 默认 `0` 关闭诊断。开启会对每个损失额外调用 `autograd.grad` 并保留图供正常反传,增加计算和临时内存开销,不修改 `.grad`、优化器或损失。这里记录梯度,不是 Adam 预条件后的更新量;两项方向相同或相反也不能由范数判断。

保留深度分母下限 `1e-8`,不默认提高下限、按预测不透明度筛除观测或启用梯度裁剪。低不透明度时独立采样权重的导数确会放大,实际参数还经过密度激活和透射率的链式导数。本次实际场压力检查中没有出现同等放大,但不构成全局梯度上界。提高到 `1e-4` 会明显改变前向深度,例如 A≈1.2e-8 的固定测试由 94.5 m 降为 0.011 m;因此先用不透明度比例和参数梯度诊断监测,保留这一数值风险。

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
- `--pose_prior_lambda_t`、`--pose_prior_lambda_r`:每帧位姿相对 tracking 的平移平方和(单位 m⁻²)、旋转平方和权重。0 表示不加位姿先验;是否含深度项由 `--depth_weight` 决定。
- `--uniform_loss_weights 1`:关闭上游逐帧 Laplacian 清晰度加权,保留有效像素遮罩（默认 1）。
- `--test_pose_opt 0`:使留出帧不参与位姿更新（默认 0,用于留出实验）。
- `--seed`:Python、NumPy 和 PyTorch 的随机种子,默认 20211202;在 `reconstruction()` 内设置,包括直接调用入口。不能保证 GPU 算子逐位确定性。
- `--grad_log_every`:梯度诊断间隔,默认 0 关闭,详见上节。
- `--downsampling`:训练分辨率。原图 1920×1200,4 倍下采样为 480×300。

## 输出

- `trajectory.tum`:导出的载体系轨迹,米制,每个选中帧一行。
- `tracking_export.json`:轨迹路径、单位、`scene_radius_m`、实际生效的 `max_drift_m`、`lr_t_init_m_per_step` 、先验权重、随机种子、深度单位/权重及实际缩放后的初始内参。
- `checkpoints.th`：训练结束的完整渲染模型；`events.*`：TensorBoard。
- `resume.th`、`resume_fields/`：周期性分片续训存档，必须一起保留。
- `run_history.jsonl`：启动/续训参数与来源记录。

## 渲染

`localTensoRF/render.py` 读取 `checkpoints.th`,按训练时保存的优化位姿渲染每个已保存视图。它通过 `LocalTensorfs.forward` 的推理模式渲染:无采样抖动,按保存的混合权重合成各局部场,并应用保存的曝光矩阵。训练时每帧的光度损失只由其最新的局部场计算,因此重叠帧的渲染(多场混合)与训练损失所见的预测可能不同。`--subsequence` 和 `--frame_step` 必须与训练一致:检查点不记录选帧,脚本会比对视图数,不一致时拒绝运行。

```bash
python localTensoRF/render.py \
  --checkpoint /path/to/new_run/checkpoints.th \
  --datadir /path/to/dataprocess/211 --subsequence 0 16 --frame_step 1 \
  --out /path/to/new_run/render --device cuda:0 --save_depth
```

`--out` 必须是新目录或空目录,输出为:

- `rgb/`:渲染图(8 位 RGB)。
- `compare/`:观测 | 渲染 | 误差(放大 4 倍),需要 `--datadir`。
- `depth/`:沿光轴的 z 深度(不是直线距离;`.npy` 与彩色 PNG),需要 `--save_depth`。深度是场景内期望深度 Σ(w·z)/Σw,只统计场景内的采样点(去掉 `alpha2weights` 强制不透明的远端边界点),并用保存的混合权重合并各场。不透明度低于 `--min_opacity`(默认 0.5)的像素记为 NaN(`.npy`)与黑色(PNG)。`opacity/` 给出累积不透明度(白色为不透明)。单位与位姿一致:用 `--trajectory` 训练的检查点为米;不带 `--trajectory`(上游归一化位姿)训练的检查点为归一化单位,不是米。深度的精度受沿射线的采样步长限制:64³ 网格、R = 8 m 时约 0.22 m。
- `metrics.json`:逐帧 PSNR 与均值,需要 `--datadir`。

内部 `render_depth_moment()` 返回未归一化的一阶深度矩 `M=Σ(field_blend × Σ(w·z))` 及场内不透明度 A,`expected_depth()` 再返回 M/A 并按导出阈值标记 NaN。M 的单位随位姿（tracking 为米）,但 M 本身不是期望深度。

这是对训练所用帧的拟合检查,不是留出评估。不给 `--datadir` 时视图按检查点中的序号命名,且不计算 PSNR。`--ray_chunk`(默认 4096)控制每次前向的射线数,用于降低显存占用;`--stride` 只渲染每隔若干个视图。

## 历史位姿先验强度（未使用稠密深度）

16 帧、每场 40 次迭代,211 数据,下采样 4 倍,绝对初始化,R = 7.49 m。偏差是相对 tracking 的每帧最大值。`s*` 是 BA 轨迹与 tracking 之间的 Sim(3) 最佳拟合尺度,不使用 GT。

| λ_t (m⁻²) | 等价归一化权重 | 最大平移偏差 | 最大旋转偏差 | 路径长度(tracking 1.753 m) | s* |
|---|---|---|---|---|---|
| 0 | 0 | 248 mm | 0.039° | 2.300 m | 1.158 |
| 0.0178 | 1 | 6.1 mm | 0.051° | 1.748 m | 1.0006 |
| 0.178 | 10 | 2.1 mm | 0.059° | 1.754 m | 1.0004 |

λ=0 时光度项约束不住尺度:最佳拟合尺度膨胀约 16%,路径长度增加约 31%。λ 达到上游归一化权重 1 对应的值后,尺度偏差在 0.1% 以内。这些是 16 帧短片段的结果,只说明先验的作用,不能证明精度提高。

## 稠密深度初次接入验证（2026-11-02，历史记录）

- `.venv/bin/python -m unittest discover -s tests -v`:49 项 CPU 测试通过。新增独立双焦距/非整数缩放投影、深度单位/遮罩/选帧/窗口释放、数据集导出、空有效区域、深度到场与位姿的反传、旧场冻结、旧单焦距 checkpoint 迁移测试。
- 本地 `211` 前 5 帧,64×40、16³ 网格、CPU、`depth_weight=0.1`、`depth_scale=0.01`、`n_iters_reg=0`:完成 21 次迭代、两个局部场,深度日志每步都有且全部标量有限,输出 5 行有限轨迹,首帧固定,四元数为单位长度。保存后重载并导出全部 5 帧 RGB、深度、不透明度及观测对比图成功。
- 这是数据与训练流程检查,不是深度质量或收敛验证。未进行本次 GPU、整序列、正式留出对照或真实轨迹精度评估。

## 历史验证（稠密深度接入前）

- `python -m unittest discover -s tests`:9 项通过。覆盖 TUM 读取与时间匹配、标定转换、首帧为 G、主点、子序列与跳帧对齐、绝对初始化、米制与归一化渲染逐像素等价、上游 `transforms.json` 路径的归一化不变。
- 米制模式与归一化模式的逐步优化对照:前约 30 步两者的损失和位姿一致到 1e-10。之后差异缓慢放大到约 1 mm。把一帧位姿加 1.2e-7 的扰动,也会产生同量级的发散。因此**小于约 1 mm 的运行间差异不可靠**。
- 多场运行(`--max_drift 0.75`,R = 8):创建两个局部场,导出 16 行有限轨迹,四元数范数为 1,首帧为单位位姿。

## 第一轮代码审查修复（2026-11-02）

- 切场重叠权重改为固定数量 `linspace`，避免 49、98、103 等帧数的浮点步长导致数组越界；零有效重叠显式报错。
- `--n_lamb_sigma 16 16 16`、`--n_lamb_sh 24 24 24` 支持三个正整数，也支持重复选项传入；显式值替换默认值，数量不符时拒绝运行。
- tracking 前端修复半像素缩放与含旋转 LiDAR 外参的平移换算；仅影响今后显式运行的 tracking，不修改历史轨迹。
- 无效 `white_bg` 接口及未调用的单场保存接口已移除。RGB 保留远端可学习背景，深度仍只统计场内采样。用 `-expm1` 计算 alpha，减少极低密度的浮点消减。训练记录 `depth_low_opacity_fraction`（有效观测中预测 opacity < 1e-4 的比例），仅作诊断、不参与筛选，以及加权后的 `depth_loss`。
- 命令行入口补齐 Python `random` 种子，导入路径按脚本目录解析；已移除无用的切场迭代变量。

第一轮数值检查、67 项测试范围及保留的设计风险见 [审查记录](neural-ba-review-2026-11-02.md)。后续权重扫描、低不透明度参数梯度检查、载体系候选框修复和当前测试结果见 [深度权重实验记录](depth-weight-study-2026-11-02.md)。

## 已知限制

- R 尚未实验选择。8 m 与归一化版本在 211 数据上的等效半径(7.5–7.9 m)一致,只是沿用,不是验证结果。
- 采样近端为 0.1R(约 0.8 m),更近的物体不被采样。这是上游行为。
- 多场:新场创建后、第一批新帧挂上之前,先验为零;重叠帧冻结在旧场上。接缝处的台阶等于重叠帧的修正量,尚未在整序列上测量。
- 留出帧在导出轨迹中是绝对 tracking 初值,未被优化。
- 首个选中帧的位姿固定在 tracking 值(规范自由度)。
- 留出图像评估尚未实现。`render.py` 只渲染检查点保存的视图(训练帧,以及 `--test_frame_every` 留下的测试帧),不渲染任意新视角。
- 目前只在 16 帧短片段上验证过,未做整序列和 8 GB 显存测试。
