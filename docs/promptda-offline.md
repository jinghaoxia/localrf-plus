# PromptDA 离线深度生成

入口：`scripts/promptda.py`。不改动既有 `scripts/estimate_pose.py`。
本脚本只生成深度，不修改 BA 目标、不启动训练、不重跑 tracking、不读取 GT。
输入文件和已有实验结果只读；`--out` 必须是不存在的新目录。

## 环境与搬到服务器

`PromptDA/` 通过 Git submodule 管理，主仓库记录官方源码的固定提交（当前为 `9161547ca7151e14fcc8f490ad3cc02fa9437483`）。克隆或拉取主仓库后，从仓库根目录执行：

```bash
git submodule update --init -- PromptDA
```

该命令获取主仓库记录的版本；不要使用 `--remote` 自动切换上游版本。
模型权重不纳入 Git，需单独传输到 `PromptDA/models/model.ckpt`，或通过 `--checkpoint` 指定其他路径。本地子模块的 `models/` 已加入 Git 本地排除规则；新克隆如需将权重放入该目录，可执行一次：

```bash
git -C PromptDA rev-parse --git-path info/exclude
```

在该命令输出的排除文件中加入 `/models/`，避免权重显示为子模块未跟踪文件。不要使用忽略全部 dirty 状态的配置，以免隐藏源码修改。
直接复制到服务器时，需包含 `scripts/`、`localTensoRF/dataLoader/tracking_poses.py`、已初始化的 `PromptDA/` 源码及本地权重；不需要复制本机 `.venv`。

选择适合服务器 CUDA 的 PyTorch 环境，再安装本脚本所需的小集合依赖：

```bash
python -m pip install numpy scipy opencv-python PyYAML huggingface-hub
```

无需为本脚本完整安装 PromptDA 上游的训练、Gradio 和视频依赖，也不必照上游 requirements 降级已有 PyTorch。
运行时设置 Hugging Face 离线模式，只接受本地 checkpoint，不会自动下载模型。
`huggingface-hub` 是上游模块导入依赖，即使不联网仍需安装。
本次测试没有安装 xFormers，普通 PyTorch 路径可以运行。
默认 checkpoint 是仓库下 `PromptDA/models/model.ckpt`，默认编码器 `vitl`。
Small 权重需要显式 `--encoder vits`；权重必须与编码器匹配。

## 先用官方样例检查环境

以下命令从仓库根目录运行。每次换一个新的输出目录：

```bash
python scripts/promptda.py \
  --image PromptDA/assets/example_images/image.jpg \
  --prompt-depth PromptDA/assets/example_images/arkit_depth.png \
  --checkpoint PromptDA/models/model.ckpt \
  --out /path/to/new_promptda_smoke \
  --device cuda:0 --max-size 504
```

无 GPU 时使用 `--device cpu --max-size 196`。支持任意工作目录：使用脚本的绝对路径即可；默认源码和权重位置以脚本所在仓库定位。
单图输入的提示文件接受单通道 uint16 PNG（毫米）或二维 float `.npy`（米）；0、负值、NaN/Inf 为无效。
提示需与图像覆盖相同视场，可以是低分辨率。常数提示和不足两个有效像素会报错，避免模型的 min/max 归一化除零。

## 数据集短片段

显式提供原有 tracking TUM，不自动挑选历史结果。先跑两帧检查结果：

```bash
python scripts/promptda.py \
  --datadir /path/to/dataprocess/211 \
  --trajectory /path/to/explicit_tracking/trajectory.tum \
  --checkpoint /path/to/localrf-plus/PromptDA/models/model.ckpt \
  --out /path/to/dataprocess/211/output/promptda_short_unique \
  --start 0 --end 2 --frame-step 1 \
  --device cuda:0 --max-size 504 --prompt-width 256
```

确认后，在**新目录**运行更长片段；省略 `--end` 表示运行至末尾，省略 `--start` 表示从 0 开始。
`--start/--end` 是原始 output_index 的左闭右开区间，然后应用 `--frame-step`。
即使只运行片段，也先校验完整 pairs.csv 和 tracking 的一对一时间匹配，容限 1 微秒。
TUM 必须是图像时刻的米制 `T_world_body`，原始第 0 帧为单位阵，不得传入相机轨迹。

### 处理流程和约定

1. 从 `meta/calib.yaml` 读取矫正左目 K、`T_body_cam`、`T_body_lidar` 与时间原点；仅使用静态标定，不读 GT 或 source_world_anchor。
2. 读取 `meta/pairs.csv`、`images/NNNNN.png` 和 `pcd/NNNNN.pcd`，按 output_index 对齐。PCD 支持本数据集的 binary float32 x/y/z/intensity，不使用 VIEWPOINT。文件尾部只允许零填充。
3. 在图像时刻 tracking 之间使用世界系线性平移与 SO(3) 最短弧插值，估计扫描时刻载体位姿。序列两端采用最近两帧作有限外推，并记录到 manifest。`--max-time-offset` 默认 0.25 秒，同时限制扫描/图像差和端点外推范围。
4. 使用 `T_cam_lidar = inv(T_world_body_image @ T_body_cam) @ T_world_body_scan @ T_body_lidar` 转到图像时刻相机系。仅作扫描整体时刻补偿，**没有逐点去畸变**。
5. 将相机 z（非欧氏距离）投到低分辨率提示网格。支持独立 fx/fy，使用 OpenCV 半像素 resize 的主点约定，同像素保留最近点。默认范围 0.8–60 米；可用 `--min-depth/--max-depth` 修改。
6. 提示空洞用图像平面最近有效像素填充，包括观测覆盖范围以外的区域。填充是工程起点，**不是新增的可靠几何观测**，没有证明适合 HDL32。使用 `--save-prompts` 时原始测量掩码与填充距离单独保存。
7. 图像最长边不超过 `--max-size`，宽高分别向下取 14 的整数倍，不裁剪视场。推理尺寸与存储尺寸独立：默认输出为原图宽高的 1/4（1920×1200 → 480×300），直接双线性缩放，不裁剪，深度值不随 resize 缩放。`--output-downsampling 1` 恢复原图尺寸；放大并不增加几何信息。实际原图、推理和输出尺寸均写入 manifest。

默认单帧 FP32 推理；`--amp` 可选 CUDA float16，未在本机验证。显存不足先减小 `--max-size`（例如 504/392），不要同时运行 BA。默认 1008 只是推理尺寸设置，不代表已经验证 8 GB 显存预算。

## 产物

| 路径 | 内容 |
| --- | --- |
| `depth/NNNNN.png` | **默认**：1/4 宽高，单通道 uint16 光轴 z 深度；默认每单位 0.01 米（厘米），可用 `--png-unit-m` 调整，读取时乘 manifest 中的 `depth_scale_to_metres` |
| `depth/NNNNN.npy` | 仅 `--depth-format npy`：float32 米制深度，不受 uint16 编码量程限制 |
| `preview/NNNNN.png` | 仅 `--save-preview`：输出分辨率的观测图与彩色深度并排；固定色标，超范围只截断颜色 |
| `prompt/NNNNN.npy` | 仅 `--save-prompts`：实际送入模型的完整提示深度，低分辨率、float32 米 |
| `prompt_sparse/NNNNN.npy` | 仅 `--save-prompts`：填充前提示，未命中处为 0 |
| `prompt_valid/NNNNN.png` | 仅 `--save-prompts`：提示网格上的真实有效像素，255 有效、0 无效；不是稠密预测的置信度 |
| `prompt_distance/NNNNN.npy` | 仅 `--save-prompts`：到最近有效提示像素的二维距离，单位为提示网格像素，不是米或概率 |
| `manifest.json` | 运行状态、输入路径和哈希、权重 SHA256、选帧、时间差、外推标记、尺寸、覆盖率、深度统计与每帧耗时 |

默认只保存 `depth/` 与 `manifest.json`，关闭预览和提示中间文件。

### 存储大小与读取

旧版每帧默认保存全分辨率浮点深度、双宽对比图和提示中间文件。1920×1200 的 float32 深度本身约 9.22 MB；减小 `--max-size` 只减少推理尺寸，旧版仍放大保存，因此不能有效控制文件大小。

新版默认 `--output-downsampling 4 --depth-format png`。480×300 单通道 16 位像素数据未压缩为 288,000 字节（约 0.288 MB），PNG 压缩后的大小随内容变化。1 cm 量化步长，四舍五入误差不超过约 5 mm；这是保存精度，不代表预测准确度。最大可表示值 655.35 m，若输出超出范围会报错，**不静默截断或溢出**；改用 `--depth-format npy`。默认数据集提示范围 0.8–60 m 在 PNG 范围内。

默认使用 `--png-unit-m 0.01`（每单位 1 cm，无需额外指定），仍为 uint16，每像素 2 字节，最大 655.35 米、四舍五入误差不超过约 5 mm。PNG 实际压缩大小仍依图像内容变化。如确需毫米编码，可显式指定 `--png-unit-m 0.001`，对应上限 65.535 米。新旧输出均以各自 manifest 为准，不能固定除以 1000。

注意输入提示也有范围限制：`--max-depth` 默认 60，会排除更远的 LiDAR 点。若需要例如 200 米内的提示，显式设置 `--max-depth 200`；这仅改变允许输入的范围，不保证传感器有该距离的有效返回或模型在该距离准确。PromptDA 使用提示的 min/max 进行归一化和反归一化，扩大文件编码范围本身不会生成可靠远深度。

单图 `--prompt-depth` 的 PNG **输入**仍遵循官方毫米约定；上述 `--png-unit-m` 只控制输出。如要把厘米编码输出再作为输入提示，应先按 manifest 转换为米制 `.npy`。

```python
import json
import cv2
import numpy as np

with open("manifest.json", encoding="utf-8") as stream:
    scale = json.load(stream)["depth_scale_to_metres"]
depth_m = cv2.imread("depth/00000.png", cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
```

不要把数值深度 PNG 当成三通道彩色图读取，也不要使用 JPEG 保存训练深度。输出视场不变；需要投影时应按 manifest 中的真实缩放比例处理内参及半像素中心。可用 `--output-downsampling 8` 进一步减少空间，但会降低空间细节。

需要旧版全部产物时显式添加：

```bash
--depth-format npy --output-downsampling 1 --save-preview --save-prompts
```

不会自动删除或转换已有结果；使用新版对新目录运行即可。

输出目录必须不存在，避免混合实验；本脚本不实现断点续跑。中断或失败后保留已生成文件，并在可捕获的异常下将 manifest 标记为 failed。重新运行使用新的输出目录，可以显式设置新的帧段。

这一步只证明生成流程可运行。PromptDA 输出是伪深度；低纹理、遮挡、远处和稀疏提示区域的精度仍需评估。不能将 PNG 颜色当几何真值，数值 PNG 和 `.npy` 都是绝对 z 深度，旧逆深度/尺度不变损失接口已删除。当前 BA 可通过 `--depth_weight` 启用米制 L1 深度监督;PNG 还需按实际编码设置 `--depth_scale`（本样本为 `0.01`）,详见 `neural-ba-usage.md`。

## 本次验证（2026-11-01）

- `.venv/bin/python -m unittest discover -s tests -p 'test_promptda_offline.py' -v`：10 项合成 CPU 测试通过，覆盖 PCD、米制投影、双焦距/半像素、时间补偿、子序列、提示填充、输出和拒绝覆盖。
- 使用本地 `PromptDA/models/model.ckpt`、官方 image.jpg + arkit_depth.png，CPU、`--max-size 196` 实际推理成功；输入缩放至 196×140，输出还原至原图，全部为有限正深度。
- 没有选择真实数据的历史 tracking、没有运行实际序列的 PromptDA 推理，也没有验证服务器 CUDA、AMP、整序列性能或深度精度。

### 节省存储改动验证

- 新增默认低分辨率输出、毫米编码往返、范围溢出拒绝与浮点格式兼容测试；PromptDA 专项测试共 12 项通过。
- 完整 CPU 测试共 32 项通过。
- 此前在毫米编码版本使用本地 Large 权重和官方样例完成 CPU 推理（196×140 推理、原图 1920×1440）：保存 480×360 的 uint16 深度 PNG，文件 70,425 字节，manifest 约 2.4 KB。该大小是样例实测，不是所有图像的保证。
- 修复入口更名为 `scripts/promptda.py` 后对上游 `promptda` namespace package 的遮蔽问题；本地权重实际加载与推理通过。

- 可配置 PNG 量程：合成测试覆盖 70/200/655.35 米的厘米编码往返及超范围拒绝。

- PNG 默认编码单位已改为厘米（0.01 米）；13 项专项测试通过，覆盖默认编码和 manifest 比例的一致性。既有输出不自动转换。
