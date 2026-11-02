#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指定数据集目录的单文件 v3 位姿估计。

输入：矫正左目、时间配对表、静态标定、原始 HDL32；不读取 GT/稠密深度。
约定：p_A = T_A_B @ p_B；输出为图像时刻载体系到首帧载体系的变换。
依赖：numpy、scipy、opencv-python、PyYAML；沿用已验证 v3 的计算顺序和参数。
运行：python estimate_pose.py --dataset-dir E:/dataprocess/211
输入目录：images/、未补偿的原始 pcd/、meta/pairs.csv、meta/calib.yaml。
--check-only 只检查输入；--end 64 只估计前 64 帧。
输出 DATASET/output/pose_estimation_RUN/ 下的两份 TUM，不覆盖已有结果。
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml


def estimate_sequence(dataset_directory, output_directory, end=None):
    """主流程：读取数据 -> 视觉运动 -> LiDAR 融合 -> 写出载体系位姿。"""
    data = load_sequence(dataset_directory, end)
    output_directory.mkdir(parents=True, exist_ok=False)

    # 1. 视觉特征 + 原始 LiDAR 深度，得到度量运动和有效性标记。
    visual_poses, visual_observed = estimate_visual_trajectory(data)
    visual_path = output_directory / "visual_trajectory.tum"
    save_trajectory(visual_path, data.image_times, visual_poses)

    # 2. 沿用原 v3 的 9 位小数 TUM 中间接口；融合只使用相邻视觉运动。
    visual_poses = load_visual_poses(visual_path)
    body_poses = estimate_lidar_trajectory(data, visual_poses, visual_observed)

    # 3. timestamp tx ty tz qx qy qz qw；秒、米、xyzw 四元数。
    trajectory_path = output_directory / "trajectory.tum"
    save_trajectory(trajectory_path, data.image_times, body_poses)
    print(f"{data.sequence} 最终轨迹：{trajectory_path}", flush=True)
    return body_poses


@dataclass
class SequenceData:
    """估计所需的路径、配对时间与静态外参，不包含参考位姿。"""

    sequence: str
    image_directory: Path
    lidar_directory: Path
    pairs: list[dict[str, int]]
    camera_matrix: np.ndarray
    T_body_camera: np.ndarray
    T_body_lidar: np.ndarray
    time_origin_us: int
    image_times: np.ndarray

    def read_image(self, frame):
        path = self.image_directory / f"{frame:05d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.resize(image, (640, 400), interpolation=cv2.INTER_AREA)

    def read_scan(self, frame):
        # 本地 pcd 已按 output_index 重编号；lidar_index 只用于追溯原始文件。
        index = self.pairs[frame]["output_index"]
        return read_lidar_scan(self.lidar_directory / f"{index:05d}.pcd")


def load_sequence(dataset_directory, end=None):
    """读取可搬移的数据集；只使用配对时间、原始 XYZ 和静态标定。"""
    sequence_directory = Path(dataset_directory).expanduser().resolve()
    if not sequence_directory.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{sequence_directory}")
    pair_path = sequence_directory / "meta" / "pairs.csv"
    with pair_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {"output_index", "camera_index", "lidar_index", "camera_utc_us",
                   "lidar_utc_us", "lidar_minus_camera_us"}
        if set(reader.fieldnames or ()) != columns:
            raise ValueError(f"pairs.csv 必须包含约定的六个字段：{pair_path}")
        pairs = [{key: int(value) for key, value in row.items()} for row in reader]
    if [pair["output_index"] for pair in pairs] != list(range(len(pairs))):
        raise ValueError("output_index 必须从 0 连续编号，并与本地 images/pcd 文件名一致")
    for key in ("camera_index", "lidar_index"):
        if any(pair[key] < 0 for pair in pairs) or len({pair[key] for pair in pairs}) != len(pairs):
            raise ValueError(f"{key} 必须为非负且不重复的一对一配对编号")
    end = len(pairs) if end is None else end
    if not 2 <= end <= len(pairs):
        raise ValueError("--end 必须在 2 和序列帧数之间，且为排他上界")
    camera_times = np.array([pair["camera_utc_us"] for pair in pairs], dtype=np.int64)
    lidar_times = np.array([pair["lidar_utc_us"] for pair in pairs], dtype=np.int64)
    if np.any(np.diff(camera_times) <= 0) or np.any(np.diff(lidar_times) <= 0):
        raise ValueError("相机和 LiDAR 的配对时间必须严格递增")
    if np.max(abs(lidar_times - camera_times)) > 250000:
        raise ValueError("相机/LiDAR 配对超过 0.25 秒")
    if any(pair["lidar_minus_camera_us"] != pair["lidar_utc_us"] - pair["camera_utc_us"]
           for pair in pairs):
        raise ValueError("lidar_minus_camera_us 与两个时间戳的差不一致")
    pairs = pairs[:end]

    calibration = yaml.safe_load((sequence_directory / "meta" / "calib.yaml").read_text(encoding="utf-8"))
    if calibration.get("pcd_motion_compensation") != "none" or calibration.get("pcd_indexing") != "output_index":
        raise ValueError("pcd/ 必须是未补偿的原始扫描，标定须声明 pcd_motion_compensation: none 和 pcd_indexing: output_index")
    if (calibration["left_camera"]["width"], calibration["left_camera"]["height"]) != (1920, 1200):
        raise ValueError("当前 v3 使用 1920×1200 的矫正图像；接入其他分辨率须调整缩放与内参")
    if not np.allclose(calibration["left_camera"]["D"], 0, atol=1e-12):
        raise ValueError("images/ 必须已经矫正，left_camera.D 应为 0")
    camera_matrix = np.asarray(calibration["left_camera"]["K"], float)
    if (camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all()
            or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0
            or not np.allclose(camera_matrix[2], [0, 0, 1])):
        raise ValueError("left_camera.K 必须是有效的 3×3 针孔内参")
    # OpenCV resize maps integer pixel centres by (p + 0.5) / 3 - 0.5.
    camera_matrix[:2] /= 3  # 1920x1200 -> 640x400; focal lengths and skew.
    camera_matrix[:2, 2] += 0.5 / 3 - 0.5
    T_body_camera = np.asarray(calibration["camera_to_body"]["T_body_cam"], float)
    T_body_lidar = np.asarray(calibration["lidar_to_body"]["T_body_lidar"], float)
    for name, matrix in (("T_body_cam", T_body_camera), ("T_body_lidar", T_body_lidar)):
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
            raise ValueError(f"{name} 必须为有效的 4×4 刚性变换")
    time_origin_us = int(calibration["time"]["origin_utc_us"])
    for pair in pairs:
        for folder, suffix in (("images", ".png"), ("pcd", ".pcd")):
            path = sequence_directory / folder / f"{pair['output_index']:05d}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
    return SequenceData(
        sequence=sequence_directory.name,
        image_directory=sequence_directory / "images",
        lidar_directory=sequence_directory / "pcd",
        pairs=pairs,
        camera_matrix=camera_matrix,
        T_body_camera=T_body_camera,
        T_body_lidar=T_body_lidar,
        time_origin_us=time_origin_us,
        image_times=np.array([(pair["camera_utc_us"] - time_origin_us) / 1e6 for pair in pairs]),
    )


def estimate_visual_trajectory(data):
    """第一阶段：视觉度量运动；16 帧原始 LiDAR 地图辅助重投影优化。"""
    previous_image = data.read_image(0)
    previous_scan = data.read_scan(0)
    body_poses = [np.eye(4)]
    observed = np.zeros(len(data.pairs), dtype=bool)
    predicted_motion, prediction_dt = np.eye(4), 0.5
    local_map = deque(maxlen=16)
    initial_points = prepare_visual_map_points(previous_scan)
    local_map.append(initial_points @ data.T_body_lidar[:3, :3].T + data.T_body_lidar[:3, 3])

    for frame in range(1, len(data.pairs)):
        current_image = data.read_image(frame)
        current_scan = data.read_scan(frame)
        body_motion, statistics, observations = estimate_visual_motion(
            previous_image,
            current_image,
            previous_scan,
            current_scan,
            data.pairs[frame - 1],
            data.pairs[frame],
            data.camera_matrix,
            data.T_body_camera,
            data.T_body_lidar,
            predicted_motion,
            prediction_dt,
        )
        observed[frame] = body_motion is not None
        image_dt = statistics["dt_s"]

        if observed[frame]:
            next_pose, _ = refine_visual_pose_with_lidar(
                body_poses[-1],
                body_motion,
                current_scan,
                data.pairs[frame],
                data.camera_matrix,
                data.T_body_camera,
                data.T_body_lidar,
                image_dt,
                observations,
                local_map,
            )
            body_motion = invert_pose(body_poses[-1]) @ next_pose
            body_motion[:3, :3] = Rotation.from_matrix(body_motion[:3, :3]).as_matrix()
            predicted_motion, prediction_dt = body_motion, image_dt
        else:
            # 视觉失败时按上一次有效运动的常扭量传播，保留无观测标记。
            body_motion = interpolate_motion(predicted_motion, image_dt / prediction_dt)
            next_pose = body_poses[-1] @ body_motion

        next_pose[:3, :3] = Rotation.from_matrix(next_pose[:3, :3]).as_matrix()
        body_poses.append(next_pose)
        if observed[frame]:
            scan_offset = (
                data.pairs[frame]["lidar_utc_us"] - data.pairs[frame]["camera_utc_us"]
            ) / 1e6
            T_world_lidar = (
                next_pose
                @ interpolate_motion(body_motion, scan_offset / image_dt)
                @ data.T_body_lidar
            )
            map_points = prepare_visual_map_points(current_scan)
            local_map.append(map_points @ T_world_lidar[:3, :3].T + T_world_lidar[:3, 3])
        previous_image, previous_scan = current_image, current_scan
        if frame % 100 == 0 or frame == len(data.pairs) - 1:
            print(f"{data.sequence} 视觉：{frame + 1}/{len(data.pairs)}", flush=True)

    return np.asarray(body_poses), observed


def estimate_lidar_trajectory(data, visual_poses, visual_observed):
    """第二阶段：LiDAR 主导姿态/高度，视觉只给局部 XY 位移软约束。"""
    T_body_lidar = data.T_body_lidar
    local_map = deque(maxlen=30)
    scan_poses, body_poses = [], []
    last_scan_motion, last_scan_dt = np.eye(4), 0.51
    predicted_frames = 0

    for frame, pair in enumerate(data.pairs):
        source_points = prepare_fusion_points(data.read_scan(frame), T_body_lidar)
        if frame == 0:
            current_scan_pose = T_body_lidar.copy()
            current_body_pose = np.eye(4)
        else:
            previous_pair = data.pairs[frame - 1]
            scan_dt = (pair["lidar_utc_us"] - previous_pair["lidar_utc_us"]) / 1e6
            image_dt = (pair["camera_utc_us"] - previous_pair["camera_utc_us"]) / 1e6

            # 将相邻图像的视觉运动缩放到相邻扫描的时间间隔。
            visual_motion = invert_pose(visual_poses[frame - 1]) @ visual_poses[frame]
            visual_motion = interpolate_motion(visual_motion, scan_dt / image_dt)
            previous_body_pose = scan_poses[-1] @ invert_pose(T_body_lidar)
            expected_body_pose = previous_body_pose @ visual_motion
            use_visual = bool(visual_observed[frame])
            visual_sigma = 0.005 * max(1.0, image_dt / 0.5) if use_visual else None

            # 正常初值：LiDAR 常扭量旋转 + 视觉平移；视觉旋转不进入优化残差。
            predicted_pose = scan_poses[-1] @ interpolate_motion(
                last_scan_motion, scan_dt / last_scan_dt
            )
            if use_visual:
                predicted_pose[:3, 3] = (
                    expected_body_pose[:3, 3] + lidar_offset_in_world(predicted_pose, T_body_lidar)
                )
            target_points = voxel_downsample(np.vstack(local_map), 0.8)
            current_scan_pose, statistics = register_scan_to_map(
                source_points,
                target_points,
                predicted_pose,
                previous_body_pose,
                expected_body_pose,
                T_body_lidar,
                visual_sigma,
            )
            success = accept_registration(statistics, predicted_pose, current_scan_pose, scan_dt)

            # 加减速会破坏常扭量初值，正常配准失败时再尝试视觉运动初值。
            if not success and use_visual:
                visual_initial = previous_body_pose @ visual_motion @ T_body_lidar
                alternative_pose, alternative_statistics = register_scan_to_map(
                    source_points,
                    target_points,
                    visual_initial,
                    previous_body_pose,
                    expected_body_pose,
                    T_body_lidar,
                    visual_sigma,
                )
                if accept_registration(
                    alternative_statistics, visual_initial, alternative_pose, scan_dt
                ):
                    current_scan_pose, success = alternative_pose, True

            if success:
                last_scan_motion = invert_pose(scan_poses[-1]) @ current_scan_pose
                last_scan_dt = scan_dt
                last_scan_motion[:3, :3] = Rotation.from_matrix(
                    last_scan_motion[:3, :3]
                ).as_matrix()
            else:
                current_scan_pose = predicted_pose
                predicted_frames += 1
                local_map.clear()  # 重建地图，避免一直对冻结的坏地图拒绝。

            # 扫描时刻的 LiDAR 位姿 -> 图像时刻的载体系位姿。
            current_body_pose = scan_pose_to_image_time(
                scan_poses[-1],
                current_scan_pose,
                (previous_pair["lidar_utc_us"] - data.time_origin_us) / 1e6,
                (pair["lidar_utc_us"] - data.time_origin_us) / 1e6,
                (pair["camera_utc_us"] - data.time_origin_us) / 1e6,
                T_body_lidar,
            )

        scan_poses.append(current_scan_pose)
        body_poses.append(current_body_pose)
        local_map.append(source_points @ current_scan_pose[:3, :3].T + current_scan_pose[:3, 3])
        if frame % 100 == 0 or frame == len(data.pairs) - 1:
            print(
                f"{data.sequence} 融合：{frame + 1}/{len(data.pairs)}，外推 {predicted_frames}",
                flush=True,
            )

    trajectory = np.asarray(body_poses)
    return invert_pose(trajectory[0]) @ trajectory


def prepare_fusion_points(raw_points, T_body_lidar):
    """保留 2~35 m 点，排除前方抬高候选点，再做 0.5 m 体素采样。"""
    radius = np.linalg.norm(raw_points[:, :2], axis=1)
    valid = np.isfinite(raw_points).all(axis=1) & (radius > 2) & (radius < 35)
    # This fixed foreground box is defined in body axes, not raw LiDAR axes.
    # It remains a dataset-specific geometric heuristic, not semantic segmentation.
    body_points = raw_points @ T_body_lidar[:3, :3].T + T_body_lidar[:3, 3]
    front_candidates = (
        (body_points[:, 0] > 2)
        & (body_points[:, 0] < 12)
        & (abs(body_points[:, 1]) < 1.3)
        & (body_points[:, 2] > 0.35)
    )
    return voxel_downsample(raw_points[valid & ~front_candidates], 0.5)


def accept_registration(statistics, initial_pose, fitted_pose, scan_dt):
    """沿用 v3 准入门限；内部通过不代表外部误差为零。"""
    correction = invert_pose(initial_pose) @ fitted_pose
    angle = float(np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude()))
    distance = float(np.linalg.norm(correction[:3, 3]))
    return (
        statistics["success"]
        and statistics["rmse_m"] < 0.15
        and statistics.get("inlier_ratio", 0) >= 0.65
        and distance < (2 if scan_dt > 1 else 1)
        and angle < (10 if scan_dt > 1 else 5)
    )


def save_trajectory(path, timestamps, body_poses):
    """TUM 为 xyzw 四元数，使用与原算法相同的 9 位小数。"""
    rows = np.column_stack(
        (
            timestamps,
            body_poses[:, :3, 3],
            Rotation.from_matrix(body_poses[:, :3, :3]).as_quat(),
        )
    )
    np.savetxt(path, rows, fmt="%.9f", header="timestamp tx ty tz qx qy qz qw")


def load_visual_poses(path):
    """保留原算法的 TUM 中间精度和四元数到旋转的处理。"""
    rows = np.loadtxt(path, ndmin=2)
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    poses[:, :3, 3] = rows[:, 1:4]
    return poses


# 位姿估计所需数值细节：几何 -> 跟踪 -> 视觉优化 -> LiDAR 融合。


# 读取原始 binary XYZ+intensity float32 的 XYZ；忽略 VIEWPOINT。
def read_lidar_scan(file_path: Path) -> np.ndarray:
    with file_path.open("rb") as stream:
        point_count = None
        while True:
            header_line = stream.readline()
            if not header_line:
                raise ValueError(f"Missing PCD DATA line: {file_path}")
            header_fields = header_line.split()
            if header_fields[:1] == [b"POINTS"]:
                point_count = int(header_fields[1])
            if header_fields[:1] == [b"DATA"]:
                if header_fields[1:] != [b"binary"] or point_count is None:
                    raise ValueError(f"Expected binary XYZ+intensity PCD: {file_path}")
                break
        payload = stream.read(point_count * 16)
    if len(payload) != point_count * 16:
        raise ValueError(f"Truncated PCD: {file_path}")
    xyz_points = np.frombuffer(payload, dtype="<f4").reshape(point_count, 4)[:, :3]
    return np.asarray(xyz_points, dtype=np.float64)


# 每个体素保留离中心最近的一点，沿用确定性排序。
def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    "每个体素保留离中心最近的一点，沿用确定性排序。"
    if len(points) == 0:
        return points
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    center_error = np.sum((points / voxel_size - voxel_indices - 0.5) ** 2, axis=1)
    sort_order = np.lexsort(
        (center_error, voxel_indices[:, 2], voxel_indices[:, 1], voxel_indices[:, 0])
    )
    sorted_voxels = voxel_indices[sort_order]
    first_in_voxel = np.r_[True, np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)]
    return points[sort_order[first_in_voxel]]


# 叉乘矩阵：[v]x @ w = v x w。
def cross_product_matrix(values: np.ndarray) -> np.ndarray:
    x, y, z = values
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# SO(3) 左雅可比；小角度用级数稳定计算。
def so3_left_jacobian(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    cross_matrix = cross_product_matrix(rotation_vector)
    if angle < 1e-5:
        first_coefficient = 0.5 - angle**2 / 24.0 + angle**4 / 720.0
        second_coefficient = 1.0 / 6.0 - angle**2 / 120.0 + angle**4 / 5040.0
    else:
        first_coefficient = (1.0 - np.cos(angle)) / angle**2
        second_coefficient = (angle - np.sin(angle)) / angle**3
    return (
        np.eye(3)
        + first_coefficient * cross_matrix
        + second_coefficient * (cross_matrix @ cross_matrix)
    )


# 常扭量 exp(time_ratio * log(motion))；平移与旋转耦合。
def interpolate_motion(motion: np.ndarray, time_ratio: float) -> np.ndarray:
    rotation_vector = Rotation.from_matrix(motion[:3, :3]).as_rotvec()
    translation_tangent = np.linalg.solve(so3_left_jacobian(rotation_vector), motion[:3, 3])
    scaled_rotation = rotation_vector * time_ratio
    scaled_motion = np.eye(4)
    scaled_motion[:3, :3] = Rotation.from_rotvec(scaled_rotation).as_matrix()
    scaled_motion[:3, 3] = so3_left_jacobian(scaled_rotation) @ (translation_tangent * time_ratio)
    return scaled_motion


# 扫描位姿按估计运动传播到图像时刻，再转为载体系位姿。
def scan_pose_to_image_time(
    previous_scan_pose: np.ndarray,
    current_scan_pose: np.ndarray,
    previous_scan_time: float,
    current_scan_time: float,
    image_time: float,
    T_body_lidar: np.ndarray,
) -> np.ndarray:
    if not previous_scan_time < current_scan_time:
        raise ValueError("LiDAR timestamps must increase")
    time_ratio = (image_time - current_scan_time) / (current_scan_time - previous_scan_time)
    scan_motion = np.linalg.inv(previous_scan_pose) @ current_scan_pose
    T_world_lidar_at_image = current_scan_pose @ interpolate_motion(scan_motion, time_ratio)
    return T_world_lidar_at_image @ np.linalg.inv(T_body_lidar)


# 刚体逆变换：R.T、-R.T @ t。
def invert_pose(pose):
    inverse_pose = np.eye(4)
    inverse_pose[:3, :3] = pose[:3, :3].T
    inverse_pose[:3, 3] = -inverse_pose[:3, :3] @ pose[:3, 3]
    return inverse_pose


# 六维优化参数（旋转向量、平移）转齐次矩阵。
def pose_from_vector(parameters):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    pose[:3, 3] = parameters[3:]
    return pose


# 齐次矩阵转六维优化参数。
def pose_to_vector(pose):
    return np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]


# 针孔投影，camera_matrix 是 640x400 的矫正内参。
def project_points(camera_points, camera_matrix):
    depth = np.maximum(camera_points[:, 2], 0.05)
    return (
        camera_points[:, :2] / depth[:, None] * np.array([camera_matrix[0, 0], camera_matrix[1, 1]])
        + camera_matrix[:2, 2]
    )


# Shi-Tomasi 角点，排除边框和极暗/饱和像素。
def detect_corners(gray_image, max_corners=2200):
    mask = np.zeros_like(gray_image)
    mask[8:-8, 8:-8] = 255
    mask[(gray_image >= 250) | (gray_image <= 5)] = 0
    corners = cv2.goodFeaturesToTrack(gray_image, max_corners, 0.01, 5, mask=mask, blockSize=7)
    return corners[:, 0] if corners is not None else np.empty((0, 2), np.float32)


# 金字塔 LK 正反向检查，回跟误差小于 0.8 px。
def track_lk(previous_image, current_image, source_pixels, initial_pixels=None):
    lk_settings = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
        minEigThreshold=1e-4,
    )
    forward_pixels, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_image,
        current_image,
        source_pixels[:, None],
        initial_pixels[:, None].astype(np.float32) if initial_pixels is not None else None,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW if initial_pixels is not None else 0,
        **lk_settings,
    )
    if forward_pixels is None:
        return np.empty((0, 2)), np.empty((0, 2))
    backward_pixels, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_image, previous_image, forward_pixels, None, **lk_settings
    )
    if backward_pixels is None:
        return np.empty((0, 2)), np.empty((0, 2))
    destination_pixels = forward_pixels[:, 0]
    valid_tracks = (
        (forward_status[:, 0] > 0)
        & (backward_status[:, 0] > 0)
        & (np.linalg.norm(backward_pixels[:, 0] - source_pixels, axis=1) < 0.8)
    )
    valid_tracks &= (
        (destination_pixels[:, 0] >= 5)
        & (destination_pixels[:, 0] < current_image.shape[1] - 5)
        & (destination_pixels[:, 1] >= 5)
        & (destination_pixels[:, 1] < current_image.shape[0] - 5)
    )
    valid_tracks &= (
        current_image[
            np.clip(np.rint(destination_pixels[:, 1]).astype(int), 0, current_image.shape[0] - 1),
            np.clip(np.rint(destination_pixels[:, 0]).astype(int), 0, current_image.shape[1] - 1),
        ]
        < 250
    )
    return source_pixels[valid_tracks], destination_pixels[valid_tracks]


# 主平面运动支持足够时才做单应性筛选。
def filter_by_homography(previous_pixels, current_pixels):
    if len(previous_pixels) < 20:
        return previous_pixels, current_pixels, None, 0.0
    homography, inlier_mask = cv2.findHomography(
        previous_pixels, current_pixels, cv2.RANSAC, 1.5, maxIters=3000, confidence=0.999
    )
    inlier_fraction = float(inlier_mask.mean()) if inlier_mask is not None else 0.0
    # Dominant local terrain motion. Keep all candidates for nonplanar scenes.
    if homography is not None and inlier_fraction >= 0.55 and int(inlier_mask.sum()) >= 80:
        inliers = inlier_mask[:, 0] > 0
        return previous_pixels[inliers], current_pixels[inliers], homography, inlier_fraction
    return previous_pixels, current_pixels, homography, inlier_fraction


# SIFT 双向匹配和比率检查，用于大间隔/弱跟踪的恢复。
def match_sift(previous_image, current_image):
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
    previous_keypoints, previous_descriptors = sift.detectAndCompute(previous_image, None)
    current_keypoints, current_descriptors = sift.detectAndCompute(current_image, None)
    if previous_descriptors is None or current_descriptors is None:
        return np.empty((0, 2)), np.empty((0, 2)), None, 0
    matcher = cv2.BFMatcher()
    forward_matches, backward_matches = matcher.knnMatch(
        previous_descriptors, current_descriptors, k=2
    ), matcher.knnMatch(current_descriptors, previous_descriptors, k=2)
    reverse_lookup = {
        best_match.queryIdx: best_match.trainIdx
        for candidate_pair in backward_matches
        if len(candidate_pair) == 2
        for best_match, second_match in [candidate_pair]
        if best_match.distance < 0.75 * second_match.distance
    }
    mutual_matches = [
        best_match
        for candidate_pair in forward_matches
        if len(candidate_pair) == 2
        for best_match, second_match in [candidate_pair]
        if best_match.distance < 0.75 * second_match.distance
        and reverse_lookup.get(best_match.trainIdx) == best_match.queryIdx
    ]
    previous_pixels = np.asarray(
        [previous_keypoints[best_match.queryIdx].pt for best_match in mutual_matches],
        dtype=np.float32,
    ).reshape(-1, 2)
    current_pixels = np.asarray(
        [current_keypoints[best_match.trainIdx].pt for best_match in mutual_matches],
        dtype=np.float32,
    ).reshape(-1, 2)
    if len(previous_pixels) < 12:
        return previous_pixels, current_pixels, None, len(mutual_matches)
    homography, inlier_mask = cv2.findHomography(
        previous_pixels, current_pixels, cv2.RANSAC, 2.0, maxIters=5000, confidence=0.999
    )
    if inlier_mask is not None:
        inliers = inlier_mask[:, 0] > 0
        previous_pixels, current_pixels = previous_pixels[inliers], current_pixels[inliers]
    return previous_pixels, current_pixels, homography, len(mutual_matches)


# 依次尝试 LK、运动引导 LK、SIFT 引导 LK 和 SIFT 匹配。
def track_features(previous_image, current_image, image_dt, predicted_homography=None):
    seed_pixels = detect_corners(previous_image)
    previous_pixels, current_pixels = track_lk(previous_image, current_image, seed_pixels)
    previous_pixels, current_pixels, homography, homography_fraction = filter_by_homography(
        previous_pixels, current_pixels
    )
    method = "lk"
    sift_count = 0
    if predicted_homography is not None and (image_dt > 0.8 or len(previous_pixels) < 300):
        predicted_pixels = cv2.perspectiveTransform(seed_pixels[:, None], predicted_homography)[
            :, 0
        ]
        guided_previous, guided_current = track_lk(
            previous_image, current_image, seed_pixels, predicted_pixels
        )
        guided_previous, guided_current, _, guided_fraction = filter_by_homography(
            guided_previous, guided_current
        )
        if len(guided_previous) >= 100 and guided_fraction >= 0.55:
            previous_pixels, current_pixels, homography_fraction, method = (
                guided_previous,
                guided_current,
                guided_fraction,
                "motion_guided_lk",
            )
    if image_dt > 0.8 or len(previous_pixels) < 200 or homography_fraction < 0.4:
        sift_previous, sift_current, sift_homography, sift_count = match_sift(
            previous_image, current_image
        )
        if sift_homography is not None and len(sift_previous) >= 25:
            predicted_pixels = cv2.perspectiveTransform(seed_pixels[:, None], sift_homography)[:, 0]
            guided_previous, guided_current = track_lk(
                previous_image, current_image, seed_pixels, predicted_pixels
            )
            guided_previous, guided_current, _, guided_fraction = filter_by_homography(
                guided_previous, guided_current
            )
            if guided_fraction >= 0.55 and len(guided_previous) >= max(80, len(previous_pixels)):
                previous_pixels, current_pixels, homography_fraction, method = (
                    guided_previous,
                    guided_current,
                    guided_fraction,
                    "sift_guided_lk",
                )
        if len(sift_previous) >= 30 and (
            len(previous_pixels) < 80 or (image_dt > 0.8 and method == "lk")
        ):
            previous_pixels, current_pixels, method = sift_previous, sift_current, "sift"
    return (
        previous_pixels,
        current_pixels,
        {
            "tracking_method": method,
            "corners": len(seed_pixels),
            "tracks": len(previous_pixels),
            "homography_fraction": homography_fraction,
            "descriptor_mutual_matches": sift_count,
            "median_flow_px": (
                float(np.median(np.linalg.norm(current_pixels - previous_pixels, axis=1)))
                if len(previous_pixels)
                else None
            ),
        },
    )


# 投影原始扫描，用 12 邻点局部平面与特征射线求交，拒绝深度边缘。
def estimate_feature_depth(
    raw_xyz,
    feature_pixels,
    camera_matrix,
    T_body_camera,
    T_body_lidar,
    body_motion,
    image_dt,
    scan_offset,
):
    "投影原始扫描，用 12 邻点局部平面与特征射线求交，拒绝深度边缘。"
    feature_count = len(feature_pixels)
    feature_points = np.full((feature_count, 3), np.nan)
    depth_uncertainty = np.full(feature_count, np.inf)
    if feature_count == 0:
        return feature_points, np.zeros(feature_count, bool), depth_uncertainty
    estimated_scan_motion = interpolate_motion(body_motion, scan_offset / max(image_dt, 1e-6))
    T_camera_lidar = invert_pose(T_body_camera) @ estimated_scan_motion @ T_body_lidar
    camera_points = raw_xyz @ T_camera_lidar[:3, :3].T + T_camera_lidar[:3, 3]
    in_front = (
        np.isfinite(camera_points).all(axis=1)
        & (camera_points[:, 2] > 0.5)
        & (camera_points[:, 2] < 60)
    )
    camera_points = camera_points[in_front]
    projected_pixels = project_points(camera_points, camera_matrix)
    in_image = (
        (projected_pixels[:, 0] >= 0)
        & (projected_pixels[:, 0] < 640)
        & (projected_pixels[:, 1] >= 0)
        & (projected_pixels[:, 1] < 400)
    )
    camera_points, projected_pixels = camera_points[in_image], projected_pixels[in_image]
    if len(camera_points) < 12:
        return feature_points, np.zeros(feature_count, bool), depth_uncertainty
    pixel_distances, neighbor_indices = cKDTree(projected_pixels).query(
        feature_pixels, k=12, workers=1
    )
    neighbors = camera_points[neighbor_indices]
    plane_centers = neighbors.mean(axis=1)
    centered_neighbors = neighbors - plane_centers[:, None]
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.einsum("nki,nkj->nij", centered_neighbors, centered_neighbors) / 12
    )
    plane_normals = eigenvectors[:, :, 0]
    feature_rays = np.column_stack(
        (
            (feature_pixels[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
            (feature_pixels[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
            np.ones(feature_count),
        )
    )
    ray_plane_dot = np.einsum("ni,ni->n", plane_normals, feature_rays)
    feature_depths = np.einsum("ni,ni->n", plane_normals, plane_centers) / np.where(
        abs(ray_plane_dot) > 1e-8, ray_plane_dot, np.nan
    )
    neighbor_median_depth = np.median(neighbors[:, :, 2], axis=1)
    valid_depth = (pixel_distances[:, 0] < 8) & (pixel_distances[:, -1] < 24)
    valid_depth &= (eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12) < 0.025) & (
        eigenvalues[:, 1] > 1e-5
    )
    valid_depth &= np.ptp(neighbors[:, :, 2], axis=1) < 0.35 * neighbor_median_depth
    valid_depth &= (feature_depths > 0.5) & (feature_depths < 45) & (abs(ray_plane_dot) > 0.15)
    valid_depth &= (feature_depths / neighbor_median_depth > 0.7) & (
        feature_depths / neighbor_median_depth < 1.35
    )
    valid_depth &= np.sqrt(np.maximum(eigenvalues[:, 0], 0)) < 0.08
    feature_points[valid_depth] = feature_rays[valid_depth] * feature_depths[valid_depth, None]
    depth_uncertainty[valid_depth] = (
        np.sqrt(np.maximum(eigenvalues[valid_depth, 0], 0)) / abs(ray_plane_dot[valid_depth]) + 0.02
    )
    return feature_points, valid_depth, depth_uncertainty


# EPNP RANSAC + VVS 精修，返回当前相机 <- 前相机的变换。
def solve_pnp(object_points, image_pixels, camera_matrix):
    if len(object_points) < 40:
        return None, np.zeros(len(object_points), bool), None
    success, rotation_vector, translation_vector, inlier_indices = cv2.solvePnPRansac(
        object_points.astype(np.float32),
        image_pixels.astype(np.float32),
        camera_matrix,
        None,
        iterationsCount=500,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inlier_indices is None or len(inlier_indices) < 40:
        return None, np.zeros(len(object_points), bool), None
    inliers = np.zeros(len(object_points), bool)
    inliers[inlier_indices[:, 0]] = True
    rotation_vector, translation_vector = cv2.solvePnPRefineVVS(
        object_points[inliers],
        image_pixels[inliers].astype(float),
        camera_matrix,
        None,
        rotation_vector,
        translation_vector,
    )
    T_current_camera_previous_camera = np.eye(4)
    T_current_camera_previous_camera[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    T_current_camera_previous_camera[:3, 3] = translation_vector[:, 0]
    reprojection_errors = np.linalg.norm(
        project_points(
            object_points @ T_current_camera_previous_camera[:3, :3].T
            + T_current_camera_previous_camera[:3, 3],
            camera_matrix,
        )
        - image_pixels,
        axis=1,
    )
    inliers = reprojection_errors < 2.0
    return T_current_camera_previous_camera, inliers, float(np.median(reprojection_errors[inliers]))


# 刷新异步补偿两轮，再优化前后双向重投影，输出前载体系 <- 当前载体系。
def estimate_visual_motion(
    previous_image,
    current_image,
    previous_raw_points,
    current_raw_points,
    previous_pair,
    pair,
    camera_matrix,
    T_body_camera,
    T_body_lidar,
    predicted_motion,
    prediction_dt,
):
    image_dt = (pair["camera_utc_us"] - previous_pair["camera_utc_us"]) / 1e6
    body_motion = interpolate_motion(predicted_motion, image_dt / max(prediction_dt, 1e-6))
    previous_offset = (previous_pair["lidar_utc_us"] - previous_pair["camera_utc_us"]) / 1e6
    current_offset = (pair["lidar_utc_us"] - pair["camera_utc_us"]) / 1e6
    predicted_homography = None
    if image_dt > 0.8:
        seed_pixels = detect_corners(previous_image)
        seed_points, valid_seeds, _ = estimate_feature_depth(
            previous_raw_points,
            seed_pixels,
            camera_matrix,
            T_body_camera,
            T_body_lidar,
            body_motion,
            image_dt,
            previous_offset,
        )
        if int(valid_seeds.sum()) >= 80:
            T_current_camera_previous_camera = (
                invert_pose(T_body_camera) @ invert_pose(body_motion) @ T_body_camera
            )
            predicted_pixels = project_points(
                seed_points[valid_seeds] @ T_current_camera_previous_camera[:3, :3].T
                + T_current_camera_previous_camera[:3, 3],
                camera_matrix,
            )
            predicted_homography, _ = cv2.findHomography(
                seed_pixels[valid_seeds], predicted_pixels, cv2.RANSAC, 2.0
            )
    previous_pixels, current_pixels, statistics = track_features(
        previous_image, current_image, image_dt, predicted_homography
    )
    statistics["dt_s"] = image_dt
    if len(previous_pixels) < 40:
        return None, {**statistics, "reason": "insufficient_tracks"}, None
    camera_motion = None
    for iteration in range(2):
        previous_points, previous_depth_valid, previous_depth_uncertainty = estimate_feature_depth(
            previous_raw_points,
            previous_pixels,
            camera_matrix,
            T_body_camera,
            T_body_lidar,
            body_motion,
            image_dt,
            previous_offset,
        )
        forward_indices = np.flatnonzero(previous_depth_valid)
        camera_motion, pnp_inliers, pnp_error = solve_pnp(
            previous_points[forward_indices], current_pixels[forward_indices], camera_matrix
        )
        statistics.update(
            depth_features=len(forward_indices),
            pnp_inliers=int(pnp_inliers.sum()),
            reprojection_median_px=pnp_error,
        )
        if camera_motion is None:
            return None, {**statistics, "reason": "pnp_rejected"}, None
        body_motion = T_body_camera @ invert_pose(camera_motion) @ invert_pose(T_body_camera)
    previous_points, previous_depth_valid, previous_depth_uncertainty = estimate_feature_depth(
        previous_raw_points,
        previous_pixels,
        camera_matrix,
        T_body_camera,
        T_body_lidar,
        body_motion,
        image_dt,
        previous_offset,
    )
    current_points, current_depth_valid, current_depth_uncertainty = estimate_feature_depth(
        current_raw_points,
        current_pixels,
        camera_matrix,
        T_body_camera,
        T_body_lidar,
        body_motion,
        image_dt,
        current_offset,
    )
    forward_errors = np.linalg.norm(
        project_points(
            previous_points[previous_depth_valid] @ camera_motion[:3, :3].T + camera_motion[:3, 3],
            camera_matrix,
        )
        - current_pixels[previous_depth_valid],
        axis=1,
    )
    forward_indices = np.flatnonzero(previous_depth_valid)[forward_errors < 2.0]
    backward_indices = forward_indices[current_depth_valid[forward_indices]]
    reverse_motion = invert_pose(camera_motion)
    backward_errors = np.linalg.norm(
        project_points(
            current_points[backward_indices] @ reverse_motion[:3, :3].T + reverse_motion[:3, 3],
            camera_matrix,
        )
        - previous_pixels[backward_indices],
        axis=1,
    )
    backward_indices = backward_indices[backward_errors < 2.5]
    statistics["symmetric_features"] = len(backward_indices)
    if len(forward_indices) < 40:
        return None, {**statistics, "reason": "depth_refresh_rejected"}, None

    def reprojection_residual(parameters):
        trial_motion = pose_from_vector(parameters)
        forward_residual = (
            project_points(
                previous_points[forward_indices] @ trial_motion[:3, :3].T + trial_motion[:3, 3],
                camera_matrix,
            )
            - current_pixels[forward_indices]
        )
        inverse_trial_motion = invert_pose(trial_motion)
        backward_residual = (
            project_points(
                current_points[backward_indices] @ inverse_trial_motion[:3, :3].T
                + inverse_trial_motion[:3, 3],
                camera_matrix,
            )
            - previous_pixels[backward_indices]
        )
        return np.r_[forward_residual.ravel(), backward_residual.ravel()]

    optimization = least_squares(
        reprojection_residual,
        pose_to_vector(camera_motion),
        loss="huber",
        f_scale=0.8,
        max_nfev=20,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    camera_motion = pose_from_vector(optimization.x)
    body_motion = T_body_camera @ invert_pose(camera_motion) @ invert_pose(T_body_camera)
    final_errors = np.linalg.norm(
        project_points(
            previous_points[forward_indices] @ camera_motion[:3, :3].T + camera_motion[:3, 3],
            camera_matrix,
        )
        - current_pixels[forward_indices],
        axis=1,
    )
    support_cells = np.floor(previous_pixels[forward_indices] / np.array([128, 100])).astype(int)
    statistics.update(
        support_cells=len(np.unique(support_cells, axis=0)),
        final_reprojection_median_px=float(np.median(final_errors)),
        relative_translation_m=float(np.linalg.norm(body_motion[:3, 3])),
        relative_rotation_deg=float(
            np.degrees(Rotation.from_matrix(body_motion[:3, :3]).magnitude())
        ),
    )
    # Broad physical sanity bounds, not tuned against GT.
    if statistics["relative_translation_m"] > max(0.5, 2 * image_dt) or statistics[
        "relative_rotation_deg"
    ] > max(10, 45 * image_dt):
        return None, {**statistics, "reason": "physical_bound_rejected"}, None
    if statistics["support_cells"] < 6 or statistics["final_reprojection_median_px"] > 1.5:
        return None, {**statistics, "reason": "insufficient_spatial_support"}, None
    observations = dict(
        pa=previous_points[forward_indices],
        b=current_pixels[forward_indices],
        pb=current_points[backward_indices],
        a=previous_pixels[backward_indices],
    )
    return body_motion, {**statistics, "reason": "observed"}, observations


# 前端地图：XY 半径 2~40 m，默认 0.3 m 体素。
def prepare_visual_map_points(raw_points, voxel_size=0.3):
    radius = np.linalg.norm(raw_points[:, :2], axis=1)
    valid = np.isfinite(raw_points).all(axis=1) & (radius > 2) & (radius < 40)
    return voxel_downsample(raw_points[valid], voxel_size)


# 联合重投影和原始扫描点到面残差；地图仅由此前估计建立。
def refine_visual_pose_with_lidar(
    previous_body_pose,
    body_motion,
    raw_points,
    pair,
    camera_matrix,
    T_body_camera,
    T_body_lidar,
    image_dt,
    observations,
    local_map,
):
    "联合重投影和原始扫描点到面残差；地图仅由此前估计建立。"
    initial_body_pose = previous_body_pose @ body_motion
    scan_offset = (pair["lidar_utc_us"] - pair["camera_utc_us"]) / 1e6
    T_body_scan = interpolate_motion(body_motion, scan_offset / image_dt) @ T_body_lidar
    source_points = prepare_visual_map_points(raw_points)
    map_points = voxel_downsample(np.vstack(local_map), 0.35)
    if len(map_points) < 100 or len(source_points) < 100:
        return initial_body_pose, {"lidar_map_used": 0}
    map_tree = cKDTree(map_points)
    current_body_pose = initial_body_pose.copy()
    previous_camera_pose = previous_body_pose @ T_body_camera
    previous_points, current_pixels, current_points, previous_pixels = [
        observations[key] for key in ("pa", "b", "pb", "a")
    ]
    source_body_points = source_points @ T_body_scan[:3, :3].T + T_body_scan[:3, 3]
    statistics = {"lidar_map_used": 0}
    for outer in range(2):
        world_points = source_body_points @ current_body_pose[:3, :3].T + current_body_pose[:3, 3]
        neighbor_distances, neighbor_indices = map_tree.query(world_points, k=8, workers=1)
        neighbors = map_points[neighbor_indices]
        plane_centers = neighbors.mean(axis=1)
        centered_neighbors = neighbors - plane_centers[:, None]
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.einsum("nki,nkj->nij", centered_neighbors, centered_neighbors) / 8
        )
        valid_planes = (neighbor_distances[:, 0] < 0.7) & (neighbor_distances[:, -1] < 1.2)
        valid_planes &= (eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12) < 0.08) & (
            eigenvalues[:, 1] > 0.001
        )
        plane_indices = np.flatnonzero(valid_planes)
        if len(plane_indices) < 150:
            break
        if len(plane_indices) > 1200:
            plane_indices = plane_indices[np.linspace(0, len(plane_indices) - 1, 1200, dtype=int)]
        plane_normals, corresponding_centers, corresponding_points = (
            eigenvectors[plane_indices, :, 0],
            plane_centers[plane_indices],
            source_body_points[plane_indices],
        )
        plane_sigma = np.sqrt(np.maximum(eigenvalues[plane_indices, 0], 0)) + 0.035
        linearization_pose = current_body_pose.copy()

        def pose_after_update(update):
            trial_pose = linearization_pose.copy()
            trial_pose[:3, :3] = (
                Rotation.from_rotvec(update[:3]).as_matrix() @ linearization_pose[:3, :3]
            )
            trial_pose[:3, 3] = linearization_pose[:3, 3] + update[3:]
            return trial_pose

        def joint_residual(update):
            trial_pose = pose_after_update(update)
            T_current_camera_previous_camera = (
                invert_pose(trial_pose @ T_body_camera) @ previous_camera_pose
            )
            forward_residual = (
                project_points(
                    previous_points @ T_current_camera_previous_camera[:3, :3].T
                    + T_current_camera_previous_camera[:3, 3],
                    camera_matrix,
                )
                - current_pixels
            ) / 0.8
            inverse_camera_motion = invert_pose(T_current_camera_previous_camera)
            backward_residual = (
                project_points(
                    current_points @ inverse_camera_motion[:3, :3].T + inverse_camera_motion[:3, 3],
                    camera_matrix,
                )
                - previous_pixels
            ) / 0.8
            transformed_points = corresponding_points @ trial_pose[:3, :3].T + trial_pose[:3, 3]
            plane_residual = (
                np.einsum("ni,ni->n", plane_normals, transformed_points - corresponding_centers)
                / plane_sigma
            )
            return np.r_[forward_residual.ravel(), backward_residual.ravel(), plane_residual]

        optimization = least_squares(
            joint_residual,
            np.zeros(6),
            loss="huber",
            f_scale=1.0,
            max_nfev=15,
            ftol=1e-4,
            xtol=1e-5,
            gtol=1e-4,
        )
        candidate_pose = pose_after_update(optimization.x)
        T_current_camera_previous_camera = (
            invert_pose(candidate_pose @ T_body_camera) @ previous_camera_pose
        )
        reprojection_errors = np.linalg.norm(
            project_points(
                previous_points @ T_current_camera_previous_camera[:3, :3].T
                + T_current_camera_previous_camera[:3, 3],
                camera_matrix,
            )
            - current_pixels,
            axis=1,
        )
        correction = invert_pose(initial_body_pose) @ candidate_pose
        rotation_correction = np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude())
        if (
            np.median(reprojection_errors) > 1.5
            or np.linalg.norm(correction[:3, 3]) > 0.35
            or rotation_correction > 4
        ):
            break
        current_body_pose = candidate_pose
        transformed_points = (
            corresponding_points @ current_body_pose[:3, :3].T + current_body_pose[:3, 3]
        )
        plane_errors = np.einsum(
            "ni,ni->n", plane_normals, transformed_points - corresponding_centers
        )
        statistics = dict(
            lidar_map_used=1,
            lidar_plane_pairs=len(plane_indices),
            lidar_plane_rmse_m=float(np.sqrt(np.mean(plane_errors**2))),
            joint_visual_median_px=float(np.median(reprojection_errors)),
            map_translation_correction_m=float(np.linalg.norm(correction[:3, 3])),
            map_rotation_correction_deg=float(rotation_correction),
        )
    return current_body_pose, statistics


def lidar_offset_in_world(T_world_lidar, T_body_lidar):
    """Vector from body origin to LiDAR origin, expressed in the world frame."""
    R_world_body = T_world_lidar[:3, :3] @ T_body_lidar[:3, :3].T
    return R_world_body @ T_body_lidar[:3, 3]


def visual_translation_residual(scan_pose, previous_body_pose, expected_body_pose, T_body_lidar):
    """Body XY residual and Jacobian for a left rotation / additive world translation step."""
    world_lidar_offset = lidar_offset_in_world(scan_pose, T_body_lidar)
    estimated_body_position = scan_pose[:3, 3] - world_lidar_offset
    previous_body_xy_axes = previous_body_pose[:3, :2].T
    residual = previous_body_xy_axes @ (estimated_body_position - expected_body_pose[:3, 3])
    jacobian = previous_body_xy_axes @ np.column_stack(
        (cross_product_matrix(world_lidar_offset), np.eye(3))
    )
    return residual, jacobian


# IRLS 点到面优化 + 前帧载体系 XY 位移软约束；不加视觉旋转残差。
def register_scan_to_map(
    source_points,
    map_points,
    initial_scan_pose,
    previous_body_pose,
    expected_body_pose,
    T_body_lidar,
    visual_sigma,
):
    map_tree = cKDTree(map_points)
    scan_pose = initial_scan_pose.copy()
    statistics = dict(success=False, pairs=0, rmse_m=None)
    for iteration in range(20):
        rotated_points = source_points @ scan_pose[:3, :3].T
        world_points = rotated_points + scan_pose[:3, 3]
        neighbor_distances, neighbor_indices = map_tree.query(world_points, k=8, workers=1)
        nearby = neighbor_distances[:, 0] < 1.0
        if int(nearby.sum()) < 500:
            return initial_scan_pose, statistics
        neighbors = map_points[neighbor_indices[nearby]]
        plane_centers = neighbors.mean(axis=1)
        centered_neighbors = neighbors - plane_centers[:, None]
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.einsum("nki,nkj->nij", centered_neighbors, centered_neighbors) / 8
        )
        planar_neighbors = (
            eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12) < 0.08
        ) & (eigenvalues[:, 1] > 0.003)
        plane_normals = eigenvectors[planar_neighbors, :, 0]
        corresponding_centers, corresponding_points, rotated_correspondences = (
            plane_centers[planar_neighbors],
            world_points[nearby][planar_neighbors],
            rotated_points[nearby][planar_neighbors],
        )
        plane_errors = np.einsum(
            "ni,ni->n", plane_normals, corresponding_points - corresponding_centers
        )
        inliers = abs(plane_errors) < 1.0
        plane_normals, plane_errors, rotated_correspondences = (
            plane_normals[inliers],
            plane_errors[inliers],
            rotated_correspondences[inliers],
        )
        if len(plane_errors) < 400:
            return initial_scan_pose, statistics
        plane_jacobian = np.column_stack(
            (np.cross(rotated_correspondences, plane_normals), plane_normals)
        )
        huber_weights = np.minimum(1.0, 0.05 / np.maximum(abs(plane_errors), 1e-12))
        weighted_jacobian = plane_jacobian * np.sqrt(huber_weights[:, None])
        weighted_rhs = -plane_errors * np.sqrt(huber_weights)
        if visual_sigma is not None:
            visual_residual, visual_jacobian = visual_translation_residual(
                scan_pose, previous_body_pose, expected_body_pose, T_body_lidar
            )
            visual_weight = 0.04 / visual_sigma
            weighted_jacobian = np.vstack((weighted_jacobian, visual_jacobian * visual_weight))
            weighted_rhs = np.r_[weighted_rhs, -visual_residual * visual_weight]
        normal_matrix = weighted_jacobian.T @ weighted_jacobian
        damping = max(float(np.trace(normal_matrix) / 6), 1.0) * 1e-6
        pose_update = np.linalg.solve(
            normal_matrix + np.eye(6) * damping, weighted_jacobian.T @ weighted_rhs
        )
        if (
            not np.isfinite(pose_update).all()
            or np.linalg.norm(pose_update[:3]) > np.deg2rad(5)
            or np.linalg.norm(pose_update[3:]) > 1
        ):
            return initial_scan_pose, statistics
        scan_pose[:3, :3] = Rotation.from_rotvec(pose_update[:3]).as_matrix() @ scan_pose[:3, :3]
        scan_pose[:3, :3] = Rotation.from_matrix(scan_pose[:3, :3]).as_matrix()
        scan_pose[:3, 3] += pose_update[3:]
        statistics = dict(
            success=True,
            pairs=len(plane_errors),
            rmse_m=float(np.sqrt(np.mean(plane_errors * plane_errors))),
            inlier_ratio=float(len(plane_errors) / len(source_points)),
            iterations=iteration + 1,
        )
        if np.linalg.norm(pose_update[:3]) < 1e-4 and np.linalg.norm(pose_update[3:]) < 1e-3:
            break
    return scan_pose, statistics


def main():
    parser = argparse.ArgumentParser(description="指定数据集目录，使用矫正左目和本地原始点云估计 v3 位姿")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="包含 images/、pcd/、meta/ 的数据集目录")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ"), help="新输出标签；默认 UTC 时间")
    parser.add_argument("--end", type=int, help="短前缀排他上界；省略则完整序列")
    parser.add_argument("--check-only", action="store_true", help="只检查输入并读取首末帧，不估计、不写出轨迹")
    args = parser.parse_args()
    dataset_directory = args.dataset_dir.expanduser().resolve()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.run_name or any(character not in allowed for character in args.run_name):
        parser.error("--run-name 只能包含字母、数字、下划线和短横线")
    if args.check_only:
        data = load_sequence(dataset_directory, args.end)
        for frame in (0, len(data.pairs) - 1):
            image_path = data.image_directory / f"{frame:05d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2] != (1200, 1920):
                raise ValueError(f"图像必须为 1920×1200：{image_path}")
            points = data.read_scan(frame)
            if len(points) == 0 or not np.isfinite(points).all(axis=1).any():
                raise ValueError(f"点云缺少有效 XYZ：第 {frame} 帧")
        print(f"输入检查通过：{dataset_directory}，{len(data.pairs)} 帧；未进行位姿估计或写出轨迹。")
        return
    output_directory = dataset_directory / "output" / ("pose_estimation_" + args.run_name)
    if output_directory.exists():
        raise FileExistsError(f"不覆盖已有轨迹，请换 --run-name：{output_directory}")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(71)
    estimate_sequence(dataset_directory, output_directory, args.end)


if __name__ == "__main__":
    main()
