"""Convert an explicit tracking TUM trajectory to LocalRF camera poses, and back.

Frames: T_world_body is the tracking output (image time, metres, xyzw quaternion).
G is the LocalRF camera frame of original output frame 0, so G is the identity
for that frame and the poses stay metric. Convention: p_A = T_A_B @ p_B.
"""
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# OpenCV camera axes (x right, y down, z forward) -> LocalRF ray axes (y up, z back).
CV_TO_RF = np.diag([1.0, -1.0, -1.0, 1.0])
MATCH_TOLERANCE_US = 1


def _check_transform(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix


def read_tum(path):
    """Return [(timestamp_us as Decimal, 4x4 T_world_body)], strictly increasing."""
    poses = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"{path}:{line_no}: expected timestamp tx ty tz qx qy qz qw")
        try:
            # Decimal keeps the TUM timestamp exact; float subtraction of large UTC values would not.
            stamp_us = Decimal(parts[0]) * 1000000
            values = np.array([float(v) for v in parts[1:]], dtype=np.float64)
        except (ValueError, InvalidOperation) as exc:
            raise ValueError(f"{path}:{line_no}: invalid number") from exc
        if not stamp_us.is_finite() or not np.isfinite(values).all():
            raise ValueError(f"{path}:{line_no}: non-finite value")
        if abs(np.linalg.norm(values[3:]) - 1) > 1e-4:
            raise ValueError(f"{path}:{line_no}: quaternion must be unit length (xyzw)")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
        pose[:3, 3] = values[:3]
        if poses and stamp_us <= poses[-1][0]:
            raise ValueError(f"{path}:{line_no}: timestamps must be strictly increasing")
        poses.append((stamp_us, pose))
    if not poses:
        raise ValueError(f"{path}: trajectory is empty")
    return poses


def match_frames(frame_times_us, trajectory):
    """Map each pairs.csv camera time to exactly one TUM pose within 1 microsecond."""
    wanted = set(frame_times_us)
    matched = {}
    for stamp, pose in trajectory:
        rounded = int(stamp.to_integral_value(rounding=ROUND_HALF_EVEN))
        candidates = [t for t in (rounded - 1, rounded, rounded + 1)
                      if t in wanted and abs(stamp - t) <= MATCH_TOLERANCE_US]
        if len(candidates) != 1 or candidates[0] in matched:
            raise ValueError(f"TUM timestamp {stamp} us does not match exactly one frame")
        matched[candidates[0]] = pose
    missing = sorted(wanted - set(matched))
    if missing:
        raise ValueError(f"TUM trajectory lacks {len(missing)} frame timestamps, first {missing[:3]}")
    return matched


def load_tracking(dataset_dir, trajectory_path):
    """Return LocalRF-style transforms (fl_x, w, h, cx, cy, frames) and the extrinsics.

    frames[i]["transform_matrix"] is the G-frame LocalRF camera-to-world pose of
    original output frame i, in metres. cx and cy use the ray-centre convention
    (OpenCV pixel centre + 0.5), so the principal point is the LocalRF centre.
    """
    root = Path(dataset_dir)
    calibration = yaml.safe_load((root / "meta/calib.yaml").read_text(encoding="utf-8"))
    camera = calibration["left_camera"]
    K = np.asarray(camera["K"], dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0):
        raise ValueError("left_camera.K must be a finite zero-skew pinhole matrix")
    if not np.allclose(np.asarray(camera["D"], dtype=np.float64), 0, atol=1e-10):
        raise ValueError("LocalRF tracking input requires rectified images (zero distortion)")
    if not np.isclose(K[0, 0], K[1, 1], rtol=1e-6, atol=0):
        raise ValueError("LocalRF uses one focal length; fx and fy must match")
    T_body_cam = _check_transform(calibration["camera_to_body"]["T_body_cam"], "T_body_cam")
    origin_us = int(calibration["time"]["origin_utc_us"])

    with (root / "meta/pairs.csv").open(encoding="utf-8", newline="") as stream:
        rows = [(int(row["output_index"]), int(row["camera_utc_us"]) - origin_us)
                for row in csv.DictReader(stream)]
    if not rows or [index for index, _ in rows] != list(range(len(rows))):
        raise ValueError("pairs.csv output_index must be contiguous from zero")
    times = [time_us for _, time_us in rows]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("pairs.csv camera times must be strictly increasing")

    trajectory = match_frames(times, read_tum(trajectory_path))
    first = trajectory[times[0]]
    if not np.allclose(first, np.eye(4), atol=1e-6):
        raise ValueError("original frame 0 must be the identity body pose (tracking convention)")
    T_world_G = first @ T_body_cam @ CV_TO_RF
    T_G_world = np.linalg.inv(T_world_G)

    frames = []
    for index, time_us in enumerate(times):
        image = root / "images" / f"{index:05d}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        T_world_cam = trajectory[time_us] @ T_body_cam @ CV_TO_RF
        frames.append({"file_path": f"images/{index:05d}.png",
                       "transform_matrix": (T_G_world @ T_world_cam).tolist(),
                       "timestamp_us": time_us})
    transforms = {"fl_x": float(K[0, 0]), "w": int(camera["width"]), "h": int(camera["height"]),
                  "principal_point_px": [float(K[0, 2] + 0.5), float(K[1, 2] + 0.5)], "frames": frames}
    return {"transforms": transforms, "T_world_G": T_world_G, "T_body_cam": T_body_cam}


def camera_to_body(cam2world_G, T_world_G, T_body_cam):
    """Inverse of load_tracking: [N,4,4] G-frame LocalRF poses in metres -> body poses."""
    return T_world_G @ np.asarray(cam2world_G) @ CV_TO_RF @ np.linalg.inv(T_body_cam)


def write_tum(path, timestamps_us, body_poses):
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for stamp_us, pose in zip(timestamps_us, body_poses):
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        values = [stamp_us / 1e6, *pose[:3, 3], *quat]
        lines.append(" ".join(f"{value:.9f}" for value in values))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
