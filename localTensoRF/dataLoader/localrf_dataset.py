# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import random

import numpy as np
import torch
import cv2

from joblib import delayed, Parallel
from torch.utils.data import Dataset
from pathlib import Path
from dataLoader.tracking_poses import (load_tracking, load_calibration, calibration_intrinsics,
                                       camera_to_body, write_tum)
import json

def concatenate_append(old, new, dim):
    new = np.concatenate(new, 0).reshape(-1, dim)
    if old is not None:
        new = np.concatenate([old, new], 0)

    return new

class LocalRFDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        frames_chunk=20,
        downsampling=-1,
        load_depth=False,
        with_preprocessed_poses=False,
        n_init_frames=7,
        subsequence=[0, -1],
        test_frame_every=10,
        frame_step=1,
        trajectory=None,
        depth_scale=None,
        uniform_loss_weights=False,
    ):
        self.root_dir = datadir
        self.split = split
        self.frames_chunk = max(frames_chunk, n_init_frames)
        self.downsampling = downsampling
        self.load_depth = load_depth
        self.depth_scale = depth_scale
        self.uniform_loss_weights = uniform_loss_weights
        self.frame_step = frame_step

        if frame_step < 1 or (downsampling != -1 and downsampling <= 0):
            raise ValueError("frame_step and downsampling must be positive (or downsampling=-1)")
        if n_init_frames < 1 or frames_chunk < 1:
            raise ValueError("frame counts must be positive")
        if depth_scale is not None and (not np.isfinite(depth_scale) or depth_scale <= 0):
            raise ValueError("depth_scale must be a finite positive number of metres per PNG unit")
        self.tracking = load_tracking(datadir, trajectory) if trajectory else None
        self.camera_prior = None
        self.pose_scale = 1.0
        root = Path(datadir)
        transforms_path = root / "transforms.json"
        transforms = None
        if self.tracking is not None:
            frames = self.tracking["frames"]
            camera = self.tracking["intrinsics"]
        else:
            if transforms_path.is_file():
                transforms = json.loads(transforms_path.read_text())
            if with_preprocessed_poses and transforms is None:
                raise FileNotFoundError(transforms_path)
            frames = transforms["frames"] if with_preprocessed_poses else None
            if (root / "meta/calib.yaml").is_file():
                camera = calibration_intrinsics(load_calibration(root))
            elif transforms is not None:
                # Legacy LocalRF metadata: fl_y may be omitted for a square-pixel camera.
                w, h = transforms["w"], transforms["h"]
                cx, cy = transforms.get("principal_point_px", [
                    transforms.get("cx", w / 2 - 0.5) + 0.5,
                    transforms.get("cy", h / 2 - 0.5) + 0.5])
                camera = {"fx": transforms["fl_x"], "fy": transforms.get("fl_y", transforms["fl_x"]),
                          "cx": cx, "cy": cy, "w": w, "h": h}
            else:
                raise FileNotFoundError("camera intrinsics require meta/calib.yaml or transforms.json")
        if not np.isfinite(list(camera.values())).all() or min(camera[k] for k in ("fx", "fy", "w", "h")) <= 0:
            raise ValueError("camera intrinsics and image dimensions must be finite, with positive focal lengths")
        if self.load_depth and self.tracking is None:
            raise ValueError("metric dense depth requires an explicit tracking trajectory; normalised poses are unsupported")
        self.source_wh = (int(camera["w"]), int(camera["h"]))
        self.image_paths = (sorted(Path(f["file_path"]).name for f in frames) if frames is not None else
                            sorted(f.name for f in (root / "images").iterdir()
                                   if f.suffix.lower() in (".png", ".jpg", ".jpeg") and f.is_file()))
        start, end = subsequence
        if start < 0 or (end != -1 and end <= start):
            raise ValueError("subsequence must be [start, end) with 0 <= start < end (or end=-1)")
        self.image_paths = self.image_paths[start:None if end == -1 else end:frame_step]
        if not self.image_paths:
            raise ValueError("no selected images")

        if frames is not None:
            if len(self.image_paths) < 2:
                raise ValueError("pose priors need at least two selected frames")
            by_name = {Path(f["file_path"]).name: f for f in frames}
            poses = np.array([by_name[name]["transform_matrix"] for name in self.image_paths], dtype=np.float64)
            rel = np.stack([np.eye(4)] + [np.linalg.inv(poses[i - 1]) @ poses[i] for i in range(1, len(poses))])
            if self.tracking is not None:
                rel[0] = poses[0]  # Keep the original frame-0 gauge for nonzero-start subsequences.
                self.timestamps_us = [by_name[name]["timestamp_us"] for name in self.image_paths]
            else:
                median_step = np.median(np.linalg.norm(rel[:, :3, 3], axis=-1))
                if median_step <= 0:
                    raise ValueError("cannot normalise a transforms.json trajectory with zero median motion")
                self.pose_scale = 2e-2 / median_step
                poses[:, :3, 3] *= self.pose_scale
                rel[:, :3, 3] *= self.pose_scale
            self.abs_poses = poses.astype(np.float32)
            self.rel_poses = rel.astype(np.float32)
            self.camera_prior = ({"abs_poses": self.abs_poses} if self.tracking is not None else
                                 {"rel_poses": torch.from_numpy(self.rel_poses)})

        self.test_mask = []
        self.test_paths = []
        for idx, image_path in enumerate(self.image_paths):
            fbase = os.path.splitext(image_path)[0]
            index = int(fbase) if fbase.isnumeric() else idx
            # The first selected frame anchors the world, so it is never held out.
            if test_frame_every > 0 and idx > 0 and index % test_frame_every == 0:
                self.test_paths.append(image_path)
                self.test_mask.append(1)
            else:
                self.test_mask.append(0)
        self.test_mask = np.array(self.test_mask)

        if split=="test":
            self.image_paths = self.test_paths
            self.frames_chunk = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self.all_fbases = {os.path.splitext(image_path)[0]: idx for idx, image_path in enumerate(self.image_paths)}

        self.near_far = [0.1, 1e3] # Dummi
        self.scene_bbox = 2 * torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        self.all_rgbs = None
        self.all_depths = None
        self.all_depth_valid = None
        self.all_loss_weights = None

        self.active_frames_bounds = [0, 0]
        self.loaded_frames = 0
        if self.num_images == 0:
            raise ValueError("no images in the requested split")
        if split == "test" and frames is not None:
            # Pose arrays below are selected using the original train/test mask, not test-local indices.
            selected = np.flatnonzero(self.test_mask)
            self.abs_poses = self.abs_poses[selected]
            self.rel_poses = self.rel_poses[selected]
            self.camera_prior = {"abs_poses": self.abs_poses}
            if self.tracking is not None:
                self.timestamps_us = [self.timestamps_us[i] for i in selected]
        self.activate_frames(n_init_frames)
        sx, sy = self.img_wh[0] / camera["w"], self.img_wh[1] / camera["h"]
        self.intrinsics = {"fx": camera["fx"] * sx, "fy": camera["fy"] * sy,
                           "cx": camera["cx"] * sx, "cy": camera["cy"] * sy}

    def export_trajectory(self, path, cam2world):
        """Export the selected, optimised camera poses as metric body poses."""
        if self.tracking is None:
            raise ValueError("trajectory export requires tracking metadata")
        cam2world = np.asarray(cam2world, dtype=np.float64)
        if cam2world.shape != (len(self.image_paths), 3, 4):
            raise ValueError("export must contain exactly one 3x4 pose per selected image")
        bottom = np.tile([[[0., 0., 0., 1.]]], (len(cam2world), 1, 1))
        poses = np.concatenate([cam2world, bottom], axis=1)
        body = camera_to_body(poses, self.tracking["T_world_G"], self.tracking["T_body_cam"])
        write_tum(path, self.timestamps_us, body)


    def activate_frames(self, n_frames=1):
        self.active_frames_bounds[1] += n_frames
        self.active_frames_bounds[1] = min(
            self.active_frames_bounds[1], self.num_images
        )

        if self.active_frames_bounds[1] > self.loaded_frames:
            self.read_meta()



    def has_left_frames(self):
        return self.active_frames_bounds[1] < self.num_images

    def deactivate_frames(self, first_frame):
        n_frames = first_frame - self.active_frames_bounds[0]
        self.active_frames_bounds[0] = first_frame

        self.all_rgbs = self.all_rgbs[n_frames * self.n_px_per_frame:] 
        if self.load_depth:
            self.all_depths = self.all_depths[n_frames * self.n_px_per_frame:]
            self.all_depth_valid = self.all_depth_valid[n_frames * self.n_px_per_frame:]
        self.all_loss_weights = self.all_loss_weights[n_frames * self.n_px_per_frame:]



    def read_meta(self):
        def read_image(i):
            image_path = os.path.join(self.root_dir, "images", self.image_paths[i])
            motion_mask_path = os.path.join(self.root_dir, "masks", 
                f"{os.path.splitext(self.image_paths[i])[0]}.png")
            if not os.path.isfile(motion_mask_path):
                motion_mask_path = os.path.join(self.root_dir, "masks/all.png")


            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(image_path)
            if img.shape[1::-1] != self.source_wh:
                raise ValueError(f"{image_path}: image dimensions do not match calibration {self.source_wh}")
            img = img[..., ::-1].astype(np.float32) / 255
            if self.downsampling != -1:
                img = cv2.resize(img, None, fx=1 / self.downsampling, fy=1 / self.downsampling,
                                 interpolation=cv2.INTER_AREA)

            depth, depth_valid = None, None
            if self.load_depth:
                base = Path(self.root_dir) / "depth" / Path(self.image_paths[i]).stem
                candidates = [base.with_suffix(ext) for ext in (".png", ".npy") if base.with_suffix(ext).is_file()]
                if len(candidates) != 1:
                    raise ValueError(f"{base}: expected exactly one depth .png or .npy")
                path = candidates[0]
                if path.suffix == ".png":
                    if self.depth_scale is None:
                        raise ValueError("PNG depth requires explicit --depth_scale in metres per integer unit")
                    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
                        raise ValueError(f"{path}: depth must be a single-channel uint16 PNG")
                    depth = raw.astype(np.float32) * self.depth_scale
                else:
                    depth = np.load(path, allow_pickle=False)
                    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
                        raise ValueError(f"{path}: depth must be a 2D floating-point array in metres")
                    depth = depth.astype(np.float32)
                # Nearest preserves invalid pixels and avoids blending foreground/background depths.
                depth = cv2.resize(depth, tuple(img.shape[1::-1]), interpolation=cv2.INTER_NEAREST)
                depth_valid = np.isfinite(depth) & (depth > 0)
                depth = np.where(depth_valid, depth, 0).astype(np.float32)

            if os.path.isfile(motion_mask_path):
                mask = cv2.imread(motion_mask_path, cv2.IMREAD_UNCHANGED)
                if len(mask.shape) != 2:
                    mask = mask[..., 0]
                mask = cv2.resize(mask, tuple(img.shape[1::-1]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = None

            if depth_valid is not None and mask is not None:
                depth_valid &= mask
            return {"img": img, "depth": depth, "depth_valid": depth_valid, "mask": mask}

        n_frames_to_load = min(self.frames_chunk, self.num_images - self.loaded_frames)
        all_data = Parallel(n_jobs=-1, backend="threading")(
            delayed(read_image)(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load) 
        )
        self.loaded_frames += n_frames_to_load
        all_rgbs = [data["img"] for data in all_data]
        all_depths = [data["depth"] for data in all_data]
        all_depth_valid = [data["depth_valid"] for data in all_data]
        all_mask = [data["mask"] for data in all_data]

        all_laplacian = [
                np.ones_like(img[..., 0]) * (1.0 if self.uniform_loss_weights else max(1e-6, cv2.Laplacian(
                            cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F
                        ).var()))
            for img in all_rgbs
        ]
        all_loss_weights = [laplacian if mask is None else laplacian * mask for laplacian, mask in zip(all_laplacian, all_mask)]

        self.img_wh = list(all_rgbs[0].shape[1::-1])
        self.n_px_per_frame = self.img_wh[0] * self.img_wh[1]

        if self.split != "train":
            self.all_rgbs = np.stack(all_rgbs, 0)
            if self.load_depth:
                self.all_depths = np.stack(all_depths, 0)
                self.all_depth_valid = np.stack(all_depth_valid, 0)
        else:
            self.all_rgbs = concatenate_append(self.all_rgbs, all_rgbs, 3)
            if self.load_depth:
                self.all_depths = concatenate_append(self.all_depths, all_depths, 1)
                self.all_depth_valid = concatenate_append(self.all_depth_valid, all_depth_valid, 1)
            self.all_loss_weights = concatenate_append(self.all_loss_weights, all_loss_weights, 1)


    def __len__(self):
        return int(1e10)

    def __getitem__(self, i):
        raise NotImplementedError

    def get_frame_fbase(self, view_id):
        return list(self.all_fbases.keys())[view_id]

    def sample(self, batch_size, is_refining, optimize_poses, n_views=16, allow_test_pose_opt=False):
        if batch_size < 1 or batch_size % n_views:
            raise ValueError("batch_size must be a positive multiple of n_views")
        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        test_ratio = active_test_mask.mean()
        # Pose-only steps on test views use the held-out images; off for held-out runs.
        if optimize_poses and allow_test_pose_opt:
            train_test_poses = test_ratio > random.uniform(0, 1)
        else:
            train_test_poses = False

        inclusion_mask = active_test_mask if train_test_poses else 1 - active_test_mask
        sample_map = np.arange(
            self.active_frames_bounds[0], 
            self.active_frames_bounds[1], 
            dtype=np.int64)[inclusion_mask == 1]
        
        if inclusion_mask.sum() == 0:
            raise ValueError("active window contains no eligible training frames")
        raw_samples = np.random.randint(0, inclusion_mask.sum(), n_views, dtype=np.int64)

        # Force having the last views during coarse optimization
        if not is_refining and inclusion_mask.sum() > 4:
            raw_samples[:2] = inclusion_mask.sum() - 1
            raw_samples[2:4] = inclusion_mask.sum() - 2
            raw_samples[4] = inclusion_mask.sum() - 3
            raw_samples[5] = inclusion_mask.sum() - 4

        view_ids = sample_map[raw_samples]

        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64)
        idx = idx.reshape(n_views, -1)
        idx = idx + view_ids[..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx - self.active_frames_bounds[0] * self.n_px_per_frame

        return {
            "rgbs": self.all_rgbs[idx_sample], 
            "loss_weights": self.all_loss_weights[idx_sample], 
            "depths": self.all_depths[idx_sample] if self.load_depth else None,
            "depth_valid": self.all_depth_valid[idx_sample] if self.load_depth else None,
            "idx": idx,
            "view_ids": view_ids,
            "train_test_poses": train_test_poses,
        }