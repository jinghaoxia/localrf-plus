"""CPU checks for tracking-to-LocalRF pose conversion, units and dataset alignment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "localTensoRF"))
from dataLoader.localrf_dataset import LocalRFDataset  # noqa: E402
from dataLoader.tracking_poses import camera_to_body, load_tracking  # noqa: E402
from local_tensorfs import LocalTensorfs  # noqa: E402

ORIGIN_US = 1544189387246410
N = 6
TINY_FIELD = {"gridSize": [8, 8, 8], "density_n_comp": [2, 2, 2], "appearance_n_comp": [2, 2, 2],
              "app_dim": 3, "shadingMode": "MLP_Fea_late_view", "near_far": [0.1, 100.],
              "density_shift": -10., "distance_scale": 25., "rayMarch_weight_thres": 1e-4,
              "pos_pe": 0, "view_pe": 0, "fea_pe": 0, "featureC": 8, "step_ratio": 0.5,
              "fea2denseAct": "softplus", "alphaMask_thres": 1e-4}


def build_field(abs_poses, transforms, scene_radius=1.0, n_init_frames=2):
    prior = {"rel_poses": torch.eye(4).repeat(len(abs_poses), 1, 1), "transforms": transforms,
             "abs_poses": abs_poses}
    return LocalTensorfs(
        camera_prior=prior, fov=85.6, n_init_frames=n_init_frames, n_overlap=1, WH=(8, 6),
        n_iters_per_frame=2, n_iters_reg=1, lr_R_init=1e-3, lr_t_init=1e-3, lr_i_init=0,
        lr_exposure_init=0, rf_lr_init=1e-2, rf_lr_basis=1e-3, lr_decay_target_ratio=0.1,
        N_voxel_list={}, update_AlphaMask_list=[], lr_upsample_reset=1, device="cpu",
        scene_radius=scene_radius, aabb=torch.tensor([[-2.] * 3, [2.] * 3]), **TINY_FIELD)


def random_pose(rng):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    pose[:3, 3] = rng.normal(size=3)
    return pose


class TrackingFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.body = [np.eye(4)] + [random_pose(rng) for _ in range(N - 1)]
        self.T_body_cam = random_pose(rng)
        (self.root / "meta").mkdir()
        (self.root / "images").mkdir()
        calibration = {
            "left_camera": {"width": 8, "height": 6, "K": [[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]],
                            "D": [0, 0, 0, 0]},
            "camera_to_body": {"T_body_cam": self.T_body_cam.tolist()},
            "time": {"origin_utc_us": ORIGIN_US},
        }
        (self.root / "meta/calib.yaml").write_text(yaml.safe_dump(calibration))
        (self.root / "meta/pairs.csv").write_text(
            "output_index,camera_utc_us\n" + "".join(f"{i},{ORIGIN_US + 100000 * i}\n" for i in range(N)))
        for i in range(N):
            cv2.imwrite(str(self.root / "images" / f"{i:05d}.png"), np.full((6, 8, 3), 20 * i, np.uint8))
        self.write_trajectory(self.body)

    def tearDown(self):
        self.tmp.cleanup()

    def write_trajectory(self, poses, times=range(N)):
        lines = []
        for i, pose in zip(times, poses):
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            lines.append(f"{i / 10:.9f} " + " ".join(f"{v:.12f}" for v in [*pose[:3, 3], *quat]))
        self.trajectory = self.root / "tracking.tum"
        self.trajectory.write_text("\n".join(lines) + "\n")


class ConversionTests(TrackingFixture):
    def test_round_trip_recovers_body_poses(self):
        tracking = load_tracking(self.root, self.trajectory)
        cameras = np.stack([np.array(f["transform_matrix"]) for f in tracking["transforms"]["frames"]])
        restored = camera_to_body(cameras, tracking["T_world_G"], tracking["T_body_cam"])
        np.testing.assert_allclose(restored, np.stack(self.body), atol=1e-6)

    def test_first_frame_is_the_camera_frame_G(self):
        tracking = load_tracking(self.root, self.trajectory)
        np.testing.assert_allclose(tracking["transforms"]["frames"][0]["transform_matrix"], np.eye(4), atol=1e-9)

    def test_principal_point_uses_continuous_pixel_centres(self):
        transforms = load_tracking(self.root, self.trajectory)["transforms"]
        self.assertEqual(transforms["principal_point_px"], [4.5, 3.5])
        self.assertEqual(transforms["fl_x"], 8.0)

    def test_rejects_non_identity_first_pose(self):
        self.write_trajectory([self.body[1]] + self.body[1:])
        with self.assertRaisesRegex(ValueError, "identity"):
            load_tracking(self.root, self.trajectory)

    def test_rejects_frame_without_tracking_pose(self):
        self.write_trajectory(self.body[:-1], times=range(N - 1))
        with self.assertRaisesRegex(ValueError, "lacks"):
            load_tracking(self.root, self.trajectory)


class DatasetAlignmentTests(TrackingFixture):
    def test_subsequence_and_step_keep_metric_poses_aligned_with_images(self):
        transforms = load_tracking(self.root, self.trajectory)["transforms"]
        data = LocalRFDataset(str(self.root), split="train", n_init_frames=2, subsequence=[1, 6],
                              frame_step=2, test_frame_every=0, transforms=transforms)
        self.assertEqual(data.image_paths, ["00001.png", "00003.png", "00005.png"])
        self.assertEqual(data.pose_scale, 1.0)
        selected = np.stack([np.array(f["transform_matrix"]) for f in transforms["frames"]
                             if f["file_path"].split("/")[-1] in data.image_paths])
        np.testing.assert_allclose(data.abs_poses, selected, atol=1e-5)
        # Chaining the relative poses from the first selected frame must rebuild the absolute poses.
        chained = [data.rel_poses[0]]
        for rel in data.rel_poses[1:]:
            chained.append(chained[-1] @ rel)
        np.testing.assert_allclose(np.stack(chained), data.abs_poses, atol=1e-5)


class UnitTests(TrackingFixture):
    def test_transforms_json_prior_keeps_upstream_normalisation(self):
        frames = [{"file_path": f"images/{i:05d}.png", "transform_matrix": pose.tolist()}
                  for i, pose in enumerate(self.body)]
        (self.root / "transforms.json").write_text(json.dumps({"fl_x": 8.0, "w": 8, "h": 6, "frames": frames}))
        data = LocalRFDataset(str(self.root), split="train", n_init_frames=2, test_frame_every=0,
                              with_preprocessed_poses=True)
        rel = np.stack([np.eye(4)] + [np.linalg.inv(self.body[i - 1]) @ self.body[i] for i in range(1, N)])
        expected = 2e-2 / np.median(np.linalg.norm(rel[:, :3, 3], axis=-1))
        self.assertAlmostEqual(data.pose_scale, expected, places=6)
        self.assertAlmostEqual(float(data.abs_poses[1, 0, 3]), float(self.body[1][0, 3] * expected), places=5)

    def test_metric_scene_radius_renders_like_normalised_poses(self):
        transforms = load_tracking(self.root, self.trajectory)["transforms"]
        data = LocalRFDataset(str(self.root), split="train", n_init_frames=3, test_frame_every=0,
                              transforms=transforms)
        radius = 4.0
        normalised = data.abs_poses.copy()
        normalised[:, :3, 3] /= radius  # the same poses expressed in units of the scene radius
        torch.manual_seed(0)
        metric = build_field(data.abs_poses, transforms, scene_radius=radius, n_init_frames=3)
        torch.manual_seed(0)
        reference = build_field(normalised, transforms, scene_radius=1.0, n_init_frames=3)
        ray_ids = torch.arange(8 * 6)
        view_ids = torch.tensor([2])
        with torch.no_grad():
            rgb_m, depth_m = metric(ray_ids, view_ids, 8, 6, is_train=False)[:2]
            rgb_n, depth_n = reference(ray_ids, view_ids, 8, 6, is_train=False)[:2]
        self.assertGreater(float(rgb_m.std()), 1e-3)  # guard against comparing trivial white images
        torch.testing.assert_close(rgb_m, rgb_n, atol=1e-5, rtol=0)
        torch.testing.assert_close(depth_m, depth_n * radius, atol=1e-4, rtol=1e-5)


class AbsoluteInitTests(TrackingFixture):
    def test_new_frame_starts_at_its_tracking_pose_not_the_optimised_chain(self):
        transforms = load_tracking(self.root, self.trajectory)["transforms"]
        data = LocalRFDataset(str(self.root), split="train", n_init_frames=2, test_frame_every=0,
                              transforms=transforms)
        field = build_field(data.abs_poses, transforms)
        np.testing.assert_allclose(field.get_cam2world().detach().numpy(), data.abs_poses[:2, :3], atol=1e-5)
        with torch.no_grad():
            # Simulate an optimisation step on the previous frame, then add the next frame.
            field.t_c2w[1].add_(0.5)
        field.append_frame()
        np.testing.assert_allclose(field.get_cam2world()[2].detach().numpy(), data.abs_poses[2, :3], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
