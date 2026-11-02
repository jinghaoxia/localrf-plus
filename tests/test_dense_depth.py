"""CPU regression checks for calibrated rays and metric dense-depth supervision."""
import contextlib
import io
import json
import unittest

import cv2
import numpy as np
import torch
import yaml

from test_tracking_poses import TrackingFixture, N, build_field
from dataLoader.localrf_dataset import LocalRFDataset
from dataLoader.tracking_poses import read_tum
from opt import config_parser
from utils.ray_utils import get_ray_directions_lean
from utils.utils import dense_depth_loss


class CalibratedDatasetTests(TrackingFixture):
    def dataset(self, **kwargs):
        options = dict(datadir=str(self.root), trajectory=self.trajectory,
                       n_init_frames=2, frames_chunk=2, test_frame_every=0)
        options.update(kwargs)
        return LocalRFDataset(**options)

    def write_depths(self):
        (self.root / 'depth').mkdir()
        for i in range(N):
            cv2.imwrite(str(self.root / 'depth' / f'{i:05d}.png'), np.full((3, 4), 100 * (i + 1), np.uint16))

    def test_dual_focal_projection_with_actual_resize_and_pixel_centres(self):
        path = self.root / 'meta/calib.yaml'
        calibration = yaml.safe_load(path.read_text())
        calibration['left_camera'].update(width=9, height=7, K=[[11., 0., 3.2], [0., 7., 2.3], [0., 0., 1.]])
        path.write_text(yaml.safe_dump(calibration))
        for i in range(N):
            cv2.imwrite(str(self.root / 'images' / f'{i:05d}.png'), np.full((7, 9, 3), 20 * i, np.uint8))
        data = self.dataset(downsampling=2)
        W, H = data.img_wh
        self.assertEqual((W, H), (4, 4))  # Rounding means sx != sy != 0.5.
        k = data.intrinsics
        np.testing.assert_allclose(list(k.values()), [11 * W / 9, 7 * H / 7, 3.7 * W / 9, 2.8 * H / 7])
        i, j = torch.tensor([0, 2, 3]), torch.tensor([1, 3, 0])
        ray = get_ray_directions_lean(i, j, **k)
        # Back-project to the full-resolution OpenCV image, then resize about pixel centres.
        u_cv = ray[:, 0] * 11 + 3.2
        v_cv = -ray[:, 1] * 7 + 2.3
        torch.testing.assert_close((u_cv + 0.5) * (W / 9) - 0.5, i.float())
        torch.testing.assert_close((v_cv + 0.5) * (H / 7) - 0.5, j.float())

    def test_selected_depths_follow_frames_when_the_window_moves(self):
        self.write_depths()
        data = self.dataset(load_depth=True, depth_scale=.01, subsequence=[1, 6], frame_step=2)
        self.assertEqual(data.image_paths, ['00001.png', '00003.png', '00005.png'])
        self.assertEqual(data.loaded_frames, 2)
        data.activate_frames()
        data.deactivate_frames(1)
        batch = data.sample(64, is_refining=True, optimize_poses=False)
        selected = np.array([2., 4., 6.])[batch['idx'] // data.n_px_per_frame]
        np.testing.assert_allclose(batch['depths'][:, 0], selected)
        self.assertTrue(batch['depth_valid'].all())
        self.assertNotIn('fwd_flow', batch)
        self.assertNotIn('invdepths', batch)

    def test_uniform_photometric_weights_preserve_masks_and_depth_validity(self):
        self.write_depths()
        (self.root / 'masks').mkdir()
        mask = np.ones((6, 8), np.uint8) * 255
        mask[:, 4:] = 0
        cv2.imwrite(str(self.root / 'masks/all.png'), mask)
        raw = np.full((3, 4), 300, np.uint16)
        raw[0, 0] = 0
        cv2.imwrite(str(self.root / 'depth/00000.png'), raw)
        data = self.dataset(load_depth=True, depth_scale=.01, uniform_loss_weights=True)
        weights = data.all_loss_weights[:48].reshape(6, 8)
        np.testing.assert_array_equal(weights, mask > 0)
        valid = data.all_depth_valid[:48].reshape(6, 8)
        self.assertFalse(valid[:, 4:].any())
        self.assertFalse(valid[:2, :2].any())
        self.assertTrue(valid[2:, :4].all())
        np.testing.assert_allclose(data.all_depths[:48].reshape(6, 8)[valid], 3.)

    def test_npy_depth_is_in_metres_and_rejects_nonfinite_observations(self):
        (self.root / 'depth').mkdir()
        raw = np.full((6, 8), 2.75, np.float32)
        raw[0, :3] = [np.nan, np.inf, -1]
        for i in range(N):
            np.save(self.root / 'depth' / f'{i:05d}.npy', raw)
        data = self.dataset(load_depth=True)
        self.assertTrue(np.isfinite(data.all_depths).all())
        self.assertFalse(data.all_depth_valid[:3].any())
        np.testing.assert_allclose(data.all_depths[data.all_depth_valid], 2.75)

    def test_png_depth_requires_explicit_units(self):
        self.write_depths()
        with self.assertRaisesRegex(ValueError, 'depth_scale'):
            self.dataset(load_depth=True)

    def test_missing_depth_is_an_error_only_when_enabled(self):
        data = self.dataset(load_depth=False)
        self.assertIsNone(data.all_depths)
        with self.assertRaisesRegex(ValueError, 'expected exactly one depth'):
            self.dataset(load_depth=True, depth_scale=.01)

    def test_metric_depth_rejects_normalised_pose_path(self):
        with self.assertRaisesRegex(ValueError, 'metric dense depth requires'):
            self.dataset(trajectory=None, load_depth=True, depth_scale=.01)

    def test_dataset_export_preserves_original_gauge_and_extrinsic_translation(self):
        data = self.dataset(subsequence=[1, 6], frame_step=2)
        path = self.root / 'export.tum'
        data.export_trajectory(path, data.abs_poses[:, :3])
        actual = read_tum(path)
        self.assertEqual([int(t) for t, _ in actual], [100000, 300000, 500000])
        np.testing.assert_allclose(np.stack([p for _, p in actual]), np.stack(self.body)[[1, 3, 5]], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            data.export_trajectory(path, data.abs_poses[:1, :3])

    def test_held_out_frames_are_not_sampled_by_default(self):
        data = self.dataset(test_frame_every=2, n_init_frames=N)
        for _ in range(4):
            batch = data.sample(64, is_refining=True, optimize_poses=True)
            self.assertFalse(batch['train_test_poses'])
            self.assertFalse(np.isin(batch['view_ids'], [2, 4]).any())

    def test_transforms_metadata_also_supplies_both_focals_without_cli(self):
        (self.root / 'meta/calib.yaml').unlink()
        (self.root / 'transforms.json').write_text(json.dumps(
            dict(fl_x=8., fl_y=6., cx=3., cy=2., w=8, h=6, frames=[])))
        data = self.dataset(trajectory=None, downsampling=2)
        self.assertEqual(data.intrinsics, dict(fx=4., fy=3., cx=1.75, cy=1.25))

    def test_depth_alone_backpropagates_into_field_and_pose(self):
        data = self.dataset()
        torch.manual_seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_field(data.abs_poses, data.intrinsics, scene_radius=4.)
        _, depth, opacity = model(torch.arange(48), torch.tensor([1]), 8, 6, is_train=True)
        self.assertTrue(torch.isfinite(depth).all())
        self.assertTrue((opacity >= 0).all())
        loss = dense_depth_loss(depth, depth.detach() + .5, torch.ones_like(depth, dtype=torch.bool))
        loss.backward()
        grads = [model.t_c2w[1].grad, model.r_c2w[1].grad, model.tensorfs[0].density_plane[0].grad]
        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)
        self.assertFalse(model.init_focal.requires_grad)
        self.assertFalse(model.focal_offset.requires_grad)
        self.assertFalse(model.center_rel.requires_grad)


    def test_depth_updates_only_current_field_and_its_new_poses(self):
        data = self.dataset(n_init_frames=4)
        torch.manual_seed(8)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_field(data.abs_poses, data.intrinsics, scene_radius=4., n_init_frames=3)
            model.append_rf(n_added_frames=2)
            model.append_frame()
        old_poses = model.get_cam2world().detach().clone()
        old_field = {k: v.clone() for k, v in model.tensorfs[0].state_dict().items()}
        _, depth, _ = model(torch.arange(48), torch.tensor([3]), 8, 6, is_train=True)
        loss = dense_depth_loss(depth, depth.detach() + .5, torch.ones_like(depth, dtype=torch.bool))
        model.optimizer_step(loss, optimize_poses=True)
        torch.testing.assert_close(model.get_cam2world()[:3], old_poses[:3], atol=0, rtol=0)
        self.assertFalse(torch.equal(model.get_cam2world()[3], old_poses[3]))
        for key, value in model.tensorfs[0].state_dict().items():
            torch.testing.assert_close(value, old_field[key], atol=0, rtol=0)


class DenseLossTests(unittest.TestCase):
    def test_absolute_residual_does_not_cancel_or_fit_scale(self):
        pred = torch.tensor([2., 4., 100., 5.], requires_grad=True)
        target = torch.tensor([3., 3., 1., float('nan')])
        loss = dense_depth_loss(pred, target, torch.tensor([True, True, False, True]))
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([-.5, .5, 0., 0.]))

    def test_empty_valid_mask_has_finite_zero_loss_and_gradient(self):
        pred = torch.tensor([2., 3.], requires_grad=True)
        loss = dense_depth_loss(pred, torch.tensor([0., float('nan')]), torch.ones(2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.zeros(2))

    def test_old_and_manual_intrinsics_cli_are_removed(self):
        args = config_parser([])
        self.assertEqual(args.depth_weight, 0.)
        for option in ('--fov', '--fx', '--fy', '--cx', '--cy', '--loss_flow_weight_inital',
                       '--loss_depth_weight_inital', '--model_name', '--ckpt', '--nSamples'):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    config_parser([option, '1'])


if __name__ == '__main__':
    unittest.main()
