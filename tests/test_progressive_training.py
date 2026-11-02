"""CPU regressions for component counts, field transitions, schedules and a tiny training run."""
import contextlib
import io
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from test_tracking_poses import TrackingFixture, N, build_field
from opt import config_parser
from render import load_model, render_view
from train import reconstruction, active_gradient_groups, loss_gradient_norms

K = dict(fx=8., fy=9., cx=4.5, cy=3.5)


def tiny_field(n_frames=2):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_field(np.repeat(np.eye(4)[None], n_frames+1, axis=0), K,
                           n_init_frames=n_frames, scene_radius=8.)


class ComponentCountTests(unittest.TestCase):
    def test_defaults_and_repeated_and_grouped_options(self):
        defaults = config_parser([])
        self.assertEqual(defaults.n_lamb_sigma, [8]*3)
        self.assertEqual(defaults.n_lamb_sh, [24]*3)
        repeated = config_parser(['--n_lamb_sigma', '16']*3 + ['--n_lamb_sh', '32']*3)
        self.assertEqual(repeated.n_lamb_sigma, [16]*3)
        self.assertEqual(repeated.n_lamb_sh, [32]*3)
        grouped = config_parser(['--n_lamb_sigma', '2', '3', '4', '--n_lamb_sh', '5', '6', '7'])
        self.assertEqual(grouped.n_lamb_sigma, [2, 3, 4])
        self.assertEqual(grouped.n_lamb_sh, [5, 6, 7])

    def test_config_file_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.txt'
            path.write_text('n_lamb_sigma = [2, 3, 4]\nn_lamb_sh = [5, 6, 7]\n')
            args = config_parser(['--config', str(path), '--n_lamb_sigma', '8', '9', '10'])
            self.assertEqual(args.n_lamb_sigma, [8, 9, 10])
            self.assertEqual(args.n_lamb_sh, [5, 6, 7])

    def test_invalid_counts_fail_before_constructing_the_field(self):
        for ranks in (['1'], ['2', '3'], ['2']*4, ['0', '2', '3'], ['-1', '2', '3']):
            for option in ('--n_lamb_sigma', '--n_lamb_sh'):
                with self.subTest(ranks=ranks, option=option), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        config_parser([option, *ranks])


class ProgressiveScheduleTests(unittest.TestCase):
    def test_large_overlap_counts_create_exact_complementary_weights(self):
        for overlap in (1, 2, 30, 49, 98, 103, 399):
            with self.subTest(overlap=overlap), contextlib.redirect_stdout(io.StringIO()):
                model = tiny_field(overlap+1)
                model.n_overlap = overlap
                model.append_rf(n_added_frames=overlap)
                weights = model.blending_weights.detach()
                self.assertEqual(weights.shape, (overlap+1, 2))
                self.assertEqual(weights.device, model.r_c2w[0].device)
                self.assertTrue(((weights >= 0) & (weights <= 1)).all())
                torch.testing.assert_close(weights.sum(1), torch.ones(overlap+1))
                torch.testing.assert_close(weights[0], torch.tensor([1., 0.]))
                torch.testing.assert_close(weights[-1], torch.tensor([0., 1.]))
                if overlap > 1:
                    self.assertTrue((torch.diff(weights[-overlap:, 1]) > 0).all())
                self.assertTrue(all(link == 0 for link in model.pose_linked_rf))

    def test_effective_overlap_is_limited_by_added_and_available_frames(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = tiny_field(5)
            model.n_overlap = 49
            model.append_rf(n_added_frames=2)
        self.assertEqual(int((model.blending_weights[:, 1] > 0).sum()), 2)
        torch.testing.assert_close(model.blending_weights[-2:], torch.tensor([[.5,.5],[0.,1.]]))
        model = tiny_field(1)
        with self.assertRaisesRegex(ValueError, 'at least one overlapping'):
            model.append_rf()
        self.assertEqual(len(model.tensorfs), 1)

    def test_refinement_schedule_scales_with_active_frames_and_decays_lr(self):
        model = tiny_field(3)
        # A normal coarse step does not advance the per-field refinement counter.
        def zero_loss():
            return model.tensorfs[-1].density_plane[0].sum()*0 + model.t_c2w[-1].sum()*0
        initial_rf_lr = model.rf_optimizer.param_groups[0]['lr']
        model.optimizer_step(zero_loss(), optimize_poses=True)
        self.assertEqual(model.rf_iter[-1], 0)
        model.is_refining = True
        model.optimizer_step(zero_loss(), optimize_poses=True)
        self.assertEqual(model.rf_iter[-1], 1)
        self.assertEqual(model.rf_optimizer.param_groups[0]['lr'], initial_rf_lr)
        # The next step expands the budget to 3 frames and starts the LR decay.
        model.optimizer_step(zero_loss(), optimize_poses=True)
        self.assertEqual(model.n_iters, model.n_iters_per_frame*3)
        self.assertEqual(model.n_iters_reg, model.n_iters_reg_per_frame*3)
        factor = model.lr_decay_target_ratio**(1/model.n_iters)
        self.assertAlmostEqual(model.rf_optimizer.param_groups[0]['lr'], initial_rf_lr*factor)
        self.assertAlmostEqual(model.t_optimizers[-1].param_groups[0]['lr'], model.lr_t_init*factor)
        while model.rf_iter[-1] < model.n_iters_reg+1:
            model.optimizer_step(zero_loss(), optimize_poses=True)
        self.assertFalse(model.regularize)


class VolumeDepthTests(unittest.TestCase):
    def test_tiny_density_keeps_nonzero_opacity_and_finite_depth_gradient(self):
        model = tiny_field(1)
        field = model.tensorfs[0]
        field.density_shift = -25.
        feature = torch.tensor(0., requires_grad=True)
        with patch.object(field, 'compute_densityfeature', side_effect=lambda xyz: feature.expand(len(xyz))):
            ray = torch.tensor([[0., 0., 0., 0., 0., -1.]])
            _, depth_sum, opacity = field(ray, is_train=False)
        _, z, _ = field.sample_ray_contracted(ray[:, :3], ray[:, 3:], is_train=False)
        sigma = torch.nn.functional.softplus(torch.tensor(-25., dtype=torch.float64))
        expected_opacity = -torch.expm1(-sigma*(z[0,-1]-z[0,0]).double()*field.distance_scale)
        self.assertGreater(opacity.item(), 0.)
        self.assertAlmostEqual(opacity.item()/expected_opacity.item(), 1., places=5)
        depth = depth_sum/opacity.clamp_min(1e-8)
        self.assertGreater(depth.item(), 0.)
        depth.sum().backward()
        self.assertTrue(torch.isfinite(feature.grad))

    def test_low_opacity_parameter_gradients_do_not_follow_independent_weight_spikes(self):
        from utils.utils import dense_depth_loss
        torch.manual_seed(20261102)
        model = tiny_field(2)
        with contextlib.redirect_stdout(io.StringIO()):
            model.tensorfs[0].upsample_volume_grid([64]*3)
        norms = []
        for shift in (-15., -25.):
            model.tensorfs[0].density_shift = shift
            _, depth, opacity = model(torch.arange(48), torch.tensor([1]), 8, 6, is_train=False)
            loss = dense_depth_loss(depth, torch.full((48,), 3.), torch.ones(48,dtype=torch.bool))
            stats = loss_gradient_norms({"depth":loss}, active_gradient_groups(model, True))
            values = np.array([stats[f'grad/depth/{g}_l2'] for g in ('density','rotation','translation')])
            self.assertTrue(np.isfinite(values).all())
            # Quantitative bound for this seeded synthetic field, not a universal safety bound.
            self.assertLess(values.max(), 100.)
            norms.append(values)
            if shift == -25.:
                self.assertGreater(opacity.min().item(), 1e-8)
                self.assertLess(opacity.max().item(), 2e-8)
        np.testing.assert_allclose(norms[1], norms[0], rtol=.05, atol=.01)

    def test_independent_weight_derivative_has_inverse_opacity_scaling(self):
        z_metres = torch.tensor([1., 5.], dtype=torch.float64)
        maxima = []
        for opacity in (.5, .01, 2e-8):
            weights = torch.tensor([opacity/2]*2, dtype=torch.float64, requires_grad=True)
            depth = (weights*z_metres).sum()/weights.sum().clamp_min(1e-8)
            grad, = torch.autograd.grad(depth, weights)
            # z is already in metres here, so dD/dw_i=(z_i-D)/A above the clamp.
            torch.testing.assert_close(grad, (z_metres-depth.detach())/opacity)
            maxima.append(grad.abs().max().item())
        self.assertAlmostEqual(maxima[1]/maxima[0], 50.)
        self.assertGreater(maxima[2], 9e7)
        # Below the floor the denominator is constant. Do not assume a particular
        # subgradient exactly at the non-differentiable clamp boundary.
        weights = torch.full((2,), 2.5e-9, dtype=torch.float64, requires_grad=True)
        depth = (weights*z_metres).sum()/weights.sum().clamp_min(1e-8)
        grad, = torch.autograd.grad(depth, weights)
        torch.testing.assert_close(grad, z_metres/1e-8)

    def test_empty_scene_uses_far_sample_colour_as_learned_background(self):
        model = tiny_field(1)
        field = model.tensorfs[0]
        # Zero foreground alpha: all RGB weight is at the forced opaque far endpoint.
        colour = torch.tensor([.2, .4, .7])
        with patch.object(field, 'feature2density', side_effect=torch.zeros_like), \
                patch.object(field.renderModule, 'forward', side_effect=lambda xyz, viewdirs, features, refine: colour.expand(len(xyz), 3)):
            rgb, depth_sum, opacity = field(torch.tensor([[0., 0., 0., 0., 0., -1.]]), is_train=False)
        torch.testing.assert_close(rgb, colour[None])
        torch.testing.assert_close(opacity, torch.zeros(1))
        torch.testing.assert_close(depth_sum, torch.zeros(1))


class GradientDiagnosticsTests(unittest.TestCase):
    def test_weighted_gradients_are_correct_without_touching_optimizer_grads(self):
        x = torch.nn.Parameter(torch.tensor([1., 2.]))
        y = torch.nn.Parameter(torch.tensor([3.]))
        x.grad = torch.tensor([7., 8.])
        before = x.grad.clone()
        losses = {"rgb": x.sum()*.25, "depth": (x.square().sum()+y.square().sum())*.1}
        stats = loss_gradient_norms(losses, {"density":[x], "translation":[y]})
        self.assertAlmostEqual(stats['grad/rgb/density_l2'], np.sqrt(2)*.25)
        self.assertAlmostEqual(stats['grad/depth/density_l2'], np.sqrt(.2**2+.4**2))
        self.assertAlmostEqual(stats['grad/depth/translation_max'], .6)
        self.assertEqual(stats['grad/rgb/translation_l2'], 0.)
        torch.testing.assert_close(x.grad, before)
        self.assertIsNone(y.grad)
        (losses['rgb']+losses['depth']).backward()
        torch.testing.assert_close(x.grad, before+torch.tensor([.45,.65]))
        torch.testing.assert_close(y.grad, torch.tensor([.6]))

    def test_diagnostics_exclude_anchored_and_old_field_parameters(self):
        model = tiny_field(3)
        for optimizer in (model.r_optimizers[0], model.t_optimizers[0]):
            optimizer.param_groups[0]['lr'] = 0.
        groups = active_gradient_groups(model, True)
        self.assertNotIn(id(model.t_c2w[0]), {id(p) for p in groups['translation']})
        with contextlib.redirect_stdout(io.StringIO()):
            model.append_rf(2)
            model.append_frame()
        groups = active_gradient_groups(model, True)
        self.assertEqual([id(p) for p in groups['translation']], [id(model.t_c2w[3])])
        old = {id(p) for p in model.tensorfs[0].parameters()}
        self.assertFalse(old & {id(p) for values in groups.values() for p in values})
        poses = active_gradient_groups(model, True, poses_only=True)
        self.assertNotIn('density', poses)
        self.assertNotIn('appearance', poses)

    def test_moment_export_below_floor_does_not_lose_numerator(self):
        from render import render_depth_moment, expected_depth
        model = tiny_field(1)
        moment = 2e-10
        opacity = 1e-10
        # The model returns a clamped ratio. The export must recover the first moment,
        # not multiply by the unclamped opacity and erroneously shrink it again.
        with patch.object(model, 'forward', return_value=(torch.zeros(48,3),
                          torch.full((48,), moment/1e-8), torch.full((48,), opacity))):
            M, A = render_depth_moment(model,0)
        np.testing.assert_allclose(M, moment, rtol=1e-6)
        np.testing.assert_allclose(expected_depth(M,A,min_opacity=1e-12), 2., rtol=1e-6)


class TinyTrainingTests(TrackingFixture):
    def test_multifield_training_exports_and_reloads_with_dense_depth(self):
        (self.root/'depth').mkdir()
        for i in range(N):
            cv2.imwrite(str(self.root/'depth'/f'{i:05d}.png'), np.full((6,8), 300, np.uint16))
        out = self.root/'run'
        args = config_parser([
            '--datadir', str(self.root), '--trajectory', str(self.trajectory), '--logdir', str(out),
            '--subsequence', '0', '5', '--device', 'cpu', '--batch_size', '16',
            '--n_init_frames', '2', '--n_overlap', '1', '--n_max_frames', '3',
            '--n_iters_per_frame', '3', '--n_iters_reg', '0', '--add_frames_every', '1',
            '--N_voxel_init', '512', '--N_voxel_final', '512',
            '--n_lamb_sigma', '2', '2', '2', '--n_lamb_sh', '2', '2', '2', '--featureC', '8',
            '--depth_weight', '.1', '--depth_scale', '.01', '--pose_prior_lambda_t', '.0156',
            '--pose_prior_lambda_r', '1', '--progress_refresh_rate', '1000000', '--grad_log_every', '1'])
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(20211202)
            np.random.seed(20211202)
            torch.manual_seed(20211202)
            with contextlib.redirect_stdout(io.StringIO()):
                reconstruction(args)
                model = load_model(out/'checkpoints.th', 'cpu')
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        self.assertEqual(len(model.tensorfs), 2)
        self.assertEqual(len(model.r_c2w), 5)
        trajectory = np.loadtxt(out/'trajectory.tum')
        self.assertEqual(trajectory.shape, (5,8))
        self.assertTrue(np.isfinite(trajectory).all())
        np.testing.assert_allclose(trajectory[:,0], np.arange(5)/10)
        np.testing.assert_allclose(trajectory[0,1:], [0,0,0,0,0,0,1], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(trajectory[:,4:], axis=1), 1, atol=1e-7)
        events = EventAccumulator(str(out)).Reload()
        self.assertTrue(any(tag.startswith('grad/depth/density_') for tag in events.Tags()['scalars']))
        self.assertGreater(len(events.Scalars('train/depth_l1_m')), 3)  # n_iters_reg=0 still supervises depth.
        for tag in events.Tags()['scalars']:
            self.assertTrue(all(np.isfinite(e.value) for e in events.Scalars(tag)), tag)
        for i in range(5):
            rgb, depth = render_view(model, i, ray_chunk=16)
            self.assertTrue(np.isfinite(rgb).all() and np.isfinite(depth).all())
        before = (out/'trajectory.tum').read_bytes()
        with self.assertRaises(FileExistsError):
            reconstruction(args)
        self.assertEqual((out/'trajectory.tum').read_bytes(), before)

    def test_entrypoint_help_works_outside_repo(self):
        entry = Path(__file__).resolve().parents[1]/'localTensoRF/train.py'
        result = subprocess.run([sys.executable, str(entry), '--help'], cwd=self.root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--depth_weight', result.stdout)


if __name__ == '__main__':
    unittest.main()
