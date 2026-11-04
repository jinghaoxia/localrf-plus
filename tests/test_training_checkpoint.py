"""Recovery must reproduce future optimization, including field transitions."""
import contextlib
import io
import os
from pathlib import Path
import signal
import unittest
import weakref
from unittest.mock import patch

import cv2
import numpy as np
import torch

from test_tracking_poses import TrackingFixture, N, build_field
from dataLoader.localrf_dataset import LocalRFDataset
from opt import config_parser
from train import reconstruction
from training_checkpoint import (atomic_torch_save, load_checkpoint,
                                 StopAtStepBoundary, resume_arguments)
from local_tensorfs import LocalTensorfs
import local_tensorfs
import train


class ResumeTrainingTests(TrackingFixture):
    def args(self, out):
        return config_parser([
            '--datadir', str(self.root), '--trajectory', str(self.trajectory), '--logdir', str(out),
            '--device', 'cpu', '--batch_size', '16', '--subsequence', '1', '6',
            '--n_init_frames', '2', '--n_overlap', '1', '--n_max_frames', '3', '--max_drift', '10000',
            '--n_iters_per_frame', '8', '--n_iters_reg', '4', '--add_frames_every', '2',
            '--refinement_speedup_factor', '2', '--prog_speedup_factor', '2',
            '--N_voxel_init', '512', '--N_voxel_final', '1000', '--upsamp_list', '2', '4',
            '--update_AlphaMask_list', '2', '--density_shift', '0',
            '--n_lamb_sigma', '2', '2', '2', '--n_lamb_sh', '2', '2', '2', '--featureC', '8',
            '--lr_i_init', '.001', '--lr_exposure_init', '.001', '--lr_upsample_reset', '0',
            '--depth_weight', '.001', '--depth_scale', '.01',
            '--pose_prior_lambda_t', '.0156', '--pose_prior_lambda_r', '1',
            '--checkpoint_every', '1', '--pose_vis_every', '0', '--progress_refresh_rate', '1000000'])

    def setUp(self):
        super().setUp()
        (self.root / 'depth').mkdir()
        for i in range(N):
            cv2.imwrite(str(self.root / 'depth' / f'{i:05d}.png'), np.full((6, 8), 300, np.uint16))

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_resume_matches_uninterrupted_across_upsampling_masks_and_fields(self):
        baseline = self.root / 'baseline'
        snapshots = {}
        real_save = train.save_checkpoint

        def capture(path, model, dataset, args, identity, loop, frozen):
            real_save(path, model, dataset, args, identity, loop, frozen)
            step = loop['iteration']
            # Save three meaningful states: progressive, mask/upsample, new field.
            if (step == 1 or (model.is_refining and model.rf_iter[-1] == 4) or
                    (len(model.tensorfs) == 2 and model.rf_iter[-1] == 0)):
                destination = Path(path).with_name(f'step_{step}.th')
                destination.write_bytes(Path(path).read_bytes())
                snapshots[step] = destination

        with contextlib.redirect_stdout(io.StringIO()), patch.object(train, 'save_checkpoint', side_effect=capture):
            reconstruction(self.args(baseline))
        self.assertGreaterEqual(len(snapshots), 3)
        expected = torch.load(baseline / 'checkpoints.th', weights_only=True)
        expected_resume = load_checkpoint(baseline / 'resume.th')
        self.assertEqual(len(expected_resume['frozen_fields']), 1)
        self.assertTrue(any('alphaMask' in key for key in expected['state_dict']))
        for step, source in snapshots.items():
            with self.subTest(step=step), contextlib.redirect_stdout(io.StringIO()):
                out = self.root / f'resumed_{step}'
                # Only the resume path and new output are needed; saved raw options
                # must not receive speedup scaling for a second time.
                reconstruction(config_parser(['--resume', str(source), '--logdir', str(out)]))
                result = torch.load(out / 'checkpoints.th', weights_only=True)
                self.assert_tree_equal(expected['state_dict'], result['state_dict'])
                actual_resume = load_checkpoint(out / 'resume.th')
                for key in ('optimizers', 'schedule', 'rng', 'loop', 'tensorf_args'):
                    self.assert_tree_equal(expected_resume[key], actual_resume[key])
                np.testing.assert_array_equal(np.loadtxt(out / 'trajectory.tum'),
                                              np.loadtxt(baseline / 'trajectory.tum'))
                self.assertEqual(len(list((out / 'resume_fields').glob('*.th'))), 1)

    def test_interruption_uses_committed_step_and_continues_in_place(self):
        out = self.root / 'interrupted'
        real_save = train.save_checkpoint

        def crash_after_commit(path, model, dataset, args, identity, loop, frozen):
            real_save(path, model, dataset, args, identity, loop, frozen)
            if loop['iteration'] == 3:
                raise torch.cuda.OutOfMemoryError('simulated OOM after committed step')

        with contextlib.redirect_stdout(io.StringIO()), patch.object(train, 'save_checkpoint', side_effect=crash_after_commit):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                reconstruction(self.args(out))
        self.assertEqual(load_checkpoint(out / 'resume.th')['loop']['iteration'], 3)
        self.assertFalse((out / 'checkpoints.th').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            reconstruction(config_parser(['--resume', str(out / 'resume.th')]))
        self.assertFalse(load_checkpoint(out / 'resume.th')['loop']['training'])
        self.assertEqual(len((out / 'run_history.jsonl').read_text().splitlines()), 2)
        self.assertTrue((out / 'trajectory.tum').is_file())

    def test_sigterm_commits_the_step_before_stopping(self):
        out = self.root / 'graceful'
        original = LocalTensorfs.optimizer_step

        def request_stop(model, loss, optimize_poses):
            result = original(model, loss, optimize_poses)
            os.kill(os.getpid(), signal.SIGTERM)
            return result

        with contextlib.redirect_stdout(io.StringIO()), patch.object(LocalTensorfs, 'optimizer_step', request_stop):
            reconstruction(self.args(out))
        state = load_checkpoint(out / 'resume.th')
        self.assertEqual(state['loop']['iteration'], 1)
        self.assertTrue(state['loop']['training'])
        self.assertFalse((out / 'checkpoints.th').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            reconstruction(config_parser(['--resume', str(out / 'resume.th'), '--batch_size', '32']))
        self.assertTrue((out / 'trajectory.tum').exists())

    def test_atomic_failure_preserves_last_checkpoint(self):
        path = self.root / 'resume.th'
        atomic_torch_save({'iteration': 7}, path)
        original = path.read_bytes()

        def fail(value, stream):
            stream.write(b'partial write')
            raise OSError('simulated disk full')

        with patch('training_checkpoint.torch.save', side_effect=fail):
            with self.assertRaises(OSError):
                atomic_torch_save({'iteration': 8}, path)
        self.assertEqual(original, path.read_bytes())
        self.assertFalse(list(self.root.glob('.resume.th.*.tmp')))

    def test_rejects_render_only_and_changed_configuration_or_data(self):
        path = self.root / 'old.th'
        torch.save({'kwargs': {}, 'state_dict': {}}, path)
        with self.assertRaisesRegex(ValueError, 'legacy'):
            load_checkpoint(path)
        out = self.root / 'run'
        with contextlib.redirect_stdout(io.StringIO()):
            reconstruction(self.args(out))
        for option, value in (('--depth_weight', '1'), ('--frame_step', '2')):
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                reconstruction(config_parser(['--resume', str(out / 'resume.th'), option, value]))
        before = (out / 'resume.th').read_bytes()
        cv2.imwrite(str(self.root / 'depth/00001.png'), np.ones((6, 8), np.uint16))
        with self.assertRaisesRegex(ValueError, 'dataset mismatch'):
            reconstruction(config_parser(['--resume', str(out / 'resume.th')]))
        self.assertEqual(before, (out / 'resume.th').read_bytes())

    def test_window_restore_keeps_absolute_indices_and_prefetched_depth(self):
        options = dict(datadir=str(self.root), trajectory=str(self.trajectory), split='train',
                       n_init_frames=2, frames_chunk=2, load_depth=True, depth_scale=.01,
                       test_frame_every=0, uniform_loss_weights=True)
        original = LocalRFDataset(**options)
        original.activate_frames(2)
        original.activate_frames(1)
        original.deactivate_frames(2)
        restored = LocalRFDataset(**options)
        restored.restore_window(original.active_frames_bounds, original.loaded_frames)
        self.assertEqual(restored.active_frames_bounds, [2, 5])
        self.assertEqual(restored.loaded_frames, 6)
        for name in ('all_rgbs', 'all_depths', 'all_depth_valid', 'all_loss_weights'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        state = np.random.get_state()
        expected = original.sample(16, is_refining=True, optimize_poses=True)
        np.random.set_state(state)
        actual = restored.sample(16, is_refining=True, optimize_poses=True)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_config_file_overrides_are_checked_on_resume(self):
        args = self.args(self.root / 'unused')
        configuration = self.root / 'changed.txt'
        configuration.write_text('depth_weight = 1\n')
        requested = config_parser(['--resume', 'unused.th', '--config', str(configuration)])
        with self.assertRaisesRegex(ValueError, 'depth_weight'):
            resume_arguments(requested, {'args': vars(args)})


class ScheduleRegressionTests(unittest.TestCase):
    def test_resize_without_lr_reset_updates_new_parameters(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_field(np.repeat(np.eye(4)[None], 3, axis=0),
                                dict(fx=8., fy=8., cx=4., cy=3.))
            model.lr_upsample_reset = False
            model.N_voxel_list = {0: 1000}
            rate = model.rf_optimizer.param_groups[0]['lr']
            model.optimizer_step(model.tensorfs[-1].density_plane[0].sum(), False)
        parameter = model.tensorfs[-1].density_plane[0]
        self.assertIn(id(parameter), {id(p) for group in model.rf_optimizer.param_groups for p in group['params']})
        self.assertEqual(rate, model.rf_optimizer.param_groups[0]['lr'])
        before = parameter.detach().clone()
        model.N_voxel_list = {}
        model.optimizer_step(parameter.sum(), False)
        self.assertFalse(torch.equal(before, parameter))

    def test_loading_fields_with_different_non_cubic_grids(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_field(np.repeat(np.eye(4)[None], 4, axis=0),
                                dict(fx=8., fy=8., cx=4., cy=3.))
            model.tensorfs[0].upsample_volume_grid([8, 9, 10])
            model.append_rf(1)
            model.tensorfs[1].upsample_volume_grid([9, 10, 11])
            restored = LocalTensorfs(**dict(model.get_kwargs(), device='cpu'))
            restored.load(model.state_dict())
        self.assertEqual(restored.tensorfs[0].gridSize.tolist(), [8, 9, 10])
        self.assertEqual(restored.tensorfs[1].gridSize.tolist(), [9, 10, 11])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_old_adam_is_released_before_next_field_allocation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_field(np.repeat(np.eye(4)[None], 4, axis=0),
                                dict(fx=8., fy=8., cx=4., cy=3.))
            model.optimizer_step(model.tensorfs[0].density_plane[0].sum() + model.t_c2w[1].sum(), True)
            self.assertTrue(model.rf_optimizer.state)
            previous = weakref.ref(model.rf_optimizer)
            constructor = local_tensorfs.TensorVMSplit

            def allocate(**kwargs):
                self.assertIsNone(previous())
                self.assertFalse(model.t_optimizers[1].state)
                return constructor(**kwargs)

            with patch.object(local_tensorfs, 'TensorVMSplit', side_effect=allocate):
                model.append_rf(1)
        self.assertFalse(any(p.requires_grad for p in model.tensorfs[0].parameters()))

    def test_signal_handler_requests_boundary_and_restores_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)
        with StopAtStepBoundary() as stop:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(stop.requested)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


if __name__ == '__main__':
    unittest.main()
