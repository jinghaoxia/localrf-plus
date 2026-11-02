"""Synthetic CPU tests; no downloaded model, real trajectory, LiDAR or GT required."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

SPEC = importlib.util.spec_from_file_location('promptda_offline', Path(__file__).resolve().parents[1] / 'scripts/promptda.py')
offline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(offline)


def write_pcd(path, points, trailing=b''):
    values = np.column_stack([points, np.zeros(len(points))]).astype('<f4')
    path.write_bytes(('FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                      'COUNT 1 1 1 1\nPOINTS %d\nDATA binary\n' % len(points)).encode()
                     + values.tobytes() + trailing)


class OfflineGeometryTests(unittest.TestCase):
    def test_pcd_ignores_zero_padding_not_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.pcd'
            points = [[1, 2, 3], [4, 5, 6]]
            write_pcd(path, points, b'\0' * 100)
            np.testing.assert_array_equal(offline.read_pcd_xyz(path), points)
            write_pcd(path, points, b'bad')
            with self.assertRaises(ValueError):
                offline.read_pcd_xyz(path)

    def test_projection_zbuffer_units_and_half_pixel(self):
        K = np.array([[4., 0, 1.5], [0, 6., 1.5], [0, 0, 1.]])
        # Resize 8x8 -> 4x4; optical axis lands in pixel (1,1), not (0,0).
        points = np.array([[0., 0, 5], [0, 0, 2], [0, 0, -1], [np.nan, 0, 1]])
        depth = offline.project_prompt(points, np.eye(4), K, (8, 8), (4, 4), .1, 10)
        self.assertEqual(depth[1, 1], 2.)
        self.assertEqual(np.count_nonzero(depth), 1)

    def test_projection_uses_extrinsic_translation(self):
        T_cam_lidar = np.eye(4)
        T_cam_lidar[:3, 3] = [1, 0, 2]
        depth = offline.project_prompt(np.array([[0., 0, 2]]), T_cam_lidar,
                                       np.diag([4., 4., 1.]), (8, 8), (8, 8), .1, 10)
        self.assertEqual(depth[0, 1], 4.)

    def test_pose_interpolation_and_bounded_extrapolation(self):
        times = np.array([0, 1000000], dtype=np.int64)
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, 3] = [2, 0, 0]
        poses[1, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        middle = offline.pose_at(times, poses, 500000, .25)
        np.testing.assert_allclose(middle[:3, 3], [1, 0, 0])
        np.testing.assert_allclose(middle[:3, :3], Rotation.from_euler('z', 45, degrees=True).as_matrix(), atol=1e-12)
        self.assertAlmostEqual(offline.pose_at(times, poses, -100000, .25)[0, 3], -.2)
        with self.assertRaises(ValueError):
            offline.pose_at(times, poses, -300000, .25)

    def test_fill_preserves_measured_values_and_records_support(self):
        sparse = np.array([[2., 0, np.nan], [0, 0, 4.]])
        filled, valid, distance = offline.fill_prompt(sparse)
        self.assertTrue(np.isfinite(filled).all())
        np.testing.assert_array_equal(filled[valid], sparse[valid])
        self.assertEqual(valid.sum(), 2)
        self.assertTrue((distance[valid] == 0).all())
        self.assertTrue((distance[~valid] > 0).all())

    def test_empty_and_constant_prompts_rejected(self):
        for sparse in (np.zeros((2, 2)), np.ones((2, 2)), np.array([[0, 1], [0, 0]])):
            with self.assertRaises(ValueError):
                offline.fill_prompt(sparse)

    def test_png_millimetres_npy_metres(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = np.array([[1.5, 3.], [0, 4]], np.float32)
            np.save(root / 'd.npy', depth)
            cv2.imwrite(str(root / 'd.png'), (depth * 1000).astype(np.uint16))
            np.testing.assert_array_equal(offline.read_prompt(root / 'd.npy'), depth)
            np.testing.assert_array_equal(offline.read_prompt(root / 'd.png'), depth)

    def test_dataset_nonzero_start_step_and_time_compensation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('meta', 'pcd', 'images'):
                (root / name).mkdir()
            calibration = {'left_camera': {'K': [[4., 0, 3], [0, 4., 3], [0, 0, 1]],
                                           'D': [0] * 5, 'width': 8, 'height': 8},
                           'camera_to_body': {'T_body_cam': np.eye(4).tolist()},
                           'lidar_to_body': {'T_body_lidar': np.eye(4).tolist()},
                           'time': {'origin_utc_us': 1000000000000000}}
            (root / 'meta/calib.yaml').write_text(yaml.safe_dump(calibration))
            with (root / 'meta/pairs.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['output_index', 'camera_utc_us', 'lidar_utc_us'])
                for i in range(4):
                    stamp = calibration['time']['origin_utc_us'] + i * 1000000
                    writer.writerow([i, stamp, stamp - 200000])
                    cv2.imwrite(str(root / 'images' / f'{i:05d}.png'), np.zeros((8, 8, 3), np.uint8))
                    write_pcd(root / 'pcd' / f'{i:05d}.pcd', [[0, 0, 2], [1, 0, 3]])
            trajectory = root / 'tracking.tum'
            trajectory.write_text('\n'.join(f'{i} {i * 2} 0 0 0 0 0 1' for i in range(4)))
            args = offline.parse_args(['--datadir', str(root), '--trajectory', str(trajectory),
                                       '--out', str(root / 'out'), '--start', '1', '--end', '4',
                                       '--frame-step', '2', '--prompt-width', '8'])
            samples = list(offline.load_dataset(args))
            self.assertEqual([info['output_index'] for _, _, info in samples], [1, 3])
            self.assertEqual(samples[0][2]['timestamp_us'], 1000000)
            # body moves +2m/s: scan point at optical axis shifts -0.4m at image time.
            self.assertEqual(samples[0][1][3, 2], 2.)
            args.trajectory = root / 'missing.tum'
            with self.assertRaises(FileNotFoundError):
                list(offline.load_dataset(args))


class OfflineOutputTests(unittest.TestCase):
    def test_single_image_outputs_and_refuses_overwrite(self):
        class DummyModel:
            def predict(self, image, prompt):
                assert image.shape[-1] % 14 == image.shape[-2] % 14 == 0
                return torch.full((1, 1, *image.shape[-2:]), 2.5, device=image.device)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / 'image.png'), np.zeros((81, 101, 3), np.uint8))
            np.save(root / 'prompt.npy', np.array([[1., 0], [0, 3.]], np.float32))
            (root / 'model.ckpt').write_bytes(b'only used for manifest hash')
            argv = ['--image', str(root / 'image.png'), '--prompt-depth', str(root / 'prompt.npy'),
                    '--checkpoint', str(root / 'model.ckpt'), '--out', str(root / 'out'),
                    '--device', 'cpu', '--max-size', '70', '--depth-format', 'npy',
                    '--output-downsampling', '1', '--save-preview', '--save-prompts']
            with patch.object(offline, 'load_model', return_value=DummyModel()):
                offline.main(argv)
            depth = np.load(root / 'out/depth/image.npy')
            self.assertEqual(depth.shape, (81, 101))
            np.testing.assert_array_equal(depth, 2.5)
            manifest = json.loads((root / 'out/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['frames'][0]['inference_wh'], [70, 56])
            self.assertTrue((root / 'out/prompt_valid/image.png').is_file())
            with self.assertRaises(FileExistsError):
                offline.main(argv)

    def test_compact_defaults_and_metric_png(self):
        class DummyModel:
            def predict(self, image, prompt):
                h, w = image.shape[-2:]
                return torch.linspace(1.0004, 40.1234, h * w).reshape(1, 1, h, w)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / 'image.png'), np.zeros((120, 192, 3), np.uint8))
            np.save(root / 'prompt.npy', np.array([[1., 0], [0, 3.]], np.float32))
            (root / 'model.ckpt').write_bytes(b'dummy')
            argv = ['--image', str(root / 'image.png'), '--prompt-depth', str(root / 'prompt.npy'),
                    '--checkpoint', str(root / 'model.ckpt'), '--out', str(root / 'out'),
                    '--device', 'cpu', '--max-size', '70']
            with patch.object(offline, 'load_model', return_value=DummyModel()):
                offline.main(argv)
            depth = cv2.imread(str(root / 'out/depth/image.png'), cv2.IMREAD_UNCHANGED)
            self.assertEqual(depth.shape, (30, 48))
            self.assertEqual(depth.dtype, np.uint16)
            self.assertEqual({p.name for p in (root / 'out').iterdir()}, {'depth', 'manifest.json'})
            manifest = json.loads((root / 'out/manifest.json').read_text())
            self.assertEqual(manifest['depth_scale_to_metres'], .01)
            self.assertEqual(manifest['frames'][0]['original_wh'], [192, 120])
            self.assertEqual(manifest['frames'][0]['output_wh'], [48, 30])
            reference, _ = offline.predict(DummyModel(), np.zeros((120, 192, 3), np.uint8),
                                           np.array([[1., 2.]], np.float32), offline.parse_args(argv))
            np.testing.assert_allclose(depth.astype(float) * .01, reference, atol=.00501)

    def test_png_rejects_overflow_and_preserves_npy_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = np.array([[2.1234, 700.]], np.float32)
            with self.assertRaisesRegex(ValueError, 'npy'):
                offline.save_depth(root / 'depth.png', depth, 'png')
            self.assertFalse((root / 'depth.png').exists())
            offline.save_depth(root / 'depth.npy', depth, 'npy')
            np.testing.assert_array_equal(np.load(root / 'depth.npy'), depth)

    def test_centimetre_png_stores_far_depth_without_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.png'
            depth = np.array([[2.1234, 70., 200., 655.35]], np.float32)
            offline.save_depth(path, depth, 'png')
            encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertEqual(encoded.dtype, np.uint16)
            np.testing.assert_array_equal(encoded, [[212, 7000, 20000, 65535]])
            np.testing.assert_allclose(encoded.astype(float) * .01, depth, atol=.00501)
            with self.assertRaisesRegex(ValueError, 'png-unit-m'):
                offline.save_depth(Path(directory) / 'overflow.png', np.array([[700.]]), 'png', .01)

    def test_missing_checkpoint_never_loads_random_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = offline.parse_args(['--image', 'x', '--prompt-depth', 'x.npy', '--out', 'out',
                                       '--checkpoint', str(Path(directory) / 'missing.ckpt')])
            with self.assertRaises(FileNotFoundError):
                offline.load_model(args)


if __name__ == '__main__':
    unittest.main()
