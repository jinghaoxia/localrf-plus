"""Synthetic geometry checks for the standalone tracking frontend; no datasets or GT."""
import csv
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import estimate_pose as tracking


def pose(rotvec, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = translation
    return result


class TrackingCalibrationTests(unittest.TestCase):
    def test_load_sequence_uses_resized_pixel_centres(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('images', 'pcd', 'meta'):
                (root / name).mkdir()
            K = np.array([[1015., 0., 959.35], [0., 1001., 623.28], [0., 0., 1.]])
            calibration = dict(
                left_camera=dict(width=1920, height=1200, K=K.tolist(), D=[0]*4),
                camera_to_body=dict(T_body_cam=np.eye(4).tolist()),
                lidar_to_body=dict(T_body_lidar=np.eye(4).tolist()),
                time=dict(origin_utc_us=0), pcd_motion_compensation='none', pcd_indexing='output_index')
            (root / 'meta/calib.yaml').write_text(yaml.safe_dump(calibration))
            columns = ('output_index', 'camera_index', 'lidar_index', 'camera_utc_us',
                       'lidar_utc_us', 'lidar_minus_camera_us')
            with (root / 'meta/pairs.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for i in range(2):
                    writer.writerow(dict(zip(columns, [i, i, i, i*500000, i*500000, 0])))
                    cv2.imwrite(str(root / 'images' / f'{i:05d}.png'), np.zeros((1200, 1920), np.uint8))
                    (root / 'pcd' / f'{i:05d}.pcd').touch()  # Loading calibration does not read scans.
            data = tracking.load_sequence(root)
            xyz = np.array([[0., 0., 2.], [1.1, -.7, 4.], [-2.1, 1.3, 6.]])
            full = tracking.project_points(xyz, K)
            small = tracking.project_points(xyz, data.camera_matrix)
            np.testing.assert_allclose(small, (full + .5) / 3 - .5, atol=1e-12)
            self.assertEqual(data.read_image(0).shape, (400, 640))
            np.testing.assert_allclose(data.camera_matrix[:2, 2] - K[:2, 2]/3, -1/3)


class LidarExtrinsicTests(unittest.TestCase):
    def setUp(self):
        self.T_body_lidar = pose([.4, -.3, .2], [.7, -.2, 1.3])
        self.T_world_body = pose([-.2, .1, .5], [3., -2., 1.])
        self.T_world_lidar = self.T_world_body @ self.T_body_lidar

    def test_rotated_extrinsic_translation_matches_full_transform_composition(self):
        offset = tracking.lidar_offset_in_world(self.T_world_lidar, self.T_body_lidar)
        # The prediction's lidar translation is body translation plus this world-frame offset.
        np.testing.assert_allclose(self.T_world_body[:3, 3] + offset, self.T_world_lidar[:3, 3], atol=1e-12)
        body = self.T_world_lidar @ tracking.invert_pose(self.T_body_lidar)
        np.testing.assert_allclose(self.T_world_lidar[:3, 3] - offset, body[:3, 3], atol=1e-12)
        naive = self.T_world_lidar[:3, :3] @ self.T_body_lidar[:3, 3]
        self.assertGreater(np.linalg.norm(offset - naive), .1)

    def test_identity_extrinsic_rotation_keeps_existing_result(self):
        extrinsic = self.T_body_lidar.copy()
        extrinsic[:3, :3] = np.eye(3)
        np.testing.assert_allclose(tracking.lidar_offset_in_world(self.T_world_lidar, extrinsic),
                                   self.T_world_lidar[:3, :3] @ extrinsic[:3, 3], atol=1e-12)

    def test_visual_prior_residual_and_jacobian_with_rotated_extrinsic(self):
        previous = pose([.2, .3, -.7], [1., 2., 3.])
        expected = self.T_world_body.copy()
        expected[:3, 3] += [.3, -.1, .2]
        residual, jacobian = tracking.visual_translation_residual(
            self.T_world_lidar, previous, expected, self.T_body_lidar)
        body = self.T_world_lidar @ tracking.invert_pose(self.T_body_lidar)
        np.testing.assert_allclose(residual, previous[:3, :2].T @ (body[:3, 3]-expected[:3, 3]), atol=1e-12)
        eps = 1e-6
        numeric = np.zeros((2, 6))
        for axis in range(6):
            differences = []
            for sign in (1, -1):
                update = np.eye(6)[axis] * eps * sign
                perturbed = self.T_world_lidar.copy()
                perturbed[:3, :3] = Rotation.from_rotvec(update[:3]).as_matrix() @ perturbed[:3, :3]
                perturbed[:3, 3] += update[3:]
                value, _ = tracking.visual_translation_residual(perturbed, previous, expected, self.T_body_lidar)
                differences.append(value)
            numeric[:, axis] = (differences[0]-differences[1])/(2*eps)
        np.testing.assert_allclose(jacobian, numeric, atol=5e-10)

    def test_motion_interpolation_and_scan_to_image_time(self):
        motion = pose([.05, -.1, .3], [.7, .2, -.1])
        half = tracking.interpolate_motion(motion, .5)
        np.testing.assert_allclose(half @ half, motion, atol=1e-12)
        np.testing.assert_allclose(tracking.interpolate_motion(motion, 0), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(tracking.interpolate_motion(motion, -1), np.linalg.inv(motion), atol=1e-12)
        previous = self.T_world_lidar
        current = previous @ motion
        result = tracking.scan_pose_to_image_time(previous, current, 1., 2., 2.5, self.T_body_lidar)
        np.testing.assert_allclose(result, current @ half @ np.linalg.inv(self.T_body_lidar), atol=1e-12)
        with self.assertRaises(ValueError):
            tracking.scan_pose_to_image_time(previous, current, 2., 2., 2.5, self.T_body_lidar)


class ForegroundCandidateTests(unittest.TestCase):
    def test_foreground_box_is_evaluated_in_body_coordinates(self):
        body_points = np.array([[5., 0., 1.], [5., 0., -.2], [5., 2., 1.], [15., 0., 1.]])
        # Keep LiDAR XY ranges within the scanner's independent [2,35] range filter.
        extrinsic = pose([0., 0., np.pi/2], [.2, -.3, 1.3])
        raw = (body_points-extrinsic[:3,3]) @ extrinsic[:3,:3]
        with patch.object(tracking, 'voxel_downsample', side_effect=lambda points, _: points):
            selected = tracking.prepare_fusion_points(raw, extrinsic)
        np.testing.assert_allclose(selected, raw[1:])

    def test_height_uses_rotated_body_z_not_raw_lidar_z(self):
        extrinsic = pose([0., np.pi/4, 0.], [0., 0., 1.3])
        body_points = np.array([[6., 0., .8], [6., 0., -.5]])
        raw = (body_points-extrinsic[:3,3]) @ extrinsic[:3,:3]
        # Both points have high raw LiDAR z; only the first lies above the body-height threshold.
        self.assertTrue(((raw[:,2]+1.3)>.35).all())
        with patch.object(tracking, 'voxel_downsample', side_effect=lambda points, _: points):
            selected = tracking.prepare_fusion_points(raw, extrinsic)
        np.testing.assert_allclose(selected, raw[1:])

    def test_identity_rotation_and_z_translation_preserve_211_filter(self):
        rng = np.random.default_rng(1)
        points = rng.uniform([-40.,-20.,-3.], [40.,20.,4.], (500,3))
        extrinsic = pose([0.,0.,0.], [0.,0.,1.29])
        radius = np.linalg.norm(points[:,:2],axis=1)
        front = (points[:,0]>2)&(points[:,0]<12)&(abs(points[:,1])<1.3)&(points[:,2]+1.29>.35)
        expected = points[(radius>2)&(radius<35)&~front]
        with patch.object(tracking, 'voxel_downsample', side_effect=lambda points, _: points):
            actual = tracking.prepare_fusion_points(points, extrinsic)
        np.testing.assert_array_equal(actual, expected)


class RegistrationAcceptanceTests(unittest.TestCase):
    def test_point_to_plane_registration_with_rotated_lidar_extrinsic(self):
        rng = np.random.default_rng(2)
        uv = rng.uniform(-2., 2., (900, 2))
        scene = np.vstack((np.column_stack((uv, np.zeros(900))),
                           np.column_stack((np.full(900, 4.), uv)),
                           np.column_stack((uv[:, 0], np.full(900, 4.), uv[:, 1]))))
        expected_scan = pose([.02, -.01, .03], [.1, -.05, .08])
        source = (scene - expected_scan[:3, 3]) @ expected_scan[:3, :3]
        extrinsic = pose([.3, -.2, .4], [.7, -.2, 1.3])
        expected_body = expected_scan @ tracking.invert_pose(extrinsic)
        initial = pose([.01, -.005, .02], [.08, -.04, .07])
        result, stats = tracking.register_scan_to_map(source, scene, initial, np.eye(4),
                                                      expected_body, extrinsic, .005)
        self.assertTrue(stats['success'])
        # Statistics describe the residual before the final <1 mm / <1e-4 rad step.
        self.assertLess(stats['rmse_m'], 1e-3)
        np.testing.assert_allclose(result, expected_scan, atol=1e-4)

    def test_registration_with_insufficient_correspondences_keeps_initial_pose(self):
        rng = np.random.default_rng(3)
        points = rng.normal(size=(100, 3))
        initial = pose([0., 0., .02], [.01, 0., 0.])
        result, stats = tracking.register_scan_to_map(points, points, initial, np.eye(4),
                                                      np.eye(4), np.eye(4), None)
        self.assertFalse(stats['success'])
        np.testing.assert_array_equal(result, initial)

    def test_rejects_bad_fit_and_large_corrections(self):
        valid = dict(success=True, rmse_m=.05, inlier_ratio=.9)
        identity = np.eye(4)
        self.assertTrue(tracking.accept_registration(valid, identity, identity, .5))
        for invalid in (dict(success=False), dict(rmse_m=.15), dict(inlier_ratio=.64)):
            self.assertFalse(tracking.accept_registration({**valid, **invalid}, identity, identity, .5))
        self.assertFalse(tracking.accept_registration(valid, identity, pose([0, 0, 0], [1.1, 0, 0]), .5))
        self.assertTrue(tracking.accept_registration(valid, identity, pose([0, 0, 0], [1.1, 0, 0]), 1.5))
        self.assertFalse(tracking.accept_registration(valid, identity, pose([0, 0, np.deg2rad(6)], [0, 0, 0]), .5))
        self.assertTrue(tracking.accept_registration(valid, identity, pose([0, 0, np.deg2rad(6)], [0, 0, 0]), 1.5))


if __name__ == '__main__':
    unittest.main()
