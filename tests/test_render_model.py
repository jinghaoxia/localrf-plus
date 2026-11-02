"""CPU checks for localTensoRF/render.py: checkpoint round trip, chunking, frame selection and outputs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "localTensoRF"))
import render  # noqa: E402
from local_tensorfs import LocalTensorfs  # noqa: E402

TINY_FIELD = {"gridSize": [8, 8, 8], "density_n_comp": [2, 2, 2], "appearance_n_comp": [2, 2, 2],
              "app_dim": 3, "shadingMode": "MLP_Fea_late_view", "near_far": [0.1, 100.],
              "density_shift": -10., "distance_scale": 25., "rayMarch_weight_thres": 1e-4,
              "pos_pe": 0, "view_pe": 0, "fea_pe": 0, "featureC": 8, "step_ratio": 0.5,
              "fea2denseAct": "softplus", "alphaMask_thres": 1e-4}
N_VIEWS = 4
WH = (16, 10)


def build_model():
    """Two fields, the second upsampled and the first alpha-masked, so the checkpoint exercises both paths."""
    model = LocalTensorfs(camera_prior=None, fov=85.6, n_init_frames=N_VIEWS, n_overlap=2, WH=WH,
                          n_iters_per_frame=2, n_iters_reg=1, lr_R_init=1e-3, lr_t_init=1e-3, lr_i_init=0,
                          lr_exposure_init=1e-3, rf_lr_init=1e-2, rf_lr_basis=1e-3, lr_decay_target_ratio=0.1,
                          N_voxel_list={}, update_AlphaMask_list=[], lr_upsample_reset=1, device="cpu",
                          scene_radius=2.0, aabb=torch.tensor([[-2.] * 3, [2.] * 3]), **TINY_FIELD)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(N_VIEWS):
            rotation = Rotation.from_rotvec(0.2 * rng.normal(size=3)).as_matrix()
            model.r_c2w[i].copy_(torch.tensor(rotation[:, :2], dtype=torch.float32))
            model.t_c2w[i].copy_(torch.tensor([0.5 * i - 0.7, 0.2 * i, -0.3], dtype=torch.float32))
            model.exposure[i].add_(0.05 * torch.tensor(rng.normal(size=(3, 3)), dtype=torch.float32))
        model.append_rf(n_added_frames=2)
        for tensorf in model.tensorfs:
            for coef in list(tensorf.density_plane) + list(tensorf.density_line):
                coef.data.uniform_(0, 2)
        model.tensorfs[1].upsample_volume_grid([12, 12, 12])
        model.tensorfs[0].updateAlphaMask((8, 8, 8))
    model.is_refining = True
    model.eval()
    return model


def wall_model(distance_m, scene_radius):
    """Field whose density is solid beyond z = -distance_m and empty in front of it; z in metres."""
    model = LocalTensorfs(camera_prior=None, fov=85.6, n_init_frames=1, n_overlap=1, WH=WH,
                          n_iters_per_frame=2, n_iters_reg=1, lr_R_init=1e-3, lr_t_init=1e-3, lr_i_init=0,
                          lr_exposure_init=0, rf_lr_init=1e-2, rf_lr_basis=1e-3, lr_decay_target_ratio=0.1,
                          N_voxel_list={}, update_AlphaMask_list=[], lr_upsample_reset=1, device="cpu",
                          scene_radius=scene_radius, aabb=torch.tensor([[-2.] * 3, [2.] * 3]),
                          **{**TINY_FIELD, "gridSize": [64, 64, 64]})
    field = model.tensorfs[0]
    grid = 64
    # Field units are scene_radius metres and the aabb spans [-2, 2] units, so grid index i sits at
    # z = 2 * scene_radius * (-1 + 2 i / (grid - 1)) metres.
    z_m = 2 * scene_radius * (-1 + 2 * torch.arange(grid, dtype=torch.float32) / (grid - 1))
    with torch.no_grad():
        for i in range(3):
            field.density_plane[i].zero_()
            field.density_line[i].zero_()
        field.density_plane[0].fill_(1.0)  # constant over (x, y), so the density depends on z only
        field.density_line[0][0, :, :, 0] = torch.where(z_m <= -distance_m, 20.0, -20.0)
    return model


class RenderModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = build_model()
        self.checkpoint = self.root / "checkpoints.th"
        self.model.save(self.checkpoint)

    def tearDown(self):
        self.tmp.cleanup()

    def make_dataset(self, n_images):
        data = self.root / "data"
        (data / "images").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        for i in range(n_images):
            cv2.imwrite(str(data / "images" / f"{i:05d}.png"), rng.integers(0, 255, (20, 32, 3), dtype=np.uint8))
        return data

    def test_checkpoint_round_trip_renders_identically(self):
        loaded = render.load_model(self.checkpoint, "cpu")
        self.assertEqual(len(loaded.tensorfs), 2)
        self.assertEqual(len(loaded.r_c2w), N_VIEWS)
        self.assertIsNotNone(loaded.tensorfs[0].alphaMask)
        for view in range(N_VIEWS):
            rgb, depth = render.render_view(self.model, view, ray_chunk=64)
            rgb_loaded, depth_loaded = render.render_view(loaded, view, ray_chunk=64)
            np.testing.assert_allclose(rgb_loaded, rgb, atol=1e-6)
            np.testing.assert_allclose(depth_loaded, depth, atol=1e-6)
        # The field must produce structure, not a flat background.
        self.assertGreater(rgb.std(), 1e-3)

    def test_ray_chunking_does_not_change_the_image(self):
        model = render.load_model(self.checkpoint, "cpu")
        whole, _ = render.render_view(model, 2, ray_chunk=WH[0] * WH[1])
        pieces, _ = render.render_view(model, 2, ray_chunk=37)
        np.testing.assert_allclose(pieces, whole, atol=1e-5)

    def test_selection_follows_the_training_order(self):
        data = self.make_dataset(6)
        self.assertEqual(render.select_frame_names(data, [0, -1], 1), [f"{i:05d}.png" for i in range(6)])
        self.assertEqual(render.select_frame_names(data, [1, 5], 2), ["00001.png", "00003.png"])

    def test_render_only_writes_rgb_and_depth_for_selected_views(self):
        out = self.root / "render"
        status = render.main(["--checkpoint", str(self.checkpoint), "--out", str(out), "--device", "cpu",
                                    "--stride", "2", "--save_depth", "--ray_chunk", "50"])
        self.assertEqual(status, 0)
        self.assertEqual(sorted(p.name for p in (out / "rgb").iterdir()), ["00000.png", "00002.png"])
        self.assertEqual(sorted(p.name for p in (out / "depth").iterdir()),
                         ["00000.npy", "00000.png", "00002.npy", "00002.png"])
        depth = np.load(out / "depth" / "00002.npy")
        self.assertEqual(depth.shape, (WH[1], WH[0]))
        # NaN exactly where the in-scene opacity is below --min_opacity (default 0.5); finite elsewhere.
        depth_sum, opacity = render.render_depth(render.load_model(self.checkpoint, "cpu"), 2, ray_chunk=50)
        np.testing.assert_allclose(depth, render.expected_depth(depth_sum, opacity, 0.5), rtol=1e-5, equal_nan=True)
        self.assertTrue(np.isfinite(depth[opacity >= 0.5]).all())

        # The saved PNG is the 8-bit version of the same render (allow one level for quantisation at chunk edges).
        expected, _ = render.render_view(render.load_model(self.checkpoint, "cpu"), 0)
        written = cv2.imread(str(out / "rgb" / "00000.png"))[..., ::-1].astype(int)
        self.assertLessEqual(np.abs(written - render.to_uint8(expected).astype(int)).max(), 1)

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual([f["view_index"] for f in metrics["frames"]], [0, 2])
        self.assertIsNone(metrics["psnr_db_mean"])

    def test_with_datadir_writes_comparisons_and_psnr(self):
        data = self.make_dataset(N_VIEWS)
        out = self.root / "render_gt"
        status = render.main(["--checkpoint", str(self.checkpoint), "--out", str(out),
                                    "--datadir", str(data), "--device", "cpu"])
        self.assertEqual(status, 0)
        self.assertEqual(sorted(p.name for p in (out / "compare").iterdir()),
                         [f"{i:05d}.png" for i in range(N_VIEWS)])
        # Panels are observed | rendering | error, each WH wide.
        self.assertEqual(cv2.imread(str(out / "compare" / "00000.png")).shape, (WH[1], 3 * WH[0], 3))
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual([f["name"] for f in metrics["frames"]], [f"{i:05d}" for i in range(N_VIEWS)])
        self.assertTrue(all(np.isfinite(f["psnr_db"]) for f in metrics["frames"]))
        self.assertAlmostEqual(metrics["psnr_db_mean"], np.mean([f["psnr_db"] for f in metrics["frames"]]))

    def test_selection_must_match_the_checkpoint_view_count(self):
        data = self.make_dataset(6)
        with self.assertRaisesRegex(SystemExit, "stores 4 views"):
            render.main(["--checkpoint", str(self.checkpoint), "--out", str(self.root / "bad"),
                               "--datadir", str(data), "--subsequence", "0", "3", "--device", "cpu"])
        self.assertFalse((self.root / "bad").exists())

    def test_refuses_to_write_into_a_non_empty_directory(self):
        out = self.root / "render"
        out.mkdir()
        (out / "keep.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "non-empty"):
            render.main(["--checkpoint", str(self.checkpoint), "--out", str(out), "--device", "cpu"])
        self.assertEqual((out / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_depth_is_metric_z_depth(self):
        # A solid wall 3 m straight ahead must read 3 m at every pixel. Range (along-ray) depth would read up to
        # about 40% more at the image corners, and a missing scene_radius factor would read 3 / 8 m.
        for scene_radius in (8.0, 4.0):
            with self.subTest(scene_radius=scene_radius):
                _, depth = render.render_view(wall_model(3.0, scene_radius), 0, ray_chunk=64)
                self.assertLess(abs(depth.mean() - 3.0), 0.25)
                self.assertLess(np.abs(depth - 3.0).max(), 0.5)

    def test_in_scene_depth_of_an_opaque_wall(self):
        # The forward pass adds the far boundary sample; on an opaque wall that sample carries no weight, so the
        # in-scene expected depth must match the renderer's depth and sit at the wall.
        model = wall_model(3.0, 8.0)
        _, depth_forward = render.render_view(model, 0, ray_chunk=64)
        depth_sum, opacity = render.render_depth(model, 0, ray_chunk=64)
        self.assertGreater(opacity.min(), 0.95)
        self.assertLess(np.abs(render.expected_depth(depth_sum, opacity, 0.5) - 3.0).max(), 0.5)
        self.assertLess(np.abs(depth_sum / opacity - depth_forward).max(), 0.05)

    def test_empty_field_has_no_in_scene_depth(self):
        # Negative density everywhere: nothing in the scene absorbs light, so the in-scene opacity is ~0 and the
        # expected depth must be NaN. The forward pass would report the far boundary (tens of metres) here.
        depth_sum, opacity = render.render_depth(wall_model(100.0, 8.0), 0, ray_chunk=64)
        self.assertLess(opacity.max(), 0.05)
        self.assertTrue(np.isnan(render.expected_depth(depth_sum, opacity, min_opacity=0.5)).all())

    def test_save_depth_writes_opacity_and_masks_transparent_pixels(self):
        out = self.root / "render_depth"
        status = render.main(["--checkpoint", str(self.checkpoint), "--out", str(out), "--device", "cpu",
                              "--stride", "3", "--save_depth", "--ray_chunk", "50"])
        self.assertEqual(status, 0)
        self.assertEqual(sorted(p.name for p in (out / "opacity").iterdir()), ["00000.png", "00003.png"])
        depth = np.load(out / "depth" / "00000.npy")
        self.assertEqual(depth.shape, (WH[1], WH[0]))
        self.assertEqual(depth.dtype, np.float32)
        colour = cv2.imread(str(out / "depth" / "00000.png"))
        self.assertTrue((colour[~np.isfinite(depth)] == 0).all())

if __name__ == "__main__":
    unittest.main()
