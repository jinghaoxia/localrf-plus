"""Render a neural-BA checkpoint at the views it stores.

Each view saved in checkpoints.th is rendered at its optimised pose through
LocalTensorfs.forward in inference mode: no sampling jitter, the saved blending
weights across fields, and the saved exposure. Output goes to a new directory:

  rgb/NAME.png       rendering, 8-bit RGB
  compare/NAME.png   observed | rendering | 4x amplified |difference|   (needs --datadir)
  depth/NAME.npy     expected z-depth in metres, float32, NaN where opacity < --min_opacity  (--save_depth)
  depth/NAME.png     colour-mapped depth, near blue to far red, black where NaN            (--save_depth)
  opacity/NAME.png   accumulated opacity, white = opaque                                    (--save_depth)
                     Depth is sum(weight * z) / sum(weight), i.e. the opacity-weighted mean z along the
                     optical axis (not range). The raw volume-rendering sum is divided by the accumulated
                     opacity, which the renderer does not return, so it is recovered from the white-background
                     difference (see render_view). Metres for checkpoints trained with --trajectory
                     (scene_radius = --scene_radius_m); checkpoints trained without it store normalised poses.
  metrics.json       per-frame PSNR against the observed image           (needs --datadir)

With --datadir, frame names come from the dataset. Pass the same --subsequence
and --frame_step as the training run: the checkpoint does not record them, and
the view count is checked against the selection. Without --datadir, views are
named by their index in the checkpoint.

These renders compare the field with the frames it was fitted to. They are not
a held-out evaluation.

Example:
    python localTensoRF/render.py --checkpoint RUN/checkpoints.th \\
        --datadir /path/to/dataprocess/211 --subsequence 0 16 --frame_step 1 \\
        --out RUN/render --device cuda:0
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from local_tensorfs import LocalTensorfs, ids2pixel
from utils.ray_utils import get_ray_directions_lean, get_rays_lean


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render a neural-BA checkpoint at its stored views.")
    parser.add_argument("--checkpoint", required=True, help="checkpoints.th written by train.py")
    parser.add_argument("--out", required=True, help="output directory; must not exist or must be empty")
    parser.add_argument("--datadir", default=None,
                        help="dataset root used for training; adds observed images, comparison panels and PSNR")
    parser.add_argument("--subsequence", type=int, nargs=2, default=[0, -1], metavar=("START", "END"),
                        help="same [start, end) selection as the training run")
    parser.add_argument("--frame_step", type=int, default=1, help="same frame step as the training run")
    parser.add_argument("--stride", type=int, default=1, help="render every STRIDE-th stored view")
    parser.add_argument("--ray_chunk", type=int, default=4096, help="rays per forward call; lower it to reduce memory")
    parser.add_argument("--save_depth", action="store_true",
                        help="also write opacity-normalised depth as .npy (metres) and a colour PNG, plus opacity PNG")
    parser.add_argument("--min_opacity", type=float, default=0.5,
                        help="with --save_depth, pixels whose accumulated opacity is below this are NaN / black")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if min(args.stride, args.ray_chunk, args.frame_step) < 1:
        parser.error("--stride, --ray_chunk and --frame_step must be at least 1")
    return args


def load_model(checkpoint, device):
    """Rebuild the trained LocalTensorfs from a train.py checkpoint, ready for inference."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    # The stored kwargs give the first field's grid size as saved; LocalTensorfs.load() creates the
    # remaining fields and frames from the state dict.
    kwargs = dict(ckpt["kwargs"], device=device)
    with contextlib.redirect_stdout(io.StringIO()):  # TensorBase prints its configuration for every field
        model = LocalTensorfs(**kwargs)
        model.load(ckpt["state_dict"])
    model.to(device)
    for tensorf in model.tensorfs:
        tensorf.to(device)  # nn.Module.to() skips TensorBase.to(), which keeps .device and stepSize in sync
    model.eval()
    # TensorBase.forward defaults to refine=True, which the upstream renderer relied on.
    # The flag only changes the MLP input when fea_pe > 0.
    model.is_refining = True
    return model


@torch.no_grad()
def render_view(model, view_index, ray_chunk=4096):
    """Render one stored view. Returns rgb [H, W, 3] in [0, 1] and depth [H, W] in metres (see render_depth)."""
    if float(model.blending_weights[view_index].sum()) <= 0:
        raise RuntimeError(f"view {view_index} has no field with a positive blending weight")
    W, H = model.W, model.H
    device = model.device
    view_ids = torch.tensor([view_index], dtype=torch.long, device=device)
    rgb_chunks, depth_chunks = [], []
    for start in range(0, W * H, ray_chunk):
        ray_ids = torch.arange(start, min(start + ray_chunk, W * H), dtype=torch.long, device=device)
        rgb, depth, _, _ = model(ray_ids, view_ids, W, H, is_train=False)
        rgb_chunks.append(rgb)
        depth_chunks.append(depth)
    rgb = torch.cat(rgb_chunks).reshape(H, W, 3).cpu().numpy()
    depth = torch.cat(depth_chunks).reshape(H, W).cpu().numpy()
    return rgb, depth


@torch.no_grad()
def render_depth(model, view_index, ray_chunk=4096):
    """In-scene weight sum and depth sum for one stored view: returns (depth_sum [H, W] m, opacity [H, W]).

    LocalTensorfs.forward cannot be used for depth directly. TensorBase.alpha2weights forces the last sample of
    every ray opaque, so the returned depth includes the far end of the ray even where nothing in the scene is
    hit. Here the same samples are weighted but the final one is left out, and the fields are blended with the
    saved blending weights as in LocalTensorfs.forward. opacity is the in-scene accumulated opacity;
    depth_sum / opacity is the expected depth of the in-scene surface (see expected_depth).
    """
    if model.fov == 360:
        raise NotImplementedError("in-scene depth is implemented for the perspective camera only")
    if float(model.blending_weights[view_index].sum()) <= 0:
        raise RuntimeError(f"view {view_index} has no field with a positive blending weight")
    W, H = model.W, model.H
    device = model.device
    view_ids = torch.tensor([view_index], dtype=torch.long, device=device)
    cam2world = model.get_cam2world(view_ids)
    blending = model.blending_weights[view_ids]
    depth_sum = torch.zeros(H * W, device=device)
    opacity = torch.zeros(H * W, device=device)
    for start in range(0, W * H, ray_chunk):
        ray_ids = torch.arange(start, min(start + ray_chunk, W * H), dtype=torch.long, device=device)
        i, j = ids2pixel(W, H, ray_ids)
        directions = get_ray_directions_lean(i, j, model.focal(W), model.center(W, H))
        for rf_id, tensorf in enumerate(model.tensorfs):
            weight = blending[:, rf_id]
            if float(weight) <= 0:
                continue
            cam2rf = cam2world.clone()
            cam2rf[:, :3, 3] += model.world2rf[rf_id]
            cam2rf = cam2rf.repeat_interleave(ray_ids.shape[0], dim=0)
            rays_o, rays_d = get_rays_lean(directions, cam2rf)
            rays = torch.cat([rays_o / model.scene_radius, rays_d], -1).view(-1, 6)
            raw, acc = _in_scene_weights(tensorf, rays)
            depth_sum[start:start + ray_ids.shape[0]] += weight * raw * model.scene_radius
            opacity[start:start + ray_ids.shape[0]] += weight * acc
    return depth_sum.reshape(H, W).cpu().numpy(), opacity.reshape(H, W).cpu().numpy()


def _in_scene_weights(tensorf, rays):
    """Per-ray in-scene weight sum and sum(weight * z) / |viewdir|, for one field. Mirrors TensorBase.forward
    up to the density, without the far sample that alpha2weights forces opaque. z is in field units."""
    viewdirs = rays[:, 3:6]
    viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = viewdirs / viewdirs_norm
    xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
        rays[:, :3], viewdirs, is_train=False, N_samples=-1)
    dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
    sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
    if tensorf.alphaMask is not None:
        alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
        alpha_mask = alphas > 0
        ray_invalid = ~ray_valid
        ray_invalid[ray_valid] |= (~alpha_mask)
        ray_valid = ~ray_invalid
    ray_valid[:, -1] = 0
    if ray_valid.any():
        xyz_sampled = tensorf.normalize_coord(xyz_sampled)
        sigma_feature = tensorf.compute_densityfeature(xyz_sampled[ray_valid])
        sigma[ray_valid] = tensorf.feature2density(sigma_feature)
    alpha = 1.0 - torch.exp(-sigma * dists * tensorf.distance_scale)
    alpha = alpha[:, :-1]  # drop the forced-opaque far sample
    ones = torch.ones(alpha.shape[0], 1, device=alpha.device)
    transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    weight = alpha * transmittance
    depth_sum = torch.sum(weight * z_vals[:, :-1], -1) / viewdirs_norm[:, 0]
    return depth_sum, torch.sum(weight, -1)


def expected_depth(depth_sum, opacity, min_opacity):
    """Expected in-scene depth in metres: depth_sum / opacity. NaN where the ray hits too little of the scene."""
    out = np.full(depth_sum.shape, np.nan, dtype=np.float32)
    ok = opacity >= min_opacity
    out[ok] = depth_sum[ok] / opacity[ok]
    return out


def select_frame_names(datadir, subsequence, frame_step):
    """Image names in the order train.py selects them: sorted, then [start:end], then every frame_step-th."""
    names = sorted(p.name for p in (Path(datadir) / "images").iterdir()
                   if p.is_file() and p.suffix.lower() == ".png")
    if list(subsequence) != [0, -1]:
        names = names[subsequence[0]:subsequence[1]]
    return names[::frame_step]


def load_observed(path, wh):
    """Observed frame as RGB float in [0, 1], resized to the training resolution the way the loader does."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read {path}")
    rgb = bgr[..., ::-1].astype(np.float32) / 255
    if rgb.shape[1::-1] != tuple(wh):
        rgb = cv2.resize(rgb, tuple(wh), interpolation=cv2.INTER_AREA)
    return rgb


def psnr(rendered, observed):
    mse = float(np.mean((rendered - observed) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))  # capped at 100 dB for identical images


def to_uint8(x):
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def write_png(path, bgr):
    if not cv2.imwrite(str(path), np.ascontiguousarray(bgr)):
        raise OSError(f"cannot write {path}")


def depth_to_colour(depth):
    """Near is blue, far is red. Pixels without a depth (NaN) are black."""
    valid = np.isfinite(depth) & (depth > 0)
    lo, hi = np.percentile(depth[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    scaled = np.clip((np.nan_to_num(depth, nan=lo) - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    colour = cv2.applyColorMap(to_uint8(scaled), cv2.COLORMAP_JET)
    colour[~valid] = 0
    return colour


def comparison_panel(observed, rendered):
    error = np.clip(np.abs(observed - rendered) * 4.0, 0.0, 1.0)
    return to_uint8(np.concatenate([observed, rendered, error], axis=1))[..., ::-1]


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty {out}; choose a new --out")

    model = load_model(args.checkpoint, args.device)
    W, H = model.W, model.H
    n_views = len(model.r_c2w)
    names = None
    if args.datadir:
        names = select_frame_names(args.datadir, args.subsequence, args.frame_step)
        if len(names) != n_views:
            raise SystemExit(f"checkpoint stores {n_views} views but the selection has {len(names)} frames; "
                             "pass the --subsequence and --frame_step used for training")
        stems = [Path(name).stem for name in names]
    else:
        stems = [f"{k:05d}" for k in range(n_views)]
    print(f"{args.checkpoint}: {n_views} views, {len(model.tensorfs)} fields, {W}x{H}, device {args.device}",
          flush=True)

    dirs = {"rgb": out / "rgb"}
    if names is not None:
        dirs["compare"] = out / "compare"
    if args.save_depth:
        dirs["depth"] = out / "depth"
        dirs["opacity"] = out / "opacity"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    frames = []
    for view in range(0, n_views, args.stride):
        stem = stems[view]
        rgb, _ = render_view(model, view, args.ray_chunk)
        if args.save_depth:
            depth_sum, acc = render_depth(model, view, args.ray_chunk)
            depth = expected_depth(depth_sum, acc, args.min_opacity)
        write_png(dirs["rgb"] / f"{stem}.png", to_uint8(rgb)[..., ::-1])
        record = {"name": stem, "view_index": view}
        if names is not None:
            observed = load_observed(Path(args.datadir) / "images" / names[view], (W, H))
            record["psnr_db"] = float(psnr(rgb, observed))
            write_png(dirs["compare"] / f"{stem}.png", comparison_panel(observed, rgb))
        if args.save_depth:
            np.save(dirs["depth"] / f"{stem}.npy", depth.astype(np.float32))
            write_png(dirs["depth"] / f"{stem}.png", depth_to_colour(depth))
            write_png(dirs["opacity"] / f"{stem}.png", to_uint8(acc))
        frames.append(record)
        suffix = f"  psnr {record['psnr_db']:.2f} dB" if "psnr_db" in record else ""
        print(f"[{len(frames)}] {stem}{suffix}", flush=True)

    psnrs = [f["psnr_db"] for f in frames if "psnr_db" in f]
    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "datadir": os.path.abspath(args.datadir) if args.datadir else None,
        "resolution": [W, H],
        "views_in_checkpoint": n_views,
        "stride": args.stride,
        "device": args.device,
        "psnr_db_mean": float(np.mean(psnrs)) if psnrs else None,
        "psnr_note": "rendering vs observed image of each selected frame; a fit measure, not a held-out evaluation",
        "frames": frames,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {len(frames)} renders to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
