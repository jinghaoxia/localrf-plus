#!/usr/bin/env python3
"""Offline PromptDA inference; see docs/promptda-offline.md. No training or GT access."""
import argparse
import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'localTensoRF'))
from dataLoader.tracking_poses import _check_transform, match_frames, read_tum


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--datadir', type=Path, help='images/, pcd/, meta/{calib.yaml,pairs.csv}')
    source.add_argument('--image', type=Path, help='single RGB/grayscale image')
    parser.add_argument('--prompt-depth', type=Path, help='single-image prompt: .npy metres or uint16 PNG millimetres')
    parser.add_argument('--trajectory', type=Path, help='explicit image-time body-frame tracking TUM, metres')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'PromptDA/models/model.ckpt')
    parser.add_argument('--promptda-root', type=Path, default=ROOT / 'PromptDA')
    parser.add_argument('--encoder', choices=['vitl', 'vits'], default='vitl')
    parser.add_argument('--out', type=Path, required=True, help='new directory (never overwrite an existing run)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None, help='exclusive; omitted means sequence end')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--max-size', type=int, default=1008, help='inference image longest side, rounded down to /14')
    parser.add_argument('--output-downsampling', type=int, default=4,
                        help='output width/height divided by this factor; 1 keeps original resolution')
    parser.add_argument('--depth-format', choices=['png', 'npy'], default='png',
                        help='png: uint16 with configurable --png-unit-m; npy: float32 metres')
    parser.add_argument('--png-unit-m', type=float, default=0.01,
                        help='metres per PNG integer unit (default .01 = centimetres, range 655.35m)')
    parser.add_argument('--save-preview', action='store_true', help='save a small RGB/depth comparison')
    parser.add_argument('--save-prompts', action='store_true', help='save prompt arrays, support mask and distance')
    parser.add_argument('--prompt-width', type=int, default=256, help='LiDAR prompt grid width')
    parser.add_argument('--min-depth', type=float, default=0.8)
    parser.add_argument('--max-depth', type=float, default=60.0)
    parser.add_argument('--max-time-offset', type=float, default=0.25, help='maximum scan/image offset and endpoint extrapolation, seconds')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    parser.add_argument('--amp', action='store_true', help='optional CUDA float16 inference (off by default)')
    args = parser.parse_args(argv)
    if args.datadir and (args.trajectory is None or args.prompt_depth is not None):
        parser.error('--datadir requires --trajectory and does not accept --prompt-depth')
    if args.image and (args.prompt_depth is None or args.trajectory is not None):
        parser.error('--image requires --prompt-depth and does not accept --trajectory')
    if (args.start < 0 or args.frame_step < 1 or args.max_size < 56 or args.prompt_width < 2
            or args.threads < 1 or args.output_downsampling < 1 or (args.end is not None and args.end <= args.start)):
        parser.error('invalid frame range, size, step or thread count')
    if not (0 < args.min_depth < args.max_depth < float('inf')) or not (0 < args.max_time_offset < float('inf')):
        parser.error('require finite 0 < min-depth < max-depth and positive max-time-offset')
    if not (0 < args.png_unit_m < float('inf')):
        parser.error('--png-unit-m must be finite and positive')
    return args


def read_pcd_xyz(path):
    """Strict binary XYZ+intensity reader for this dataset; accept only zero padding."""
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'{path}: missing DATA header')
            parts = line.decode('ascii').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0]] = parts[1:]
            if parts[0] == 'DATA':
                break
        expected = {'FIELDS': ['x', 'y', 'z', 'intensity'], 'SIZE': ['4'] * 4,
                    'TYPE': ['F'] * 4, 'COUNT': ['1'] * 4, 'DATA': ['binary']}
        if any(header.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{path}: expected binary float32 XYZ+intensity PCD')
        count = int(header['POINTS'][0])
        if count <= 0:
            raise ValueError(f'{path}: no points')
        payload = stream.read(count * 16)
        if len(payload) != count * 16 or any(stream.read()):
            raise ValueError(f'{path}: truncated data or nonzero trailing payload')
    return np.frombuffer(payload, dtype='<f4').reshape(count, 4)[:, :3].astype(np.float64)


def pose_at(times_us, poses, query_us, max_extrapolation_s):
    """Linear world translation + shortest-arc rotation interpolation; bounded endpoint extrapolation."""
    outside_us = max(int(times_us[0]) - query_us, query_us - int(times_us[-1]), 0)
    if outside_us > max_extrapolation_s * 1e6:
        raise ValueError('scan timestamp outside tracking coverage / extrapolation limit')
    i = int(np.clip(np.searchsorted(times_us, query_us, side='right') - 1, 0, len(times_us) - 2))
    fraction = (query_us - int(times_us[i])) / int(times_us[i + 1] - times_us[i])
    first, second = poses[i:i + 2]
    delta = Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).as_rotvec()
    result = np.eye(4)
    result[:3, :3] = first[:3, :3] @ Rotation.from_rotvec(fraction * delta).as_matrix()
    result[:3, 3] = first[:3, 3] + fraction * (second[:3, 3] - first[:3, 3])
    return result


def project_prompt(points, T_cam_lidar, K, image_wh, prompt_wh, min_depth, max_depth):
    """Z-buffer in OpenCV integer-centre pixels, with half-pixel resize alignment."""
    points = points[np.isfinite(points).all(axis=1)]
    camera = points @ T_cam_lidar[:3, :3].T + T_cam_lidar[:3, 3]
    camera = camera[(camera[:, 2] >= min_depth) & (camera[:, 2] <= max_depth)]
    width, height = prompt_wh
    sparse = np.full((height, width), np.inf, dtype=np.float32)
    if len(camera):
        pixels = camera @ K.T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        scale = np.array(prompt_wh) / np.array(image_wh)
        pixels = (pixels + 0.5) * scale - 0.5
        indices = np.floor(pixels + 0.5).astype(np.int64)
        x, y = indices.T
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.minimum.at(sparse, (y[valid], x[valid]), camera[valid, 2])
    sparse[~np.isfinite(sparse)] = 0
    return sparse


def fill_prompt(sparse):
    """Nearest valid pixel fill, including outside support; not a confidence estimate."""
    sparse = np.asarray(sparse, dtype=np.float32)
    if sparse.ndim != 2:
        raise ValueError('prompt depth must be a 2D array')
    valid = np.isfinite(sparse) & (sparse > 0)
    if valid.sum() < 2:
        raise ValueError('prompt needs at least two valid pixels')
    values = sparse[valid]
    if float(values.max() - values.min()) <= 1e-6:
        raise ValueError('constant prompt: PromptDA min/max normalization would divide by zero')
    distances, nearest = distance_transform_edt(~valid, return_indices=True)
    dense = sparse[tuple(nearest)].astype(np.float32)
    return dense, valid, distances.astype(np.float32)


def read_prompt(path):
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False).astype(np.float32)
    if path.suffix.lower() == '.png':
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None or value.dtype != np.uint16 or value.ndim != 2:
            raise ValueError('prompt PNG must be single-channel uint16 millimetres')
        return value.astype(np.float32) / 1000.0
    raise ValueError('prompt must be .npy (metres) or uint16 .png (millimetres)')


def load_dataset(args):
    root = args.datadir
    calibration = yaml.safe_load((root / 'meta/calib.yaml').read_text())
    camera = calibration['left_camera']
    K = np.asarray(camera['K'], dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or not np.allclose(camera['D'], 0)):
        raise ValueError('requires valid rectified pinhole calibration')
    T_body_cam = _check_transform(calibration['camera_to_body']['T_body_cam'], 'T_body_cam')
    T_body_lidar = _check_transform(calibration['lidar_to_body']['T_body_lidar'], 'T_body_lidar')
    origin_us = int(calibration['time']['origin_utc_us'])
    with (root / 'meta/pairs.csv').open(newline='') as stream:
        rows = [{key: int(row[key]) for key in ('output_index', 'camera_utc_us', 'lidar_utc_us')}
                for row in csv.DictReader(stream)]
    if len(rows) < 2 or [r['output_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('pairs.csv must contain >=2 frames, contiguous output_index from 0')
    times = np.array([row['camera_utc_us'] - origin_us for row in rows], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError('camera timestamps must be strictly increasing')
    matched = match_frames(times.tolist(), read_tum(args.trajectory))
    poses = np.stack([matched[int(t)] for t in times])
    if not np.allclose(poses[0], np.eye(4), atol=1e-6):
        raise ValueError('tracking frame 0 must be the identity body pose')
    end = len(rows) if args.end is None else args.end
    if args.start >= len(rows) or end > len(rows):
        raise ValueError('frame selection outside pairs.csv')
    image_wh = (int(camera['width']), int(camera['height']))
    if min(image_wh) <= 0:
        raise ValueError('invalid calibrated image dimensions')
    prompt_wh = (args.prompt_width, max(2, round(args.prompt_width * image_wh[1] / image_wh[0])))
    for row in rows[args.start:end:args.frame_step]:
        index = row['output_index']
        image = root / 'images' / f'{index:05d}.png'
        scan = root / 'pcd' / f'{index:05d}.pcd'
        if not image.is_file() or not scan.is_file():
            raise FileNotFoundError(f'missing image/PCD for frame {index}')
        offset_s = (row['lidar_utc_us'] - row['camera_utc_us']) / 1e6
        if abs(offset_s) > args.max_time_offset:
            raise ValueError(f'frame {index}: scan/image offset {offset_s}s exceeds limit')
        scan_time = row['lidar_utc_us'] - origin_us
        T_world_body_scan = pose_at(times, poses, scan_time, args.max_time_offset)
        T_cam_lidar = np.linalg.inv(poses[index] @ T_body_cam) @ T_world_body_scan @ T_body_lidar
        sparse = project_prompt(read_pcd_xyz(scan), T_cam_lidar, K, image_wh, prompt_wh,
                                args.min_depth, args.max_depth)
        yield image, sparse, {'output_index': index, 'timestamp_us': int(times[index]),
                              'scan_image_offset_s': offset_s, 'calibrated_wh': list(image_wh),
                              'scan_pose_extrapolated': bool(scan_time < times[0] or scan_time > times[-1])}


def load_model(args):
    # Only a local file is accepted. Never fall back to random parameters / Hub download.
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.promptda_root / 'promptda/promptda.py').is_file():
        raise FileNotFoundError(f'PromptDA source missing: {args.promptda_root}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    source_root = str(args.promptda_root.resolve())
    sys.path.insert(0, source_root)
    # Upstream is a namespace package; scripts/promptda.py otherwise shadows it.
    spec = importlib.machinery.PathFinder.find_spec('promptda', [source_root])
    sys.modules['promptda'] = importlib.util.module_from_spec(spec)
    from promptda.promptda import PromptDA
    # Upstream constructor loads a local checkpoint and constructs DINOv2 with pretrained=False.
    return PromptDA(encoder=args.encoder, ckpt_path=str(args.checkpoint.resolve())).to(args.device).eval()


def predict(model, bgr, prompt, args):
    height, width = bgr.shape[:2]
    scale = min(1.0, args.max_size / max(height, width))
    wh = (max(14, int(width * scale) // 14 * 14), max(14, int(height * scale) // 14 * 14))
    rgb = cv2.resize(bgr, wh, interpolation=cv2.INTER_AREA)[..., ::-1].copy()
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device, dtype=torch.float32) / 255
    prompt_tensor = torch.from_numpy(prompt)[None, None].to(args.device)
    with torch.inference_mode(), torch.autocast(device_type=torch.device(args.device).type,
                                                dtype=torch.float16, enabled=args.amp):
        depth = model.predict(image_tensor, prompt_tensor)[0, 0].float().cpu().numpy()
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise RuntimeError('model returned non-finite or non-positive depth')
    # Resize directly to storage resolution; never multiply depth by the image scale.
    output_wh = (max(1, round(width / args.output_downsampling)),
                 max(1, round(height / args.output_downsampling)))
    return cv2.resize(depth, output_wh, interpolation=cv2.INTER_LINEAR), list(wh)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'could not write {path}')


def save_depth(path, depth, depth_format, png_unit_m=0.01):
    """PNG stores rounded depth / png_unit_m, with no silent clipping or wraparound."""
    if depth_format == 'npy':
        np.save(path, depth.astype(np.float32))
    else:
        if not (0 < png_unit_m < float('inf')):
            raise ValueError('png_unit_m must be finite and positive')
        encoded = np.rint(depth.astype(np.float64) / png_unit_m)
        if not np.isfinite(encoded).all() or np.any(encoded < 1) or np.any(encoded > 65535):
            raise ValueError(f'depth outside uint16 range (max {65535 * png_unit_m:g}m, 0 reserved for invalid); '
                             'increase --png-unit-m or use --depth-format npy')
        if not cv2.imwrite(str(path), encoded.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise OSError(f'could not write {path}')


def main(argv=None):
    args = parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f'refusing existing output directory: {args.out}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or install a compatible CUDA PyTorch')
    if device.type not in ('cpu', 'cuda') or (args.amp and device.type != 'cuda'):
        raise ValueError('supported devices: cpu/cuda; --amp requires CUDA')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    samples = (load_dataset(args) if args.datadir else
               iter([(args.image, read_prompt(args.prompt_depth), {})]))
    # Validate the first prompt before allocating a large network.
    first = next(samples)
    fill_prompt(first[1])
    model = load_model(args)
    args.out.mkdir(parents=True, exist_ok=False)
    subdirs = ['depth']
    if args.save_preview:
        subdirs.append('preview')
    if args.save_prompts:
        subdirs.extend(['prompt', 'prompt_sparse', 'prompt_valid', 'prompt_distance'])
    for subdir in subdirs:
        (args.out / subdir).mkdir()
    metadata = {'status': 'running', 'args': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'units': 'scaled_integer' if args.depth_format == 'png' else 'metres',
                'depth_scale_to_metres': args.png_unit_m if args.depth_format == 'png' else 1.0,
                'depth_dtype': 'uint16' if args.depth_format == 'png' else 'float32',
                'depth_definition': 'OpenCV optical-axis z; not range',
                'checkpoint_sha256': sha256(args.checkpoint), 'torch_version': torch.__version__,
                'prompt_fill': 'nearest valid pixel, including outside measured support; NOT confidence',
                'time_alignment': 'world-linear translation + SO(3) interpolation; bounded endpoint extrapolation; NO per-point deskew',
                'output_resolution': 'original image dimensions divided by output_downsampling, rounded; bilinear resize without crop',
                'preview_range_m': [args.min_depth, args.max_depth], 'frames': []}
    if args.datadir:
        metadata['input_sha256'] = {name: sha256(path) for name, path in {
            'trajectory': args.trajectory, 'calibration': args.datadir / 'meta/calib.yaml',
            'pairs': args.datadir / 'meta/pairs.csv'}.items()}
    else:
        metadata['input_sha256'] = {'image': sha256(args.image), 'prompt': sha256(args.prompt_depth)}
    manifest = args.out / 'manifest.json'
    def save_manifest():
        temporary = args.out / 'manifest.tmp'
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(manifest)
    save_manifest()
    import itertools
    try:
        for image_path, sparse, info in itertools.chain([first], samples):
            start = time.monotonic()
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            if 'calibrated_wh' in info and list(bgr.shape[1::-1]) != info['calibrated_wh']:
                raise ValueError(f'{image_path}: image dimensions disagree with calibration')
            prompt, valid, distance = fill_prompt(sparse)
            depth, inference_wh = predict(model, bgr, prompt, args)
            stem = image_path.stem
            depth_path = args.out / 'depth' / f'{stem}.{args.depth_format}'
            save_depth(depth_path, depth, args.depth_format, args.png_unit_m)
            if args.save_prompts:
                np.save(args.out / 'prompt' / f'{stem}.npy', prompt)
                np.save(args.out / 'prompt_sparse' / f'{stem}.npy', np.where(valid, sparse, 0).astype(np.float32))
                np.save(args.out / 'prompt_distance' / f'{stem}.npy', distance)
                write_png(args.out / 'prompt_valid' / f'{stem}.png', valid.astype(np.uint8) * 255)
            if args.save_preview:
                normalized = np.clip((depth - args.min_depth) / (args.max_depth - args.min_depth), 0, 1)
                colour = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                observed = cv2.resize(bgr, depth.shape[::-1], interpolation=cv2.INTER_AREA)
                write_png(args.out / 'preview' / f'{stem}.png', np.concatenate([observed, colour], axis=1))
            info.update(name=stem, image=str(image_path.resolve()), inference_wh=inference_wh,
                        original_wh=list(bgr.shape[1::-1]), output_wh=list(depth.shape[::-1]),
                        depth_file=str(depth_path.relative_to(args.out)), depth_bytes=depth_path.stat().st_size,
                        prompt_wh=list(prompt.shape[::-1]),
                        prompt_coverage=float(valid.mean()), depth_percentiles_m=np.percentile(depth, [0, 5, 50, 95, 100]).tolist(),
                        seconds=time.monotonic() - start)
            metadata['frames'].append(info)
            save_manifest()
            print(f'{stem}: {info["seconds"]:.2f}s, prompt coverage {valid.mean():.1%}, '
                  f'depth median {np.median(depth):.3f} m', flush=True)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error'] = str(exc)
        raise
    finally:
        save_manifest()
    print(f'Wrote {len(metadata["frames"])} frame(s) to {args.out.resolve()}')


if __name__ == '__main__':
    main()
