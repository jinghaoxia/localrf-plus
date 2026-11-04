"""Atomic, versioned training checkpoints with immutable CPU field shards."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import tempfile

import numpy as np
import torch

from local_tensorfs import LocalTensorfs

VERSION = 1
SCHEDULE_KEYS = ("rf_iter", "pose_linked_rf", "is_refining", "lr_factor", "regularize",
                 "n_iters", "n_iters_reg", "N_voxel_list", "update_AlphaMask_list")
RUNTIME_OPTIONS = {"resume", "logdir", "device", "datadir", "trajectory", "batch_size",
                   "checkpoint_every", "vis_every", "pose_vis_every", "grad_log_every",
                   "progress_refresh_rate", "config"}


def cpu_tree(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch_save(value, path):
    """A failed or interrupted write leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("training_checkpoint_version") != VERSION:
        raise ValueError("--resume requires a versioned resume.th; legacy render checkpoints/events lack training state")
    return checkpoint


def resume_arguments(args, checkpoint):
    saved = copy.deepcopy(checkpoint["args"])
    for name in getattr(args, "_explicit_args", []):
        if name not in saved or name == "config":
            continue
        value = getattr(args, name)
        if name not in RUNTIME_OPTIONS and value != saved[name]:
            raise ValueError(f"cannot change --{name} on resume: saved {saved[name]!r}, requested {value!r}")
        saved[name] = value
    # An unspecified destination continues in the checkpoint's directory.
    if "logdir" not in getattr(args, "_explicit_args", []):
        saved["logdir"] = str(Path(args.resume).resolve().parent)
    saved["resume"] = args.resume
    saved["_explicit_args"] = getattr(args, "_explicit_args", [])
    return argparse.Namespace(**saved)


def dataset_identity(dataset, args):
    """Bind the checkpoint to selected observations, geometry and metadata.

    Hash small metadata files by content; use size/mtime for large observations.
    Moving a dataset is supported if observation timestamps are preserved.
    """
    digest = hashlib.sha256()
    root = Path(args.datadir)
    digest.update(json.dumps({"images": dataset.image_paths, "intrinsics": dataset.intrinsics,
                              "test_mask": dataset.test_mask.tolist()}, sort_keys=True).encode())
    for relative in ("meta/calib.yaml", "meta/pairs.csv", "transforms.json"):
        path = root / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    if args.trajectory:
        digest.update(Path(args.trajectory).read_bytes())
    for name in dataset.image_paths:
        mask = root / "masks" / f"{Path(name).stem}.png"
        paths = [root / "images" / name, mask if mask.is_file() else root / "masks/all.png"]
        if dataset.load_depth:
            paths += [root / "depth" / f"{Path(name).stem}{suffix}" for suffix in (".png", ".npy")]
        for path in paths:
            digest.update(str(path.relative_to(root)).encode())
            if path.exists():
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            else:
                digest.update(b"missing")
    return digest.hexdigest()


def rng_state(device):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(path, model, dataset, args, identity, loop, frozen_fields):
    """Write each frozen field once, then atomically commit the small mutable state."""
    path = Path(path)
    for index in range(len(frozen_fields), len(model.tensorfs) - 1):
        # Unique names prevent a branched resume from replacing a shard used
        # by another valid checkpoint. No cleanup of existing experiment data.
        shard_dir = path.parent / "resume_fields"
        shard_dir.mkdir(parents=True, exist_ok=True)
        fd, shard_path = tempfile.mkstemp(prefix=f"rf{index:05d}_", suffix=".th", dir=shard_dir)
        os.close(fd)
        try:
            atomic_torch_save(cpu_tree(model.tensorfs[index].state_dict()), shard_path)
        except BaseException:
            Path(shard_path).unlink(missing_ok=True)
            raise
        frozen_fields.append(str(Path(shard_path).relative_to(path.parent)))
    current = len(model.tensorfs) - 1
    # Avoid constructing an all-fields state_dict on every save.
    state = {name: cpu_tree(value) for name, value in model.state_dict().items()
             if not name.startswith("tensorfs.") or name.startswith(f"tensorfs.{current}.")}
    checkpoint = {
        "training_checkpoint_version": VERSION,
        "kwargs": cpu_tree(model.get_kwargs()), "state_dict": state,
        "tensorf_args": cpu_tree(model.tensorf_args), "frozen_fields": list(frozen_fields),
        "schedule": {key: copy.deepcopy(getattr(model, key)) for key in SCHEDULE_KEYS},
        "optimizers": {"rf": cpu_tree(model.rf_optimizer.state_dict()),
                       "rotation": [cpu_tree(opt.state_dict()) for opt in model.r_optimizers],
                       "translation": [cpu_tree(opt.state_dict()) for opt in model.t_optimizers],
                       "exposure": [cpu_tree(opt.state_dict()) for opt in model.exp_optimizers]},
        "requires_grad": {name: param.requires_grad for name, param in model.named_parameters()
                          if not name.startswith("tensorfs.")},
        "dataset": {"bounds": list(dataset.active_frames_bounds), "loaded_frames": dataset.loaded_frames,
                    "identity": identity},
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "loop": dict(loop), "rng": rng_state(model.device),
    }
    if model.lr_i_init > 0:
        checkpoint["optimizers"]["intrinsics"] = cpu_tree(model.intrinsic_optimizer.state_dict())
    atomic_torch_save(cpu_tree(checkpoint), path)


def restore_model(checkpoint, path, device, camera_prior=None):
    """Stream historical fields onto CPU and keep only the current field on device."""
    kwargs = dict(checkpoint["kwargs"], device=device, camera_prior=camera_prior)
    # The first constructor field is temporary and should use the small initial grid.
    kwargs.update(checkpoint["tensorf_args"])
    kwargs["aabb"] = kwargs["aabb"].to(device)
    model = LocalTensorfs(**kwargs)
    current = len(checkpoint["frozen_fields"])

    def fields():
        for relative in checkpoint["frozen_fields"]:
            shard = (Path(path).resolve().parent / relative).resolve()
            if not shard.is_relative_to(Path(path).resolve().parent):
                raise ValueError("field shard escapes checkpoint directory")
            yield torch.load(shard, map_location="cpu", weights_only=True)
        prefix = f"tensorfs.{current}."
        yield {key[len(prefix):]: value for key, value in checkpoint["state_dict"].items()
               if key.startswith(prefix)}

    model.load(checkpoint["state_dict"], field_states=fields())
    model.tensorf_args = dict(checkpoint["tensorf_args"])
    for key in SCHEDULE_KEYS:
        setattr(model, key, copy.deepcopy(checkpoint["schedule"][key]))
    for name, parameter in model.named_parameters():
        if name in checkpoint["requires_grad"]:
            parameter.requires_grad_(checkpoint["requires_grad"][name])
    opts = checkpoint["optimizers"]
    model.rf_optimizer.load_state_dict(opts["rf"])
    for key, optimizers in (("rotation", model.r_optimizers), ("translation", model.t_optimizers),
                            ("exposure", model.exp_optimizers)):
        if len(optimizers) != len(opts[key]):
            raise ValueError(f"invalid {key} optimizer count")
        for optimizer, state in zip(optimizers, opts[key]):
            optimizer.load_state_dict(state)
    if model.lr_i_init > 0:
        model.intrinsic_optimizer.load_state_dict(opts["intrinsics"])
    return model


class StopAtStepBoundary:
    """First SIGINT/SIGTERM requests a consistent save after the current step."""
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        def request(signum, frame):
            self.requested = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, request)
        return self

    def __exit__(self, *exc):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)
