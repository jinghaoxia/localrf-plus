# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
from pathlib import Path
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys
import time

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataLoader.localrf_dataset import LocalRFDataset
from local_tensorfs import LocalTensorfs
from opt import config_parser
from utils.utils import N_to_reso, TVLoss, draw_poses, dense_depth_loss


def pose_prior_loss(local_tensorfs, prior_poses, lambda_t, lambda_r):
    """Penalise poses of frames the current field optimises, relative to tracking.

    Translation is squared distance in metres; rotation is the
    squared Frobenius distance of the 3x3 matrix, which is about the squared
    angle for small errors and has no branch at zero.
    """
    current = len(local_tensorfs.rf_iter) - 1
    active = [i for i, rf in enumerate(local_tensorfs.pose_linked_rf) if rf == current]
    if not active:
        # Right after a new field is created no frame is linked to it yet; overlap frames stay frozen on the old field.
        return torch.zeros((), device=local_tensorfs.device)
    cam2world = local_tensorfs.get_cam2world(view_ids=active)
    prior = torch.as_tensor(prior_poses[active][:, :3, :], dtype=cam2world.dtype, device=cam2world.device)
    translation = (cam2world[:, :, 3] - prior[:, :, 3]).square().sum()
    rotation = (cam2world[:, :, :3] - prior[:, :, :3]).square().sum()
    return lambda_t * translation + lambda_r * rotation


def active_gradient_groups(model, optimize_poses, poses_only=False):
    """Parameters stepped by the current window, separated by role and units."""
    groups = {}
    current = len(model.tensorfs) - 1
    if not poses_only:
        field = model.tensorfs[-1]
        density = list(field.density_plane) + list(field.density_line)
        density_ids = {id(p) for p in density}
        groups["density"] = density
        groups["appearance"] = [p for p in field.parameters() if id(p) not in density_ids and p.requires_grad]
    if model.rf_iter[-1] < model.n_iters:
        active = [i for i, linked in enumerate(model.pose_linked_rf) if linked == current]
        if optimize_poses:
            for name, optimizers in (("rotation", model.r_optimizers), ("translation", model.t_optimizers)):
                groups[name] = [p for i in active for group in optimizers[i].param_groups
                                if group["lr"] > 0 for p in group["params"] if p.requires_grad]
        if not poses_only and model.lr_exposure_init > 0:
            groups["exposure"] = [model.exposure[i] for i in active]
    if not poses_only and model.lr_i_init > 0 and current == 0 and model.is_refining:
        groups["intrinsics"] = [model.focal_offset, model.center_rel]
    return {name: params for name, params in groups.items() if params}


def loss_gradient_norms(losses, parameter_groups):
    """L2 and maximum absolute gradients of weighted terms, without altering .grad.

    Each loss uses the same forward samples. Groups are not combined across units.
    autograd.grad retains the graph for the usual training backward/optimiser step.
    """
    parameters = [p for group in parameter_groups.values() for p in group]
    metrics = {}
    if not parameters:
        return metrics
    for loss_name, loss in losses.items():
        if loss.requires_grad:
            grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        else:
            grads = [None] * len(parameters)
        offset = 0
        for group_name, group in parameter_groups.items():
            values = [g.detach() for g in grads[offset:offset+len(group)] if g is not None]
            offset += len(group)
            if values:
                # float64 accumulation avoids overflow when diagnosing a float32 gradient spike.
                l2 = torch.stack([g.double().square().sum() for g in values]).sum().sqrt().item()
                maximum = max(g.abs().max().item() for g in values)
            else:
                l2, maximum = 0., 0.
            metrics[f"grad/{loss_name}/{group_name}_l2"] = l2
            metrics[f"grad/{loss_name}/{group_name}_max"] = maximum
    return metrics


def export_tracking_trajectory(args, local_tensorfs, train_dataset, logfolder):
    """Write optimised camera poses as a metric body-frame TUM trajectory."""
    trajectory_path = f"{logfolder}/trajectory.tum"
    cam2world = local_tensorfs.get_cam2world().detach().cpu().numpy().astype(np.float64)
    train_dataset.export_trajectory(trajectory_path, cam2world)
    with open(f"{logfolder}/tracking_export.json", "w", encoding="utf-8") as f:
        json.dump({"trajectory": os.path.abspath(args.trajectory), "units": "metres",
                   "scene_radius_m": local_tensorfs.scene_radius, "max_drift_m": args.max_drift,
                   "lr_t_init_m_per_step": args.lr_t_init, "lr_R_init": args.lr_R_init,
                   "pose_prior_lambda_t_per_m2": args.pose_prior_lambda_t,
                   "pose_prior_lambda_r": args.pose_prior_lambda_r,
                   "seed": args.seed,
                   "depth_weight": args.depth_weight, "depth_scale_m_per_unit": args.depth_scale,
                   "depth_definition": "in-scene expected camera z in metres",
                   "intrinsics": train_dataset.intrinsics}, f, indent=2)
    print(f"tracking trajectory written to {trajectory_path}", flush=True)


def reconstruction(args):
    if os.path.exists(args.logdir) and os.listdir(args.logdir):
        raise FileExistsError(f"refusing to write into non-empty {args.logdir}; choose a new --logdir")
    if args.n_overlap < 1 or args.n_init_frames < 1:
        raise ValueError("n_overlap and n_init_frames must be at least 1")
    if args.grad_log_every < 0:
        raise ValueError("grad_log_every must be nonnegative")
    if args.batch_size <= 0 or args.batch_size % 16:
        raise ValueError("batch_size must be a positive multiple of 16")
    if not np.isfinite(args.depth_weight) or args.depth_weight < 0:
        raise ValueError("depth_weight must be finite and nonnegative")
    if (args.pose_prior_lambda_t > 0 or args.pose_prior_lambda_r > 0) and not args.trajectory:
        raise ValueError("pose priors require --trajectory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scene_radius = args.scene_radius_m if args.trajectory else 1.0
    if args.max_drift is None:
        args.max_drift = scene_radius
    if args.lr_t_init is None:
        args.lr_t_init = 5e-4 * scene_radius

    # Apply speedup factors
    args.n_iters_per_frame = int(args.n_iters_per_frame / args.refinement_speedup_factor)
    args.n_iters_reg = int(args.n_iters_reg / args.refinement_speedup_factor)
    args.upsamp_list = [int(upsamp / args.refinement_speedup_factor) for upsamp in args.upsamp_list]
    args.update_AlphaMask_list = [int(update_AlphaMask / args.refinement_speedup_factor) 
                                  for update_AlphaMask in args.update_AlphaMask_list]
    
    args.add_frames_every = int(args.add_frames_every / args.prog_speedup_factor)
    args.lr_R_init = args.lr_R_init * args.prog_speedup_factor
    args.lr_t_init = args.lr_t_init * args.prog_speedup_factor
    args.L1_weight = args.L1_weight * args.prog_speedup_factor
    args.TV_weight_density = args.TV_weight_density * args.prog_speedup_factor
    args.TV_weight_app = args.TV_weight_app * args.prog_speedup_factor
    
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        load_depth=args.depth_weight > 0,
        depth_scale=args.depth_scale,
        uniform_loss_weights=args.uniform_loss_weights,
        with_preprocessed_poses=args.with_preprocessed_poses,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
        trajectory=args.trajectory,
    )
    near_far = train_dataset.near_far

    # init resolution
    upsamp_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f"{args.logdir}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    writer = SummaryWriter(log_dir=logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(args.device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

    print("lr decay", args.lr_decay_target_ratio)

    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]
    N_voxel_list = {
        usamp_idx: round(N_voxel**(1/3))**3 for usamp_idx, N_voxel in zip(upsamp_list, N_voxel_list)
    }

    local_tensorfs = LocalTensorfs(
        camera_prior=train_dataset.camera_prior,
        intrinsics=train_dataset.intrinsics,
        n_init_frames=min(args.n_init_frames, train_dataset.num_images),
        n_overlap=args.n_overlap,
        WH=train_dataset.img_wh,
        n_iters_per_frame=args.n_iters_per_frame,
        n_iters_reg=args.n_iters_reg,
        lr_R_init=args.lr_R_init,
        lr_t_init=args.lr_t_init,
        lr_i_init=args.lr_i_init,
        lr_exposure_init=args.lr_exposure_init,
        rf_lr_init=args.lr_init,
        rf_lr_basis=args.lr_basis,
        lr_decay_target_ratio=args.lr_decay_target_ratio,
        N_voxel_list=N_voxel_list,
        update_AlphaMask_list=args.update_AlphaMask_list,
        lr_upsample_reset=args.lr_upsample_reset,
        device=args.device,
        alphaMask_thres=args.alpha_mask_thre,
        shadingMode=args.shadingMode,
        aabb=aabb,
        gridSize=reso_cur,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        rayMarch_weight_thres=args.rm_weight_mask_thre,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
        scene_radius=scene_radius,
    )
    local_tensorfs = local_tensorfs.to(args.device)
    if train_dataset.tracking is not None:
        # The first selected frame fixes the gauge: its pose stays at the tracking value.
        for optimizer in (local_tensorfs.r_optimizers[0], local_tensorfs.t_optimizers[0]):
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0

    torch.cuda.empty_cache()

    tvreg = TVLoss()
    W, H = train_dataset.img_wh

    training = True
    n_added_frames = 0
    last_add_iter = 0
    iteration = 0
    start_time = time.time()
    while training:
        optimize_poses = args.lr_R_init > 0 or args.lr_t_init > 0
        data_blob = train_dataset.sample(args.batch_size, local_tensorfs.is_refining, optimize_poses,
                                         allow_test_pose_opt=args.test_pose_opt)
        view_ids = torch.from_numpy(data_blob["view_ids"]).to(args.device)
        rgb_train = torch.from_numpy(data_blob["rgbs"]).to(args.device)
        loss_weights = torch.from_numpy(data_blob["loss_weights"]).to(args.device)
        train_test_poses = data_blob["train_test_poses"]
        ray_idx = torch.from_numpy(data_blob["idx"]).to(args.device)

        rgb_map, depth_map, opacity_map = local_tensorfs(
            ray_idx,
            view_ids,
            W,
            H,
            is_train=True,
            test_id=train_test_poses,
        )

        # loss
        # Uniform weighting still respects observation masks; an all-masked batch contributes zero.
        loss = 0.25 * (torch.abs(rgb_map - rgb_train) * loss_weights).sum()
        loss = loss / (3 * loss_weights.sum()).clamp_min(1e-8)
        total_loss = loss
        loss_terms = {"rgb": loss}
        writer.add_scalar("train/rgb_loss", loss, global_step=iteration)

        if train_dataset.tracking is not None and optimize_poses and not train_test_poses and (
                args.pose_prior_lambda_t > 0 or args.pose_prior_lambda_r > 0):
            prior_loss = pose_prior_loss(local_tensorfs, train_dataset.abs_poses,
                                         args.pose_prior_lambda_t, args.pose_prior_lambda_r)
            total_loss = total_loss + prior_loss
            loss_terms["pose_prior"] = prior_loss
            writer.add_scalar("train/pose_prior_loss", prior_loss, global_step=iteration)

        # Metric depth is an observation term throughout optimisation, with a constant weight.
        if args.depth_weight > 0:
            target_depth = torch.from_numpy(data_blob["depths"]).to(args.device)
            depth_valid = torch.from_numpy(data_blob["depth_valid"]).to(args.device)
            depth_l1 = dense_depth_loss(depth_map, target_depth, depth_valid)
            loss_terms["depth"] = args.depth_weight * depth_l1
            total_loss = total_loss + loss_terms["depth"]
            writer.add_scalar("train/depth_l1_m", depth_l1, global_step=iteration)
            writer.add_scalar("train/depth_valid_fraction", depth_valid.float().mean(), global_step=iteration)
            writer.add_scalar("train/opacity_mean", opacity_map.mean(), global_step=iteration)
            observed = depth_valid.reshape(-1).bool()
            low_opacity = (opacity_map.detach() < 1e-4) & observed
            writer.add_scalar("train/depth_low_opacity_fraction",
                              low_opacity.sum() / observed.sum().clamp_min(1), global_step=iteration)
            writer.add_scalar("train/depth_loss", args.depth_weight * depth_l1, global_step=iteration)

        # Optional field regularisation keeps the original progressive schedule.
        if  local_tensorfs.regularize:
            loss_tv, l1_loss = local_tensorfs.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight)
            total_loss = total_loss + loss_tv + l1_loss
            writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
            writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)

        if args.grad_log_every and iteration % args.grad_log_every == 0:
            groups = active_gradient_groups(local_tensorfs, optimize_poses, poses_only=train_test_poses)
            for tag, value in loss_gradient_norms(loss_terms, groups).items():
                writer.add_scalar(tag, value, global_step=iteration)
            for name, parameters in groups.items():
                writer.add_scalar(f"grad/parameter_count/{name}", sum(p.numel() for p in parameters), iteration)

        # Optimizes
        if train_test_poses:
            can_add_rf = False
            if optimize_poses:
                local_tensorfs.optimizer_step_poses_only(total_loss)
        else:
            can_add_rf = local_tensorfs.optimizer_step(total_loss, optimize_poses)
            training |= train_dataset.active_frames_bounds[1] != train_dataset.num_images

        ## Progressive optimization
        if not local_tensorfs.is_refining:
            should_refine = (not train_dataset.has_left_frames() or (
                n_added_frames > args.n_overlap and (
                    local_tensorfs.get_dist_to_last_rf().cpu().item() > args.max_drift
                    or (train_dataset.active_frames_bounds[1] - train_dataset.active_frames_bounds[0]) >= args.n_max_frames
                )))
            if should_refine and (iteration - last_add_iter) >= args.add_frames_every:
                local_tensorfs.is_refining = True

            should_add_frame = train_dataset.has_left_frames()
            should_add_frame &= (iteration - last_add_iter + 1) % args.add_frames_every == 0

            should_add_frame &= not should_refine
            should_add_frame &= not local_tensorfs.is_refining
            # Add supervising frames
            if should_add_frame:
                local_tensorfs.append_frame()
                train_dataset.activate_frames()
                n_added_frames += 1
                last_add_iter = iteration

        # Add new RF
        if can_add_rf:
            if train_dataset.has_left_frames():
                local_tensorfs.append_rf(n_added_frames)
                n_added_frames = 0

                # Remove supervising frames
                training_frames = (local_tensorfs.blending_weights[:, -1] > 0)
                train_dataset.deactivate_frames(
                    np.argmax(training_frames.cpu().numpy(), axis=0))
            else:
                training = False
        ## Log
        loss = loss.detach().item()

        writer.add_scalar(
            "train/density_app_plane_lr",
            local_tensorfs.rf_optimizer.param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/basis_mat_lr",
            local_tensorfs.rf_optimizer.param_groups[4]["lr"],
            global_step=iteration,
        )

        writer.add_scalar(
            "train/lr_r",
            local_tensorfs.r_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/lr_t",
            local_tensorfs.t_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )

        for name, value in zip(("fx", "fy", "cx", "cy"), local_tensorfs.get_intrinsics(W, H)):
            writer.add_scalar(f"train/{name}", value.item(), global_step=iteration)

        writer.add_scalar(
            "active_frames_bounds/0", train_dataset.active_frames_bounds[0], global_step=iteration
        )
        writer.add_scalar(
            "active_frames_bounds/1", train_dataset.active_frames_bounds[1], global_step=iteration
        )

        for index, blending_weights in enumerate(
            torch.permute(local_tensorfs.blending_weights, [1, 0])
        ):
            active_cam_indices = torch.nonzero(blending_weights)
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b0", active_cam_indices[0], global_step=iteration
            )
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b1", active_cam_indices[-1], global_step=iteration
            )

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            # All poses visualization
            poses_mtx = local_tensorfs.get_cam2world().detach().cpu()
            t_w2rf = torch.stack(list(local_tensorfs.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), -t_w2rf.clone()[..., None]], axis=-1)

            all_poses = torch.cat([poses_mtx,  RF_mtx_inv], dim=0)
            colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * RF_mtx_inv.shape[0]
            img = draw_poses(all_poses, colours)
            writer.add_image("poses/all", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)

            # Get runtime 
            ips = min(args.progress_refresh_rate, iteration + 1) / (time.time() - start_time)
            writer.add_scalar(f"train/iter_per_sec", ips, global_step=iteration)
            print(f"Iteration {iteration:06d}: {ips:.2f} it/s")
            start_time = time.time()

        if (iteration % args.vis_every == args.vis_every - 1):
            with open(f"{logfolder}/checkpoints_tmp.th", "wb") as f:
                local_tensorfs.save(f)

        iteration += 1

    with open(f"{logfolder}/checkpoints.th", "wb") as f:
        local_tensorfs.save(f)

    if train_dataset.tracking is not None:
        export_tracking_trajectory(args, local_tensorfs, train_dataset, logfolder)

    writer.close()


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)

    args = config_parser()
    print(args)

    reconstruction(args)
