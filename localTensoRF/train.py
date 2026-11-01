# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import warnings

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys
import time

from torch.utils.tensorboard import SummaryWriter

sys.path.append("localTensoRF")
from dataLoader.localrf_dataset import LocalRFDataset
from dataLoader.tracking_poses import camera_to_body, load_tracking, write_tum
from local_tensorfs import LocalTensorfs
from opt import config_parser
from utils.utils import (get_fwd_bwd_cam2cams)
from utils.utils import (N_to_reso, TVLoss, draw_poses, get_pred_flow,
                         compute_depth_loss)


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


def export_tracking_trajectory(args, local_tensorfs, train_dataset, tracking, logfolder):
    """Write optimised camera poses as a metric body-frame TUM trajectory."""
    trajectory_path = f"{logfolder}/trajectory.tum"
    cam2world = local_tensorfs.get_cam2world().detach().cpu().numpy().astype(np.float64)
    # get_cam2world returns [N,3,4]; complete it to homogeneous [N,4,4] before inverting.
    bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(cam2world), 1, 1))
    cam2world = np.concatenate([cam2world, bottom], axis=1)
    body_poses = camera_to_body(cam2world, tracking["T_world_G"], tracking["T_body_cam"])
    timestamps = {os.path.basename(f["file_path"]): f["timestamp_us"] for f in tracking["transforms"]["frames"]}
    stamps = [timestamps[p] for p in train_dataset.image_paths]
    if len(stamps) != len(body_poses):
        # zip() would silently truncate the trajectory, so check the counts first.
        raise RuntimeError(f"{len(body_poses)} poses for {len(stamps)} selected frames")
    write_tum(trajectory_path, stamps, body_poses)
    with open(f"{logfolder}/tracking_export.json", "w", encoding="utf-8") as f:
        json.dump({"trajectory": os.path.abspath(args.trajectory), "units": "metres",
                   "scene_radius_m": local_tensorfs.scene_radius, "max_drift_m": args.max_drift,
                   "lr_t_init_m_per_step": args.lr_t_init, "lr_R_init": args.lr_R_init,
                   "pose_prior_lambda_t_per_m2": args.pose_prior_lambda_t,
                   "pose_prior_lambda_r": args.pose_prior_lambda_r}, f, indent=2)
    print(f"tracking trajectory written to {trajectory_path}", flush=True)


def reconstruction(args):
    # An explicit tracking trajectory replaces the transforms.json prior. Its poses are metric; the model
    # works in units of scene_radius, so the unit-dependent defaults are derived here.
    tracking = load_tracking(args.datadir, args.trajectory) if args.trajectory else None
    if tracking is not None and os.path.exists(f"{args.logdir}/trajectory.tum"):
        raise FileExistsError(f"refusing to overwrite {args.logdir}/trajectory.tum; choose a new --logdir")
    scene_radius = args.scene_radius_m if tracking is not None else 1.0
    if args.max_drift is None:
        args.max_drift = scene_radius  # upstream default: 1 normalised unit, which is one scene radius
    if args.lr_t_init is None:
        args.lr_t_init = 5e-4 * scene_radius  # upstream default: 5e-4 per normalised unit, now per metre

    # Apply speedup factors
    args.n_iters_per_frame = int(args.n_iters_per_frame / args.refinement_speedup_factor)
    args.n_iters_reg = int(args.n_iters_reg / args.refinement_speedup_factor)
    args.upsamp_list = [int(upsamp / args.refinement_speedup_factor) for upsamp in args.upsamp_list]
    args.update_AlphaMask_list = [int(update_AlphaMask / args.refinement_speedup_factor) 
                                  for update_AlphaMask in args.update_AlphaMask_list]
    
    args.add_frames_every = int(args.add_frames_every / args.prog_speedup_factor)
    args.lr_R_init = args.lr_R_init * args.prog_speedup_factor
    args.lr_t_init = args.lr_t_init * args.prog_speedup_factor
    args.loss_flow_weight_inital = args.loss_flow_weight_inital * args.prog_speedup_factor
    args.L1_weight = args.L1_weight * args.prog_speedup_factor
    args.TV_weight_density = args.TV_weight_density * args.prog_speedup_factor
    args.TV_weight_app = args.TV_weight_app * args.prog_speedup_factor
    
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        with_preprocessed_poses=args.with_preprocessed_poses,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
        transforms=tracking["transforms"] if tracking is not None else None,
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

    # TODO: Add midpoint loading
    # if args.ckpt is not None:
    #     ckpt = torch.load(args.ckpt, map_location=args.device)
    #     kwargs = ckpt["kwargs"]
    #     kwargs.update({"device": args.device})
    #     tensorf = eval(args.model_name)(**kwargs)
    #     tensorf.load(ckpt)
    # else:

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

    if tracking is not None or args.with_preprocessed_poses:
        camera_prior = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
            "transforms": train_dataset.transforms
        }
        if tracking is not None:
            camera_prior["abs_poses"] = train_dataset.abs_poses
    else:
        camera_prior = None


    local_tensorfs = LocalTensorfs(
        camera_prior=camera_prior,
        fov=args.fov,
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
    if tracking is not None:
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
        if args.uniform_loss_weights:
            # Plain L1 over sampled pixels: no per-frame Laplacian sharpness weighting.
            loss_weights = torch.ones_like(loss_weights)
        train_test_poses = data_blob["train_test_poses"]
        ray_idx = torch.from_numpy(data_blob["idx"]).to(args.device)
        reg_loss_weight = local_tensorfs.lr_factor ** (local_tensorfs.rf_iter[-1])

        rgb_map, depth_map, directions, ij = local_tensorfs(
            ray_idx,
            view_ids,
            W,
            H,
            is_train=True,
            test_id=train_test_poses,
        )

        # loss
        loss = 0.25 * ((torch.abs(rgb_map - rgb_train)) * loss_weights) / loss_weights.mean()
               
        loss = loss.mean()
        total_loss = loss
        writer.add_scalar("train/rgb_loss", loss, global_step=iteration)

        if tracking is not None and optimize_poses and not train_test_poses and (
                args.pose_prior_lambda_t > 0 or args.pose_prior_lambda_r > 0):
            prior_loss = pose_prior_loss(local_tensorfs, train_dataset.abs_poses,
                                         args.pose_prior_lambda_t, args.pose_prior_lambda_r)
            total_loss = total_loss + prior_loss
            writer.add_scalar("train/pose_prior_loss", prior_loss, global_step=iteration)

        ## Regularization
        # Get rendered rays schedule
        if local_tensorfs.regularize and args.loss_flow_weight_inital > 0 or args.loss_depth_weight_inital > 0:
            depth_map = depth_map.view(view_ids.shape[0], -1)
            loss_weights = loss_weights.view(view_ids.shape[0], -1)
            depth_map = depth_map.view(view_ids.shape[0], -1)

            writer.add_scalar("train/reg_loss_weights", reg_loss_weight, global_step=iteration)

        # Optical flow
        if local_tensorfs.regularize and args.loss_flow_weight_inital > 0:
            if args.fov == 360:
                raise NotImplementedError
            starting_frame_id = max(train_dataset.active_frames_bounds[0] - 1, 0)
            cam2world = local_tensorfs.get_cam2world(starting_id=starting_frame_id)
            directions = directions.view(view_ids.shape[0], -1, 3)
            ij = ij.view(view_ids.shape[0], -1, 2)
            fwd_flow = torch.from_numpy(data_blob["fwd_flow"]).to(args.device).view(view_ids.shape[0], -1, 2)
            fwd_mask = torch.from_numpy(data_blob["fwd_mask"]).to(args.device).view(view_ids.shape[0], -1)
            fwd_mask[view_ids == len(cam2world) - 1] = 0
            bwd_flow = torch.from_numpy(data_blob["bwd_flow"]).to(args.device).view(view_ids.shape[0], -1, 2)
            bwd_mask = torch.from_numpy(data_blob["bwd_mask"]).to(args.device).view(view_ids.shape[0], -1)
            fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids - starting_frame_id)
                       
            pts = directions * depth_map[..., None]
            pred_fwd_flow = get_pred_flow(
                pts, ij, fwd_cam2cams, local_tensorfs.focal(W), local_tensorfs.center(W, H))
            pred_bwd_flow = get_pred_flow(
                pts, ij, bwd_cam2cams, local_tensorfs.focal(W), local_tensorfs.center(W, H))
            flow_loss_arr =  torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask
            flow_loss_arr += torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask
            flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.9, dim=1)[..., None]] = 0

            flow_loss = (flow_loss_arr).mean() * args.loss_flow_weight_inital * reg_loss_weight / ((W + H) / 2)
            total_loss = total_loss + flow_loss
            writer.add_scalar("train/flow_loss", flow_loss, global_step=iteration)

        # Monocular Depth 
        if local_tensorfs.regularize and args.loss_depth_weight_inital > 0:
            if args.fov == 360:
                raise NotImplementedError
            invdepths = torch.from_numpy(data_blob["invdepths"]).to(args.device)
            invdepths = invdepths.view(view_ids.shape[0], -1)
            _, _, depth_loss_arr = compute_depth_loss(1 / depth_map.clamp(1e-6), invdepths)
            depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.8, dim=1)[..., None]] = 0

            depth_loss = (depth_loss_arr).mean() * args.loss_depth_weight_inital * reg_loss_weight
            total_loss = total_loss + depth_loss 
            writer.add_scalar("train/depth_loss", depth_loss, global_step=iteration)

        if  local_tensorfs.regularize:
            loss_tv, l1_loss = local_tensorfs.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight)
            total_loss = total_loss + loss_tv + l1_loss
            writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
            writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)

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
                last_add_rf_iter = iteration

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

        writer.add_scalar(
            "train/focal", local_tensorfs.focal(W), global_step=iteration
        )
        writer.add_scalar(
            "train/center0", local_tensorfs.center(W, H)[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "train/center1", local_tensorfs.center(W, H)[1].item(), global_step=iteration
        )

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

    if tracking is not None:
        export_tracking_trajectory(args, local_tensorfs, train_dataset, tracking, logfolder)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    reconstruction(args)
