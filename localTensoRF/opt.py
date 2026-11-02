# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--logdir", type=str, default="./log", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )
    parser.add_argument(
        "--progress_refresh_rate",
        type=int,
        default=200,
        help="how many iterations to show iters",
    )

    parser.add_argument(
        "--downsampling", 
        type=float, 
        default=-1, 
        help="Image downsampling ratio; -1 preserves source resolution."
    )

    parser.add_argument("--seed", type=int, default=20211202, help="Python, NumPy and PyTorch random seed")

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)

    ## training options
    # learning rate
    parser.add_argument("--lr_R_init", type=float, default=5e-3, help="Rotation learning rate")
    parser.add_argument("--lr_t_init", type=float, default=None,
                        help="Translation learning rate; default 5e-4 per normalised unit, i.e. 5e-4 * scene_radius_m per step with --trajectory")
    parser.add_argument("--lr_i_init", type=float, default=0, help="Intrinsics learning rate")
    parser.add_argument("--lr_exposure_init", type=float, default=0, help="Exposure compensation learning rate")
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate")
    parser.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=0.1,
        help="the target decay ratio; after decay_iters initial lr decays to lr*ratio",
    )
    parser.add_argument(
        "--lr_upsample_reset",
        type=int,
        default=1,
        help="reset lr to initial after upsampling",
    )

    # Basic scheduling options
    parser.add_argument("--N_voxel_init", type=int, default=64**3)
    parser.add_argument("--N_voxel_final", type=int, default=640**3)
    parser.add_argument("--n_iters_per_frame", type=int, default=600)
    parser.add_argument("--n_iters_reg", type=int, default=100)
    parser.add_argument(
        "--upsamp_list", 
        type=int, 
        default=[100, 150, 200, 250, 300],
        nargs='+')
    parser.add_argument("--update_AlphaMask_list", type=int, default=[100, 200, 300], nargs='+')
    parser.add_argument("--refinement_speedup_factor", type=float, default=1.0, 
                        help="Divides number of iterations in scheduling. Does not apply to progressive optimization.")

    # Progressive optimization scheduling options
    parser.add_argument(
        "--n_init_frames",
        type=int,
        default=5,
        help="Number of initial frames for the first RF optimization",
    )
    parser.add_argument(
        "--max_drift",
        type=float,
        default=None,
        help="Create a new RF once the camera pose shifts this amount w.r.t the last created RF; metres with --trajectory (default scene_radius_m), normalised units otherwise (default 1)",
    )
    parser.add_argument(
        "--n_max_frames",
        type=int,
        default=100,
        help="Maximum number of frames added before optimizing a new RF",
    )
    parser.add_argument(
        "--add_frames_every",
        type=int,
        default=100,
        help="Number of iterations before adding another frame",
    )
    parser.add_argument(
        "--n_overlap",
        type=int,
        default=30,
        help="Number of frames supervising two neighbour RFs",
    )
    parser.add_argument("--prog_speedup_factor", type=float, default=1.0, 
                        help="Divides number of iterations in progressive optimization scheduling. Multiplies pose lr.")

    # losses
    parser.add_argument("--depth_weight", type=float, default=0.0,
                        help="weight of mean absolute metric z-depth error; 0 disables loading depth")
    parser.add_argument("--depth_scale", type=float, default=None,
                        help="metres per uint16 depth PNG unit (211 sample: 0.01); .npy depth is already in metres")
    parser.add_argument("--grad_log_every", type=int, default=0,
                        help="log per-loss gradients on active parameter groups every N steps; 0 disables extra backward passes")
    parser.add_argument("--L1_weight", type=float, default=0.0, help="loss weight")
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help="loss weight")#0.1
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help="loss weight")#0.01

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, default=None, nargs="+", action="extend",
                        help="three positive density ranks, e.g. --n_lamb_sigma 8 8 8")
    parser.add_argument("--n_lamb_sh", type=int, default=None, nargs="+", action="extend",
                        help="three positive appearance ranks, e.g. --n_lamb_sh 24 24 24")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument(
        "--rm_weight_mask_thre",
        type=float,
        default=0.001,
        help="mask points in ray marching",
    )
    parser.add_argument(
        "--alpha_mask_thre",
        type=float,
        default=0.0001,
        help="threshold for creating alpha mask volume",
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=25,
        help="scaling sampling distance for computation",
    )
    parser.add_argument(
        "--density_shift",
        type=float,
        default=-5,
        help="shift density in softplus; making density = 0  when feature == 0",
    )

    # network decoder
    parser.add_argument(
        "--shadingMode", type=str, default="MLP_Fea_late_view", help="which shading mode to use"
    )
    parser.add_argument("--pos_pe", type=int, default=0, help="number of pe for pos")
    parser.add_argument("--view_pe", type=int, default=0, help="number of pe for view")
    parser.add_argument(
        "--fea_pe", type=int, default=0, help="number of pe for features"
    )
    parser.add_argument(
        "--featureC", type=int, default=128, help="hidden feature channel in MLP"
    )

    # rendering options
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument("--step_ratio", type=float, default=0.5)

    # Camera model options
    parser.add_argument("--with_preprocessed_poses", type=int, default=0)
    parser.add_argument("--trajectory", type=str, default="",
                        help="explicit tracking TUM trajectory (body poses in metres); replaces transforms.json as the pose prior")
    parser.add_argument("--pose_prior_lambda_t", type=float, default=0.0,
                        help="weight on squared translation distance to the tracking prior, per metre^2; a normalised-unit weight w becomes w / scene_radius_m^2 (needs --trajectory)")
    parser.add_argument("--pose_prior_lambda_r", type=float, default=0.0,
                        help="weight on squared rotation distance to the tracking prior (needs --trajectory)")
    parser.add_argument("--scene_radius_m", type=float, default=8.0,
                        help="metres at which the contraction starts; full spatial resolution lies inside about this radius (used with --trajectory)")
    parser.add_argument("--uniform_loss_weights", type=int, default=1,
                        help="1 disables Laplacian sharpness weights while preserving masks")
    parser.add_argument("--test_pose_opt", type=int, default=0,
                        help="0 forbids pose-only steps on held-out frames, so they never affect poses")

    parser.add_argument("--subsequence", default=[0, -1], type=int, nargs=2)
    parser.add_argument('--frame_step', type=int, default=1, help="Step between retained frames")
    parser.add_argument("--test_frame_every", default=0, type=int, help="Every test_frame_every-th frame is a test frame.")
    # logging/saving options
    parser.add_argument(
        "--vis_every", type=int, default=10000, help="Iterations between temporary checkpoint saves."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args(cmd)
    for name, default in (("n_lamb_sigma", [8, 8, 8]), ("n_lamb_sh", [24, 24, 24])):
        ranks = getattr(args, name)
        if ranks is None:
            setattr(args, name, default)
        elif len(ranks) != 3 or any(rank <= 0 for rank in ranks):
            parser.error(f"--{name} requires exactly three positive ranks")
    return args
