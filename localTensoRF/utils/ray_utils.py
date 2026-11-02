# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import torch

def contract(x):
    x_norm = torch.clamp(x.abs().amax(dim=-1, keepdim=True), 1e-6)
    z = torch.where(x_norm <= 1, x, ((2 * x_norm - 1) / (x_norm**2)) * x)
    return z

def get_ray_directions_lean(i, j, fx, fy, cx, cy):
    """Unnormalised pinhole rays in LocalRF axes; camera z is -1.

    Intrinsics use i+0.5 pixel centres and are already scaled to the image size.
    """
    i, j = i.float() + 0.5, j.float() + 0.5
    return torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)


def get_rays_lean(directions, c2w):
    """
    get_rays but returns only relevant rays
    Inputs:
        directions: (B, 3) ray directions in camera coordinate
        c2w: (B, 3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (B, 3), the origin of the rays in world coordinate
        rays_d: (B, 3), the unnormalised direction of the rays in world coordinate
    """
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, :3, 3]  # (B, 3)
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.bmm(c2w[:, :3, :3], directions[..., None])[..., 0]  # (B, 3)

    return rays_o, rays_d