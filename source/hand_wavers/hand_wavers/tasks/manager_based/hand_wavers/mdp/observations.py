# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_yaw_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_up_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)


def body_pos_in_robot_root_frame(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Position of specified bodies relative to the robot's root frame.
    
    This is useful for tracking hand position relative to the body center,
    which helps the policy learn coordinated waving motions.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices for the specified bodies
    body_indices = [asset.body_names.index(name) for name in asset_cfg.body_names]
    
    # Get body positions in world frame
    body_pos_w = asset.data.body_pos_w[:, body_indices, :]
    
    # Get root position and orientation
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    
    # Transform body positions to root frame
    # First, translate to root frame
    body_pos_rel = body_pos_w - root_pos_w.unsqueeze(1)
    
    # Then, rotate to root frame (inverse rotation)
    root_quat_inv = math_utils.quat_inv(root_quat_w)
    
    # Apply rotation to each body position
    num_bodies = len(body_indices)
    body_pos_root_frame = torch.zeros_like(body_pos_rel)
    
    for i in range(num_bodies):
        body_pos_root_frame[:, i, :] = math_utils.quat_apply(root_quat_inv, body_pos_rel[:, i, :])
    
    # Reshape to (num_envs, num_bodies * 3)
    return body_pos_root_frame.view(env.num_envs, -1)