# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


def feet_on_ground_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])
) -> torch.Tensor:
    """Reward for keeping both feet in contact with the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get foot body indices
    foot_indices = [asset.body_names.index(name) for name in asset_cfg.body_names]
    
    # Check if feet have contact forces (using net forces on feet)
    foot_forces = asset.data.body_net_forces_w[:, foot_indices, 2]  # Z-axis forces
    
    # Both feet should have upward forces (supporting weight)
    feet_in_contact = (foot_forces > 5.0).float()  # Threshold: 5N minimum contact force
    
    # Return reward: 1.0 if both feet touching, 0.5 if one foot, 0.0 if airborne
    return feet_in_contact.mean(dim=1)


def base_height_stability(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for deviating from target standing height (prevents jumping)."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    
    # Squared error from target height
    height_error = torch.square(current_height - target_height)
    
    return height_error


class hand_waving_reward(ManagerTermBase):
    """Reward for hand waving motion while maintaining balance.
    
    Rewards:
    - Hand reaches target height (shoulder level)
    - Rhythmic side-to-side motion (waving)
    - Smooth oscillation without jerky movements
    """
    
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        
        # Parameters
        self.target_height = cfg.params.get("target_height", 1.3)  # Shoulder height
        self.wave_amplitude = cfg.params.get("wave_amplitude", 0.25)  # Side motion range
        self.wave_frequency = cfg.params.get("wave_frequency", 2.0)  # Target Hz
        
        # History tracking
        self.prev_hand_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self.hand_x_history = torch.zeros(env.num_envs, 20, device=env.device)  # Last 20 timesteps
        self.history_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.direction_changes = torch.zeros(env.num_envs, device=env.device)
        self.prev_x_vel = torch.zeros(env.num_envs, device=env.device)
        
    def reset(self, env_ids: torch.Tensor):
        """Reset tracking for new episodes."""
        self.prev_hand_pos[env_ids] = 0.0
        self.hand_x_history[env_ids] = 0.0
        self.history_idx[env_ids] = 0
        self.direction_changes[env_ids] = 0.0
        self.prev_x_vel[env_ids] = 0.0
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        hand_name: str = "right_hand",
        target_height: float = 1.3,
        wave_amplitude: float = 0.25,
    ) -> torch.Tensor:
        # Get hand position in world frame
        asset: Articulation = env.scene[asset_cfg.name]
        hand_idx = asset.body_names.index(hand_name)
        hand_pos = asset.data.body_pos_w[:, hand_idx, :]
        
        reward = torch.zeros(env.num_envs, device=env.device)
        
        # --- Component 1: Height Reward ---
        # Encourage hand to be at shoulder height
        height_error = torch.abs(hand_pos[:, 2] - target_height)
        height_reward = torch.exp(-3.0 * height_error)  # Gaussian reward, tight tolerance
        reward += height_reward * 0.4
        
        # --- Component 2: Waving Motion Reward ---
        # Get hand velocity
        hand_vel = (hand_pos - self.prev_hand_pos) / env.step_dt
        x_vel = hand_vel[:, 0]  # Side-to-side velocity
        
        # Detect direction changes (waving back and forth)
        sign_change = (x_vel * self.prev_x_vel) < 0
        self.direction_changes += sign_change.float()
        
        # Reward oscillation: check if hand is moving side-to-side
        oscillation_reward = torch.abs(x_vel).clamp(0, 1.0)  # Reward velocity magnitude
        reward += oscillation_reward * 0.3
        
        # Bonus for completing wave cycles (direction changes)
        if env.common_step_counter > 0 and env.common_step_counter % 50 == 0:  # Check every ~1 second
            wave_cycles = self.direction_changes / 50.0  # Normalize by timesteps
            cycle_bonus = (wave_cycles * 2.0).clamp(0, 1.0)  # Target ~2Hz
            reward += cycle_bonus * 0.2
            self.direction_changes[:] = 0  # Reset counter
        
        # --- Component 3: Amplitude Reward ---
        # Track horizontal position range
        self.hand_x_history[torch.arange(env.num_envs, device=env.device), self.history_idx] = hand_pos[:, 0]
        self.history_idx = (self.history_idx + 1) % 20
        
        # Calculate range of motion
        x_range = self.hand_x_history.max(dim=1)[0] - self.hand_x_history.min(dim=1)[0]
        amplitude_reward = torch.exp(-torch.abs(x_range - wave_amplitude) / 0.1)
        reward += amplitude_reward * 0.1
        
        # Update tracking
        self.prev_hand_pos[:] = hand_pos
        self.prev_x_vel[:] = x_vel
        
        return reward


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)