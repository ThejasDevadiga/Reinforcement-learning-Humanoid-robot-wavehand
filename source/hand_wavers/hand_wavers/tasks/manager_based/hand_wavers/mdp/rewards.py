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

class hand_lifting_reward(ManagerTermBase):
    """Reward for lifting hand to target height with smooth upward motion.
    
    Rewards:
    - Height achievement (reaches target lift_height)
    - Upward velocity (bonus for lifting motion)
    - Smooth trajectory (penalizes jerky movements)
    """
    
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        
        # Parameters from config
        self.lift_height = cfg.params.get("lift_height", 1.5)  # Target height in meters
        self.min_lift_height = cfg.params.get("min_lift_height", 1.0)  # Minimum acceptable height
        self.max_lift_vel = cfg.params.get("max_lift_vel", 2.0)  # Max desired lift speed
        
        # Track previous height for velocity calculation
        self.prev_height = torch.zeros(env.num_envs, device=env.device)
        
    def reset(self, env_ids: torch.Tensor):
        """Reset height tracking for new episodes."""
        self.prev_height[env_ids] = 0.0
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        hand_name: str = "right_hand",
    ) -> torch.Tensor:
        # Get hand position
        asset: Articulation = env.scene[asset_cfg.name]
        hand_idx = asset.body_names.index(hand_name)
        hand_pos = asset.data.body_pos_w[:, hand_idx, :3]
        current_height = hand_pos[:, 2]
        
        # --- Lift Reward Components ---
        reward = torch.zeros(env.num_envs, device=env.device)
        
        # 1. Height achievement reward (dominant component)
        # Ranges from 0 (at min_lift_height) to 1.0 (at lift_height)
        height_progress = (current_height - self.min_lift_height) / (self.lift_height - self.min_lift_height)
        height_progress = torch.clamp(height_progress, 0.0, 1.0)
        
        # Use a shaped reward: small reward for getting off ground, big reward for reaching target
        lift_reward = torch.where(
            current_height >= self.lift_height,
            1.0,  # Full reward at target height
            height_progress * 0.3  # Partial reward for partial lifting
        )
        reward += lift_reward * 0.5  # Weight this component
        
        # 2. Upward velocity bonus (encourages active lifting, not just holding)
        upward_vel = (current_height - self.prev_height) / env.step_dt
        self.prev_height[:] = current_height
        
        # Only reward positive (upward) velocity, capped at max_lift_vel
        upward_bonus = torch.clamp(upward_vel / self.max_lift_vel, 0.0, 1.0)
        reward += upward_bonus * 0.2  # Smaller weight than height reward
        
        # 3. Height maintenance bonus (once at target, keep it there)
        at_target = current_height >= self.lift_height
        maintenance_bonus = torch.where(
            at_target,
            0.1,  # Small continuous reward for maintaining height
            0.0
        )
        reward += maintenance_bonus
        
        return reward


class hand_waving_reward(ManagerTermBase):
    """Reward for lifting hand and performing waving motion.
    Rewards:
    - Vertical hand position (lifting)
    - Rhythmic oscillation (waving frequency)
    - Smooth motion (penalizes jerky movements)
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)   
        # Buffers for tracking hand motion
        self.hand_pos_history = torch.zeros(env.num_envs, 3, 10, device=env.device)  # Last 10 positions
        self.hand_vel_history = torch.zeros(env.num_envs, 3, 10, device=env.device)  # Last 10 velocities
        self.prev_hand_pos = torch.zeros(env.num_envs, 3, device=env.device)

        # Wave parameters
        self.target_wave_freq = cfg.params.get("target_frequency", 2.0)  # 2 Hz waving
        self.target_wave_amp = cfg.params.get("target_amplitude", 0.3)   # 30cm amplitude
        self.lift_height = cfg.params.get("lift_height", 1.5)            # Target lift height

    def reset(self, env_ids: torch.Tensor):
        """Reset history buffers for new episodes."""
        self.hand_pos_history[env_ids] = 0
        self.hand_vel_history[env_ids] = 0
        self.prev_hand_pos[env_ids] = 0
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        hand_name: str = "right_hand",
    ) -> torch.Tensor:
        # Get hand position
        asset: Articulation = env.scene[asset_cfg.name]
        hand_idx = asset.body_names.index(hand_name)
        hand_pos = asset.data.body_pos_w[:, hand_idx, :3]
        
        # Store in history
        self.hand_pos_history = torch.roll(self.hand_pos_history, 1, dims=2)
        self.hand_pos_history[:, :, 0] = hand_pos
        
        # Compute velocity
        hand_vel = (hand_pos - self.prev_hand_pos) / env.step_dt
        self.prev_hand_pos[:] = hand_pos
        
        self.hand_vel_history = torch.roll(self.hand_vel_history, 1, dims=2)
        self.hand_vel_history[:, :, 0] = hand_vel
        
        # --- Reward components ---
        reward = torch.zeros(env.num_envs, device=env.device)
        
        # 1. Lift reward: encourage hand to reach target height
        lift_error = torch.abs(hand_pos[:, 2] - self.lift_height)
        lift_reward = torch.exp(-lift_error / 0.2)  # Gaussian-like reward
        reward += lift_reward * 0.5
        
        # 2. Wave reward: encourage oscillating motion
        if env.episode_length_buf[0] > 20:  # Need enough history
            # Extract horizontal motion (x-axis)
            x_motion = self.hand_pos_history[:, 0, :20]
            
            # Compute oscillation frequency using FFT
            fft = torch.fft.rfft(x_motion, dim=1)
            freq_magnitudes = torch.abs(fft)
            
            # Find peak frequency
            freqs = torch.fft.rfftfreq(20, d=env.step_dt, device=env.device)
            target_freq_idx = torch.argmin(torch.abs(freqs - self.target_wave_freq))
            wave_reward = freq_magnitudes[:, target_freq_idx] / (torch.sum(freq_magnitudes, dim=1) + 1e-6)
            reward += wave_reward * 0.3
            
        # 3. Smoothness penalty: penalize high accelerations
        if env.episode_length_buf[0] > 2:
            acceleration = (self.hand_vel_history[:, :, 0] - self.hand_vel_history[:, :, 1]) / env.step_dt
            accel_penalty = torch.norm(acceleration, dim=1)
            reward -= accel_penalty * 0.05
        
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
