# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.humanoid import HUMANOID_CFG  # isort:skip


##
# Scene definition
##


@configclass
class HandWaversSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
        )
        # Add hand position observation
        hand_pos = ObsTerm(
            func=mdp.body_pos_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand"])}
        )
        actions = ObsTerm(func=mdp.last_action)
        hand_pos = ObsTerm(
            func=mdp.body_pos_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand"])}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Primary: Hand waving motion
    hand_waving = RewTerm(
        func=mdp.hand_waving_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hand_name": "right_hand",
            "target_height": 1.4,  # Reasonable shoulder height
            "wave_amplitude": 0.25,  # 25cm side-to-side motion
        }
    )

    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=1.5)
    
    # (3) Reward for maintaining upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=1.5, params={"threshold": 0.93})
    
    # (4) CRITICAL: Keep feet on ground
    feet_contact = RewTerm(
        func=mdp.feet_on_ground_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])}
    )
    
    # (5) Penalize base height changes (prevent jumping)
    base_height_penalty = RewTerm(
        func=mdp.base_height_stability,
        weight=-2.0,
        params={"target_height": 1.1}  # Typical humanoid standing height
    )
    
    # (6) Penalize excessive base velocity (no jumping/moving around)
    base_lin_vel_penalty = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0
    )
    
    # (7) Penalize body rotation instability
    body_orientation_penalty = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5
    )
    
    # (8) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    
    # (9) Penalty for action rate (smoothness)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # (10) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.005,
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )
    
    # (11) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            },
        },
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})


##
# Environment configuration
##


@configclass
class HandWaversEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid waving environment."""

    # Scene settings
    scene: HandWaversSceneCfg = HandWaversSceneCfg(num_envs=1000, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0