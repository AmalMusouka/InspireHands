import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

DRIVERS = [
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
]

FOLLOWERS = {
    "right_thumb_3_joint": ("right_thumb_2_joint", 0.0, 1.1425),
    "right_index_2_joint": ("right_index_1_joint", -0.15, 1.1169),
    "right_middle_2_joint": ("right_middle_1_joint", -0.15, 1.1169),
    "right_ring_2_joint": ("right_ring_1_joint", -0.15, 1.1169),
    "right_little_2_joint": ("right_little_1_joint", -0.15, 1.1169),
}


class HandActionTerm(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._hand = env.scene[cfg.asset_name]
        names = list(self._hand.joint_names)
        self._n_joints = len(names)
        self._drv = [names.index(n) for n in DRIVERS]
        self._fol = [(names.index(k), names.index(v[0]), v[1], v[2]) for k, v in FOLLOWERS.items()]
        self._t3 = names.index("right_thumb_3_joint")
        self._t4 = names.index("right_thumb_4_joint")
        limits = self._hand.data.joint_pos_limits.torch[0]
        self._lo = limits[self._drv, 0]
        self._hi = limits[self._drv, 1]
        default_pose = self._hand.data.default_root_pose.torch
        self._pos0 = default_pose[:, :3].clone()
        self._quat0 = default_pose[:, 3:7].clone()
        self._pos_t = self._pos0.clone()
        self._eul_t = torch.zeros(self.num_envs, 3, device=self.device)
        self._raw = torch.zeros(self.num_envs, 12, device=self.device)
        self._proc = torch.zeros(self.num_envs, 12, device=self.device)
        self._zero_vel = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self):
        return 12

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._proc

    def process_actions(self, actions):
        a = torch.clamp(actions, -1.0, 1.0)
        self._raw = a
        c = self.cfg
        self._pos_t = torch.clamp(
            self._pos_t + a[:, :3] * c.pos_step, self._pos0 - c.pos_limit, self._pos0 + c.pos_limit
        )
        self._eul_t = torch.clamp(self._eul_t + a[:, 3:6] * c.rot_step, -c.rot_limit, c.rot_limit)
        self._proc = a

    def apply_actions(self):
        a = self._proc
        quat = quat_mul(
            quat_from_euler_xyz(self._eul_t[:, 0], self._eul_t[:, 1], self._eul_t[:, 2]), self._quat0
        )
        pose = torch.cat([self._pos_t + self._env.scene.env_origins, quat], dim=1)
        self._hand.write_root_pose_to_sim_index(root_pose=pose)
        self._hand.write_root_velocity_to_sim_index(root_velocity=self._zero_vel)
        q = torch.zeros(self.num_envs, self._n_joints, device=self.device)
        q[:, self._drv] = self._lo + 0.5 * (a[:, 6:12] + 1.0) * (self._hi - self._lo)
        for child, parent, c0, c1 in self._fol:
            q[:, child] = c0 + c1 * q[:, parent]
        q[:, self._t4] = 0.1 + 0.7508 * q[:, self._t3]
        lim = self._hand.data.joint_pos_limits.torch[0]
        q = torch.max(torch.min(q, lim[:, 1]), lim[:, 0])
        self._hand.set_joint_position_target_index(target=q)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._pos_t[env_ids] = self._pos0[env_ids]
        self._eul_t[env_ids] = 0.0
        self._raw[env_ids] = 0.0
        self._proc[env_ids] = 0.0


@configclass
class HandActionCfg(ActionTermCfg):
    class_type: type = HandActionTerm
    asset_name: str = "hand"
    pos_step: float = 0.02
    rot_step: float = 0.1
    pos_limit: float = 0.6
    rot_limit: float = 1.0


def hand_pos_local(env, hand_name="hand"):
    return env.scene[hand_name].data.root_pos_w.torch - env.scene.env_origins


def cube_pos_rel_hand(env, hand_name="hand", cube_name="cube"):
    return env.scene[cube_name].data.root_pos_w.torch - env.scene[hand_name].data.root_pos_w.torch


def cube_to_hand_distance(env, hand_name="hand", cube_name="cube"):
    return torch.norm(cube_pos_rel_hand(env, hand_name, cube_name), dim=1)


def cube_height(env, rest_z, cap, cube_name="cube"):
    z = env.scene[cube_name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(z - rest_z, 0.0, cap)


def cube_lifted(env, rest_z, threshold, cube_name="cube"):
    return (cube_height(env, rest_z, 1.0, cube_name) > threshold).float()
