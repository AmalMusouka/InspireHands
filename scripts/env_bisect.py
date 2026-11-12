import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="none")
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        import torch

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

        from InspireHandsManipulation.tasks import relocate_mdp as rmdp
        from InspireHandsManipulation.tasks.relocate_env_cfg import RelocateEnvCfg

        term = rmdp.HandActionTerm
        printed = []

        def apply_root(self):
            quat = quat_mul(
                quat_from_euler_xyz(self._eul_t[:, 0], self._eul_t[:, 1], self._eul_t[:, 2]), self._quat0
            )
            pose = torch.cat([self._pos_t + self._env.scene.env_origins, quat], dim=1)
            if not printed:
                print("POSE", pose[0].tolist(), "quat0", self._quat0[0].tolist())
                printed.append(1)
            self._hand.write_root_pose_to_sim_index(root_pose=pose)
            self._hand.write_root_velocity_to_sim_index(root_velocity=self._zero_vel)

        def apply_fingers(self):
            a = self._proc
            q = torch.zeros(self.num_envs, self._n_joints, device=self.device)
            q[:, self._drv] = self._lo + 0.5 * (a[:, 6:12] + 1.0) * (self._hi - self._lo)
            for child, parent, c0, c1 in self._fol:
                q[:, child] = c0 + c1 * q[:, parent]
            q[:, self._t4] = 0.1 + 0.7508 * q[:, self._t3]
            if not printed:
                print("TARGETS", q[0].tolist())
                printed.append(1)
            self._hand.set_joint_position_target_index(target=q)

        full = term.apply_actions
        if args_cli.mode == "none":
            term.apply_actions = lambda self: None
        elif args_cli.mode == "root":
            term.apply_actions = apply_root
        elif args_cli.mode == "fingers":
            term.apply_actions = apply_fingers

        cfg = RelocateEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        hand = env.scene["hand"]
        actions = torch.zeros(1, 12, device=env.device)
        for step in range(100):
            env.step(actions)
            root = hand.data.root_pos_w.torch
            jp = hand.data.joint_pos.torch
            if not (torch.isfinite(root).all() and torch.isfinite(jp).all()):
                print("NaN at step", step, "mode", args_cli.mode)
                break
        else:
            print("mode", args_cli.mode, "finite for 100 steps")


if __name__ == "__main__":
    main()
