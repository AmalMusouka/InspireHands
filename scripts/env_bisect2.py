import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="drivers")
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        import torch

        from isaaclab.envs import ManagerBasedRLEnv

        from InspireHandsManipulation.tasks import relocate_mdp as rmdp
        from InspireHandsManipulation.tasks.relocate_env_cfg import RelocateEnvCfg

        mode = args_cli.mode
        counter = [0]

        def apply(self):
            a = self._proc
            lim = self._hand.data.joint_pos_limits.torch[0]
            drv = self._lo + 0.5 * (a[:, 6:12] + 1.0) * (self._hi - self._lo)
            base = torch.zeros(self.num_envs, self._n_joints, device=self.device)
            base[:, self._drv] = drv
            q = torch.zeros_like(base) if mode == "followers" else base.clone()
            if mode != "drivers":
                for child, parent, c0, c1 in self._fol:
                    q[:, child] = c0 + c1 * base[:, parent]
                q[:, self._t4] = 0.1 + 0.7508 * q[:, self._t3]
            if mode in ("clamped", "ramp"):
                q = torch.max(torch.min(q, lim[:, 1]), lim[:, 0])
            if mode == "ramp":
                counter[0] += 1
                q = q * min(1.0, counter[0] / 60.0)
            self._hand.set_joint_position_target_index(target=q)

        rmdp.HandActionTerm.apply_actions = apply

        cfg = RelocateEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        hand = env.scene["hand"]
        actions = torch.zeros(1, 12, device=env.device)
        for step in range(150):
            env.step(actions)
            jp = hand.data.joint_pos.torch
            if not torch.isfinite(jp).all():
                print("NaN at step", step, "mode", mode)
                break
        else:
            print("mode", mode, "finite, max joint", round(jp.abs().max().item(), 3))


if __name__ == "__main__":
    main()
