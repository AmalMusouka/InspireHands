import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=0)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        import torch

        from isaaclab.envs import ManagerBasedRLEnv

        from InspireHandsManipulation.tasks.relocate_env_cfg import RelocateEnvCfg

        cfg = RelocateEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        hand = env.scene["hand"]
        actions = torch.zeros(1, 12, device=env.device)
        for step in range(150):
            actions.zero_()
            if args_cli.mode == 1:
                actions[:, 6:] = 1.0
            if args_cli.mode == 2 and step >= 20:
                actions[:, 2] = -1.0
            env.step(actions)
            root = hand.data.root_pos_w.torch
            jp = hand.data.joint_pos.torch
            if not (torch.isfinite(root).all() and torch.isfinite(jp).all()):
                print("NaN at step", step, "mode", args_cli.mode)
                print(root.tolist())
                print(jp.tolist())
                break
            if step % 10 == 0:
                print(step, round(root[0, 2].item(), 3), round(jp.abs().max().item(), 3))
        else:
            print("mode", args_cli.mode, "stayed finite for 150 steps")


if __name__ == "__main__":
    main()
