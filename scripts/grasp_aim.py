import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--offx", type=float, default=0.0)
parser.add_argument("--offy", type=float, default=0.09)
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
        cube = env.scene["cube"]
        actions = torch.zeros(1, 12, device=env.device)
        for step in range(140):
            actions.zero_()
            actions[:, 6:] = -1.0
            c = cube.data.root_pos_w.torch[0]
            h = hand.data.root_pos_w.torch[0]
            if step < 40:
                err = (c[:2] - h[:2]) + torch.tensor([-args_cli.offx, -args_cli.offy], device=env.device)
                actions[0, :2] = torch.clamp(err / 0.02, -1.0, 1.0)
            elif step < 58:
                actions[0, 2] = -1.0
            elif step < 75:
                actions[:, 6:] = 1.0
            else:
                actions[:, 6:] = 1.0
                if step < 93:
                    actions[0, 2] = 1.0
            env.step(actions)
            if step % 10 == 0 or step == 139:
                print("S", step, "hand", [round(v, 3) for v in hand.data.root_pos_w.torch[0].tolist()],
                      "cube", [round(v, 3) for v in cube.data.root_pos_w.torch[0].tolist()])
        print("DONE")


if __name__ == "__main__":
    main()
