import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
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
        names = list(hand.body_names)
        actions = torch.zeros(1, 12, device=env.device)
        for step in range(100):
            phase = step
            actions.zero_()
            actions[:, 6:] = -1.0
            if 30 <= phase < 48:
                actions[:, 2] = -1.0
            if phase >= 60:
                actions[:, 6:] = 1.0
            env.step(actions)
            if phase in (50, 58, 95):
                c = cube.data.root_pos_w.torch[0]
                bp = hand.data.body_pos_w.torch[0]
                print("PHASE", phase, "cube", [round(v, 3) for v in c.tolist()],
                      "hand", [round(v, 3) for v in hand.data.root_pos_w.torch[0].tolist()])
                order = torch.argsort(bp[:, 2])[:8]
                for i in order.tolist():
                    d = bp[i] - c
                    print("BODY", phase, names[i], "dx", round(d[0].item(), 3),
                          "dy", round(d[1].item(), 3), "dz", round(d[2].item(), 3))
        print("DONE")


if __name__ == "__main__":
    main()
