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
        cfg.episode_length_s = 60.0
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        hand = env.scene["hand"]
        names = list(hand.body_names)
        tips = [i for i, n in enumerate(names) if n.endswith("_tip")]
        actions = torch.zeros(1, 12, device=env.device)
        for val in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 1.0):
            actions.zero_()
            actions[:, 6:] = val
            for _ in range(40):
                env.step(actions)
            root = hand.data.root_pos_w.torch[0]
            bp = hand.data.body_pos_w.torch[0]
            row = []
            for i in tips:
                d = bp[i] - root
                row.append(names[i].replace("right_", "").replace("_tip", "")
                           + "(%.3f,%.3f,%.3f)" % (d[0].item(), d[1].item(), d[2].item()))
            print("F", val, " ".join(row))
        print("DONE")


if __name__ == "__main__":
    main()
