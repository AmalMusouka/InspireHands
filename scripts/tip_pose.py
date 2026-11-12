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
        names = list(hand.body_names)
        tips = [i for i, n in enumerate(names) if n.endswith("_tip")]
        actions = torch.zeros(1, 12, device=env.device)
        for label, val in (("OPEN", -1.0), ("CLOSED", 1.0)):
            actions.zero_()
            actions[:, 6:] = val
            for _ in range(40):
                env.step(actions)
            root = hand.data.root_pos_w.torch[0]
            bp = hand.data.body_pos_w.torch[0]
            print(label, "root", [round(v, 3) for v in root.tolist()])
            for i in tips:
                d = bp[i] - root
                print(label, names[i], [round(v, 3) for v in d.tolist()])
        print("DONE")


if __name__ == "__main__":
    main()
