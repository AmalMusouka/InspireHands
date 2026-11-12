import argparse
import time

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
        print("PROBE sim", cfg.sim)
        print("PROBE viewer", getattr(cfg, "viewer", "none"))
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            print("PROBE has_gui", env.sim.has_gui)
        except Exception as e:
            print("PROBE has_gui error", e)
        env.reset()
        actions = torch.zeros(env.num_envs, 12, device=env.device)
        for i in range(300):
            env.step(actions)
            time.sleep(0.02)
            if i % 50 == 0:
                print("PROBE step", i)
        print("PROBE DONE")


if __name__ == "__main__":
    main()
