import argparse
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
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
        cfg.seed = 42
        cfg.scene.num_envs = args_cli.num_envs
        cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        actions = torch.zeros(env.num_envs, 12, device=env.device)
        step = 0
        for _ in range(600):
            phase = step % 150
            actions.zero_()
            actions[:, 6:] = -1.0
            if 30 <= phase < 48:
                actions[:, 2] = -1.0
            if phase >= 60:
                actions[:, 6:] = 1.0
            if 100 <= phase < 118:
                actions[:, 2] = 1.0
            obs, rew, term, trunc, info = env.step(actions)
            time.sleep(0.02)
            step += 1
            if step % 15 == 0:
                hand_z = env.scene["hand"].data.root_pos_w.torch[0, 2].item()
                cube_z = env.scene["cube"].data.root_pos_w.torch[0, 2].item()
                print(phase, round(hand_z, 3), round(cube_z, 3), rew[0].item())


if __name__ == "__main__":
    main()
