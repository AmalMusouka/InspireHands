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
        hand = env.scene["hand"]
        cube = env.scene["cube"]
        dev = env.device
        actions = torch.zeros(1, 12, device=dev)
        zero6 = torch.zeros(1, 6, device=dev)

        def place(off):
            pose = torch.zeros(1, 7, device=dev)
            pose[0, :3] = hand.data.root_pos_w.torch[0] + torch.tensor(off, device=dev)
            pose[0, 3] = 1.0
            cube.write_root_pose_to_sim_index(root_pose=pose)
            cube.write_root_velocity_to_sim_index(root_velocity=zero6)

        for dx in (-0.08, -0.05, -0.02, 0.01, 0.04):
            for dz in (-0.07, -0.10, -0.13, -0.16):
                env.reset()
                actions.zero_()
                actions[:, 6:] = -1.0
                for _ in range(20):
                    env.step(actions)
                actions[:, 6:] = 1.0
                for _ in range(25):
                    place((dx, 0.0, dz))
                    env.step(actions)
                for _ in range(30):
                    env.step(actions)
                rel = cube.data.root_pos_w.torch[0] - hand.data.root_pos_w.torch[0]
                held = "HELD" if cube.data.root_pos_w.torch[0, 2].item() > 1.0 else "dropped"
                print("R", dx, dz, [round(v, 3) for v in rel.tolist()], held)
        print("DONE")


if __name__ == "__main__":
    main()
