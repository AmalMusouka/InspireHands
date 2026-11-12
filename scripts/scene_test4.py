import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="pose")
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0)
    with launch_simulation(sim_cfg, args_cli):
        import torch

        from isaaclab.sim import SimulationContext
        from isaaclab.utils import instantiate

        from InspireHandsManipulation.tasks.table_scene_cfg import TableSceneCfg

        sim = SimulationContext(sim_cfg)
        scene = instantiate(TableSceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        hand = scene["hand"]
        dt = sim.get_physics_dt()
        print("FIXED", getattr(hand, "is_fixed_base", "unknown"))
        base = hand.data.default_root_pose.torch.clone()
        zero = torch.zeros(1, 6, device=args_cli.device)
        vel = zero.clone()
        vel[:, 2] = -0.2
        for step in range(60):
            if args_cli.mode == "pose":
                pose = base.clone()
                pose[:, 2] = base[:, 2] - 0.005 * step
                hand.write_root_pose_to_sim_index(root_pose=pose)
                hand.write_root_velocity_to_sim_index(root_velocity=zero)
            elif args_cli.mode == "once":
                if step == 0:
                    pose = base.clone()
                    pose[:, 2] = 1.0
                    hand.write_root_pose_to_sim_index(root_pose=pose)
            elif args_cli.mode == "vel":
                hand.write_root_velocity_to_sim_index(root_velocity=vel)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if step % 10 == 0 or step == 59:
                print("Z", args_cli.mode, step, round(hand.data.root_pos_w.torch[0, 2].item(), 3))


if __name__ == "__main__":
    main()
