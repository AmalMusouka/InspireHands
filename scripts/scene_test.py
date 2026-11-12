import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--followers", type=str, default="keep")
parser.add_argument("--hold", type=int, default=1)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        import torch

        from isaaclab.sim import SimulationContext
        from isaaclab.utils import instantiate

        from InspireHandsManipulation.tasks.table_scene_cfg import TableSceneCfg

        sim = SimulationContext(sim_cfg)
        scene_cfg = TableSceneCfg(num_envs=1, env_spacing=3.0)
        if args_cli.followers == "drop":
            scene_cfg.hand.actuators.pop("followers")
        scene = instantiate(scene_cfg)
        sim.reset()
        hand = scene["hand"]
        dt = sim.get_physics_dt()
        root_pose = hand.data.default_root_pose.torch.clone()
        zero_vel = torch.zeros(1, 6, device=args_cli.device)
        for step in range(150):
            if args_cli.hold:
                hand.write_root_pose_to_sim_index(root_pose=root_pose)
                hand.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            jp = hand.data.joint_pos.torch
            if not torch.isfinite(jp).all():
                print("NaN at step", step)
                break
            if step % 10 == 0:
                print(step, round(jp.abs().max().item(), 3))
        else:
            print("finite for 150 steps")


if __name__ == "__main__":
    main()
