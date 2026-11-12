import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="none")
parser.add_argument("--dt", type=float, default=0.0)
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    kwargs = {"dt": args_cli.dt} if args_cli.dt > 0 else {}
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, **kwargs)
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
        names = list(hand.joint_names)
        n = len(names)
        dev = args_cli.device
        root_pose = hand.data.default_root_pose.torch.clone()
        zero_vel = torch.zeros(1, 6, device=dev)
        q = torch.zeros(1, n, device=dev)
        zeros_n = torch.zeros(1, n, device=dev)
        if args_cli.mode == "one":
            q[:, names.index("right_index_1_joint")] = 0.2
        if args_cli.mode == "all":
            q[:] = 0.05
        if args_cli.mode == "hard":
            q[:, names.index("right_index_1_joint")] = 0.2
        print("dt", dt, "mode", args_cli.mode)
        for step in range(100):
            hand.write_root_pose_to_sim_index(root_pose=root_pose)
            hand.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            if args_cli.mode in ("one", "all"):
                hand.set_joint_position_target_index(target=q)
            if args_cli.mode == "hard":
                hand.write_joint_position_to_sim_index(position=q)
                hand.write_joint_velocity_to_sim_index(velocity=zeros_n)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            jp = hand.data.joint_pos.torch
            if not torch.isfinite(jp).all():
                print("NaN at step", step)
                break
        else:
            print("finite, max joint", round(jp.abs().max().item(), 3))


if __name__ == "__main__":
    main()
