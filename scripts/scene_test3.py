import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="one")
parser.add_argument("--selfcol", type=str, default="keep")
parser.add_argument("--iters", type=int, default=0)
parser.add_argument("--arm", type=float, default=0.0)
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
        scene_cfg = TableSceneCfg(num_envs=1, env_spacing=3.0)
        props = {}
        if args_cli.selfcol == "off":
            props["enabled_self_collisions"] = False
        if args_cli.iters > 0:
            props["solver_position_iteration_count"] = args_cli.iters
            props["solver_velocity_iteration_count"] = 1
        if props:
            scene_cfg.hand.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(**props)
        if args_cli.arm > 0:
            for act in scene_cfg.hand.actuators.values():
                act.armature = args_cli.arm
        scene = instantiate(scene_cfg)
        sim.reset()
        hand = scene["hand"]
        dt = sim.get_physics_dt()
        names = list(hand.joint_names)
        dev = args_cli.device
        base_pose = hand.data.default_root_pose.torch.clone()
        zero_vel = torch.zeros(1, 6, device=dev)
        q = torch.zeros(1, len(names), device=dev)
        q[:, names.index("right_index_1_joint")] = 0.2
        for step in range(100):
            pose = base_pose.clone()
            if args_cli.mode == "move":
                pose[:, 2] = base_pose[:, 2] - 0.002 * step
            hand.write_root_pose_to_sim_index(root_pose=pose)
            hand.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            if args_cli.mode == "one":
                hand.set_joint_position_target_index(target=q)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            jp = hand.data.joint_pos.torch
            if not torch.isfinite(jp).all() or not torch.isfinite(hand.data.root_pos_w.torch).all():
                print("NaN at step", step)
                break
        else:
            print("finite, root z", round(hand.data.root_pos_w.torch[0, 2].item(), 3),
                  "(expect 1.102 for move)", "max joint", round(jp.abs().max().item(), 3))


if __name__ == "__main__":
    main()
