import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stage", type=int, default=0)
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils import instantiate

from InspireHandsManipulation.tasks.table_scene_cfg import TableSceneCfg

STAGE = args_cli.stage


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([1.8, 1.2, 1.4], [0.9, 0.0, 1.0])
        scene = instantiate(TableSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
        sim.reset()

        hand = scene["hand"]
        n = args_cli.num_envs
        dev = args_cli.device
        sim_dt = sim.get_physics_dt()
        names = list(hand.joint_names)
        lim = hand.data.joint_pos_limits.torch[0]
        root_pose = hand.data.default_root_pose.torch.clone()
        root_pose[:, :3] += scene.env_origins
        zero_vel = torch.zeros(n, 6, device=dev)
        q = torch.zeros(n, len(names), device=dev)
        if STAGE >= 3:
            i = names.index("right_thumb_1_joint")
            q[:, i] = 0.8 * lim[i, 1]

        print("STAGE", STAGE, "start pose", root_pose[0].tolist())
        step = 0
        while sim.is_running():
            if STAGE >= 1:
                hand.write_root_pose_to_sim_index(root_pose=root_pose)
                hand.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            if STAGE >= 2:
                hand.set_joint_position_target_index(target=q)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            step += 1
            if step % 100 == 0:
                print(step, hand.data.root_pos_w.torch[0].tolist())


if __name__ == "__main__":
    main()