import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--reroot", type=int, default=1)
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
        term = env.action_manager.get_term("hand")

        if args_cli.reroot:
            orig_root = hand.write_root_pose_to_sim_index
            orig_joint = hand.write_joint_position_to_sim_index
            last = {}

            def root(**kw):
                last.update(kw)
                return orig_root(**kw)

            def joint(**kw):
                r = orig_joint(**kw)
                if last:
                    orig_root(**last)
                return r

            hand.write_root_pose_to_sim_index = root
            hand.write_joint_position_to_sim_index = joint

        actions = torch.zeros(1, 12, device=env.device)
        for step in range(50):
            actions.zero_()
            if 10 <= step < 40:
                actions[:, 2] = -1.0
            env.step(actions)
            if step % 5 == 0:
                print(step, "target_z", round(term._pos_t[0, 2].item(), 3),
                      "root_z", round(hand.data.root_pos_w.torch[0, 2].item(), 3))


if __name__ == "__main__":
    main()
