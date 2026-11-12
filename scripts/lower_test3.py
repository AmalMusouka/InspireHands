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
        term = env.action_manager.get_term("hand")
        calls = {"root": 0, "joint": 0}
        orig_root = hand.write_root_pose_to_sim_index
        orig_joint = hand.write_joint_position_to_sim_index

        def root(**kw):
            calls["root"] += 1
            calls["written_z"] = kw["root_pose"][0, 2].item()
            return orig_root(**kw)

        def joint(**kw):
            calls["joint"] += 1
            return orig_joint(**kw)

        hand.write_root_pose_to_sim_index = root
        hand.write_joint_position_to_sim_index = joint
        actions = torch.zeros(1, 12, device=env.device)
        for step in range(30):
            actions.zero_()
            if step >= 10:
                actions[:, 2] = -1.0
            env.step(actions)
            if step % 5 == 0:
                print(step, "written_z", round(calls.get("written_z", -1), 3),
                      "read_z", round(hand.data.root_pos_w.torch[0, 2].item(), 3),
                      "calls", calls["root"], calls["joint"])


if __name__ == "__main__":
    main()
