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
        cfg.episode_length_s = 60.0
        env = ManagerBasedRLEnv(cfg=cfg)
        hand = env.scene["hand"]
        cube = env.scene["cube"]
        dev = env.device
        actions = torch.zeros(1, 12, device=dev)

        def goto(tgt, fingers, steps):
            for _ in range(steps):
                actions.zero_()
                actions[:, 6:] = fingers
                err = tgt - hand.data.root_pos_w.torch[0]
                actions[0, :3] = torch.clamp(err / 0.02, -1.0, 1.0)
                env.step(actions)

        def show(tag, rz, op):
            c = cube.data.root_pos_w.torch[0]
            h = hand.data.root_pos_w.torch[0]
            print("G", rz, op, tag, "hand", [round(v, 3) for v in h.tolist()],
                  "cube", [round(v, 3) for v in c.tolist()], "ep_step", int(env.episode_length_buf[0].item()))

        for rz in (0.89, 0.93, 0.97):
            for op in (-1.0, -0.5):
                env.reset()
                c0 = cube.data.root_pos_w.torch[0].clone()
                x, y = c0[0].item() + 0.05, c0[1].item()
                dev_t = lambda z: torch.tensor([x, y, z], device=dev)
                goto(dev_t(1.3), op, 40)
                show("aligned", rz, op)
                goto(dev_t(rz), op, 60)
                show("down", rz, op)
                goto(dev_t(rz), 1.0, 30)
                show("closed", rz, op)
                goto(dev_t(rz + 0.25), 1.0, 50)
                show("lifted", rz, op)
        print("DONE")


if __name__ == "__main__":
    main()
