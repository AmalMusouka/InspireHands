import argparse
import time
from dataclasses import replace

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        import torch
        import omni.kit.app

        from isaaclab.envs import ManagerBasedRLEnv

        from InspireHandsManipulation.tasks.relocate_env_cfg import RelocateEnvCfg

        cfg = RelocateEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args_cli.device
        cfg.episode_length_s = 600.0
        cfg.scene.hand.init_state = replace(cfg.scene.hand.init_state, rot=(0.0, 0.7071, 0.0, 0.7071))
        env = ManagerBasedRLEnv(cfg=cfg)
        hand = env.scene["hand"]
        cube = env.scene["cube"]
        dev = env.device
        actions = torch.zeros(1, 12, device=dev)
        warned = []

        def step():
            env.step(actions)
            try:
                omni.kit.app.get_app().update()
            except Exception as e:
                if not warned:
                    print("RENDER error:", e)
                    warned.append(1)
            time.sleep(0.03)

        def goto(tgt, fingers, steps):
            for _ in range(steps):
                actions.zero_()
                actions[:, 6:] = fingers
                err = tgt - hand.data.root_pos_w.torch[0]
                actions[0, :3] = torch.clamp(err / 0.02, -1.0, 1.0)
                step()

        env.reset()
        for _ in range(30):
            step()
        c0 = cube.data.root_pos_w.torch[0].clone()
        d, h = 0.125, 0.057
        x, y, z0 = c0[0].item() - d, c0[1].item(), c0[2].item()
        tgt = lambda z: torch.tensor([x, y, z], device=dev)
        print("PHASE approach"); goto(tgt(1.0), -1.0, 40)
        print("PHASE lower"); goto(tgt(z0 + h), -1.0, 50)
        print("PHASE close"); goto(tgt(z0 + h), 1.0, 30)
        print("PHASE lift"); goto(tgt(z0 + h + 0.3), 1.0, 50)
        print("PHASE hold, cube", [round(v, 3) for v in cube.data.root_pos_w.torch[0].tolist()])
        for _ in range(500):
            goto(tgt(z0 + h + 0.3), 1.0, 1)
        print("DONE")


if __name__ == "__main__":
    main()
