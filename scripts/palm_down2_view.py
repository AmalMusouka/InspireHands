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

        from isaaclab.envs import ManagerBasedRLEnv

        from InspireHandsManipulation.tasks.relocate_env_cfg import RelocateEnvCfg

        cfg = RelocateEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args_cli.device
        cfg.episode_length_s = 60.0
        cfg.scene.hand.init_state = replace(cfg.scene.hand.init_state, rot=(0.0, 0.7071, 0.0, 0.7071))
        env = ManagerBasedRLEnv(cfg=cfg)
        _step = env.step
        def slow(act):
            time.sleep(0.02)
            return _step(act)
        env.step = slow
        hand = env.scene["hand"]
        cube = env.scene["cube"]
        dev = env.device
        names = list(hand.body_names)
        tips = [i for i, n in enumerate(names) if n.endswith("_tip")]
        actions = torch.zeros(1, 12, device=dev)

        def goto(tgt, fingers, steps):
            for _ in range(steps):
                actions.zero_()
                actions[:, 6:] = fingers
                err = tgt - hand.data.root_pos_w.torch[0]
                actions[0, :3] = torch.clamp(err / 0.02, -1.0, 1.0)
                env.step(actions)

        for d in (0.11, 0.125, 0.14):
            for h in (0.05, 0.057, 0.065):
                env.reset()
                c0 = cube.data.root_pos_w.torch[0].clone()
                x, y, z0 = c0[0].item() - d, c0[1].item(), c0[2].item()
                tgt = lambda z: torch.tensor([x, y, z], device=dev)
                goto(tgt(1.0), -1.0, 40)
                goto(tgt(z0 + h), -1.0, 50)
                c1 = cube.data.root_pos_w.torch[0].clone()
                goto(tgt(z0 + h), 1.0, 30)
                c2 = cube.data.root_pos_w.torch[0].clone()
                goto(tgt(z0 + h + 0.3), 1.0, 50)
                c3 = cube.data.root_pos_w.torch[0]
                print("T d", d, "h", h,
                      "desc_move", round((c1[:2] - c0[:2]).norm().item(), 3),
                      "close_move", round((c2[:2] - c1[:2]).norm().item(), 3),
                      "final_z", round(c3[2].item(), 3),
                      "LIFTED" if c3[2].item() > 0.85 else "no")
        time.sleep(30)
        print("DONE")


if __name__ == "__main__":
    main()
