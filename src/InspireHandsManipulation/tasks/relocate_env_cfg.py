import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import relocate_mdp as rmdp
from .table_scene_cfg import TableSceneCfg

CUBE_REST_Z = 0.7785


@configclass
class ActionsCfg:
    hand = rmdp.HandActionCfg(asset_name="hand")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        hand_pos = ObsTerm(func=rmdp.hand_pos_local)
        finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("hand", joint_names=rmdp.DRIVERS)},
        )
        cube_rel = ObsTerm(func=rmdp.cube_pos_rel_hand)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    approach = RewTerm(func=rmdp.cube_to_hand_distance, weight=-0.1)
    lifted = RewTerm(func=rmdp.cube_lifted, weight=1.0, params={"rest_z": CUBE_REST_Z, "threshold": 0.04})
    height = RewTerm(func=rmdp.cube_height, weight=5.0, params={"rest_z": CUBE_REST_Z, "cap": 0.2})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_fell = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RelocateEnvCfg(ManagerBasedRLEnvCfg):
    scene: TableSceneCfg = TableSceneCfg(num_envs=64, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
