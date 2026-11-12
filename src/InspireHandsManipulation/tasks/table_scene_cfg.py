import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import MassCfg
from isaaclab.utils import configclass, replace

from InspireHandsManipulation.assets.inspire_hands import INSPIRE_HAND_CFG


@configclass
class TableSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.6, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.22)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.9, 0.0, 0.375)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.057, 0.057, 0.057),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=MassCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.0, 0.78)),
    )

    hand: ArticulationCfg = replace(INSPIRE_HAND_CFG, prim_path="{ENV_REGEX_NS}/Hand")
