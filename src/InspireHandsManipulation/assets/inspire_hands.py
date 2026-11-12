import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

DRIVEN = [
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
]

FOLLOWERS = [
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_2_joint",
    "right_middle_2_joint",
    "right_ring_2_joint",
    "right_little_2_joint",
]

INSPIRE_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/amalmusouka/inspire_assets/module_5_end/inspire_hand/inspire_hand.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.9, 0.0, 1.3), rot=(0.0, 1.0, 0.0, 0.0)),
    actuators={
        "driven": ImplicitActuatorCfg(
            joint_names_expr=DRIVEN,
            stiffness=None,
            damping=None,
            armature=0.01,
        ),
        "followers": ImplicitActuatorCfg(
            joint_names_expr=FOLLOWERS,
            stiffness=0.3,
            damping=0.03,
            effort_limit_sim=2.0,
            velocity_limit_sim=5.0,
            armature=0.01,
        ),
    },
)
