"""Published Cosmos3 embodiment-domain IDs and raw action widths.

``ACTION_DOMAIN_IDS`` maps recipe / dataset names onto the integer
``action_domain_id`` tensor the omni transformer expects.
``ACTION_RAW_DIMS`` is the per-domain action-vector width used by
:class:`~...initializers.cosmos3.Cosmos3LatentInitializer` when allocating
action latents.  Several aliases share an ID (for example Agibot variants
all use 15).
"""

ACTION_DOMAIN_IDS = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}

ACTION_RAW_DIMS = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "galbot": 30,
    "agibotworld": 29,
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    "fractal": 10,
    "hand_pose": 57,
}

__all__ = ["ACTION_DOMAIN_IDS", "ACTION_RAW_DIMS"]
