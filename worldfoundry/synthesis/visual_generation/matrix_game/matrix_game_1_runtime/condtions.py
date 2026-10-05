def Bench_actions_76(action_names=None, num_frames=65):
    actions_single_action = [
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "attack"
    ]
    actions_double_action = [
        "forward_attack",
        "back_attack",
        "left_attack",
        "right_attack",
        "jump_attack",
        "forward_left",
        "forward_right",
        "back_left",
        "back_right",
        "forward_jump",
        "back_jump",
        "left_jump",
        "right_jump",

    ]

    actions_single_camera = [    
        "camera_up",
        "camera_down",
        "camera_l",
        "camera_r",
        "camera_ur",
        "camera_ul",
        "camera_dl",
        "camera_dr"
    ]
    actions_to_test = actions_double_action
    for action in actions_single_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    base_action = actions_single_action + actions_single_camera
    if action_names is not None:
        aliases = {"backward": "back", "backward_left": "back_left", "backward_right": "back_right"}
        requested = [aliases.get(action, action) for action in action_names]
        supported = set(actions_to_test + base_action + ["idle", "nomove"])
        unknown = set(requested) - supported
        if unknown:
            raise ValueError(f"Unsupported Matrix-Game-1 interactions: {sorted(unknown)}")
        actions_to_test = requested

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3,
        "jump": 4,  "attack": 5
    }

    CAM_VALUE = 0.05
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }


    num_samples_per_action = int(num_frames)
    if num_samples_per_action < 1:
        raise ValueError("num_frames must be positive")

    data = []

    for action_name in actions_to_test:
        # 前，后，左，右，跳跃，攻击
        keyboard_condition = [[0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if sub_act not in action_name: # 只处理action_name包含的动作
                continue
            if sub_act in CAMERA_VALUE_MAP: # camera_dr
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act == "attack":
                # to do 只有帧数 (idx % 16 >= 8) & (idx % 16 < 16)才为1
                for idx in range(num_samples_per_action):
                    if idx % 8 == 0:
                        keyboard_condition[idx][KEYBOARD_IDX["attack"]] = 1

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "action_name": action_name,
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition
        })

    return data


def interaction_condition(interactions, num_frames):
    """Spread requested command segments across one video using official controls."""
    if not interactions or len(interactions) > num_frames:
        raise ValueError("Provide between one and num_frames interaction segments.")
    conditions = Bench_actions_76(interactions, num_frames=num_frames)
    result = {"action_name": "requested", "keyboard_condition": [], "mouse_condition": []}
    count, remainder = divmod(num_frames, len(conditions))
    for index, condition in enumerate(conditions):
        length = count + (index < remainder)
        for key in ("keyboard_condition", "mouse_condition"):
            result[key].extend(condition[key][:length])
    return result
