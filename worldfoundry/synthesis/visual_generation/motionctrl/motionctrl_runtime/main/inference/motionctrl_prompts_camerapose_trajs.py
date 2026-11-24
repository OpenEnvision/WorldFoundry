from __future__ import annotations

complex_camera_poses = [
    "test_camera_d971457c81bca597",
    "test_camera_d971457c81bca597",
    "test_camera_d971457c81bca597",
    "test_camera_Round-ZoomIn",
    "test_camera_Round-ZoomIn",
    "test_camera_Round-ZoomIn",
]
complex_camera_pose_prompt = [
    "a temple on a mountain, bird's view",
    "Effiel Tower in Paris, bird's view",
    "a castle in a forest, bird's view",
    "a temple on a mountain, bird's view",
    "Effiel Tower in Paris, bird's view",
    "a castle in a forest, bird's view",
]

basic_camera_poses = [
    "test_camera_L",
    "test_camera_D",
    "test_camera_I",
    "test_camera_O",
    "test_camera_R",
    "test_camera_U",
    "test_camera_SPIN-CW-60",
    "test_camera_SPIN-ACW-60",
]
basic_camera_pose_prompt = [
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
    "coastline, rocks, storm weather, wind, waves, lightning",
]

diff_speeds_camera_poses = [
    "test_camera_I_0.2x",
    "test_camera_I_0.4x",
    "test_camera_I_1.0x",
    "test_camera_I_2.0x",
    "test_camera_O_0.2x",
    "test_camera_O_0.4x",
    "test_camera_O_1.0x",
    "test_camera_O_2.0x",
]
diff_speeds_camera_pose_prompt = [
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
    "A sunrise landscape features mountains and lakes",
]

cmcm_prompt_camerapose = {
    "prompts": complex_camera_pose_prompt + basic_camera_pose_prompt + diff_speeds_camera_pose_prompt,
    "camera_poses": complex_camera_poses + basic_camera_poses + diff_speeds_camera_poses,
}

omom_prompt_traj = {
    "prompts": [
        "a sunflower swaying in the wind",
        "a rose swaying in the wind",
        "a wind chime swaying in the wind",
        "a man surfing",
        "a man skateboarding",
        "a girl skiing",
    ],
    "trajs": ["shake_1", "shake_1", "shake_1", "curve_2", "curve_2", "curve_2"],
}

both_prompt_camerapose_traj = {
    "prompts": ["a rose swaying in the wind"],
    "camera_poses": ["test_camera_O"],
    "trajs": ["shaking_10"],
}

assert len(cmcm_prompt_camerapose["prompts"]) == len(cmcm_prompt_camerapose["camera_poses"])
assert len(omom_prompt_traj["prompts"]) == len(omom_prompt_traj["trajs"])
assert (
    len(both_prompt_camerapose_traj["prompts"])
    == len(both_prompt_camerapose_traj["camera_poses"])
    == len(both_prompt_camerapose_traj["trajs"])
)
