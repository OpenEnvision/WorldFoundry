import numpy as np
from scipy.spatial import distance


def analyze_lengths_over_time(instance_info, threshold=0.45, min_length=1e-6, confidence_threshold=0.3, window_size=1):
    """
    Calculate the normalized motion amplitude for each batch sample

    Parameters:
    keypoints: torch.Tensor, shape [batch_size, 49, 792, 2]
    video_width: int, width of the video
    video_height: int, height of the video

    Returns:
    motion_amplitudes: torch.Tensor, shape [batch_size], containing the normalized motion amplitude for each batch sample
    """
    video_keypoints = [instance["instances"][0]["keypoints"] for instance in instance_info]
    video_keypoint_scores = [instance["instances"][0]["keypoint_scores"] for instance in instance_info]

    num_frames = len(video_keypoints)
    if num_frames < window_size:
        return "Not enough frames for analysis", 0  # Return error message and score of 0

    # Ensure window size is an odd number
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window_size // 2

    body_parts = [
        ("torso", 5, 11),
        ("left_upper_arm", 5, 7),
        ("left_forearm", 7, 9),
        ("right_upper_arm", 6, 8),
        ("right_forearm", 8, 10),
        ("left_thigh", 11, 13),
        ("left_calf", 13, 15),
        ("right_thigh", 12, 14),
        ("right_calf", 14, 16),
        ("shoulder_width", 5, 6),
        ("hip_width", 11, 12),
        ("neck", 0, 5),
    ]

    lengths = {part[0]: [] for part in body_parts}
    valid_measurements = {part[0]: [] for part in body_parts}

    # Calculate lengths for each frame, considering confidence scores
    for frame_keypoints, frame_scores in zip(video_keypoints, video_keypoint_scores, strict=False):
        for part_name, start_idx, end_idx in body_parts:
            start_point = np.array(frame_keypoints[start_idx])
            end_point = np.array(frame_keypoints[end_idx])
            start_score = frame_scores[start_idx]
            end_score = frame_scores[end_idx]

            if start_score > confidence_threshold and end_score > confidence_threshold:
                length = distance.euclidean(start_point, end_point)
                lengths[part_name].append(length)
                valid_measurements[part_name].append(True)
            else:
                lengths[part_name].append(None)
                valid_measurements[part_name].append(False)

    # Analyze length changes using a sliding window centered on each frame
    anomalies = []
    normal_parts = 0  # Count of normal body parts
    total_parts = len(body_parts)  # Total number of body parts

    for part_name in lengths:
        valid_lengths = [
            length
            for length, is_valid in zip(lengths[part_name], valid_measurements[part_name], strict=False)
            if is_valid and length is not None and length > min_length
        ]

        if len(valid_lengths) < num_frames // 2:
            # If there are not enough valid measurements, consider the part to be normal
            normal_parts += 1
            continue

        window_averages = []
        for i in range(len(valid_lengths)):
            start = max(0, i - half_window)
            end = min(len(valid_lengths), i + half_window + 1)
            window = valid_lengths[start:end]
            window_averages.append(np.mean(window))

        # Calculate relative change between adjacent window averages
        part_normal = True
        for i in range(len(window_averages) - 1):
            change = abs(window_averages[i + 1] - window_averages[i]) / window_averages[i]
            if change > threshold:
                anomalies.append(f"{part_name} shows significant change around frame {i}, change: {change:.2%}")
                part_normal = False
                break

        if part_normal:
            normal_parts += 1

    # Calculate score
    score = normal_parts / total_parts

    return anomalies, score


def calculate_angle(point1, point2, point3):
    """Calculate the angle formed by three points"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def analyze_joint_angles(instance_info, threshold=30, confidence_threshold=0.3, window_size=1):
    video_keypoints = [instance["instances"][0]["keypoints"] for instance in instance_info]
    video_keypoint_scores = [instance["instances"][0]["keypoint_scores"] for instance in instance_info]

    num_frames = len(video_keypoints)
    if num_frames < window_size:
        return "Not enough frames for analysis", 0  # Return error message and score of 0

    # Ensure window size is an odd number
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window_size // 2

    # Define joint angles to analyze
    joint_angles = [
        ("left_elbow", 5, 7, 9),  # Left shoulder - left elbow - left wrist
        ("right_elbow", 6, 8, 10),  # Right shoulder - right elbow - right wrist
        ("left_shoulder", 7, 5, 11),  # Left elbow - left shoulder - left hip
        ("right_shoulder", 8, 6, 12),  # Right elbow - right shoulder - right hip
        ("left_hip", 5, 11, 13),  # Left shoulder - left hip - left knee
        ("right_hip", 6, 12, 14),  # Right shoulder - right hip - right knee
        ("left_knee", 11, 13, 15),  # Left hip - left knee - left ankle
        ("right_knee", 12, 14, 16),  # Right hip - right knee - right ankle
    ]

    angles = {angle[0]: [] for angle in joint_angles}
    valid_measurements = {angle[0]: [] for angle in joint_angles}

    # Calculate angles for each frame, considering confidence scores
    for frame_keypoints, frame_scores in zip(video_keypoints, video_keypoint_scores, strict=False):
        for angle_name, p1, p2, p3 in joint_angles:
            if all(frame_scores[i] > confidence_threshold for i in [p1, p2, p3]):
                angle = calculate_angle(frame_keypoints[p1], frame_keypoints[p2], frame_keypoints[p3])
                angles[angle_name].append(angle)
                valid_measurements[angle_name].append(True)
            else:
                angles[angle_name].append(None)
                valid_measurements[angle_name].append(False)

    # Analyze angle changes using a sliding window centered on each frame
    anomalies = []
    normal_joints = 0
    total_joints = len(joint_angles)

    for angle_name in angles:
        valid_angles = [
            a for a, v in zip(angles[angle_name], valid_measurements[angle_name], strict=False) if v and a is not None
        ]

        if len(valid_angles) < num_frames // 3:
            # If there are not enough valid measurements, consider the joint to be normal
            normal_joints += 1
            continue

        window_averages = []
        for i in range(len(valid_angles)):
            start = max(0, i - half_window)
            end = min(len(valid_angles), i + half_window + 1)
            window = valid_angles[start:end]
            window_averages.append(np.mean(window))

        # Calculate changes between adjacent window averages
        changes = np.abs(np.diff(window_averages))

        if np.max(changes) <= threshold:
            normal_joints += 1
        else:
            for i, change in enumerate(changes):
                if change > threshold:
                    anomalies.append(
                        f"{angle_name} shows significant change around frame {i}, change: {change:.2f} degrees"
                    )
                    break

    # Calculate score
    score = normal_joints / total_joints

    return anomalies, score
