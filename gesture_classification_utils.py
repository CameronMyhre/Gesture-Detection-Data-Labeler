"""
Utilities for preparing YOLO pose-estimation keypoints for gesture classification.

This module converts raw human-pose detections into a consistent feature
representation suitable for input to a gesture-classification model. It
normalizes joint positions relative to the detected person's torso midpoint
and torso height, removes facial keypoints, and computes selected anatomical
angle features for the arms.

The resulting feature vector contains normalized joint coordinates, keypoint
confidence values, and sine/cosine representations of elbow and upper-arm
angles. Invalid or degenerate poses are rejected to prevent malformed values
from entering the classification pipeline.
"""

__author__ = "Cameron Myhre"

from ultralytics.engine.results import Keypoints
import numpy as np
from numpy.typing import NDArray
import warnings

from nicegui import app, events, ui

# Constants
MIN_CONFIDENCE = 0.25
MIN_STRAIGHTNESS = 0.5

T_POSE_MAX_ANGLE = np.radians(26)
Y_POSE_MIN_ANGLE = np.radians(30)
FETCH_ANGLE_OFFSET = np.radians(45)
DOWN_ANGLE = -np.radians(70)


def normalize_figure(
    keypoints: Keypoints,
    person_index=0,
) -> NDArray | None:
    """
    Normalize a person's pose relative to their shoulders.

    The midpoint of the torso becomes (0, 0), and the distance between
    the shoulder and hip midpoints becomes 1. The skeleton is rotated
    such that the shoulder-to-hip vector aligns with the positive y-axis.

    Args:
        keypoints: Ultralytics Keypoints object.

    Returns:
        NumPy array with shape (3, 12):
            [normalized_x, normalized_y, confidence]

        Facial keypoints are omitted.
    """

    # 0) Verify that the keypoints aren't malformed.
    if keypoints is None or keypoints.conf is None:
        return None

    if len(keypoints.xy) <= person_index:
        return None

    # 1) Get the numpy representation of the joint coordinates and confidence.
    points_xy = keypoints.xy[person_index].cpu().numpy()
    confidence = keypoints.conf[person_index].cpu().numpy()

    # 1a) Verify that our keypoint data is reasonably accurate.
    if np.any(confidence[[5, 6, 11, 12]] < MIN_CONFIDENCE):
        return None

    # 2) Compute the torso midpoint position and torso height.
    #    All points will be scaled to be measurable in torso height
    #    and positioned relative to the torso center.
    left_shoulder = points_xy[5]
    right_shoulder = points_xy[6]

    left_hip = points_xy[11]
    right_hip = points_xy[12]

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    torso_height = np.hypot(
        hip_center_x - shoulder_center_x,
        hip_center_y - shoulder_center_y,
    )

    center_x = (hip_center_x + shoulder_center_x) / 2
    center_y = (hip_center_y + shoulder_center_y) / 2

    # Torso height of zero would lead to division by zero.
    # Further, no gesture would require this condition.
    # Thus it's safe to omit this frame.
    if torso_height < 1e-10:
        return None

    # 3) Normalize for camera roll.
    rotation_angle = np.atan2(
        hip_center_x - shoulder_center_x,
        hip_center_y - shoulder_center_y,
    )

    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    # 4) Normalize all data points.
    normalized_xy = (points_xy - np.array((center_x, center_y))) / torso_height

    # 5) Rotate all keypoints.
    normalized_xy = normalized_xy @ rotation_matrix.T

    normalized = np.vstack(
        (
            normalized_xy.T,
            confidence,
        )
    )

    # Remove YOLO facial keypoints 0-4.
    return normalized[:, 5:17]


def compute_joint_angles(
    normalized_joints: NDArray,
) -> NDArray | None:
    """
    Computes the anatomical elbow angles and upper-arm orientations
    for the left and right arms.

    Each angle is represented using cosine and sine.

    Args:
        normalized_joints:
            NumPy array containing normalized joint data.

    Returns:
        NumPy array containing:
            [right_elbow_cos, right_elbow_sin,
             left_elbow_cos, left_elbow_sin,
             right_upper_cos, right_upper_sin,
             left_upper_cos, left_upper_sin]

        Shape: (8,)

        Returns None if a required joint vector has near-zero magnitude.
    """

    # 0) Ensure that the joints have reasonable confidence before computing
    #    any angles.
    if np.any(normalized_joints[2, :6] < MIN_CONFIDENCE):
        return None

    # 1) Compute cos(theta) and sin(theta) for the right elbow.
    # 1a) Vectorize the right arm joints.
    right_elbow_to_shoulder = np.array(
        [
            normalized_joints[0, 1] - normalized_joints[0, 3],
            normalized_joints[1, 1] - normalized_joints[1, 3],
        ]
    )

    right_elbow_to_wrist = np.array(
        [
            normalized_joints[0, 5] - normalized_joints[0, 3],
            normalized_joints[1, 5] - normalized_joints[1, 3],
        ]
    )

    # 1b) Compute the magnitudes of the joint vectors.
    right_shoulder_mag = np.linalg.norm(right_elbow_to_shoulder)
    right_wrist_mag = np.linalg.norm(right_elbow_to_wrist)

    # We don't want to train NaN in our data.
    if right_shoulder_mag < 1e-10 or right_wrist_mag < 1e-10:
        return None

    # 1c) Use vector prop   erties to compute cos(theta) and sin(theta).
    # We represent the angle using cosine and sine so angles near -pi and pi
    # are not interpreted as being far apart.
    right_elbow_cos = np.dot(right_elbow_to_shoulder, right_elbow_to_wrist) / (
        right_shoulder_mag * right_wrist_mag
    )

    # Use the signed 2D cross product to preserve the sign of sin(theta).
    right_elbow_sin = (
        right_elbow_to_shoulder[0] * right_elbow_to_wrist[1]
        - right_elbow_to_shoulder[1] * right_elbow_to_wrist[0]
    ) / (right_shoulder_mag * right_wrist_mag)

    # 2) Compute cos(theta) and sin(theta) for the left elbow.
    # 2a) Vectorize the left arm joints.
    left_elbow_to_shoulder = np.array(
        [
            normalized_joints[0, 0] - normalized_joints[0, 2],
            normalized_joints[1, 0] - normalized_joints[1, 2],
        ]
    )

    left_elbow_to_wrist = np.array(
        [
            normalized_joints[0, 4] - normalized_joints[0, 2],
            normalized_joints[1, 4] - normalized_joints[1, 2],
        ]
    )

    # 2b) Compute the magnitudes of the joint vectors.
    left_shoulder_mag = np.linalg.norm(left_elbow_to_shoulder)
    left_wrist_mag = np.linalg.norm(left_elbow_to_wrist)

    # We don't want to train NaN in our data.
    if left_shoulder_mag < 1e-10 or left_wrist_mag < 1e-10:
        return None

    # 2c) Use vector properties to compute cos(theta) and sin(theta).
    left_elbow_cos = np.dot(left_elbow_to_shoulder, left_elbow_to_wrist) / (
        left_shoulder_mag * left_wrist_mag
    )

    # Use the signed 2D cross product to preserve the sign of sin(theta).
    left_elbow_sin = (
        left_elbow_to_shoulder[0] * left_elbow_to_wrist[1]
        - left_elbow_to_shoulder[1] * left_elbow_to_wrist[0]
    ) / (left_shoulder_mag * left_wrist_mag)

    # 3) Compute cos(theta) and sin(theta) for the right upper arm.
    # 3a) Vectorize the right shoulder-to-elbow segment.
    right_upper_arm = np.array(
        [
            normalized_joints[0, 3] - normalized_joints[0, 1],
            normalized_joints[1, 3] - normalized_joints[1, 1],
        ]
    )

    # 3b) Compute the magnitude of the upper-arm vector.
    right_upper_mag = np.linalg.norm(right_upper_arm)

    # We don't want to train NaN in our data.
    if right_upper_mag < 1e-10:
        return None

    # 3c) Compute cos(theta) and sin(theta) relative to the positive x-axis.
    right_upper_cos = right_upper_arm[0] / right_upper_mag
    right_upper_sin = right_upper_arm[1] / right_upper_mag

    # 4) Compute cos(theta) and sin(theta) for the left upper arm.
    # 4a) Vectorize the left shoulder-to-elbow segment.
    left_upper_arm = np.array(
        [
            normalized_joints[0, 2] - normalized_joints[0, 0],
            normalized_joints[1, 2] - normalized_joints[1, 0],
        ]
    )

    # 4b) Compute the magnitude of the upper-arm vector.
    left_upper_mag = np.linalg.norm(left_upper_arm)

    # We don't want to train NaN in our data.
    if left_upper_mag < 1e-10:
        return None

    # 4c) Compute cos(theta) and sin(theta) relative to the positive x-axis.
    left_upper_cos = left_upper_arm[0] / left_upper_mag
    left_upper_sin = left_upper_arm[1] / left_upper_mag

    # 5) Return all angle features.
    return np.array(
        [
            right_elbow_cos,
            right_elbow_sin,
            left_elbow_cos,
            left_elbow_sin,
            right_upper_cos,
            right_upper_sin,
            left_upper_cos,
            left_upper_sin,
        ]
    )


def classical_classify(normalized_joints: NDArray) -> int:

    # 1) Extract the joint points.
    right_shoulder = normalized_joints[:2, 1]
    right_elbow = normalized_joints[:2, 3]
    right_wrist = normalized_joints[:2, 5]

    left_shoulder = normalized_joints[:2, 0]
    left_elbow = normalized_joints[:2, 2]
    left_wrist = normalized_joints[:2, 4]

    # 2) Compute the bend of each joint.
    right_arm_bend = compute_menger_straightness(
        right_shoulder, right_elbow, right_wrist
    )

    left_arm_bend = compute_menger_straightness(left_shoulder, left_elbow, left_wrist)

    # 3) Normalize the x coordinates so that angles lie in the first and fourth quadrants..
    right_x_multiplier = 1
    if right_shoulder[0] < left_shoulder[0]:
        right_x_multiplier = -1

    # 4) Compute the joint angles from shoulder to elbow.
    right_shoulder_elbow = np.arctan2(  # Y is inverted, so invert it here.
        -1 * (right_elbow[1] - right_shoulder[1]),
        right_x_multiplier * (right_elbow[0] - right_shoulder[0]),
    )

    left_shoulder_elbow = np.arctan2(  # Y is inverted, so invert it here.
        -1 * (left_elbow[1] - left_shoulder[1]),
        -right_x_multiplier * (left_elbow[0] - left_shoulder[0]),
    )

    # 5) Reject overly bent limbs.
    print(f"right: {right_arm_bend}, left: {left_arm_bend}")
    print(f"Angle: R: {right_shoulder_elbow} L: {left_shoulder_elbow}")
    if right_arm_bend < MIN_STRAIGHTNESS or left_arm_bend < MIN_STRAIGHTNESS:
        print("Not straight enough!")
        ui.notify("Breakpoint A!")

        return 0  # None
    print(f"Angle <Normalized>: R: {right_shoulder_elbow} L: {left_shoulder_elbow}")

    # 7) Classify all gestures.
    # 7a) Classify the follow gesture.
    if (right_shoulder_elbow > Y_POSE_MIN_ANGLE) and (
        left_shoulder_elbow > Y_POSE_MIN_ANGLE
    ):
        return 1

    # 7b) Classify the stay gesture.
    if (np.abs(right_shoulder_elbow) < T_POSE_MAX_ANGLE) and (
        np.abs(left_shoulder_elbow) < T_POSE_MAX_ANGLE
    ):
        return 2

    # 7c) Classify the fetch gesture.
    if (np.abs(left_shoulder_elbow + FETCH_ANGLE_OFFSET) < T_POSE_MAX_ANGLE) and (
        np.abs(right_shoulder_elbow) < T_POSE_MAX_ANGLE
    ):
        return 3

    # 7d) Classify the come gesture.
    if (left_shoulder_elbow < DOWN_ANGLE) and (
        np.abs(right_shoulder_elbow) < T_POSE_MAX_ANGLE
    ):
        return 4

    # 7e) Classify the give gesture.
    if (right_shoulder_elbow < DOWN_ANGLE) and (
        np.abs(left_shoulder_elbow) < T_POSE_MAX_ANGLE
    ):
        return 5

    # 7f) Action didn't match any gesture.
    print("No gesture found.")
    return 0


def compute_menger_straightness(
    point_a: NDArray, point_b: NDArray, point_c: NDArray
) -> float:
    """
    Computes the menger straightness between three given points.

    Args:
        point_a: A numpy array representing an x,y point shape (,2)
        point_b: A numpy array representing an x,y point shape (,2)
        point_c: A numpy array representing an x,y point shape (,2)

    Returns:
        The Menger straightness of the three given points, as a float.
    """

    # Compute the area of the formed triangle.
    area = 0.5 * np.abs(
        point_a[0] * (point_b[1] - point_c[1])
        + point_b[0] * (point_c[1] - point_a[1])
        + point_c[0] * (point_a[1] - point_b[1])
    )

    # Finally, compute the curvature.
    return 1 - 2 * area / (
        np.dot(point_a, point_b) + np.dot(point_b, point_c) + np.dot(point_a, point_c)
    )


def process_keypoints_for_classification(
    keypoints: Keypoints,
) -> NDArray | None:
    """
    Converts YOLO pose-estimation keypoints into features for gesture classification.

    Joint coordinates are normalized relative to the person's shoulder midpoint
    and shoulder width. Facial keypoints are removed, and selected arm angles
    are appended as sine/cosine pairs.

    Args:
        keypoints:
            Ultralytics Keypoints object containing the detected pose.

    Returns:
        NumPy array with shape (44,) containing:
            - 12 normalized non-facial joint x-coordinates
            - 12 normalized non-facial joint y-coordinates
            - 12 joint confidence values
            - 8 anatomical angle features

        Returns None if normalization or angle computation fails.
    """

    # 1) Normalize the keypoints.
    normalized_joints = normalize_figure(keypoints)
    if normalized_joints is None:
        warnings.warn("[gesture_classification_utils]: Failed to normalize keypoints.")
        return None

    # 2) Compute the anatomical angles for the normalized joints.
    joint_angles = compute_joint_angles(normalized_joints)
    if joint_angles is None:
        warnings.warn(
            "[gesture_classification_utils]: Failed to compute anatomical joint angles for keypoints."
        )
        return None

    # 3) Flatten and combine joint and angle features into one feature vector.
    return np.concatenate(
        (
            normalized_joints.ravel(),
            joint_angles,
        )
    )
