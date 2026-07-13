import numpy as np
from scipy.spatial.transform import Rotation

from spatialwm.geometry.tartanair import (
    compute_se3_error,
    derive_relative_transform,
    parse_pose_to_transform,
    select_target_frames,
)


def test_parse_pose_to_transform_identity():
    # Identity pose (zero translation, identity rotation quaternion)
    pose_row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    T = parse_pose_to_transform(pose_row)

    assert T.shape == (4, 4)
    # Check that it is an orthogonal matrix with translation 0
    assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(T[:3, 3], 0.0)

    # Check rotation structure:
    # T_cam = T_ned2cam @ T_ned @ T_cam2ned
    # Since T_ned = Identity, T_cam should be T_ned2cam @ T_cam2ned = Identity
    assert np.allclose(T[:3, :3], np.identity(3))


def test_parse_pose_to_transform_translation():
    # Only translation
    pose_row = np.array([1.5, -2.5, 3.0, 0.0, 0.0, 0.0, 1.0])
    T = parse_pose_to_transform(pose_row)

    # Check translation mapping
    # T_cam = T_ned2cam @ T_ned @ T_cam2ned
    # T_ned = [ I | t_ned ], T_ned2cam is:
    # [ 0 1 0 0 ]
    # [ 0 0 1 0 ]
    # [ 1 0 0 0 ]
    # [ 0 0 0 1 ]
    # Thus the translation component of T_cam is T_ned2cam[:3, :4] @ [t_ned | 1]^T
    # = [ t_ned[1], t_ned[2], t_ned[0] ] = [ -2.5, 3.0, 1.5 ]
    expected_t = np.array([-2.5, 3.0, 1.5])
    assert np.allclose(T[:3, 3], expected_t)
    assert np.allclose(T[:3, :3], np.identity(3))


def test_derive_relative_transform():
    # Two synthetic poses
    pose1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # Pose 2: translated and rotated 90 deg around local Z of the camera (which is forward)
    # 90 degrees around Z: cos(45) = sin(45) = 0.70710678
    # Camera RDF local Z is Z_cam = X_ned. So in NED, rotation is 90 deg around X_ned.
    # Quaternion around X_ned (North): [sin(45), 0, 0, cos(45)] = [0.7071, 0, 0, 0.7071]
    pose2 = np.array([1.0, 2.0, 3.0, 0.70710678, 0.0, 0.0, 0.70710678])

    T1 = parse_pose_to_transform(pose1)
    T2 = parse_pose_to_transform(pose2)

    T_rel = derive_relative_transform(T1, T2)

    # Compose: T2 @ T_rel should equal T1
    assert np.allclose(T2 @ T_rel, T1)


def test_compute_se3_error():
    # Identity transform
    T_gt = np.identity(4)

    # Translation only
    T_est_t = np.identity(4)
    T_est_t[:3, 3] = [0.1, -0.2, 0.3]  # norm = sqrt(0.01 + 0.04 + 0.09) = sqrt(0.14) = 0.3741657
    t_err, r_err = compute_se3_error(T_est_t, T_gt)
    assert np.allclose(t_err, np.sqrt(0.14))
    assert np.allclose(r_err, 0.0)

    # Rotation only: 10 degrees around X
    R = Rotation.from_euler("x", 10.0, degrees=True).as_matrix()
    T_est_r = np.identity(4)
    T_est_r[:3, :3] = R
    t_err, r_err = compute_se3_error(T_est_r, T_gt)
    assert np.allclose(t_err, 0.0)
    assert np.allclose(r_err, 10.0)


def test_select_target_frames_deterministic():
    # Generate synthetic sequence of 10 poses with constant translation steps
    # translation along X_ned: t_ned = [i * 0.1, 0, 0]
    poses = []
    for i in range(10):
        poses.append([i * 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    poses = np.array(poses)

    # Let's select for target baseline 0.25m from source index 0.
    # The actual distances are:
    # idx 1: 0.1m (diff = 0.15)
    # idx 2: 0.2m (diff = 0.05)
    # idx 3: 0.3m (diff = 0.05)
    # idx 4: 0.4m (diff = 0.15)
    # Distance to target 0.25 is equal for idx 2 and idx 3 (diff = 0.05).
    # Since we use strict inequality, our tie-breaker should choose index 2.
    results = select_target_frames(
        poses, source_idx=0, target_baselines=[0.25], max_search_frames=5
    )

    assert len(results) == 1
    assert results[0]["target_idx"] == 2
    assert np.allclose(results[0]["actual_baseline"], 0.2)
    assert np.allclose(results[0]["rotation_deg"], 0.0)
