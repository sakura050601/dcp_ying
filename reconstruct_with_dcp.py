import os

# import sys

"""
Baseline RGB-D 3D reconstruction pipeline (Python, ICP + TSDF).

High-level logic:
1. Load a TUM RGB-D sequence (depth + camera poses T_wB_i).
2. Estimate a global TSDF volume bounding box from the first M frames
   using ground-truth cam->world poses.
3. Choose how to obtain camera poses for fusion:
   - POSE_MODE = "gt":  use ground-truth T_wB_i (cam->world).
   - POSE_MODE = "icp": use poses accumulated from pairwise ICP only.
   - POSE_MODE = "icp_noise": use ICP poses + small synthetic noise.
4. Integrate all depth frames into a TSDF volume with the chosen poses.
5. Extract and save a mesh as the baseline 3D-scanner result.

This script is intended as a baseline for later experiments
(e.g., DCP-based pose estimation, multi-scale ICP, improved TSDF, etc.).
"""

# If needed, add project root to Python path:
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../project_root/debug
# ROOT_DIR = os.path.dirname(THIS_DIR)  # .../project_root
# if ROOT_DIR not in sys.path:
#     sys.path.insert(0, ROOT_DIR)

import numpy as np
import open3d as o3d

from data_tum import TUMRGBDDepthPairs
from tsdf_fusion import TSDFVolumeNumpy as TSDFVolume
from util import get_logger

log = get_logger()

from icp_utils import (
    skew,
    build_icp_poses,
)


def add_pose_noise(T, rot_sigma_deg=1.0, trans_sigma=0.01):
    """
    Add a small left-multiplicative noise ΔT to a cam->world pose T:

        T_noisy = ΔT * T

    rot_sigma_deg : std. dev. of rotation noise (degrees, axis-angle magnitude)
    trans_sigma   : std. dev. of translation noise (meters)
    """
    # ---- 1) Sample a small axis-angle rotation ----
    theta = np.deg2rad(rot_sigma_deg) * np.random.randn()  # |θ| ~ rot_sigma_deg
    axis = np.random.randn(3).astype(np.float32)
    norm = np.linalg.norm(axis) + 1e-8
    axis = axis / norm

    K = skew(axis)
    I = np.eye(3, dtype=np.float32)
    # Rodrigues formula to get δR
    delta_R = I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    # ---- 2) Sample a small translation noise ----
    delta_t = trans_sigma * np.random.randn(3).astype(np.float32)

    # ---- 3) Assemble ΔT and left-multiply T ----
    Delta = np.eye(4, dtype=np.float32)
    Delta[:3, :3] = delta_R
    Delta[:3, 3] = delta_t

    # Still a cam->world pose
    T_noisy = Delta @ T
    theta_deg = float(np.rad2deg(theta))
    log.debug(f"[NOISE] rot≈{theta_deg:.3f} deg, |dt|={np.linalg.norm(delta_t):.4f} m")
    return T_noisy


def compute_scene_bounds(dataset, max_frames=None):
    """
    Estimate a global scene bounding box in world coordinates.

    Use the first `max_frames` depth frames and ground-truth cam->world poses T_wB_i
    to accumulate the min/max of all 3D points in world space, and return:

        pts_min, pts_max : 3D vectors defining the bounding box.
    """
    pts_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    pts_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    N = len(dataset) if max_frames is None else min(max_frames, len(dataset))

    for idx in range(N):
        # Load depth and convert to float meters
        depth_u16 = dataset._load_depth(dataset.depth_items[idx][1])
        depth_f = depth_u16.astype(np.float32) * dataset.scale

        # Backproject to cam-space points (simple pinhole model)
        H, W = depth_f.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth_f
        m = z > 0
        u = u[m]
        v = v[m]
        z = z[m]

        fx, fy = dataset.K[0, 0], dataset.K[1, 1]
        cx, cy = dataset.K[0, 2], dataset.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)

        # Transform cam-space points to world-space using GT T_wB_i
        data_i = dataset[idx]
        *_, T_wA_i, T_wB_i = data_i
        R_wc = T_wB_i[:3, :3]
        t_wc = T_wB_i[:3, 3]

        # cam -> world (same as in world_frame_0000 export)
        pts_world_i = pts_cam @ R_wc.T + t_wc

        # Update global min/max
        pts_min = np.minimum(pts_min, pts_world_i.min(axis=0))
        pts_max = np.maximum(pts_max, pts_world_i.max(axis=0))

        if idx % 50 == 0:
            log.info(f"[BBOX] pass1 frame {idx}/{N}")

    return pts_min, pts_max


def main():
    # Pose selection mode for TSDF fusion:
    #   "gt"        : use ground-truth T_wB_i (cam->world)
    #   "icp"       : use poses from chained ICP
    #   "icp_noise" : use ICP poses with added random noise
    POSE_MODE = "gt"  # ["gt", "icp", "icp_noise"]

    tum_root = "dataset/rgbd_dataset_freiburg1_xyz"

    # ====== Step 1: Load TUM RGB-D dataset ======
    dataset = TUMRGBDDepthPairs(
        root=tum_root,
        depth_list_txt="depth.txt",
        pose_txt="poses.txt",
        intrinsics_path="intrinsics.txt",
        num_points=2048,
    )
    log.info(f"Loaded dataset, pairs = {len(dataset)}")

    out_dir = "debug/debug_single_tsdf"
    os.makedirs(out_dir, exist_ok=True)

    # ====== Step 2: Estimate global TSDF volume bounds from first M frames ======
    N = max(1, len(dataset))
    M = N  # Use the whole sequence for bounding box estimation
    pts_min, pts_max = compute_scene_bounds(dataset, max_frames=M)
    padding = 0.1
    vol_min = pts_min - padding
    vol_max = pts_max + padding
    vol_size = vol_max - vol_min

    voxel_res = 256
    voxel_size = np.max(vol_size) / voxel_res
    trunc_margin = 5 * voxel_size

    vol_bounds = np.stack([vol_min, vol_max], axis=1)  # (3, 2)

    log.info(f"[TSDF-frame] voxel_size={voxel_size}, trunc={trunc_margin}")

    # Initialize TSDF volume
    tsdf = TSDFVolume(
        vol_bounds=vol_bounds,
        voxel_size=voxel_size,
        trunc_margin=trunc_margin,
    )

    # ====== Step 3: Prepare pose source (GT / ICP / ICP+noise) ======
    ROT_SIGMA_DEG = 1.0  # ~1 deg per frame
    TRANS_SIGMA_M = 0.01  # ~1 cm per frame

    # Pre-compute ICP-based poses (cam->world)
    T_icp_list = build_icp_poses(dataset, N)

    use_noise = POSE_MODE == "icp_noise"
    print(f"[TSDF] Integrating {N} frames... (USE_NOISE={use_noise})")

    # ====== Step 4: TSDF integration over all frames ======
    for idx in range(N):
        # (1) Load this frame's depth
        depth_u16 = dataset._load_depth(dataset.depth_items[idx][1])
        depth_f = depth_u16.astype(np.float32) * dataset.scale

        # (2) Select cam->world pose according to POSE_MODE
        *_, T_wA_i, T_wB_i = dataset[idx]

        if POSE_MODE == "gt":
            cam_pose = T_wB_i
        elif POSE_MODE == "icp":
            cam_pose = T_icp_list[idx]
        elif POSE_MODE == "icp_noise":
            cam_pose = add_pose_noise(
                T_icp_list[idx],
                rot_sigma_deg=ROT_SIGMA_DEG,
                trans_sigma=TRANS_SIGMA_M,
            )
            # Log noise injection every 50 frames
            if idx % 50 == 0:
                log.info(f"[NOISE] frame {idx}: pose perturbed.")
        else:
            raise ValueError(f"Unknown POSE_MODE={POSE_MODE}")

        # (3) Integrate into TSDF
        try:
            tsdf.integrate(depth_f, dataset.K, cam_pose)
        except Exception as e:
            log.warning(f"[TSDF] Frame {idx}: integrate failed: {e}")
            continue

    # ====== Step 5: Extract and save mesh ======
    verts, faces = tsdf.extract_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    tsdf_path = os.path.join(out_dir, f"tsdf_{N}.stl")
    o3d.io.write_triangle_mesh(tsdf_path, mesh)
    log.info(
        f"[Step] Saved TSDF mesh → {tsdf_path}, "
        f"verts={verts.shape[0]}, faces={faces.shape[0]}"
    )


if __name__ == "__main__":
    np.random.seed(0)
    main()
