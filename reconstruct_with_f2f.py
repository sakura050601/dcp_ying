#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KinFu-like RGB-D 3D reconstruction (Python, frame-to-model ICP + TSDF).

Core logic aligned with the Unity TestKinfu pipeline:
- Maintain a current camera pose `curr_pose` (cam -> world).
- For each frame:
    * Use `curr_pose` to generate a "model surface" from the TSDF (here approximated by a mesh point cloud).
    * Run frame-to-model ICP and get an incremental transform `Delta`.
    * Update the pose: `curr_pose = Delta @ curr_pose`.
    * Integrate the current depth frame into TSDF using the updated `curr_pose`.

Main differences from the old pipeline:
- No more frame-to-frame pose chaining via `build_icp_poses`.
- Use frame-to-model ICP instead; the error is always measured against the currently accumulated TSDF model,
  similar to Unity's RenderSurfacePrediction + ICPTracker.Track.
"""

import os

import numpy as np
import open3d as o3d

from data_tum import TUMRGBDDepthPairs
from tsdf_fusion import TSDFVolumeNumpy as TSDFVolume
from util import get_logger

log = get_logger()

# -------------------- small helpers --------------------


def depth_to_cam_points(
    depth_u16: np.ndarray, K: np.ndarray, scale: float
) -> np.ndarray:
    """
    Convert depth image (uint16) -> point cloud in camera coordinates (N, 3).
    Same geometric logic as the previous version.
    """
    depth_f = depth_u16.astype(np.float32) * scale

    H, W = depth_f.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z = depth_f
    m = z > 0
    u = u[m]
    v = v[m]
    z = z[m]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=1)
    return pts_cam.astype(np.float64)


def downsample_random(pc: np.ndarray, max_n: int = 20000) -> np.ndarray:
    """Randomly downsample a point cloud to at most `max_n` points."""
    if pc.shape[0] <= max_n:
        return pc
    idx = np.random.choice(pc.shape[0], max_n, replace=False)
    return pc[idx]


def make_o3d_pcd(points_world: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert (N, 3) numpy array (world-space) to an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    return pcd


def compute_scene_bounds(
    dataset: TUMRGBDDepthPairs, max_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roughly estimate the global scene bounding box (pts_min, pts_max) in world coordinates.

    Uses ground-truth cam->world poses T_wB_i for the first `max_frames` depth frames,
    following the same logic as the original `compute_scene_bounds`.
    """
    pts_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    pts_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    N = len(dataset) if max_frames is None else min(max_frames, len(dataset))

    for idx in range(N):
        depth_u16 = dataset._load_depth(dataset.depth_items[idx][1])
        depth_f = depth_u16.astype(np.float32) * dataset.scale

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

        # GT cam->world
        *_, T_wA_i, T_wB_i = dataset[idx]
        R_wc = T_wB_i[:3, :3]
        t_wc = T_wB_i[:3, 3]

        pts_world_i = pts_cam @ R_wc.T + t_wc

        pts_min = np.minimum(pts_min, pts_world_i.min(axis=0))
        pts_max = np.maximum(pts_max, pts_world_i.max(axis=0))

        if idx % 50 == 0:
            log.info(f"[BBOX] pass1 frame {idx}/{N}")

    return pts_min, pts_max


# -------------------- frame-to-model ICP  --------------------


def icp_frame_to_model_o3d(
    src_pts_world: np.ndarray,
    tgt_pts_world: np.ndarray,
    voxel_sizes=(0.05, 0.02, 0.01),
    max_iters=(20, 15, 10),
    distance_ths=(0.10, 0.05, 0.02),
    fitness_min: float = 0.4,
    rmse_max: float = 0.05,
) -> np.ndarray:
    """
    Multi-scale (pyramid) ICP for frame-to-model alignment in world coordinates.

    src_pts_world : current frame point cloud already transformed by `curr_pose` (world-space).
    tgt_pts_world : model point cloud extracted from the TSDF mesh (world-space).

    Returns:
        Delta (4x4): incremental transform such that
            X_world_aligned ≈ Delta @ X_world_src

        Final camera pose update:
            curr_pose_new = Delta @ curr_pose
    """
    if src_pts_world.shape[0] < 1000 or tgt_pts_world.shape[0] < 1000:
        # Too few points -> skip ICP (equivalent to Delta = I)
        log.warning("[ICP f2m] too few points, skip (Delta = I).")
        return np.eye(4, dtype=np.float64)

    assert len(voxel_sizes) == len(max_iters) == len(distance_ths)

    pcd_src_full = make_o3d_pcd(src_pts_world)
    pcd_tgt_full = make_o3d_pcd(tgt_pts_world)

    # Estimate normals on the model (used as planes)
    pcd_tgt_full.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    T = np.eye(4, dtype=np.float64)

    try:
        for level, (voxel, it, d_th) in enumerate(
            zip(voxel_sizes, max_iters, distance_ths)
        ):
            pcd_src = pcd_src_full.voxel_down_sample(voxel)
            pcd_tgt = pcd_tgt_full.voxel_down_sample(voxel)

            if len(pcd_src.points) < 1000 or len(pcd_tgt.points) < 1000:
                log.warning(
                    f"[ICP f2m] level {level}: too few points after voxel={voxel}, skip."
                )
                continue

            pcd_tgt.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )

            reg = o3d.pipelines.registration.registration_icp(
                pcd_src,
                pcd_tgt,
                max_correspondence_distance=d_th,
                init=T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=it
                ),
            )

            T = reg.transformation
            log.info(
                f"[ICP f2m] level {level}: "
                f"voxel={voxel}, it={it}, dist_th={d_th}, "
                f"fitness={reg.fitness:.3f}, rmse={reg.inlier_rmse:.4f}"
            )

        if not np.all(np.isfinite(T)):
            raise FloatingPointError("non-finite ICP transform")

        # If all levels were skipped, T stays identity -> "no pose update".
        return T.astype(np.float64)

    except Exception as e:
        log.warning(f"[ICP f2m] failed with error {e}, use Delta = I.")
        return np.eye(4, dtype=np.float64)


# -------------------- main pipeline --------------------


def main():
    np.random.seed(0)

    tum_root = "dataset/rgbd_dataset_freiburg1_xyz"

    # ====== 1. Load dataset ======
    dataset = TUMRGBDDepthPairs(
        root=tum_root,
        depth_list_txt="depth.txt",
        pose_txt="poses.txt",
        intrinsics_path="intrinsics.txt",
        num_points=2048,
    )
    N = len(dataset)
    log.info(f"Loaded dataset, pairs = {N}")

    out_dir = "debug/icp_f2f"
    os.makedirs(out_dir, exist_ok=True)

    # ====== 2. Estimate TSDF volume bounds (same as old version, using GT) ======
    M = N  # You can also use only the first M frames to estimate the bbox
    pts_min, pts_max = compute_scene_bounds(dataset, max_frames=M)
    padding = 0.1
    vol_min = pts_min - padding
    vol_max = pts_max + padding
    vol_size = vol_max - vol_min

    voxel_res = 256
    voxel_size = float(np.max(vol_size) / voxel_res)
    trunc_margin = 5.0 * voxel_size
    vol_bounds = np.stack([vol_min, vol_max], axis=1)  # (3, 2)

    log.info(
        f"[TSDF] voxel_size={voxel_size:.5f}, trunc={trunc_margin:.5f}, "
        f"vol_min={vol_min}, vol_max={vol_max}"
    )

    tsdf = TSDFVolume(
        vol_bounds=vol_bounds,
        voxel_size=voxel_size,
        trunc_margin=trunc_margin,
    )

    # ====== 3. Initialize camera pose `curr_pose` (cam->world) ======
    # Use the first-frame GT pose T_wB_0 as initial pose, anchoring the world to TUM's world.
    *_, T_wA_0, T_wB_0 = dataset[0]
    curr_pose = T_wB_0.astype(np.float64)

    # -------------------- 3.1 Visualization setup --------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="KinFu Live", width=960, height=720)

    # render options: white background & softer look
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([1.0, 1.0, 1.0])  # white
    render_opt.light_on = False  # turn off strong shading

    # view control – we drive it using the current camera pose (like previous version)
    view_ctrl = vis.get_view_control()
    cam_params = view_ctrl.convert_to_pinhole_camera_parameters()

    # TSDF model mesh used for frame-to-model ICP
    model_mesh = o3d.geometry.TriangleMesh()
    model_mesh_valid = False

    # Visualization point cloud for TSDF model (semi-transparent look)
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd_added = False

    # Camera trajectory as LineSet
    traj = o3d.geometry.LineSet()
    traj_points: list[np.ndarray] = []
    traj_lines: list[list[int]] = []
    traj_added = False

    # Small box representing the camera / rigid body
    cam_box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.03, depth=0.02)
    cam_box.compute_vertex_normals()
    cam_box.paint_uniform_color([0.0, 0.0, 1.0])  # blue box

    # Center the box on the origin so that world transform is applied around its center
    cam_box.translate(-cam_box.get_center())

    vis.add_geometry(cam_box)
    last_cam_pose = np.eye(4, dtype=np.float64)  # for incremental transform of cam_box

    # ================== 4. Main loop (Unity-style) ==================
    for idx in range(N):
        # ---- 4.1 Load depth ----
        depth_u16 = dataset._load_depth(dataset.depth_items[idx][1])
        depth_f = depth_u16.astype(np.float32) * dataset.scale

        log.info(f"[Frame] {idx+1}/{N}")

        # ---- 4.2 First frame: no ICP, just integrate using the initial pose ----
        if idx == 0:
            tsdf.integrate(depth_f, dataset.K, curr_pose.astype(np.float32))

            # Extract an initial mesh to serve as "model surface" for later ICP
            verts, faces = tsdf.extract_mesh()
            if verts.size > 0 and faces.size > 0:
                model_mesh.vertices = o3d.utility.Vector3dVector(verts)
                model_mesh.triangles = o3d.utility.Vector3iVector(faces)
                model_mesh.compute_vertex_normals()
                model_mesh_valid = True

                # visualization: use point cloud (black) instead of shaded mesh
                vis_pcd.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
                colors = np.tile(np.array([[0.0, 0.0, 0.0]]), (verts.shape[0], 1))
                vis_pcd.colors = o3d.utility.Vector3dVector(colors)

                if not vis_pcd_added:
                    vis.add_geometry(vis_pcd)
                    vis_pcd_added = True
                else:
                    vis.update_geometry(vis_pcd)

            # ---- update camera trajectory & box for frame 0 ----
            cam_center = curr_pose[:3, 3].copy()
            traj_points.append(cam_center)
            traj.points = o3d.utility.Vector3dVector(np.asarray(traj_points))

            # move box from identity to current pose
            Delta_box = curr_pose @ np.linalg.inv(last_cam_pose)
            cam_box.transform(Delta_box)
            last_cam_pose = curr_pose.copy()
            vis.update_geometry(cam_box)

            # drive view by current camera pose
            cam_extrinsic = np.linalg.inv(curr_pose).astype(np.float64)
            cam_params.extrinsic = cam_extrinsic
            view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

            vis.poll_events()
            vis.update_renderer()
            continue

        # ---- 4.3 If the TSDF model is still too small, skip ICP and only integrate ----
        if (not model_mesh_valid) or np.asarray(model_mesh.vertices).shape[0] < 5000:
            log.info("[Frame] model mesh too small, skip ICP, only integrate.")
            tsdf.integrate(depth_f, dataset.K, curr_pose.astype(np.float32))
            verts, faces = tsdf.extract_mesh()
            if verts.size > 0 and faces.size > 0:
                model_mesh.vertices = o3d.utility.Vector3dVector(verts)
                model_mesh.triangles = o3d.utility.Vector3iVector(faces)
                model_mesh.compute_vertex_normals()
                model_mesh_valid = True

                vis_pcd.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
                colors = np.tile(np.array([[0.0, 0.0, 0.0]]), (verts.shape[0], 1))
                vis_pcd.colors = o3d.utility.Vector3dVector(colors)

                if not vis_pcd_added:
                    vis.add_geometry(vis_pcd)
                    vis_pcd_added = True
                else:
                    vis.update_geometry(vis_pcd)

            # update camera trajectory (pose unchanged this frame)
            cam_center = curr_pose[:3, 3].copy()
            traj_points.append(cam_center)
            if len(traj_points) >= 2:
                traj_lines.append([len(traj_points) - 2, len(traj_points) - 1])

            traj.points = o3d.utility.Vector3dVector(np.asarray(traj_points))
            if len(traj_lines) > 0:
                traj.lines = o3d.utility.Vector2iVector(
                    np.asarray(traj_lines, dtype=np.int32)
                )
                colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(traj_lines), 1))
                traj.colors = o3d.utility.Vector3dVector(colors)

            if not traj_added:
                vis.add_geometry(traj)
                traj_added = True
            else:
                vis.update_geometry(traj)

            # view from current camera pose
            cam_extrinsic = np.linalg.inv(curr_pose).astype(np.float64)
            cam_params.extrinsic = cam_extrinsic
            view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

            vis.poll_events()
            vis.update_renderer()
            continue

        # ---- 4.4 Build current frame world-space point cloud and model point cloud ----
        # depth -> camera coordinates -> world coordinates (via current pose)
        pts_cam = depth_to_cam_points(depth_u16, dataset.K, dataset.scale)  # (N, 3)
        R_cw = curr_pose[:3, :3]  # cam->world
        t_cw = curr_pose[:3, 3]
        pts_world = (R_cw @ pts_cam.T).T + t_cw  # (N, 3) in world

        pts_world_ds = downsample_random(pts_world, max_n=50000)

        model_verts = np.asarray(model_mesh.vertices)
        model_verts_ds = downsample_random(model_verts, max_n=80000)

        # ---- 4.5 Frame-to-model ICP: estimate incremental Delta ----
        Delta = icp_frame_to_model_o3d(
            src_pts_world=pts_world_ds,
            tgt_pts_world=model_verts_ds,
            voxel_sizes=(0.08, 0.04, 0.02),
            max_iters=(20, 15, 10),
            distance_ths=(0.15, 0.08, 0.04),
        )

        # Optional: sanity check on translation magnitude to avoid large jumps
        dt = np.linalg.norm(Delta[:3, 3])
        if dt > 0.2:
            log.warning(
                f"[ICP f2m] large step |dt|={dt:.3f} m, clamp to identity (skip update)."
            )
            Delta = np.eye(4, dtype=np.float64)

        # ---- 4.6 Update camera pose (Unity style: currPose = T10 * currPose) ----
        curr_pose = (Delta @ curr_pose).astype(np.float64)

        # ---- 4.7 Integrate current depth into TSDF ----
        tsdf.integrate(depth_f, dataset.K, curr_pose.astype(np.float32))

        # ---- 4.8 Update model mesh for the next frame ----
        verts, faces = tsdf.extract_mesh()
        if verts.size > 0 and faces.size > 0:
            model_mesh.vertices = o3d.utility.Vector3dVector(verts)
            model_mesh.triangles = o3d.utility.Vector3iVector(faces)
            model_mesh.compute_vertex_normals()
            model_mesh_valid = True

            vis_pcd.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
            colors = np.tile(np.array([[0.0, 0.0, 0.0]]), (verts.shape[0], 1))
            vis_pcd.colors = o3d.utility.Vector3dVector(colors)

            if not vis_pcd_added:
                vis.add_geometry(vis_pcd)
                vis_pcd_added = True
            else:
                vis.update_geometry(vis_pcd)

        # ---- 4.9 Update camera trajectory (world-space) ----
        cam_center = curr_pose[:3, 3].copy()
        traj_points.append(cam_center)
        if len(traj_points) >= 2:
            traj_lines.append([len(traj_points) - 2, len(traj_points) - 1])

        traj.points = o3d.utility.Vector3dVector(np.asarray(traj_points))
        if len(traj_lines) > 0:
            traj.lines = o3d.utility.Vector2iVector(
                np.asarray(traj_lines, dtype=np.int32)
            )
            colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(traj_lines), 1))
            traj.colors = o3d.utility.Vector3dVector(colors)

        if not traj_added:
            vis.add_geometry(traj)
            traj_added = True
        else:
            vis.update_geometry(traj)

        # ---- 4.10 Update camera box pose via incremental transform ----
        Delta_box = curr_pose @ np.linalg.inv(last_cam_pose)
        cam_box.transform(Delta_box)
        last_cam_pose = curr_pose.copy()
        vis.update_geometry(cam_box)

        # ---- 4.11 Drive view & render ----
        cam_extrinsic = np.linalg.inv(curr_pose).astype(np.float64)
        cam_params.extrinsic = cam_extrinsic
        view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

        vis.poll_events()
        vis.update_renderer()

    # Close window after loop
    vis.destroy_window()

    # ================== 5. Extract final mesh ==================
    verts, faces = tsdf.extract_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    tsdf_path = os.path.join(out_dir, f"icp_f2f_{N}.stl")
    o3d.io.write_triangle_mesh(tsdf_path, mesh)
    log.info(
        f"[DONE] Saved TSDF mesh → {tsdf_path}, "
        f"verts={verts.shape[0]}, faces={faces.shape[0]}"
    )


if __name__ == "__main__":
    main()
