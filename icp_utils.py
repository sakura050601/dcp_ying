import logging

log = logging.getLogger("PIPELINE")

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d


def make_o3d_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert (N,3) numpy array to an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def icp_point_to_plane_o3d(
    src: np.ndarray,
    tgt: np.ndarray,
    max_iter: int = 20,
    distance_thresh: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-level point-to-plane ICP using Open3D.

    src, tgt : (N, 3) in the same coordinate frame (e.g., cam_i, cam_j).
    Returns:
        R (3x3), t (3,)
    """
    if src.shape[0] < 20 or tgt.shape[0] < 20:
        # Too few points
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    try:
        pcd_src = make_o3d_pcd(src)
        pcd_tgt = make_o3d_pcd(tgt)

        # Estimate normals on target (used as planes)
        pcd_tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        init = np.eye(4)

        reg = o3d.pipelines.registration.registration_icp(
            pcd_src,
            pcd_tgt,
            max_correspondence_distance=distance_thresh,  # ✅ 改关键字名称
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter
            ),
        )

        T = reg.transformation  # 4x4
        if not np.all(np.isfinite(T)):
            raise FloatingPointError("non-finite transformation")

        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        return R, t

    except Exception as e:
        log.warning(f"[ICP p2plane] failed with error {e}, fallback to identity.")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def icp_pyramid_o3d(
    src: np.ndarray,
    tgt: np.ndarray,
    voxel_sizes=(0.05, 0.02, 0.01),
    max_iters=(20, 15, 10),
    distance_ths=(0.10, 0.05, 0.02),
    use_point_to_plane: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale (pyramid) ICP using Open3D.

    - Coarse → fine:
        * Downsample with larger voxel at coarse level.
        * Larger distance_threshold at coarse level.
        * Result of each level is used as init of next level.

    Returns:
        R (3x3), t (3,)
    """
    if src.shape[0] < 20 or tgt.shape[0] < 20:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    assert len(voxel_sizes) == len(max_iters) == len(distance_ths)

    try:
        pcd_src_full = make_o3d_pcd(src)
        pcd_tgt_full = make_o3d_pcd(tgt)

        if use_point_to_plane:
            pcd_tgt_full.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        T = np.eye(4, dtype=np.float64)

        for level, (voxel, it, d_th) in enumerate(
            zip(voxel_sizes, max_iters, distance_ths)
        ):
            pcd_src = pcd_src_full.voxel_down_sample(voxel)
            pcd_tgt = pcd_tgt_full.voxel_down_sample(voxel)

            if len(pcd_src.points) < 20 or len(pcd_tgt.points) < 20:
                log.warning(f"[PyramidICP] level {level}: too few points, skipping.")
                continue

            if use_point_to_plane:
                pcd_tgt.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
                )
                estimation = (
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
            else:
                estimation = (
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )

            # reg = o3d.pipelines.registration.registration_icp(
            #     pcd_src,
            #     pcd_tgt,
            #     distance_threshold=d_th,
            #     init=T,
            #     estimation_method=estimation,
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            #         max_iteration=it
            #     ),
            # )
            reg = o3d.pipelines.registration.registration_icp(
                pcd_src,
                pcd_tgt,
                max_correspondence_distance=d_th,  # ✅ 改关键字名称
                init=T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=it
                ),
            )

            T = reg.transformation
            log.info(
                f"[PyramidICP] level {level}: voxel={voxel}, it={it}, "
                f"dist_th={d_th}, fitness={reg.fitness:.3f}, inlier_rmse={reg.inlier_rmse:.4f}"
            )
            # 在 icp_pyramid_o3d 返回前，检查 reg（最后一层）的质量
            if reg.fitness < 0.6 or reg.inlier_rmse > 0.03:
                log.warning(
                    f"[PyramidICP] bad quality (fitness={reg.fitness:.3f}, rmse={reg.inlier_rmse:.4f}), fallback to identity."
                )
                return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        if not np.all(np.isfinite(T)):
            raise FloatingPointError("non-finite transformation")

        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        return R, t

    except Exception as e:
        log.warning(f"[PyramidICP] failed with error {e}, fallback to identity.")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def icp_point_to_point(src, tgt, max_iter=20, tol=1e-5, max_corr_dist=None):
    """
    src: Nx3 (source)
    tgt: Nx3 (target)
    return R(3x3), t(3,)
    """
    src_copy = src.copy()
    tgt_tree = cKDTree(tgt)

    R_total = np.eye(3, dtype=np.float32)
    t_total = np.zeros(3, dtype=np.float32)

    if src_copy.shape[0] < 20 or tgt.shape[0] < 20:
        # 点太少，直接认为失败，返回恒等
        return R_total, t_total

    for it in range(max_iter):
        dists, idx = tgt_tree.query(src_copy)

        if max_corr_dist is not None:
            mask = dists < max_corr_dist
            if mask.sum() < 20:
                # 有效对应太少
                break
            src_corr = src_copy[mask]
            tgt_corr = tgt[idx[mask]]
        else:
            src_corr = src_copy
            tgt_corr = tgt[idx]

        # 如果有 nan/inf 就直接退出
        if (not np.all(np.isfinite(src_corr))) or (not np.all(np.isfinite(tgt_corr))):
            # 返回恒等，交给上层处理
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        mu_src = np.mean(src_corr, axis=0)
        mu_tgt = np.mean(tgt_corr, axis=0)

        X = src_corr - mu_src
        Y = tgt_corr - mu_tgt
        H = X.T @ Y

        if not np.all(np.isfinite(H)):
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src

        # 再检查一次 R,t
        if (not np.all(np.isfinite(R))) or (not np.all(np.isfinite(t))):
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        src_copy = (R @ src_copy.T).T + t

        R_total = R @ R_total
        t_total = R @ t_total + t

        if np.linalg.norm(t) < tol:
            break

    return R_total, t_total


def build_icp_poses(dataset, N, icp_mode: str = "p2p"):
    """
    Build per-frame cam->world poses by chaining frame-to-frame ICP.

    icp_mode:
        - "p2p"             : custom point-to-point ICP (with max_corr_dist).
        - "p2plane"         : single-level Open3D point-to-plane ICP.
        - "pyramid_p2plane" : multi-scale point-to-plane ICP.
    """
    _, _, _, _, _, _, _, _, _, _, T_wA_0, T_wB_0 = dataset[0]
    T_icp = [T_wB_0.copy()]

    for i in range(N - 1):
        depth_i = dataset._load_depth(dataset.depth_items[i][1])
        depth_j = dataset._load_depth(dataset.depth_items[i + 1][1])
        pc_i = depth_to_cam_points(depth_i, dataset.K, dataset.scale)
        pc_j = depth_to_cam_points(depth_j, dataset.K, dataset.scale)

        pc_i_ds = downsample(pc_i, 20000)
        pc_j_ds = downsample(pc_j, 20000)

        # --- select ICP variant ---
        if icp_mode == "p2p":
            R_ji, t_ji = icp_point_to_point(
                pc_i_ds, pc_j_ds, max_iter=20, tol=1e-5, max_corr_dist=0.1
            )
        elif icp_mode == "p2plane":
            R_ji, t_ji = icp_point_to_plane_o3d(
                pc_i_ds, pc_j_ds, max_iter=20, distance_thresh=0.05
            )
        elif icp_mode == "pyramid_p2plane":
            R_ji, t_ji = icp_pyramid_o3d(
                pc_i_ds,
                pc_j_ds,
                voxel_sizes=(0.05, 0.02, 0.01),
                max_iters=(20, 15, 10),
                distance_ths=(0.10, 0.05, 0.02),
                use_point_to_plane=True,
            )
        else:
            raise ValueError(f"Unknown icp_mode={icp_mode}")

        # if ICP explodes -> fallback to GT pose
        if (not np.all(np.isfinite(R_ji))) or (not np.all(np.isfinite(t_ji))):
            log.warning(f"[ICP] pair {i}->{i+1} gave NaN/Inf, fallback to GT pose.")
            *_, T_wA_j, T_wB_j = dataset[i + 1]
            T_icp.append(T_wB_j.copy())
            continue

        T_ji = np.eye(4, dtype=np.float32)
        T_ji[:3, :3] = R_ji
        T_ji[:3, 3] = t_ji

        # Correct chain: cam_j = T_ji * cam_i
        T_j = T_ji @ T_icp[-1]
        T_icp.append(T_j)

    return T_icp


# def build_icp_poses(dataset, N):
#     _, _, _, _, _, _, _, _, _, _, T_wA_0, T_wB_0 = dataset[0]
#     T_icp = [T_wB_0.copy()]

#     for i in range(N - 1):
#         depth_i = dataset._load_depth(dataset.depth_items[i][1])
#         depth_j = dataset._load_depth(dataset.depth_items[i + 1][1])
#         pc_i = depth_to_cam_points(depth_i, dataset.K, dataset.scale)
#         pc_j = depth_to_cam_points(depth_j, dataset.K, dataset.scale)

#         pc_i_ds = downsample(pc_i, 20000)
#         pc_j_ds = downsample(pc_j, 20000)

#         R_ji, t_ji = icp_point_to_point(pc_i_ds, pc_j_ds, max_corr_dist=0.1)

#         if (not np.all(np.isfinite(R_ji))) or (not np.all(np.isfinite(t_ji))):
#             log.warning(f"[ICP] pair {i}->{i+1} gave NaN/Inf, fallback to GT pose.")
#             # 用真值 T_wB_{i+1} 兜底，避免轨迹崩坏
#             *_, T_wA_j, T_wB_j = dataset[i + 1]
#             T_icp.append(T_wB_j.copy())
#             continue

#         T_ji = np.eye(4, dtype=np.float32)
#         T_ji[:3, :3] = R_ji
#         T_ji[:3, 3] = t_ji

#         # 正确的累积方向：cam_j = T_ji * cam_i
#         T_j = T_ji @ T_icp[-1]
#         T_icp.append(T_j)

#     return T_icp


# 简单下采样
def downsample(pc, max_n=20000):
    if pc.shape[0] <= max_n:
        return pc
    idx = np.random.choice(pc.shape[0], max_n, replace=False)
    return pc[idx]


def skew(v):
    """R^3 -> so(3) 反对称矩阵"""
    x, y, z = v
    return np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ],
        dtype=np.float32,
    )


def depth_to_cam_points(depth_u16, K, scale):
    """深度图 (uint16) -> 相机坐标系点云 (N,3)"""
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
    return pts_cam
