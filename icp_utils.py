import numpy as np
from scipy.spatial import cKDTree


# def icp_point_to_point(src, tgt, max_iter=20, tol=1e-5):
#     """
#     src: Nx3 (source)
#     tgt: Nx3 (target)
#     return R(3x3), t(3,)
#     """

#     src_copy = src.copy()
#     tgt_tree = cKDTree(tgt)

#     R_total = np.eye(3)
#     t_total = np.zeros(3)

#     for i in range(max_iter):
#         # 1 最近邻对应
#         dists, idx = tgt_tree.query(src_copy)
#         correspond_tgt = tgt[idx]

#         # 2 去均值
#         mu_src = np.mean(src_copy, axis=0)
#         mu_tgt = np.mean(correspond_tgt, axis=0)

#         X = src_copy - mu_src
#         Y = correspond_tgt - mu_tgt

#         # 3 SVD 求 R
#         H = X.T @ Y
#         U, S, Vt = np.linalg.svd(H)
#         R = Vt.T @ U.T

#         # 处理反射
#         if np.linalg.det(R) < 0:
#             Vt[2, :] *= -1
#             R = Vt.T @ U.T

#         t = mu_tgt - R @ mu_src

#         # 更新
#         src_copy = (R @ src_copy.T).T + t

#         # 累积全局 R 和 t
#         R_total = R @ R_total
#         t_total = R @ t_total + t

#         # 收敛检查
#         if np.linalg.norm(t) < tol:
#             break

#     return R_total, t_total


def icp_point_to_point(src, tgt, max_iter=20, tol=1e-5, max_corr_dist=None):
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


def build_icp_poses(dataset, N):
    _, _, _, _, _, _, _, _, _, _, T_wA_0, T_wB_0 = dataset[0]
    T_icp = [T_wB_0.copy()]

    for i in range(N - 1):
        depth_i = dataset._load_depth(dataset.depth_items[i][1])
        depth_j = dataset._load_depth(dataset.depth_items[i + 1][1])
        pc_i = depth_to_cam_points(depth_i, dataset.K, dataset.scale)
        pc_j = depth_to_cam_points(depth_j, dataset.K, dataset.scale)

        pc_i_ds = downsample(pc_i, 20000)
        pc_j_ds = downsample(pc_j, 20000)

        R_ji, t_ji = icp_point_to_point(pc_i_ds, pc_j_ds, max_corr_dist=0.1)

        if (not np.all(np.isfinite(R_ji))) or (not np.all(np.isfinite(t_ji))):
            log.warning(f"[ICP] pair {i}->{i+1} gave NaN/Inf, fallback to GT pose.")
            # 用真值 T_wB_{i+1} 兜底，避免轨迹崩坏
            *_, T_wA_j, T_wB_j = dataset[i + 1]
            T_icp.append(T_wB_j.copy())
            continue

        T_ji = np.eye(4, dtype=np.float32)
        T_ji[:3, :3] = R_ji
        T_ji[:3, 3] = t_ji

        # 正确的累积方向：cam_j = T_ji * cam_i
        T_j = T_ji @ T_icp[-1]
        T_icp.append(T_j)

    return T_icp
