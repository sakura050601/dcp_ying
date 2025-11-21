import numpy as np


def depth_to_pointcloud(depth, K, scale=1.0 / 5000.0, max_depth=5.0):
    """
    depth: HxW uint16
    K: 相机内参 (3x3)
    return: Nx3 点云（相机坐标系）
    """

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 转为米
    depth_m = depth.astype(np.float32) * scale

    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth_m
    valid = (Z > 0) & (Z < max_depth)

    u = u[valid]
    v = v[valid]
    Z = Z[valid]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pc = np.stack([X, Y, Z], axis=1)
    return pc  # Nx3 float32
