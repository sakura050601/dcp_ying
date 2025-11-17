import numpy as np
from skimage import measure
from util import logging

log = logging.getLogger("PIPELINE")


class TSDFVolumeNumpy:
    """
    简化的 TSDF: 使用 dense volume（NumPy）
    """

    def __init__(self, vol_bounds, voxel_size=0.02, trunc_margin=0.06):
        """
        vol_bounds: (3,2) xyz min/max
        """
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin

        vol_min = vol_bounds[:, 0]
        vol_max = vol_bounds[:, 1]

        self.vol_dim = np.ceil((vol_max - vol_min) / voxel_size).astype(int)
        self.vol_origin = vol_min

        self.tsdf = np.ones(self.vol_dim, dtype=np.float32)
        self.weights = np.zeros(self.vol_dim, dtype=np.float32)

    def integrate(self, depth, K, cam_pose):
        """
        depth: HxW (float32, meters)
        K: 3x3 intrinsics
        cam_pose: 4x4, camera -> world (和你 debug 脚本保持一致)
        """

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        H, W = depth.shape

        # ===== 1. 构造体素世界坐标 =====
        xv, yv, zv = np.meshgrid(
            np.arange(self.vol_dim[0]),
            np.arange(self.vol_dim[1]),
            np.arange(self.vol_dim[2]),
            indexing="ij",
        )
        vox_world = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        vox_world = vox_world * self.voxel_size + self.vol_origin  # (N,3)

        # ===== 2. world -> cam （注意 cam_pose 是 cam->world，这里显式做 inverse）=====
        R_cw = cam_pose[:3, :3]  # cam -> world
        t_cw = cam_pose[:3, 3]

        # X_cam = R_cw^T (X_world - t_cw)
        # vox_cam = (vox_world - t_cw) @ R_cw.T
        vox_cam = (vox_world - t_cw) @ R_cw
        z = vox_cam[:, 2]

        # 只保留在相机前方的体素
        valid_z = z > 0.0
        if not np.any(valid_z):
            return

        vox_cam = vox_cam[valid_z]
        z = z[valid_z]

        # ===== 3. 投影到图像平面 =====
        u = fx * (vox_cam[:, 0] / z) + cx
        v = fy * (vox_cam[:, 1] / z) + cy

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(inside):
            return

        u = u[inside].astype(np.int32)
        v = v[inside].astype(np.int32)
        z = z[inside]

        # 对 flat 索引也同步过滤
        flat = np.flatnonzero(valid_z)[inside]

        # ===== 4. 读取对应深度，并过滤掉 depth<=0 或太远的 =====
        depth_sample = depth[v, u]

        # 类似 Unity 里 minDepth / maxDepth，这里先写死一个范围试试
        min_depth = 0.2
        max_depth = 5.0

        valid_depth = (depth_sample > min_depth) & (depth_sample < max_depth)
        if not np.any(valid_depth):
            return

        depth_sample = depth_sample[valid_depth]
        z = z[valid_depth]
        flat = flat[valid_depth]

        # ===== 5. 计算 SDF，只保留在截断区间内的体素 =====
        sdf = depth_sample - z  # >0: 体素在表面前方; <0: 在后方

        # 只更新 sdf >= -trunc_margin 的体素（太靠后的直接忽略）
        valid_sdf = sdf >= -self.trunc_margin
        if not np.any(valid_sdf):
            return

        sdf = sdf[valid_sdf]
        flat = flat[valid_sdf]

        tsdf_new = np.clip(sdf / self.trunc_margin, -1.0, 1.0)

        # ===== 6. 融合权重 =====
        w_old = self.weights.flat[flat]
        ts_old = self.tsdf.flat[flat]
        w_new = w_old + 1.0

        self.weights.flat[flat] = w_new
        self.tsdf.flat[flat] = (w_old * ts_old + tsdf_new) / w_new

        # ===== 7. Debug 统计（可选）=====
        log.debug(
            f"[TSDF] updated_voxels={flat.size}, "
            f"z_range=({z.min():.3f},{z.max():.3f}), "
            f"sdf_range=({sdf.min():.3f},{sdf.max():.3f})"
        )

    def extract_mesh(self):
        if self.tsdf.min() > 0 or self.tsdf.max() < 0:
            raise ValueError("Surface level must cross zero for marching cubes.")

        verts, faces, _, _ = measure.marching_cubes(self.tsdf, level=0)
        verts = verts * self.voxel_size + self.vol_origin
        return verts, faces
