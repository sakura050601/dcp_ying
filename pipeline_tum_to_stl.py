import numpy as np
import open3d as o3d
import tqdm

from data_tum import TUMRGBDDepthPairs
from dcp_loader import DCPMatcher
from icp_utils import icp_point_to_point


from tsdf_fusion import TSDFVolumeNumpy as TSDFVolume
from util import get_logger
from vis_utils import (
    visualize_cloud,
    visualize_two_clouds_save,
    visualize_two_clouds_save_color,
)

log = get_logger()


def depth_to_pointcloud(depth, K):
    """NUMPY 版本：深度图 → 点云 Nx3"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z = depth
    valid = z > 0
    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def main():

    # ====== Step 1: 参数 ======
    tum_root = "dataset/rgbd_dataset_freiburg1_xyz"
    max_frames = 50
    use_dcp = False

    # ====== Step 2: 读取 TUM ======
    dataset = TUMRGBDDepthPairs(
        root=tum_root,
        depth_list_txt="depth.txt",
        pose_txt="poses.txt",
        intrinsics_path="intrinsics.txt",
        num_points=2048,
    )

    log.info(f"Loaded dataset, pairs = {len(dataset)}")

    # ====== Step 3: 加载 DCP ======
    matcher = DCPMatcher("pretrained/dcp_v2.t7") if use_dcp else None

    # ====== Step 4: 创建 TSDF ======
    vol_bounds = np.array(
        [
            [-1.0, 1.0],  # x 左右
            [-1.0, 1.0],  # y 上下
            [0.2, 3.0],  # z 前方
        ]
    )
    # vol_bounds = np.array(
    #     [
    #         [-0.2, 1.0],  # x
    #         [-1.5, 1.5],  # y
    #         [-0.1, 3.0],  # z
    #     ]
    # )

    tsdf = TSDFVolume(vol_bounds, voxel_size=0.02, trunc_margin=0.08)
    # voxel=0.005   # 5mm trunc = 0.02  # trunc = 4 × voxel

    # ====== Step 5: 初始化位姿 ======
    T_global = np.eye(4, dtype=np.float32)
    poses = []

    prev_pc = None
    prev_depth = None
    prev_rgb = None

    # ====== Step 6: 主循环 ======
    pbar = tqdm.tqdm(range(min(max_frames, len(dataset))), desc="Reconstruction")
    for idx in pbar:

        data = dataset[idx]

        # TODO: 修复解包（dataset 不再返回 rgb_src / rgb_tgt）
        # 原来：
        src, tgt, rgb_src, rgb_tgt, R_ab, t_ab, _, _, _, _, T_wA, T_wB = data
        # src, tgt, R_ab, t_ab, _, _, _, _, T_wA, T_wB = data

        # src, tgt 是 torch (3, N)
        pc = tgt.T.numpy()  # (N,3)
        pc_prev = src.T.numpy()  # (N,3)

        # --------- Debug: 检查点数 ---------
        if pc.shape[0] < 200:
            log.warning(f"Frame {idx}: too few points ({pc.shape[0]}), skip")
            # log.warning(f"TSDF integrate failed at frame {idx}: {e}")
            continue

        if pc_prev.shape[0] < 200:
            log.warning(
                f"Frame {idx}: previous frame too few points ({pc_prev.shape[0]}), skip"
            )
            continue

        # --------- 第一帧 ---------
        if prev_pc is None:
            prev_pc = pc
            poses.append(T_global.copy())
            continue
        # ===== Debug 1: 点云范围 =====
        if idx < 5 or idx % 20 == 0:
            log.debug(
                f"[PC Range] Frame {idx}: min={pc.min(axis=0)}, max={pc.max(axis=0)}"
            )

        # ===== Step 6.1: 先尝试 DCP =====
    #     use_icp = False

    #     if matcher is not None:
    #         try:
    #             R_pred, t_pred = matcher.match(prev_pc, pc)

    #             log.debug(f"[DCP/ICP] Frame {idx} R_pred=\n{R_pred}")
    #             log.debug(f"[DCP/ICP] Frame {idx} t_pred={t_pred}")

    #             if np.isnan(R_pred).any() or np.isnan(t_pred).any():
    #                 log.warning(f"Frame {idx}: DCP produced NaN → fallback to ICP")
    #                 use_icp = True
    #         except Exception as e:
    #             log.warning(f"Frame {idx}: DCP failed → fallback to ICP\n{e}")
    #             use_icp = True
    #     else:
    #         use_icp = True

    #     # ===== Step 6.2: ICP fallback =====
    #     if use_icp:
    #         R_pred, t_pred = dataset.icp_pair(prev_pc, pc)
    #         log.info(f"Frame {idx}: Using ICP")

    #     # ===== Step 6.3: 累积位姿 =====
    #     # TODO
    #     # T_ab = np.eye(4)
    #     # T_ab[:3, :3] = np.array(R_pred, dtype=float)
    #     # T_ab[:3, 3] = np.array(t_pred, dtype=float)

    #     # ===== Debug 3: 检查 T_ab =====
    #     # if np.abs(np.linalg.det(T_ab[:3, :3]) - 1.0) > 1e-3:
    #     #     log.error(
    #     #         f"[T_ab] Frame {idx}: rotation det != 1, det = {np.linalg.det(T_ab[:3,:3])}"
    #     #     )

    #     # if np.isnan(T_ab).any():
    #     #     log.error(f"[T_ab] Frame {idx}: T_ab contains NaN")
    #     # T_global = T_global @ T_ab

    #     # T_ab = np.linalg.inv(T_wA) @ T_wB
    #     T_global = T_wB

    #     # ===== Debug 4: 累积位姿监控 =====
    #     t = T_global[:3, 3]
    #     if idx < 10 or idx % 20 == 0:
    #         log.debug(f"[T_global] Frame {idx} T_global t = {t}")

    #     poses.append(T_global.copy())

    #     # ===== Step 6.4: TSDF 融入 =====
    #     try:
    #         depth_u16 = dataset._load_depth(dataset.depth_items[idx][1])
    #         depth_f = depth_u16.astype(np.float32) * dataset.scale

    #         K = dataset.K
    #         # T_world_cam = T_global
    #         # cam_pose = np.linalg.inv(T_world_cam)
    #         # cam_pose = T_global.copy()  # cam_to_world
    #         cam_pose = np.linalg.inv(T_wB)

    #         # ===== Debug 5: cam_pose =====
    #         if np.isnan(cam_pose).any() or np.isinf(cam_pose).any():
    #             log.error(f"[cam_pose] Frame {idx}: cam_pose invalid (NaN/Inf)")

    #         if abs(np.linalg.det(cam_pose[:3, :3])) < 1e-6:
    #             log.error(
    #                 f"[cam_pose] Frame {idx}: cam_pose is singular, det={np.linalg.det(cam_pose[:3,:3])}"
    #             )

    #         # ===== Debug 6: 深度图范围 =====
    #         if idx < 10 or idx % 30 == 0:
    #             log.debug(
    #                 f"[depth] Frame {idx} depth range: min={depth_f[depth_f>0].min()}, max={depth_f.max()}"
    #             )

    #         tsdf.integrate(depth_f, K, cam_pose)

    #     except Exception as e:
    #         log.warning(f"[TSDF] Frame {idx}: integrate failed: {e}")

    #     prev_pc = pc

    # # ====== Step 7: 导出 mesh ======
    # mesh = tsdf.extract_mesh()
    # # TODO: convert (verts, faces) numpy → Open3D TriangleMesh
    # verts, faces = mesh
    # mesh_o3d = o3d.geometry.TriangleMesh()
    # mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    # mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    # mesh_o3d.compute_vertex_normals()

    # from datetime import datetime

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"reconstructed_tum_{timestamp}.stl"

    # # TODO: write Open3D mesh instead of raw numpy
    # o3d.io.write_triangle_mesh(filename, mesh_o3d)
    # log.info(f"输出 mesh → {filename}")


if __name__ == "__main__":
    main()
