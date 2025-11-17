import os, math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d


def load_intrinsics(path):
    with open(path, "r") as f:
        txt = f.read()
    nums = [float(x) for x in txt.replace(",", " ").split() if _is_float(x)]
    if len(nums) < 4:
        raise ValueError(f"Cannot parse intrinsics: {path}")
    return nums[0], nums[1], nums[2], nums[3]


def _is_float(s):
    try:
        float(s)
        return True
    except:
        return False


def quat_wxyz_to_R(qw, qx, qy, qz):
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def mat2euler_xyz(R):
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    return np.array(
        [math.atan2(R[2, 1], R[2, 2]), math.asin(sy), math.atan2(R[1, 0], R[0, 0])],
        dtype=np.float32,
    )


class TUMRGBDDepthPairs(Dataset):

    def __init__(
        self,
        root,
        depth_list_txt="depth.txt",
        rgb_list_txt="rgb.txt",
        pose_txt="poses.txt",
        intrinsics_path="intrinsics.txt",
        num_points=1024,
        stride=1,
        scale_depth_to_meter=1 / 5000.0,
        max_depth_m=5.0,
    ):
        self.root = root
        self.depth_txt = os.path.join(root, depth_list_txt)
        self.rgb_txt = os.path.join(root, rgb_list_txt)
        self.pose_txt = os.path.join(root, pose_txt)
        self.intrinsics_path = os.path.join(root, intrinsics_path)

        self.num_points = num_points
        self.stride = stride
        self.scale = scale_depth_to_meter
        self.max_depth = max_depth_m

        # --- intrinsics ---
        self.fx, self.fy, self.cx, self.cy = load_intrinsics(self.intrinsics_path)
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )

        # --- load depth list ---
        self.depth_items = self._load_list(self.depth_txt)
        # --- load rgb list ---
        self.rgb_items = self._load_list(self.rgb_txt)

        # --- load poses ---
        self.poses = self._load_poses(self.pose_txt)

        # match pairs
        self.pairs = []
        for i in range(len(self.depth_items) - stride):
            tA, _ = self.depth_items[i]
            tB, _ = self.depth_items[i + stride]
            pA = self._nearest_pose_ts(tA)
            pB = self._nearest_pose_ts(tB)
            if pA is None or pB is None:
                continue
            self.pairs.append((i, i + stride, pA, pB))

    def _load_list(self, txt):
        items = []
        with open(txt, "r") as f:
            for line in f:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                t, p = line.split()[:2]
                path = p if os.path.isabs(p) else os.path.join(self.root, p)
                items.append((float(t), path))
        return items

    def _load_poses(self, txt):
        poses = {}
        with open(txt, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                vals = [float(x) for x in line.split()]
                if len(vals) != 8:
                    continue
                t, tx, ty, tz, qx, qy, qz, qw = vals
                poses[t] = np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float32)
        return poses

    def _nearest_pose_ts(self, t, tol=0.02):
        best, best_err = None, 1e9
        for ts in self.poses.keys():
            e = abs(ts - t)
            if e < best_err:
                best, best_err = ts, e
        return best if best_err <= tol else None

    def _pose_vec_to_T(self, vec7):
        tx, ty, tz, qx, qy, qz, qw = vec7
        R = quat_wxyz_to_R(qw, qx, qy, qz)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        return T

    def _load_depth(self, path):
        return np.array(Image.open(path), dtype=np.uint16)

    def _load_rgb(self, path):
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _pc_from_depth_with_uv(self, depth):
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth.astype(np.float32) * self.scale

        valid = (z > 0) & (z < self.max_depth)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        pts = np.stack([x, y, z], axis=1)
        uv = np.stack([u, v], axis=1)
        return pts, uv

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        iA, iB, tA, tB = self.pairs[idx]

        # --- load depth / rgb ---
        _, depth_path_A = self.depth_items[iA]
        _, depth_path_B = self.depth_items[iB]
        _, rgb_path_A = self.rgb_items[iA]
        _, rgb_path_B = self.rgb_items[iB]

        depthA = self._load_depth(depth_path_A)
        depthB = self._load_depth(depth_path_B)
        rgbA = self._load_rgb(rgb_path_A)
        rgbB = self._load_rgb(rgb_path_B)

        ptsA, uvA = self._pc_from_depth_with_uv(depthA)
        ptsB, uvB = self._pc_from_depth_with_uv(depthB)

        # subsample
        selA = np.random.choice(len(ptsA), self.num_points, replace=True)
        selB = np.random.choice(len(ptsB), self.num_points, replace=True)

        ptsA = ptsA[selA]
        ptsB = ptsB[selB]
        uvA = uvA[selA]
        uvB = uvB[selB]

        # color
        rgb_src = rgbA[uvA[:, 1], uvA[:, 0]] / 255.0
        rgb_tgt = rgbB[uvB[:, 1], uvB[:, 0]] / 255.0

        # camera poses
        T_wA = self._pose_vec_to_T(self.poses[tA])
        T_wB = self._pose_vec_to_T(self.poses[tB])

        # relative pose A→B
        T_ab = np.linalg.inv(T_wA) @ T_wB
        R_ab = T_ab[:3, :3]
        t_ab = T_ab[:3, 3]

        # reverse
        R_ba = R_ab.T
        t_ba = -R_ba @ t_ab

        euler_ab = mat2euler_xyz(R_ab)
        euler_ba = mat2euler_xyz(R_ba)

        # convert to torch
        src = torch.from_numpy(ptsA.T.astype(np.float32))
        tgt = torch.from_numpy(ptsB.T.astype(np.float32))
        rgb_src = torch.from_numpy(rgb_src.T.astype(np.float32))
        rgb_tgt = torch.from_numpy(rgb_tgt.T.astype(np.float32))

        return (
            src,
            tgt,
            rgb_src,
            rgb_tgt,
            R_ab.astype(np.float32),
            t_ab.astype(np.float32),
            R_ba.astype(np.float32),
            t_ba.astype(np.float32),
            euler_ab,
            euler_ba,
            T_wA,
            T_wB,
        )
