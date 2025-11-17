# tum_loader.py
import os
import numpy as np
from PIL import Image
import cv2


class TUMRGBDDataset:
    """
    返回：
    - rgb: HxWx3 uint8
    - depth: HxW uint16
    - pose: 4x4 世界坐标下相机位姿
    """

    def __init__(self, root):
        self.root = root

        # 文件路径
        self.rgb_file = os.path.join(root, "rgb.txt")
        self.depth_file = os.path.join(root, "depth.txt")
        self.pose_file = os.path.join(root, "poses.txt")

        # 相机内参（默认TUM fr1）
        self.K = np.array(
            [[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], dtype=np.float32
        )

        # 加载所有 RGB 和 depth 文件名
        self.rgb_list = self._load_list(self.rgb_file)
        self.depth_list = self._load_list(self.depth_file)
        self.poses = self._load_poses(self.pose_file)

    def _load_list(self, txt):
        items = []
        with open(txt, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                ts, path = line.strip().split()
                items.append((float(ts), os.path.join(self.root, path)))
        return items

    def _load_poses(self, path):
        poses = {}
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                vals = line.split()
                if len(vals) != 8:
                    continue
                t, tx, ty, tz, qx, qy, qz, qw = [float(v) for v in vals]
                poses[t] = self._quat_pose(tx, ty, tz, qx, qy, qz, qw)
        return poses

    def _quat_pose(self, tx, ty, tz, qx, qy, qz, qw):
        """四元数 → 4x4 pose"""
        R = self._quat_to_rot(qw, qx, qy, qz)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz])
        return T

    def _quat_to_rot(self, qw, qx, qy, qz):
        """wxyz"""
        R = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ]
        )
        return R

    def __len__(self):
        return min(len(self.rgb_list), len(self.depth_list))

    def __getitem__(self, idx):
        # 加载 rgb
        ts_rgb, rgb_path = self.rgb_list[idx]
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        # 加载 depth
        ts_depth, depth_path = self.depth_list[idx]
        depth = np.array(Image.open(depth_path))

        # 匹配最接近的 pose
        ts_closest = min(self.poses.keys(), key=lambda t: abs(t - ts_rgb))
        T = self.poses[ts_closest]

        return rgb, depth, T, self.K
