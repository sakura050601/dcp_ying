#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

import logging
import sys

# ============================================
# 彩色 Logger（控制台 + 文件）
# ============================================


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        msg = super().format(record)
        color = self.COLORS.get(levelname, "")
        return f"{color}{msg}{self.RESET}"


def get_logger(name="PIPELINE"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ---- 控制台 Handler（彩色） ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColorFormatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # ---- 文件 Handler ----
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"log/debug_{timestamp}.log"
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(fh)

    return logger


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


# def npmat2euler(mats, seq='zyx'):
#     eulers = []
#     for i in range(mats.shape[0]):
#         r = Rotation.from_dcm(mats[i])
#         eulers.append(r.as_euler(seq, degrees=True))
#     return np.asarray(eulers, dtype='float32')


def npmat2euler(mats, seq="zyx", ensure_so3=True):
    """
    mats: (...,3,3) 旋转矩阵批量
    返回: 欧拉角(度)
    """
    mats = np.asarray(mats)
    if mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {mats.shape}")

    flat = mats.reshape(-1, 3, 3)

    if ensure_so3:
        # 极分解: 让每个 3x3 矩阵投影到最近的正交矩阵，并强制 det=+1
        U, _, Vt = np.linalg.svd(flat)
        Rproj = U @ Vt
        det = np.linalg.det(Rproj)
        neg = det < 0
        # 对 det<0 的样本，翻转 U 的最后一列再重组
        if np.any(neg):
            U[neg, :, -1] *= -1
            Rproj[neg] = U[neg] @ Vt[neg]
        flat = Rproj

    # 兼容新旧 SciPy
    try:
        rots = R.from_matrix(flat)
    except AttributeError:
        rots = R.from_dcm(flat)

    eulers_deg = rots.as_euler(seq, degrees=True)
    return eulers_deg.reshape(*mats.shape[:-2], 3)
