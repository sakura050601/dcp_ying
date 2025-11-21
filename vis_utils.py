import open3d as o3d
import numpy as np


def visualize_cloud(pc):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    o3d.visualization.draw_geometries([pcd])


def visualize_two_clouds(pc1, pc2):
    p1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))
    p1.paint_uniform_color([1, 0, 0])
    p2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2))
    p2.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([p1, p2])


def visualize_two_clouds_save(pc1, pc2, filename):
    p1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))
    p1.paint_uniform_color([1, 0, 0])
    p2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2))
    p2.paint_uniform_color([0, 0, 1])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 不弹窗
    vis.add_geometry(p1)
    vis.add_geometry(p2)
    vis.update_geometry(p1)
    vis.update_geometry(p2)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()


import open3d as o3d
import numpy as np


def visualize_two_clouds_save_color(pc1, rgb1, pc2, rgb2, filename):
    """
    pc1, pc2: Nx3 numpy
    rgb1, rgb2: Nx3 uint8 or float in [0,1]
    filename: "xxx.png"
    """

    # 转 Open3D 格式
    p1 = o3d.geometry.PointCloud()
    p1.points = o3d.utility.Vector3dVector(pc1)

    if rgb1.max() > 1.0:
        rgb1 = rgb1 / 255.0
    p1.colors = o3d.utility.Vector3dVector(rgb1)

    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(pc2)
    if rgb2.max() > 1.0:
        rgb2 = rgb2 / 255.0
    p2.colors = o3d.utility.Vector3dVector(rgb2)

    # ---- Off-screen Visualization ----
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(p1)
    vis.add_geometry(p2)

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(filename)
    vis.destroy_window()
