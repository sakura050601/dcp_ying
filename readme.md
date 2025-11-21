<!-- # Deep Closest Point

## Prerequisites 
PyTorch>=1.0: https://pytorch.org

scipy>=1.2 

numpy

h5py

tqdm

TensorboardX: https://github.com/lanpa/tensorboardX

## Training

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

or 

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval

or 

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy

where xx/yy is the pretrained model

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_ICCV,
	  title={Deep Closest Point: Learning Representations for Point Cloud Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year={2019}
	}

## License
MIT License -->

# ICP + TSDF RGB-D 3D Reconstruction Baseline

This repo contains a **pure Python baseline** for RGB-D 3D reconstruction using:

- **ICP (Iterative Closest Point)** for frame-to-frame pose estimation  
- **TSDF (Truncated Signed Distance Function)** for volumetric fusion  
- A standard **TUM RGB-D** sequence (`rgbd_dataset_freiburg1_xyz`) as input

The goal of this baseline is:

- To have a **fully working end-to-end 3D scanner pipeline** in Python  
- To provide a clean starting point for **further research**, e.g. DCP, multi-scale ICP, pose graph optimization, etc.

---

## 1. Pipeline Overview

High-level flow:

1. **Load TUM RGB-D dataset**
   - `TUMRGBDDepthPairs` reads depth images, intrinsic matrix `K`, and ground truth poses `T_wB_i` (cam → world).

2. **Estimate global TSDF volume bounds**
   - For the first `M` frames, back-project depth to world coordinates using GT poses and compute an overall bounding box.
   - This determines:
     - `vol_bounds` (min/max of the scene in world coordinates)
     - `voxel_size`, `trunc_margin` for TSDF.

3. **Build a pose trajectory**
   - Three pose modes are supported:
     - `gt` — use ground truth poses `T_wB_i`
     - `icp` — use poses estimated by point-to-point ICP
     - `icp_noise` — ICP poses + small synthetic noise (for robustness experiments)
   - `build_icp_poses(...)` accumulates ICP relative transforms into a global cam→world trajectory.

4. **TSDF integration**
   - For each frame:
     - Load depth, convert to metric depth
     - Select camera pose according to `POSE_MODE`
     - Integrate into `TSDFVolumeNumpy` with `tsdf.integrate(...)`.

5. **Mesh extraction**
   - After all frames are fused:
     - Extract mesh via `tsdf.extract_mesh()`
     - Export as `.stl` to `debug_single_tsdf/tsdf_{N}.stl`.

---

## 2. File Structure (core components)

Key files used in the pipeline:

- `data_tum.py`
  - `TUMRGBDDepthPairs`:  
    - Handles reading depth images, intrinsics, and ground truth poses from a TUM RGB-D sequence.
    - Provides `dataset[idx]` → returns data including `T_wA_i`, `T_wB_i`.

- `tsdf_fusion.py`
  - `TSDFVolumeNumpy`:
    - Numpy implementation of a TSDF volume.
    - API:
      - `TSDFVolume(vol_bounds, voxel_size, trunc_margin)`
      - `integrate(depth_f, K, cam_pose)`
      - `extract_mesh()` → `(verts, faces)`

- `icp_utils.py`
  - `skew(axis)`: helper to build skew-symmetric matrix.
  - `depth_to_cam_points(depth_u16, K, scale)`:
    - Back-projects a depth map (uint16) to camera space point cloud `(N, 3)`.
  - `downsample(pc, max_n)`:
    - Randomly subsamples a point cloud to at most `max_n` points.
  - `icp_point_to_point(src, tgt, max_iter=20, tol=1e-5)`:
    - Classic point-to-point ICP in numpy + cKDTree.
    - Returns `(R, t)` that aligns `src` to `tgt`.
  - `build_icp_poses(dataset, N)`:
    - Uses consecutive depth frames and ICP to build a cam→world pose list of length `N`, starting from frame 0 GT pose.

- `util.py`
  - `get_logger()`:
    - Provides a configured logger used across the pipeline.

- `baseline_tsdf.py` (this script, name as you like)
  - Orchestrates: dataset loading → volume bounds → pose selection → TSDF fusion → mesh export.

---

## 3. Dependencies

Tested with:

- Python 3.9+
- [Open3D](http://www.open3d.org/)
- NumPy
- SciPy (for `cKDTree` in ICP)
- (Optional) other small utilities depending on your environment

Install (example):

```bash
pip install numpy scipy open3d
```

---

## 4. KinectFusion-style frame-to-model ICP (new pipeline)

In addition to the frame-to-frame ICP baseline, this repo also provides a
**KinectFusion-style RGB-D pipeline** that uses **frame-to-model ICP** and a live
visualization:

- Script: `kinfu_icp_f2m.py`
- Input: TUM RGB-D sequence `rgbd_dataset_freiburg1_xyz`
- Pose representation: `curr_pose` (cam → world), updated per frame

### 4.1. Core idea

Instead of chaining relative poses between consecutive frames, the KinectFusion-style
pipeline always aligns the current depth frame against the **accumulated TSDF
model**:

1. Maintain a global camera pose `curr_pose` (cam → world).
2. For each incoming depth frame:
   - Use `curr_pose` to transform the depth point cloud into world space.
   - Run multi-scale **point-to-plane ICP** against the TSDF mesh vertices
     (treated as the current model surface).
   - Obtain an incremental transform `Delta` and update:

     ```text
     curr_pose_new = Delta @ curr_pose
     ```

   - Integrate the depth into the TSDF volume using `curr_pose_new`.

This is conceptually close to the original KinFu / Unity-style pipeline:

> RenderSurfacePrediction + ICPTracker.Track → update camera pose → TSDF fusion

### 4.2. Live visualization

`reconstruct_with_f2f.py` opens an **Open3D visualizer** and shows:

- TSDF model as a **black point cloud** (white background, light turned off).
- A **blue box** representing the current camera pose (rigid body).
- A **green line** (LineSet) for the camera trajectory in world coordinates.
- The Open3D camera view is driven by the current `curr_pose` to mimic a
  first-person / follow-camera view.

This makes it easy to visually inspect:

- How the camera moves through the scene,
- Where ICP alignment might fail,
- How the reconstructed geometry gradually converges.

### 4.3. How to run

From the project root:

```bash
python reconstruct_with_f2f.py
```