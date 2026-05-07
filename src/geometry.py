import numpy as np

'''
Taken from gsplat/examples/datasets/colmap.py
Args:
    pts3D: point cloud to project numpy array (M, 3) 
    c2w: cam2world camera pose (M, 3)
    K_mat: camera intrinsics (4,4)
    H: image height (int)
    W: image width (int)
    create_image: see below 

Returns:
    (M,) valid pixels if create_image=False
    (H, W) depth image if create_image=True
'''
def project_pts3D(pts3D, c2w, K_mat, H, W, create_image=False):
    w2c = np.linalg.inv(c2w)

    if create_image:
        M = pts3D.shape[0]
        pts_h = np.vstack([pts3D.T, np.ones((1, M))])  # (4, M)

        # 3) camera-frame points
        cam_h = w2c @ pts_h             # (4, M)
        cam = cam_h[:3, :]              # (3, M)
        zs = cam[2, :]                  # (M,)

        # 4) project to pixels
        proj = K_mat @ cam              # (3, M)
        us = proj[0, :] / proj[2, :]
        vs = proj[1, :] / proj[2, :]

        # 5) discrete pixel coords
        ui = np.floor(us).astype(int)
        vi = np.floor(vs).astype(int)

        # 6) validity mask
        valid = (
            (zs > 0) &
            (ui >= 0) & (ui < W) &
            (vi >= 0) & (vi < H)
        )

        # 7) prepare depth buffer
        depth_map = np.zeros((H, W), dtype=float)
        depth_buffer = np.full((H, W), np.inf, dtype=float)

        # 8) rasterize each valid point, keeping the minimum depth
        for x, y, z in zip(ui[valid], vi[valid], zs[valid]):
            if z < depth_buffer[y, x]:
                depth_buffer[y, x] = z

        # 9) convert infinities back to zero
        depth_map = np.where(np.isfinite(depth_buffer), depth_buffer, 0.0)

        return depth_map
    else: 
        points_cam = (w2c[:3, :3] @ pts3D.T + w2c[:3, 3:4]).T
        points_proj = (K_mat @ points_cam.T).T 
        points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
        depths = points_cam[:, 2]  # (M,)
        mask = (
            (points[:, 0] >= 0)
            & (points[:, 0] < W)
            & (points[:, 1] >= 0)
            & (points[:, 1] < H)
            & (depths > 0)
        )
        points = points[mask]
        depths = depths[mask]
        return points, depths
        
def scale_poses(c2ws, s):
    for c2w in c2ws:
        c2w[:3,3] *= s
    return c2ws

def unproject(depth, c2w, K_mat):
    H, W = depth.shape

    # 1) create a grid of pixel coordinates
    js, is_ = np.meshgrid(np.arange(W), np.arange(H))  # is_: rows (y), js: cols (x)

    # 2) mask out invalid depths
    valid = depth > 0
    zs = depth[valid]                          # (N,)
    us = js[valid].astype(np.float64)          # (N,)
    vs = is_[valid].astype(np.float64)         # (N,)

    # 3) backproject to camera‐frame
    #    [u, v, 1]^T * z  →  camera‐space ray scaled by depth
    inv_K = np.linalg.inv(K_mat)
    pix_h = np.stack([us, vs, np.ones_like(us)], axis=0)  # (3, N)
    cam_pts = inv_K @ pix_h                              # (3, N) unit vectors
    cam_pts *= zs                                        # scale each ray by its depth

    # 4) convert to homogeneous then to world‐space
    ones = np.ones((1, cam_pts.shape[1]))
    cam_pts_h = np.vstack([cam_pts, ones])               # (4, N)
    world_pts_h = c2w @ cam_pts_h                        # (4, N)
    pts_w = world_pts_h[:3].T                            # (N, 3)

    return pts_w