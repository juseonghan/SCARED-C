"""
Convert COLMAP outputs to SCARED dataset format.

Takes metricized COLMAP camera poses (estimated on original, undistorted left images)
and converts them to the SCARED dataset structure.

COLMAP was run directly on undistorted, non-rectified left images, so all estimated
quantities are already in the original left camera coordinate system.

Inputs:
    --colmap_dir:    Contains <key>/sparse/ (COLMAP reconstruction) and <key>/images/
    --original_dir:  Contains dataset_x/keyframe_y/{left_depth_map.tiff, Left_Image.png}

Generates per keyframe in output_dir/dataset_x/keyframe_y/:
    - intrinsics_colmap.yaml:       COLMAP-estimated intrinsics
    - data/frame_data.tar.gz:       Camera extrinsics relative to keyframe
    - data/scene_points.tar.gz:     Unprojected depth maps as (H, W, 3) float TIFFs
    - data/rgb_frames.tar.gz:       RGB frames from COLMAP reconstruction
    - overlayed.mp4:                RGB-D overlay video
    - frame_log.json:               Log of included/excluded frames
"""

import argparse
import json
import glob
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import re
import tifffile
import tqdm

import src.geometry as geometry
from pycolmap import SceneManager


# ---------------------------------------------------------------------------
# COLMAP parsing
# ---------------------------------------------------------------------------

class COLMAPInterface:
    """Parse a COLMAP sparse reconstruction."""

    def __init__(self, colmap_dir: str):
        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            Ks_dict[camera_id] = K

            type_ = cam.camera_type
            if type_ in (0, "SIMPLE_PINHOLE"):
                params = np.empty(0, dtype=np.float32)
            elif type_ in (1, "PINHOLE"):
                params = np.empty(0, dtype=np.float32)
            elif type_ in (2, "SIMPLE_RADIAL"):
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
            elif type_ in (3, "RADIAL"):
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
            elif type_ in (4, "OPENCV"):
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            elif type_ in (5, "OPENCV_FISHEYE"):
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width, cam.height)

        print(f"[COLMAP] {len(imdata)} images, {len(set(camera_ids))} camera(s).")
        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP reconstruction.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)
        image_names = [imdata[k].name for k in imdata]

        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        self.c2ws = camtoworlds
        self.image_names = image_names
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_frame_number(image_name: str) -> int:
    """Extract frame number from image name like 'frame000001.png'."""
    basename = os.path.splitext(image_name)[0]
    if basename.startswith("frame"):
        digits = basename.replace("frame", "").lstrip("_")
        return int(digits)
    elif basename == "keyframe":
        return -1
    numbers = re.findall(r"\d+", basename)
    return int(numbers[0]) if numbers else -1


def unproject_depth_to_scene_points(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject depth map to (H, W, 3) camera-space points. Invalid pixels -> (0,0,0)."""
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    pts[depth_map == 0] = 0.0
    return pts


def create_depth_overlay(
    rgb: np.ndarray, depth: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay turbo-colormapped depth onto RGB. Returns (H,W,3) uint8."""
    valid = depth > 0
    if valid.sum() == 0:
        return rgb
    lo, hi = np.percentile(depth[valid], [1, 99])
    clipped = np.clip(depth, lo, hi)
    if hi > lo:
        normed = (clipped - lo) / (hi - lo)
        normed[~valid] = 0.0
    else:
        normed = np.zeros_like(depth)
    colored = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    out = rgb.copy()
    out[valid] = (alpha * colored[valid] + (1 - alpha) * rgb[valid]).astype(np.uint8)
    return out


def package_tar_gz(src_dir: Path, tar_path: Path):
    """Package all files in src_dir into a flat tar.gz."""
    files = sorted(src_dir.iterdir())
    with tarfile.open(str(tar_path), "w:gz") as tar:
        for f in files:
            tar.add(str(f), arcname=f.name)
    print(f"  Packaged {len(files)} files -> {tar_path.name}")


def save_intrinsics_yaml(path: str, K: np.ndarray, width: int, height: int):
    """Save COLMAP intrinsics as a standalone OpenCV YAML."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("K", K.astype(np.float64))
    fs.write("image_width", width)
    fs.write("image_height", height)
    fs.release()


def load_best_colmap_reconstruction(sparse_dir: Path) -> COLMAPInterface:
    """Load the COLMAP reconstruction with the most registered images."""
    recos = sorted(glob.glob(str(sparse_dir / "*")))
    print(f"Found {len(recos)} reconstruction(s) in {sparse_dir}")
    best, best_n = None, 0
    for reco in recos:
        mgr = COLMAPInterface(reco)
        n = len(mgr.image_names)
        if n > best_n:
            best, best_n = mgr, n
    print(f"Using reconstruction with {best_n} images")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    key = args.key
    assert len(key) == 3 and "_" in key, "key must be in form x_y"
    ds, kf = key.split("_")

    # --- Resolve paths ---
    colmap_root = Path(args.colmap_dir)
    original_kf = Path(args.original_dir) / f"dataset_{ds}" / f"keyframe_{kf}"
    assert original_kf.exists(), f"Original keyframe dir not found: {original_kf}"

    # COLMAP dir: try <key>/ then dataset_x/keyframe_y/
    colmap_kf = colmap_root / key
    if not colmap_kf.exists():
        colmap_kf = colmap_root / f"dataset_{ds}" / f"keyframe_{kf}"
    assert colmap_kf.exists(), f"COLMAP dir not found: {colmap_kf}"
    sparse_dir = colmap_kf / "sparse"
    assert sparse_dir.exists(), f"No sparse/ in {colmap_kf}"

    # COLMAP images directory
    colmap_images_dir = colmap_kf / "images"
    assert colmap_images_dir.exists(), f"No images/ in {colmap_kf}"

    # Output mirrors SCARED layout
    out_kf = Path(args.output_dir) / f"dataset_{ds}" / f"keyframe_{kf}"
    out_data = out_kf / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    print(f"Key:          {key}")
    print(f"COLMAP:       {colmap_kf}")
    print(f"Original:     {original_kf}")
    print(f"Output:       {out_kf}")

    # ------------------------------------------------------------------
    # 1. Load COLMAP reconstruction
    # ------------------------------------------------------------------
    colmap = load_best_colmap_reconstruction(sparse_dir)

    assert "keyframe.png" in colmap.image_names, \
        "keyframe.png not found in COLMAP reconstruction"

    camera_id = colmap.camera_ids[0]
    if len(set(colmap.camera_ids)) > 1:
        print(f"Warning: multiple camera IDs: {set(colmap.camera_ids)}")

    K_colmap = colmap.Ks_dict[camera_id]
    img_w, img_h = colmap.imsize_dict[camera_id]
    print(f"Image size: {img_w}x{img_h}")
    print(f"COLMAP K:\n{K_colmap}")

    # ------------------------------------------------------------------
    # 2. Load keyframe depth map (already metric, no scaling needed)
    # ------------------------------------------------------------------
    depth_path = original_kf / "left_depth_map.tiff"
    assert depth_path.exists(), f"Depth map not found: {depth_path}"
    depth_raw = tifffile.imread(str(depth_path)).astype(np.float32)
    # SCARED stores (H, W, 3) XYZ scene points — extract Z as depth
    if depth_raw.ndim == 3 and depth_raw.shape[2] == 3:
        print(f"Depth TIFF is (H,W,3) scene points, extracting Z channel")
        keyframe_depth = depth_raw[..., 2]
    else:
        keyframe_depth = depth_raw
    keyframe_depth[np.isnan(keyframe_depth)] = 0.0
    print(f"Keyframe depth shape: {keyframe_depth.shape}, "
          f"range: [{keyframe_depth[keyframe_depth > 0].min():.3f}, "
          f"{keyframe_depth.max():.3f}]")

    # ------------------------------------------------------------------
    # 3. Metricize COLMAP poses
    # ------------------------------------------------------------------
    kf_idx = colmap.image_names.index("keyframe.png")
    kf_c2w = colmap.c2ws[kf_idx]

    # Project COLMAP 3D points visible in keyframe into image
    kf_pt_indices = colmap.point_indices["keyframe.png"]
    kf_pts3D = colmap.points[kf_pt_indices]
    colmap_pixels, colmap_depths = geometry.project_pts3D(
        kf_pts3D, kf_c2w, K_colmap, img_h, img_w
    )
    colmap_pixels = np.floor(colmap_pixels).astype(int)

    # Filter to pixels with valid GT depth
    xs, ys = colmap_pixels[:, 0], colmap_pixels[:, 1]
    valid = keyframe_depth[ys, xs] != 0
    colmap_pixels = colmap_pixels[valid]
    colmap_depths = colmap_depths[valid]
    gt_depths = keyframe_depth[colmap_pixels[:, 1], colmap_pixels[:, 0]]

    ratios = gt_depths / colmap_depths
    scale = np.median(ratios)
    print(f"\nScale factor (median): {scale:.6f}")
    print(f"Scale std: {ratios.std():.6f} ({len(ratios)} points)")

    # Apply scale
    scaled_c2ws = geometry.scale_poses(colmap.c2ws, scale)
    scaled_kf_c2w = scaled_c2ws[kf_idx]
    kf_w2c = np.linalg.inv(scaled_kf_c2w)

    # Unproject keyframe depth to world-space 3D points
    world_pts = geometry.unproject(keyframe_depth, scaled_kf_c2w, K_colmap)

    # Sanity check
    sanity_depth = geometry.project_pts3D(
        world_pts, scaled_kf_c2w, K_colmap, img_h, img_w, create_image=True
    )
    valid_check = (sanity_depth > 0) & (keyframe_depth > 0)
    error = np.abs(sanity_depth - keyframe_depth)
    print(f"Sanity: mean error = {error[valid_check].mean():.6f}, "
          f"coverage = {valid_check.sum()}/{(keyframe_depth > 0).sum()}")

    # ------------------------------------------------------------------
    # 4. Save intrinsics
    # ------------------------------------------------------------------
    intrinsics_path = out_kf / "intrinsics_colmap.yaml"
    save_intrinsics_yaml(intrinsics_path, K_colmap, img_w, img_h)
    print(f"\nIntrinsics saved: {intrinsics_path}")

    # ------------------------------------------------------------------
    # 5. Discover all frames from COLMAP images directory
    # ------------------------------------------------------------------
    # Build set of all frame images present on disk (the full pool COLMAP
    # could have drawn from). These use the same naming as COLMAP image_names.
    all_image_files = {
        f.name for f in colmap_images_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    }
    # Registered = in the reconstruction (excluding keyframe)
    registered_names = set(colmap.image_names)

    # Build frame number -> registration status for the log
    all_frame_nums = {}
    for fname in sorted(all_image_files):
        fnum = get_frame_number(fname)
        if fnum >= 0:
            all_frame_nums[fnum] = fname

    registered_frame_nums = set()
    for name in registered_names:
        fnum = get_frame_number(name)
        if fnum >= 0:
            registered_frame_nums.add(fnum)

    excluded_frame_nums = set(all_frame_nums.keys()) - registered_frame_nums
    print(f"\nFrames on disk: {len(all_frame_nums)}")
    print(f"Registered in reconstruction: {len(registered_frame_nums)}")
    print(f"Excluded (not registered): {len(excluded_frame_nums)}")

    # Look for original frame_data JSONs (for timestamps)
    original_data_subdir = original_kf / "data"

    # ------------------------------------------------------------------
    # 6. Process each frame: extrinsics, scene points, RGB, overlay
    # ------------------------------------------------------------------
    # Overlay video writer
    overlay_path = out_kf / "overlayed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_path), fourcc, 15, (img_w, img_h))

    tmp_fd = Path(tempfile.mkdtemp(prefix="frame_data_"))
    tmp_sp = Path(tempfile.mkdtemp(prefix="scene_pts_"))
    tmp_rgb = Path(tempfile.mkdtemp(prefix="rgb_frames_"))

    # Sort frames by number for deterministic output
    frame_items = []
    for i, name in enumerate(colmap.image_names):
        if name == "keyframe.png":
            continue
        fnum = get_frame_number(name)
        if fnum < 0:
            print(f"Warning: skipping {name} (no frame number)")
            continue
        frame_items.append((fnum, i, name))
    frame_items.sort(key=lambda x: x[0])

    frame_count = 0
    for fnum, idx, name in tqdm.tqdm(frame_items, desc="Processing frames"):
        # --- Load RGB from COLMAP images directory ---
        rgb_path = colmap_images_dir / name
        if not rgb_path.exists():
            print(f"Warning: image not found on disk: {rgb_path}, skipping")
            continue
        left_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if left_rgb is None:
            print(f"Warning: failed to read {rgb_path}, skipping")
            continue
        left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)

        # Resize if image resolution differs from COLMAP image size
        if left_rgb.shape[:2] != (img_h, img_w):
            left_rgb = cv2.resize(left_rgb, (img_w, img_h))

        # --- Relative pose ---
        scaled_c2w = scaled_c2ws[idx]
        relative_c2w = kf_w2c @ scaled_c2w
        relative_w2c = np.linalg.inv(relative_c2w)

        # --- Timestamp (from original frame_data if available) ---
        ts_path = original_data_subdir / f"frame_data{fnum:06d}.json"
        if ts_path.exists():
            with open(ts_path) as f:
                timestamp = json.load(f)["timestamp"]
        else:
            timestamp = fnum * 1000000  # fallback: microseconds

        # Save frame_data JSON
        fd = {
            "camera-calibration": "intrinsics_colmap.yaml",
            "camera-pose": relative_w2c.tolist(),
            "timestamp": timestamp,
        }
        with open(tmp_fd / f"frame_data{fnum:06d}.json", "w") as f:
            json.dump(fd, f, indent=4)

        # --- Depth map: project keyframe world points into this frame ---
        depth_map = geometry.project_pts3D(
            world_pts, scaled_c2w, K_colmap, img_h, img_w, create_image=True
        )

        # Scene points (camera-space)
        scene_pts = unproject_depth_to_scene_points(depth_map, K_colmap)
        tifffile.imwrite(str(tmp_sp / f"scene_points{fnum:06d}.tiff"), scene_pts)

        # --- Save RGB frame (preserve original naming with gaps) ---
        cv2.imwrite(
            str(tmp_rgb / f"frame{fnum:06d}.png"),
            cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR),
        )

        # --- Overlay ---
        overlay = create_depth_overlay(left_rgb, depth_map, alpha=0.35)
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        frame_count += 1

    writer.release()

    # ------------------------------------------------------------------
    # 7. Package tar.gz archives
    # ------------------------------------------------------------------
    print(f"\nPackaging {frame_count} frames...")
    package_tar_gz(tmp_fd, out_data / "frame_data.tar.gz")
    package_tar_gz(tmp_sp, out_data / "scene_points.tar.gz")
    package_tar_gz(tmp_rgb, out_data / "rgb_frames.tar.gz")

    shutil.rmtree(tmp_fd)
    shutil.rmtree(tmp_sp)
    shutil.rmtree(tmp_rgb)

    # ------------------------------------------------------------------
    # 8. Export frame log
    # ------------------------------------------------------------------
    included_sorted = sorted(registered_frame_nums)
    excluded_sorted = sorted(excluded_frame_nums)

    frame_log = {
        "key": key,
        "total_frames_on_disk": len(all_frame_nums),
        "included_count": len(included_sorted),
        "excluded_count": len(excluded_sorted),
        "included_frames": included_sorted,
        "excluded_frames": excluded_sorted,
    }
    log_path = out_kf / "frame_log.json"
    with open(log_path, "w") as f:
        json.dump(frame_log, f, indent=2)
    print(f"  Frame log:    {log_path}")

    print(f"\nDone!")
    print(f"  Intrinsics:   {intrinsics_path}")
    print(f"  Frame data:   {out_data / 'frame_data.tar.gz'}")
    print(f"  Scene points: {out_data / 'scene_points.tar.gz'}")
    print(f"  RGB frames:   {out_data / 'rgb_frames.tar.gz'}")
    print(f"  Overlay:      {overlay_path}")
    print(f"  Frame log:    {log_path}")
    print(f"  Total frames: {frame_count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert COLMAP outputs to SCARED format")
    p.add_argument("--colmap_dir", required=True,
                   help="COLMAP output root (contains <key>/sparse/ and <key>/images/)")
    p.add_argument("--original_dir",
                   default="/nfs/home/jhan3/scared_data/original")
    p.add_argument("--output_dir", default="./output",
                   help="Output directory (SCARED structure)")
    p.add_argument("--key", required=True,
                   help="x_y identifier (e.g. 1_1 for dataset_1/keyframe_1)")
    args = p.parse_args()
    main(args)