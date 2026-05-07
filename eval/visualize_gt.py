"""
Create GT depth overlay videos for each keyframe sequence.

Mirrors the evaluation loop in main.py exactly, but skips the stereo model
and metrics — just overlays GT depth onto rectified left RGB and writes
a video per sequence at 15 fps.

Usage:
    python visualize_gt_depth_videos.py \
        --dataset corrected \
        --output_dir /path/to/output \
        [--alpha 0.7] \
        [--debug]
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

from dataset import OriginalSCAREDDataset, CorrectedSCAREDDataseWithRGBmp4
import geometry

import torch


# ============================================================
#  Depth overlay utility
# ============================================================

def create_depth_overlay(
    rgb: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Overlay turbo-colormapped depth onto an RGB image."""
    valid = depth > 0
    if valid.sum() == 0:
        return rgb

    lower, upper = np.percentile(depth[valid], [1, 99])
    normed = np.clip(depth, lower, upper)
    if upper > lower:
        normed = (normed - lower) / (upper - lower)
        normed[~valid] = 0.0
    else:
        normed = np.zeros_like(depth)

    colored = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    out = rgb.copy()
    out[valid] = (alpha * colored[valid] + (1 - alpha) * rgb[valid]).astype(np.uint8)
    return out


# ============================================================
#  Original dataset visualization
# ============================================================

def visualize_original(output_dir: Path, alpha: float, debug: bool):
    original_ds = OriginalSCAREDDataset()
    original_dl = torch.utils.data.DataLoader(original_ds, batch_size=1, shuffle=False)

    vis_dir = output_dir / "original"
    vis_dir.mkdir(exist_ok=True, parents=True)

    total = 0
    frames_for_video = []
    prev_seq = None

    def flush_video(seq_name, frames, out_dir):
        if not frames:
            return
        safe_name = seq_name.replace("/", "_")
        video_path = out_dir / f"{safe_name}.mp4"
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 15, (W, H))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Wrote {len(frames)} frames -> {video_path}")

    for sample in tqdm(original_dl, desc="Original"):
        for b in range(sample["img0"].shape[0]):
            name = sample["fname"][b]

            # Detect sequence boundary from name to split videos
            seq = "/".join(name.split("/")[:-1]) if "/" in name else "all"
            if prev_seq is not None and seq != prev_seq:
                flush_video(prev_seq, frames_for_video, vis_dir)
                frames_for_video = []
            prev_seq = seq

            rgb_np = sample["img0"][b].numpy().astype(np.uint8)
            gt_depth_full = sample["depth"][b].numpy().astype(np.float64)

            overlay = create_depth_overlay(rgb_np, gt_depth_full, alpha=alpha)
            frames_for_video.append(overlay)
            total += 1

            if debug and total >= 50:
                break
        if debug and total >= 50:
            break

    flush_video(prev_seq, frames_for_video, vis_dir)
    print(f"\nDone! {total} frames visualized.")


# ============================================================
#  Corrected dataset visualization
# ============================================================

def visualize_corrected(output_dir: Path, alpha: float, debug: bool):
    corrected_ds = CorrectedSCAREDDataseWithRGBmp4()

    vis_dir = output_dir / "corrected"
    vis_dir.mkdir(exist_ok=True, parents=True)

    num_seq = 1 if debug else len(corrected_ds)
    total = 0

    for i in tqdm(range(num_seq), desc="Sequences"):
        vr, tar_reader, calib = corrected_ds.get_readers(i)

        # Pre-extract geometry matrices (same as eval code)
        Rot = np.array(calib['R1']['data']).reshape((3, 3))
        RT = geometry.create_RT(R=Rot)
        P1 = np.array(calib['P1']['data']).reshape(3, 4)
        P2 = np.array(calib['P2']['data']).reshape(3, 4)
        K1 = P1[:, :3]

        # Rectification setup (computed once per sequence on first frame)
        K1_raw = np.array(calib['K1']['data']).reshape(3, 3)
        K2_raw = np.array(calib['K2']['data']).reshape(3, 3)
        D1 = np.array(calib.get('D1', {}).get('data', [0, 0, 0, 0, 0])).astype(np.float64)
        D2 = np.array(calib.get('D2', {}).get('data', [0, 0, 0, 0, 0])).astype(np.float64)
        R = np.array(calib['R']['data']).reshape(3, 3).astype(np.float64)
        T = np.array(calib['T']['data']).reshape(3, 1).astype(np.float64)
        left_rect_map = None

        num_frames = min(len(tar_reader), len(vr))
        if debug:
            num_frames = min(num_frames, 50)

        seq_name = corrected_ds.gt_tar_paths[i].split('/')
        seq_label = f"{seq_name[-4]}/{seq_name[-3]}"

        # Prepare video writer (deferred until we know frame size)
        safe_name = seq_label.replace("/", "_")
        video_path = vis_dir / f"{safe_name}.mp4"
        writer = None
        written = 0

        for j in tqdm(range(num_frames), desc=f"  {seq_label}"):
            frame = vr[j].asnumpy()
            h = frame.shape[0] // 2
            imgL = frame[:h]

            # Build rectification maps on first frame
            if left_rect_map is None:
                image_size = imgL.shape[:2]
                R1_rect, R2_rect, P1_rect, P2_rect, Q, roi1, roi2 = cv2.stereoRectify(
                    K1_raw, D1, K2_raw, D2,
                    image_size[::-1],  # (W, H) for OpenCV
                    R, T,
                    alpha=-1,
                )
                left_rect_map = cv2.initUndistortRectifyMap(
                    K1_raw, D1, R1_rect, P1_rect,
                    image_size[::-1], cv2.CV_32FC1,
                )

            imgL = cv2.remap(imgL, left_rect_map[0], left_rect_map[1], cv2.INTER_LINEAR)

            # GT depth (same pipeline as eval code)
            gt_img3d = tar_reader[j+1]
            gt_ptcloud = geometry.img3d_to_ptcloud(gt_img3d)
            ptcloud_rotated = geometry.transform_pts(gt_ptcloud, RT)
            gt_depth = geometry.ptcloud_to_depthmap(ptcloud_rotated, K1, np.zeros(5), imgL.shape[:2])
            del gt_img3d, gt_ptcloud, ptcloud_rotated

            # imgL from vr is RGB (from video decode), overlay expects RGB
            overlay = create_depth_overlay(imgL, gt_depth, alpha=alpha)

            # Init writer on first frame
            if writer is None:
                H, W = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, 15, (W, H))

            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            written += 1
            total += 1

        if writer is not None:
            writer.release()
            print(f"  Wrote {written} frames -> {video_path}")

        tar_reader.__exit__()

    print(f"\nDone! {total} total frames visualized.")


# ============================================================
#  Main
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.dataset == "original":
        visualize_original(output_dir, alpha=args.alpha, debug=args.debug)
    elif args.dataset == "corrected":
        visualize_corrected(output_dir, alpha=args.alpha, debug=args.debug)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create GT depth overlay videos for SCARED keyframe sequences"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/jhan3/scared_correction/eval/gt_depth_overleayd_vids",
        help="Where to save output videos",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="corrected",
        choices=["original", "corrected"],
        help="Which dataset to visualize",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Depth overlay blending factor (default: 0.7)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only process first 50 frames per sequence (1 seq for corrected)",
    )
    args = parser.parse_args()
    main(args)