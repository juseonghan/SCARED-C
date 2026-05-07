# main.py 
import argparse 
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image 
import cv2 

import torch 
import torch.nn.functional as F 

from disparity_estimator import StereoDisparityEstimator
from dataset import SCAREDStereoDataset
import geometry 

VIS_FREQ = 1000

def save_depth_overlay_comparison(
    rgb: np.ndarray,         # (H, W, 3) uint8
    pred_depth: np.ndarray,  # (H, W)
    gt_depth: np.ndarray,    # (H, W)
    valid: np.ndarray,       # (H, W) bool
    save_path: str,
    alpha: float = 0.7,
):
    """Save side-by-side pred vs GT depth overlay on RGB."""
    def overlay(image, depth, mask):
        if mask.sum() == 0:
            return image
        lower, upper = np.percentile(depth[mask], [1, 99])
        normed = np.clip(depth, lower, upper)
        if upper > lower:
            normed = (normed - lower) / (upper - lower)
            normed[~mask] = 0.0
        else:
            normed = np.zeros_like(depth)
        colored = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        out = image.copy()
        out[mask] = (alpha * colored[mask] + (1 - alpha) * image[mask]).astype(np.uint8)
        return out

    # Resize RGB to match depth resolution if needed
    dh, dw = pred_depth.shape[:2]
    if rgb.shape[:2] != (dh, dw):
        rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)

    gt_valid = valid & (gt_depth > 0)
    pred_valid = valid & (pred_depth > 0)

    pred_vis = overlay(rgb, pred_depth, pred_valid)
    gt_vis = overlay(rgb, gt_depth, gt_valid)

    # Side-by-side: GT left, Pred right
    combined = np.concatenate([gt_vis, pred_vis], axis=1)

    # Add labels
    cv2.putText(combined, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined, "Pred", (dw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    Image.fromarray(combined).save(save_path)

def build_valid_mask(gt_full, scale):
    """Downsample GT and build conservative valid mask via nearest-neighbor."""
    gt_tensor = torch.from_numpy(gt_full).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    valid_orig = (np.isfinite(gt_full) & (gt_full > 0))
    valid_tensor = torch.from_numpy(valid_orig.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    valid_down = F.interpolate(valid_tensor, scale_factor=scale, mode='nearest')

    gt_safe = gt_tensor.clone()
    gt_safe[torch.isnan(gt_safe)] = 0.0
    gt_down = F.interpolate(gt_safe, scale_factor=scale, mode='bilinear', align_corners=False)

    gt_np = gt_down.squeeze().numpy().astype(np.float64)
    valid = (valid_down.squeeze().numpy() > 0) & (gt_np > 0)
    return gt_np, valid


def compute_disp_metrics(pred_disp, gt_disp, valid):
    """Compute disparity metrics for a single image. All values in pixel units."""
    p = pred_disp[valid]
    g = gt_disp[valid]
    n = valid.sum()
    if n == 0:
        return None

    abs_err = np.abs(p - g)
    sq_err = (p - g) ** 2
    mean_err = abs_err.mean()
    mean_sq = sq_err.mean()

    return {
        "epe": float(mean_err),
        "rmse": float(np.sqrt(mean_sq)),
        "std": float(np.sqrt(max(mean_sq - mean_err ** 2, 0.0))),
        "bad_1": float((abs_err > np.maximum(1, 0.05 * g)).mean()) * 100.0,
        "bad_2": float((abs_err > np.maximum(2, 0.05 * g)).mean()) * 100.0,
        "bad_3": float((abs_err > np.maximum(3, 0.05 * g)).mean()) * 100.0,
        "n_valid": int(n),
        "_abs_err": abs_err,  # kept for accumulation, stripped before JSON
        "_sq_err": sq_err,
        "disp_bad_1_count": int((abs_err > np.maximum(1, 0.05 * g)).sum()),
        "disp_bad_2_count": int((abs_err > np.maximum(2, 0.05 * g)).sum()),
        "disp_bad_3_count": int((abs_err > np.maximum(3, 0.05 * g)).sum()),
    }


def compute_depth_metrics(pred_depth, gt_depth, valid):
    """Compute depth metrics for a single image. Units match GT (typically mm)."""
    mask = valid & (pred_depth > 0)
    p = pred_depth[mask]
    g = gt_depth[mask]
    n = mask.sum()
    if n == 0:
        return None

    abs_err = np.abs(p - g)
    sq_err = (p - g) ** 2
    ratio = np.maximum(p / g, g / p)

    return {
        "abs_rel": float((abs_err / g).mean()),
        "sq_rel": float((sq_err / g).mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "rmse_log": float(np.sqrt(((np.log(p) - np.log(g)) ** 2).mean())),
        "mae": float(abs_err.mean()),
        "median": float(np.median(abs_err)),
        "delta_1": float((ratio < 1.25).mean()) * 100.0,
        "delta_2": float((ratio < 1.25 ** 2).mean()) * 100.0,
        "delta_3": float((ratio < 1.25 ** 3).mean()) * 100.0,
        "n_valid": int(n),
        "_abs_err": abs_err,
        "_sq_err": sq_err,
        "_abs_rel_arr": abs_err / g,
        "_sq_rel_arr": sq_err / g,
        "_log_sq_arr": (np.log(p) - np.log(g)) ** 2,
        "_delta_1_arr": (ratio < 1.25),
        "_delta_2_arr": (ratio < 1.25 ** 2),
        "_delta_3_arr": (ratio < 1.25 ** 3),
    }


class MetricsAccumulator:
    def __init__(self):
        # Disparity – running sums
        self.disp_sum_abs = 0.0
        self.disp_sum_sq = 0.0
        self.disp_bad_1 = 0
        self.disp_bad_2 = 0
        self.disp_bad_3 = 0
        self.disp_total = 0

        # Depth – running sums
        self.depth_sum_abs = 0.0
        self.depth_sum_sq = 0.0
        self.depth_sum_abs_rel = 0.0
        self.depth_sum_sq_rel = 0.0
        self.depth_sum_log_sq = 0.0
        self.depth_delta_1 = 0
        self.depth_delta_2 = 0
        self.depth_delta_3 = 0
        self.depth_total = 0

        self.per_image = []
        self.total_samples = 0

    def add_disp(self, m):
        n = m["n_valid"]
        self.disp_sum_abs += m["_abs_err"].sum()
        self.disp_sum_sq += m["_sq_err"].sum()
        self.disp_bad_1 += m["disp_bad_1_count"]
        self.disp_bad_2 += m["disp_bad_2_count"]
        self.disp_bad_3 += m["disp_bad_3_count"]
        self.disp_total += n

    def add_depth(self, m):
        n = m["n_valid"]
        self.depth_sum_abs += m["_abs_err"].sum()
        self.depth_sum_sq += m["_sq_err"].sum()
        self.depth_sum_abs_rel += m["_abs_rel_arr"].sum()
        self.depth_sum_sq_rel += m["_sq_rel_arr"].sum()
        self.depth_sum_log_sq += m["_log_sq_arr"].sum()
        self.depth_delta_1 += m["_delta_1_arr"].sum()
        self.depth_delta_2 += m["_delta_2_arr"].sum()
        self.depth_delta_3 += m["_delta_3_arr"].sum()
        self.depth_total += n

    def add_image(self, name, disp_m, depth_m):
        entry = {"name": name}
        if disp_m is not None:
            entry["disp"] = {k: v for k, v in disp_m.items() if not k.startswith("_")}
            self.add_disp(disp_m)
        if depth_m is not None:
            entry["depth"] = {k: v for k, v in depth_m.items() if not k.startswith("_")}
            self.add_depth(depth_m)
        self.per_image.append(entry)
        self.total_samples += 1

    def aggregate(self):
        results = {"per_image": self.per_image, "total_samples": self.total_samples}

        if self.disp_total > 0:
            mean_err = self.disp_sum_abs / self.disp_total
            mean_sq = self.disp_sum_sq / self.disp_total

            total_bad_1 = sum(m["disp"]["bad_1"] / 100.0 * m["disp"]["n_valid"] for m in self.per_image if "disp" in m)
            total_bad_2 = sum(m["disp"]["bad_2"] / 100.0 * m["disp"]["n_valid"] for m in self.per_image if "disp" in m)
            total_bad_3 = sum(m["disp"]["bad_3"] / 100.0 * m["disp"]["n_valid"] for m in self.per_image if "disp" in m)

            results["disparity"] = {
                "epe": float(mean_err),
                "rmse": float(np.sqrt(mean_sq)),
                "std": float(np.sqrt(max(mean_sq - mean_err ** 2, 0.0))),
                "bad_1": float(total_bad_1 / self.disp_total) * 100.0,
                "bad_2": float(total_bad_2 / self.disp_total) * 100.0,
                "bad_3": float(total_bad_3 / self.disp_total) * 100.0,
                "total_valid_pixels": int(self.disp_total),
            }

        if self.depth_total > 0:
            mean_abs_rel = self.depth_sum_abs_rel / self.depth_total
            mean_sq_rel = self.depth_sum_sq_rel / self.depth_total
            mean_sq = self.depth_sum_sq / self.depth_total
            mean_sq_log = self.depth_sum_log_sq / self.depth_total 
            mean_abs = self.depth_sum_abs / self.depth_total 
            mean_d1 = self.depth_delta_1 / self.depth_total 
            mean_d2 = self.depth_delta_2 / self.depth_total 
            mean_d3 = self.depth_delta_3 / self.depth_total 

            results["depth"] = {
                "abs_rel": float(mean_abs_rel),
                "sq_rel": float(mean_sq_rel),
                "rmse": float(np.sqrt(mean_sq)),
                "rmse_log": float(np.sqrt(mean_sq_log)),
                "mae": float(mean_abs),
                "delta_1": float(mean_d1) * 100.0,
                "delta_2": float(mean_d2) * 100.0,
                "delta_3": float(mean_d3) * 100.0,
                "total_valid_pixels": int(self.depth_total),
            }

        return results


# ============================================================
#  Rectification cache
# ============================================================

def build_rect_cache(calib: dict, image_size: tuple) -> dict:
    """
    Compute rectification maps and geometry matrices from stereo_calib.json.
    Cached per sequence (keyframe) since all frames share the same calibration.

    Args:
        calib: raw stereo_calib.json dict
        image_size: (H, W) of the images

    Returns dict with:
        RT, P1, P2, K1_rect, fx, baseline, left_map, right_map
    """
    # GT projection geometry (from pre-computed rectification in calib)
    Rot = np.array(calib['R1']['data']).reshape(3, 3)
    RT = geometry.create_RT(R=Rot)
    P1 = np.array(calib['P1']['data']).reshape(3, 4)
    P2 = np.array(calib['P2']['data']).reshape(3, 4)
    K1_rect = P1[:, :3]
    fx = calib['K1']['data'][0]
    baseline = -calib['T']['data'][0]

    # Image rectification maps (recomputed with alpha=-1 for full coverage)
    K1_raw = np.array(calib['K1']['data']).reshape(3, 3)
    K2_raw = np.array(calib['K2']['data']).reshape(3, 3)
    D1 = np.array(calib.get('D1', {}).get('data', [0, 0, 0, 0, 0])).astype(np.float64)
    D2 = np.array(calib.get('D2', {}).get('data', [0, 0, 0, 0, 0])).astype(np.float64)
    R = np.array(calib['R']['data']).reshape(3, 3).astype(np.float64)
    T = np.array(calib['T']['data']).reshape(3, 1).astype(np.float64)

    H, W = image_size
    R1_rect, R2_rect, P1_rect, P2_rect, Q, roi1, roi2 = cv2.stereoRectify(
        K1_raw, D1, K2_raw, D2, (W, H), R, T, alpha=-1,
    )
    left_map = cv2.initUndistortRectifyMap(
        K1_raw, D1, R1_rect, P1_rect, (W, H), cv2.CV_32FC1,
    )
    right_map = cv2.initUndistortRectifyMap(
        K2_raw, D2, R2_rect, P2_rect, (W, H), cv2.CV_32FC1,
    )

    return {
        "RT": RT,
        "P1": P1,
        "P2": P2,
        "K1_rect": K1_rect,
        "fx": fx,
        "baseline": baseline,
        "left_map": left_map,
        "right_map": right_map,
    }


# ============================================================
#  Evaluation
# ============================================================

def evaluate_corrected(dataset, stereo_model, scale=1.0, batch_size=1, debug=False, output_dir=None):
    accum = MetricsAccumulator()

    vis_dir = None
    if output_dir is not None:
        vis_dir = Path(output_dir) / "visualizations" / "corrected"
        vis_dir.mkdir(exist_ok=True, parents=True)

    # Rectification cache keyed by "ds_kf"
    rect_cache = {}

    N = len(dataset)
    if debug:
        N = min(N, 10)

    with torch.no_grad():
        for batch_start in tqdm(range(0, N, batch_size), desc="Evaluating corrected"):
            batch_end = min(batch_start + batch_size, N)
            B = batch_end - batch_start

            imgL_list, imgR_list = [], []
            gt_disp_list, gt_depth_list = [], []
            names = []
            cache_entries = []

            for i in range(batch_start, batch_end):
                left_rgb, right_rgb, scene_pts, calib, frame_num = dataset[i]

                # Identify sequence for rectification caching
                seq_idx = dataset.index_map[i][0]
                seq = dataset.sequences[seq_idx]
                seq_key = f"{seq.ds}_{seq.kf}"

                # Build rectification cache on first encounter
                if seq_key not in rect_cache:
                    rect_cache[seq_key] = build_rect_cache(calib, left_rgb.shape[:2])
                rc = rect_cache[seq_key]

                # Rectify images
                imgL = cv2.remap(left_rgb, rc['left_map'][0], rc['left_map'][1], cv2.INTER_LINEAR)
                imgR = cv2.remap(right_rgb, rc['right_map'][0], rc['right_map'][1], cv2.INTER_LINEAR)

                # Transform GT scene points to rectified space -> disparity & depth
                gt_ptcloud = geometry.img3d_to_ptcloud(scene_pts)
                ptcloud_rect = geometry.transform_pts(gt_ptcloud, rc['RT'])
                gt_disp = geometry.ptcloud_to_disparity(ptcloud_rect, rc['P1'], rc['P2'], imgL.shape[:2])
                gt_depth = geometry.ptcloud_to_depthmap(ptcloud_rect, rc['K1_rect'], np.zeros(5), imgL.shape[:2])
                del gt_ptcloud, ptcloud_rect

                imgL_list.append(imgL)
                imgR_list.append(imgR)
                gt_disp_list.append(gt_disp)
                gt_depth_list.append(gt_depth)
                names.append(f"dataset_{seq.ds}/keyframe_{seq.kf}/frame_{frame_num:06d}")
                cache_entries.append(rc)

            # --- Run stereo model ---
            img0 = torch.from_numpy(np.stack(imgL_list, axis=0)).cuda().float()
            img1 = torch.from_numpy(np.stack(imgR_list, axis=0)).cuda().float()
            result = stereo_model.estimate_batch(img0, img1)
            pred_disp = result["disp"].cpu().numpy().astype(np.float64)  # [B, H', W']
            del img0, img1, result

            # --- Metrics per sample ---
            for b in range(B):
                rc = cache_entries[b]
                fx_scaled = rc['fx'] * scale
                baseline = rc['baseline']

                # Disparity
                gt_d, dv = build_valid_mask(gt_disp_list[b], scale)
                gt_d = gt_d * scale
                disp_m = compute_disp_metrics(pred_disp[b], gt_d, dv)

                # Depth
                gt_z, zv = build_valid_mask(gt_depth_list[b], scale)
                pred_depth_map = np.where(
                    pred_disp[b] > 0,
                    (fx_scaled * baseline) / pred_disp[b],
                    0.0,
                )
                depth_m = compute_depth_metrics(pred_depth_map, gt_z, zv)

                accum.add_image(names[b], disp_m, depth_m)

                # Visualize
                if vis_dir is not None and accum.total_samples % VIS_FREQ == 0:
                    save_depth_overlay_comparison(
                        imgL_list[b], pred_depth_map, gt_z, zv,
                        str(vis_dir / f"{accum.total_samples:06d}.png"),
                    )

            del imgL_list, imgR_list, gt_disp_list, gt_depth_list, pred_disp

    return accum.aggregate()


# ============================================================
#  Printing
# ============================================================

def print_results(label, results):
    per = results["per_image"]

    if "disparity" in results:
        d = results["disparity"]
        print(f"\n{'='*100}")
        print(f"  {label} — DISPARITY METRICS (px)")
        print(f"{'='*100}")
        print(f"  {'Image':<35s} {'EPE':>7s} {'RMSE':>7s} {'Std':>7s} {'Bad1%':>7s} {'Bad2%':>7s} {'D1%':>7s}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for m in per:
            if "disp" not in m:
                continue
            md = m["disp"]
            print(f"  {m['name']:<35s} {md['epe']:7.2f} {md['rmse']:7.2f} {md['std']:7.2f} {md['bad_1']:7.2f} {md['bad_2']:7.2f} {md['bad_3']:7.2f}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        print(f"  {'MEAN':<35s} {d['epe']:7.2f} {d['rmse']:7.2f} {d['std']:7.2f} {d['bad_1']:7.2f} {d['bad_2']:7.2f} {d['bad_3']:7.2f}")

    if "depth" in results:
        z = results["depth"]
        print(f"\n{'='*120}")
        print(f"  {label} — DEPTH METRICS (mm)")
        print(f"{'='*120}")
        print(f"  {'Image':<35s} {'AbsRel':>7s} {'SqRel':>7s} {'RMSE':>8s} {'RMSElog':>8s} {'MAE':>8s} {'Med':>8s} {'d1':>7s} {'d2':>7s} {'d3':>7s}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for m in per:
            if "depth" not in m:
                continue
            mz = m["depth"]
            print(f"  {m['name']:<35s} {mz['abs_rel']:7.4f} {mz['sq_rel']:7.3f} {mz['rmse']:8.3f} {mz['rmse_log']:8.4f} {mz['mae']:8.3f} {mz['median']:8.3f} {mz['delta_1']:7.2f} {mz['delta_2']:7.2f} {mz['delta_3']:7.2f}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        print(f"  {'MEAN':<35s} {z['abs_rel']:7.4f} {z['sq_rel']:7.3f} {z['rmse']:8.3f} {z['rmse_log']:8.4f} {z['mae']:8.3f} {z['delta_1']:7.2f} {z['delta_2']:7.2f} {z['delta_3']:7.2f}")
        print(f"{'='*120}\n")


# ============================================================
#  Main
# ============================================================

def main(args):
    args.dataset = 'corrected'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    stereo_model = StereoDisparityEstimator(
        scale=args.scale, 
        max_batch_size=args.batch_size, 
        model_name='fast-foundationstereo', 
        ckpt_dir='/home/jhan3/endo_stereo/Fast-FoundationStereo/weights/23-36-37'
    )

    ds = SCAREDStereoDataset(keys=[
        '1_1', '1_2', '1_3',
        '2_1', '2_2', '2_3', '2_4',
        '3_1', '3_2', '3_3', '3_4',
        '6_1', '6_2', '6_3', '6_4',
        '7_1', '7_2', '7_3', 
        # '7_4',
    ])

    results = evaluate_corrected(
        ds, stereo_model,
        scale=args.scale,
        batch_size=args.batch_size,
        debug=args.debug,
        output_dir=args.output_dir,
    )

    ds.close()

    print_results(args.dataset, results)

    results_path = output_dir / f'eval_results_{args.dataset}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=1.0, help='resolution scaling factor')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--output_dir', default='/home/jhan3/scared_correction/eval/results_corrected', help='output directory')
    parser.add_argument('--debug', action='store_true', help='run on small subset')
    args = parser.parse_args()
    main(args)